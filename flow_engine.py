from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from statistics import mean, pstdev
from time import time
from typing import Dict, Tuple

from scapy.layers.inet import IP, TCP, UDP

FlowKey = Tuple[str, str, int, int, str]


@dataclass
class Flow:
    key: FlowKey
    first_seen: float
    last_seen: float
    forward_packets: int = 0
    reverse_packets: int = 0
    forward_bytes: int = 0
    reverse_bytes: int = 0
    forward_sizes: list[int] = field(default_factory=list)
    reverse_sizes: list[int] = field(default_factory=list)
    forward_times: list[float] = field(default_factory=list)
    reverse_times: list[float] = field(default_factory=list)
    forward_ttl: int | None = None
    reverse_ttl: int | None = None
    syn_seen: bool = False
    synack_seen: bool = False
    ack_seen: bool = False
    syn_time: float | None = None
    synack_time: float | None = None
    ack_time: float | None = None
    fin_seen: bool = False
    rst_seen: bool = False

    def add(self, ts: float, forward: bool, size: int, ttl: int | None, flags: str = "") -> None:
        self.last_seen = ts
        if forward:
            self.forward_packets += 1
            self.forward_bytes += size
            self.forward_sizes.append(size)
            self.forward_times.append(ts)
            if ttl is not None and self.forward_ttl is None:
                self.forward_ttl = ttl
        else:
            self.reverse_packets += 1
            self.reverse_bytes += size
            self.reverse_sizes.append(size)
            self.reverse_times.append(ts)
            if ttl is not None and self.reverse_ttl is None:
                self.reverse_ttl = ttl

        if "S" in flags and "A" not in flags:
            self.syn_seen = True
            self.syn_time = self.syn_time or ts
        elif "S" in flags and "A" in flags:
            self.synack_seen = True
            self.synack_time = self.synack_time or ts
        if "A" in flags:
            self.ack_seen = True
            self.ack_time = self.ack_time or ts
        self.fin_seen = self.fin_seen or "F" in flags
        self.rst_seen = self.rst_seen or "R" in flags

    def duration(self) -> float:
        return max(self.last_seen - self.first_seen, 1e-6)

    @staticmethod
    def _iat(values: list[float]) -> float:
        if len(values) < 2:
            return 0.0
        return mean(b - a for a, b in zip(values, values[1:]))

    @staticmethod
    def _jit(values: list[float]) -> float:
        if len(values) < 3:
            return 0.0
        iats = [b - a for a, b in zip(values, values[1:])]
        return pstdev(iats) if len(iats) > 1 else 0.0

    def state(self) -> str:
        proto = self.key[4]
        if proto != "tcp":
            return "CON"
        if self.rst_seen:
            return "RST"
        if self.fin_seen:
            return "FIN"
        if self.synack_seen:
            return "EST"
        if self.syn_seen:
            return "REQ"
        return "CON"

    def to_features(self, packet_count_window: int = 0) -> dict:
        src, dst, sport, dport, proto = self.key
        dur = self.duration()
        spkts = self.forward_packets
        dpkts = self.reverse_packets
        sbytes = self.forward_bytes
        dbytes = self.reverse_bytes
        total_packets = spkts + dpkts

        service = service_from_port(dport)
        rate = total_packets / dur
        sload = (sbytes * 8.0) / dur
        dload = (dbytes * 8.0) / dur

        tcprtt = (
            self.synack_time - self.syn_time
            if self.syn_time is not None and self.synack_time is not None
            else 0.0
        )
        synack = (
            self.ack_time - self.synack_time
            if self.synack_time is not None and self.ack_time is not None
            else 0.0
        )
        ackdat = (
            self.ack_time - self.syn_time
            if self.syn_time is not None and self.ack_time is not None
            else 0.0
        )

        return {
            "srcip": src,
            "dstip": dst,
            "sport": sport,
            "dsport": dport,
            "proto": proto,
            "service": service,
            "state": self.state(),
            "dur": dur,
            "spkts": spkts,
            "dpkts": dpkts,
            "sbytes": sbytes,
            "dbytes": dbytes,
            "rate": rate,
            "sttl": self.forward_ttl or 0,
            "dttl": self.reverse_ttl or 0,
            "sload": sload,
            "dload": dload,
            "sloss": 0,
            "dloss": 0,
            "sinpkt": self._iat(self.forward_times),
            "dinpkt": self._iat(self.reverse_times),
            "sjit": self._jit(self.forward_times),
            "djit": self._jit(self.reverse_times),
            "swin": 0,
            "stcpb": 0,
            "dtcpb": 0,
            "dwin": 0,
            "tcprtt": tcprtt,
            "synack": synack,
            "ackdat": ackdat,
            "smean": mean(self.forward_sizes) if self.forward_sizes else 0,
            "dmean": mean(self.reverse_sizes) if self.reverse_sizes else 0,
            "trans_depth": 0,
            "response_body_len": 0,
            "ct_srv_src": packet_count_window,
            "ct_state_ttl": 0,
            "ct_dst_ltm": 0,
            "ct_src_dport_ltm": 0,
            "ct_dst_sport_ltm": 0,
            "ct_dst_src_ltm": 0,
            "is_ftp_login": 0,
            "ct_ftp_cmd": 0,
            "ct_flw_http_mthd": 0,
            "ct_src_ltm": 0,
            "ct_srv_dst": 0,
            "is_sm_ips_ports": int(src == dst and sport == dport),
        }


def service_from_port(port: int) -> str:
    mapping = {
        20: "ftp-data", 21: "ftp", 22: "ssh", 23: "telnet", 25: "smtp",
        53: "dns", 67: "dhcp", 68: "dhcp", 80: "http", 110: "pop3",
        123: "ntp", 143: "imap", 443: "https", 445: "smb", 514: "shell",
        3306: "mysql", 3389: "rdp", 8080: "http",
    }
    return mapping.get(int(port), "-")


class FlowTracker:
    def __init__(self, idle_timeout: float = 15.0, active_emit: float = 3.0) -> None:
        self.idle_timeout = idle_timeout
        self.active_emit = active_emit
        self.flows: Dict[FlowKey, Flow] = {}
        self.last_emitted: Dict[FlowKey, float] = defaultdict(float)
        self.closed: list[Flow] = []

    # IMPORTANT: this must be an instance method because it uses self.flows.
    # The old file incorrectly declared it @staticmethod.
    def _key(self, packet) -> tuple[FlowKey, bool] | None:
        if IP not in packet:
            return None

        ip = packet[IP]

        if TCP in packet:
            proto = "tcp"
            layer = packet[TCP]
        elif UDP in packet:
            proto = "udp"
            layer = packet[UDP]
        else:
            proto = str(ip.proto)
            return ((ip.src, ip.dst, 0, 0, proto), True)

        forward_key = (
            ip.src,
            ip.dst,
            int(layer.sport),
            int(layer.dport),
            proto,
        )
        reverse_key = (
            ip.dst,
            ip.src,
            int(layer.dport),
            int(layer.sport),
            proto,
        )

        if forward_key in self.flows:
            return forward_key, True

        if reverse_key in self.flows:
            return reverse_key, False

        return forward_key, True

    def add_packet(self, packet, ts: float | None = None) -> None:
        ts = time() if ts is None else ts

        result = self._key(packet)
        if result is None:
            return

        key, forward = result

        if key not in self.flows:
            self.flows[key] = Flow(
                key=key,
                first_seen=ts,
                last_seen=ts,
            )

        flags = str(packet[TCP].flags) if TCP in packet else ""
        ttl = (
            int(packet[IP].ttl)
            if IP in packet and hasattr(packet[IP], "ttl")
            else None
        )

        self.flows[key].add(
            ts,
            forward,
            len(packet),
            ttl,
            flags,
        )

    def emit_ready(self, now: float | None = None) -> list[Flow]:
        now = time() if now is None else now
        ready: list[Flow] = []

        for key, flow in list(self.flows.items()):
            idle = now - flow.last_seen
            active = now - self.last_emitted[key]

            closed = (
                flow.fin_seen
                or flow.rst_seen
                or idle >= self.idle_timeout
            )

            if closed or active >= self.active_emit:
                ready.append(flow)
                self.last_emitted[key] = now

                if closed:
                    self.closed.append(flow)
                    self.flows.pop(key, None)
                    self.last_emitted.pop(key, None)

        return ready

    def active_count(self) -> int:
        return len(self.flows)
