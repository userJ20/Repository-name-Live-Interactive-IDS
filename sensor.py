from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scapy.utils import PcapReader

from flow_engine import FlowTracker
from model_io import load_models

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"
ALERTS_FILE = DATA_DIR / "live_alerts.jsonl"
STATE_FILE = DATA_DIR / "live_state.json"

INTERFACE = os.getenv("IDS_INTERFACE", "").strip()
PCAP_FILE = os.getenv("PCAP_FILE", "").strip()
IDLE_TIMEOUT = float(os.getenv("FLOW_IDLE_TIMEOUT", "15"))
EMIT_SECONDS = float(os.getenv("FLOW_EMIT_SECONDS", "3"))
FLUSH_SECONDS = float(os.getenv("STATE_FLUSH_SECONDS", "1"))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(record: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with ALERTS_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, separators=(",", ":")) + "\n")


def write_state(state: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(STATE_FILE)


def auto_interface() -> str:
    """Find the Linux interface used for the default route."""
    try:
        result = subprocess.run(
            ["ip", "route", "show", "default"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        for line in result.stdout.splitlines():
            parts = line.split()
            if "dev" in parts:
                i = parts.index("dev")
                if i + 1 < len(parts):
                    return parts[i + 1]
    except Exception:
        pass

    # Docker/Linux fallback.
    for candidate in ("eth0", "ens3", "enp0s3"):
        try:
            socket.if_nametoindex(candidate)
            return candidate
        except OSError:
            continue

    return ""


def process_packet(packet, tracker, preprocessor, rf, feature_names, counters, state):
    counters["packets_seen"] += 1

    try:
        tracker.add_packet(packet)
    except Exception as exc:
        counters["packet_errors"] += 1
        state["last_error"] = f"flow processing: {type(exc).__name__}: {exc}"
        return

    now = time.time()

    try:
        ready = tracker.emit_ready(now)
    except Exception as exc:
        counters["flow_errors"] += 1
        state["last_error"] = f"flow emission: {type(exc).__name__}: {exc}"
        return

    for flow in ready:
        try:
            raw = flow.to_features()
            row = pd.DataFrame([raw])

            for col in feature_names:
                if col not in row.columns:
                    row[col] = np.nan

            X = row[feature_names]
            Xp = np.asarray(preprocessor.transform(X), dtype=np.float32)

            pred = int(rf.predict(Xp)[0])
            prob = float(rf.predict_proba(Xp)[0, 1])

            if pred == 1 and prob >= 0.90:
                severity = "CRITICAL"
            elif pred == 1 and prob >= 0.70:
                severity = "HIGH"
            elif pred == 1 and prob >= 0.50:
                severity = "MEDIUM"
            else:
                severity = "LOW"

            record = {
                "timestamp": utc_now(),
                "srcip": raw["srcip"],
                "dstip": raw["dstip"],
                "sport": raw["sport"],
                "dsport": raw["dsport"],
                "proto": raw["proto"],
                "state": raw["state"],
                "prediction": pred,
                "class_name": "Attack" if pred else "Normal",
                "attack_probability": prob,
                "severity": severity if pred else "NORMAL",
                "duration": raw["dur"],
                "spkts": raw["spkts"],
                "dpkts": raw["dpkts"],
                "sbytes": raw["sbytes"],
                "dbytes": raw["dbytes"],
            }

            append_jsonl(record)

            counters["flows_scored"] += 1
            counters["attacks"] += pred
            counters["normal"] += 1 - pred
            counters["alerts"] += int(pred == 1)
            counters["last_flow_at"] = record["timestamp"]

        except Exception as exc:
            counters["prediction_errors"] += 1
            state["last_error"] = f"prediction: {type(exc).__name__}: {exc}"

    state.update(counters)
    state["active_flows"] = tracker.active_count()
    state["last_packet_at"] = utc_now()


def capture_pcap_file(path, callback):
    print(f"[sensor] Reading PCAP file: {path}", flush=True)
    with PcapReader(path) as reader:
        for packet in reader:
            callback(packet)


def capture_live(interface, callback):
    """
    Capture through tcpdump/libpcap instead of Scapy's Linux L2ListenSocket.

    This avoids the L2ListenSocket failure seen in the original sensor and
    also makes the capture path easier to diagnose inside Docker.
    """
    tcpdump = shutil.which("tcpdump")
    if not tcpdump:
        raise RuntimeError(
            "tcpdump is not installed in the sensor image. "
            "The Dockerfile must install tcpdump."
        )

    if not interface:
        raise RuntimeError(
            "No capture interface was found. Set IDS_INTERFACE, e.g. IDS_INTERFACE=eth0."
        )

    try:
        socket.if_nametoindex(interface)
    except OSError as exc:
        raise RuntimeError(
            f"Interface '{interface}' is not visible inside the sensor container."
        ) from exc

    cmd = [
        tcpdump,
        "-i", interface,
        "-U",
        "-n",
        "-s", "0",
        "-w", "-",
        "ip",
    ]

    print("[sensor] Starting capture:", " ".join(cmd), flush=True)

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )

    assert proc.stdout is not None
    assert proc.stderr is not None

    # Drain tcpdump diagnostics so stderr cannot fill and block the process.
    import threading

    def drain_stderr():
        for raw_line in iter(proc.stderr.readline, b""):
            line = raw_line.decode("utf-8", errors="replace").strip()
            if line:
                print(f"[tcpdump] {line}", flush=True)

    threading.Thread(target=drain_stderr, daemon=True).start()

    try:
        with PcapReader(proc.stdout) as reader:
            for packet in reader:
                callback(packet)
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    loaded = load_models()
    if loaded is None:
        raise SystemExit(
            "No trained model found. Train the IDS from the dashboard first."
        )

    preprocessor, rf, metadata = loaded
    feature_names = metadata.get("feature_names", [])

    if not feature_names:
        raise SystemExit("Model metadata does not contain feature_names.")

    interface = INTERFACE or auto_interface()

    tracker = FlowTracker(
        idle_timeout=IDLE_TIMEOUT,
        active_emit=EMIT_SECONDS,
    )

    counters = {
        "started_at": utc_now(),
        "packets_seen": 0,
        "flows_scored": 0,
        "attacks": 0,
        "normal": 0,
        "alerts": 0,
        "packet_errors": 0,
        "flow_errors": 0,
        "prediction_errors": 0,
        "interface": interface or "unknown",
        "model_trained_at": metadata.get("trained_at"),
    }

    state = {
        **counters,
        "active_flows": 0,
        "last_packet_at": None,
        "last_flow_at": None,
        "sensor_status": "STARTING",
        "last_error": None,
    }

    write_state(state)

    last_state = 0.0

    def on_packet(packet):
        nonlocal last_state

        process_packet(
            packet,
            tracker,
            preprocessor,
            rf,
            feature_names,
            counters,
            state,
        )

        now = time.time()
        state["active_flows"] = tracker.active_count()

        if now - last_state >= FLUSH_SECONDS:
            state["sensor_status"] = "ONLINE"
            state.update(counters)
            write_state(state)
            last_state = now

    print("[sensor] Model loaded successfully.", flush=True)
    print(f"[sensor] Capture interface: {interface or 'NONE'}", flush=True)

    state["sensor_status"] = "ONLINE"
    write_state(state)

    if PCAP_FILE:
        capture_pcap_file(PCAP_FILE, on_packet)
    else:
        capture_live(interface, on_packet)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("[sensor] Stopped.", flush=True)
    except Exception as exc:
        print(f"[sensor] FATAL: {type(exc).__name__}: {exc}", flush=True)
        try:
            state = {
                "sensor_status": "OFFLINE",
                "last_error": f"{type(exc).__name__}: {exc}",
                "timestamp": utc_now(),
            }
            write_state(state)
        except Exception:
            pass
        raise
