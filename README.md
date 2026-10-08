# Live Adversarial IDS

An interactive, Dockerized Intrusion Detection System (IDS) and adversarial ML laboratory for network-traffic analysis.

The project combines:

- Live packet capture
- Network-flow aggregation
- UNSW-NB15-style feature engineering
- Random Forest intrusion detection
- Live threat/alert visualization
- Offline traffic analysis
- Adversarial robustness experiments using FGSM and PGD
- Docker-based deployment

> **Important:** This is a research/educational prototype. A model prediction of `Attack` is not proof that a real malicious action occurred. Live feature generation is an approximation of the UNSW-NB15 feature space and can produce false positives, especially for internal/container traffic.

---

## 1. Project Workflow

```text
Network Traffic
      │
      ▼
Packet Capture
(tcpdump / PCAP)
      │
      ▼
FlowTracker
      │
      ▼
Flow Features
(UNSW-NB15-style)
      │
      ▼
Preprocessor
      │
      ▼
Random Forest
      │
      ├──────────────► Normal
      │
      └──────────────► Attack
                         │
                         ▼
                   Probability
                         │
              ┌──────────┼──────────┐
              ▼          ▼          ▼
           MEDIUM      HIGH      CRITICAL
```

The Streamlit dashboard displays the resulting flow predictions and operational statistics.

---

# 2. Main Features

## Live Network Sensor

The Live Sensor provides real-time network visibility:

- Packet capture
- Automatic interface detection
- Flow tracking
- Bidirectional flow aggregation
- Flow feature extraction
- Random Forest inference
- Attack probability
- Severity classification
- Live threat feed
- Persistent JSONL alert records
- Persistent sensor state

The sensor now uses a `tcpdump` → PCAP stream → Scapy `PcapReader` pipeline for live capture instead of relying on Scapy's live `sniff()` socket.

This was introduced to avoid the previous Linux `L2ListenSocket` capture failure.

### Live capture pipeline

```text
tcpdump
   │
   │ PCAP stream
   ▼
PcapReader
   │
   ▼
FlowTracker
   │
   ▼
ML inference
```

---

# 3. Live Sensor Dashboard

The Live Sensor page reports:

### Packets Seen

Number of packets successfully received by the sensor.

### Active Flows

Flows currently being tracked by the flow engine.

### Flows Scored

Number of flows that have been emitted from the flow tracker and passed to the trained ML model.

### Attacks

Number of scored flows for which the Random Forest predicted:

```text
1 = Attack
```

The attack percentage shown on the dashboard is:

```text
attacks / flows scored × 100
```

---

# 4. How an Attack Is Determined

The live IDS does **not** use a simple rule such as:

```text
port 80 = normal
port 2375 = attack
```

Instead, each flow is converted into a feature vector and passed through the trained Random Forest.

The model performs:

```python
pred = rf.predict(Xp)
prob = rf.predict_proba(Xp)
```

The binary classification is:

```text
0 → Normal
1 → Attack
```

The attack probability is the Random Forest probability for class `1`.

Therefore:

> An `Attack` shown by the dashboard means that the trained model classified that observed flow as class 1.

It does not independently prove that the traffic is malicious.

---

# 5. Severity Levels

For predicted attacks, the sensor uses the following confidence thresholds:

| Attack Probability | Severity |
|---:|---|
| < 50% | Not classified as Attack by the binary prediction |
| 50–69.9% | MEDIUM |
| 70–89.9% | HIGH |
| ≥ 90% | CRITICAL |

Normal predictions are displayed as:

```text
NORMAL
```

The dashboard can therefore distinguish between a model prediction and the confidence associated with that prediction.

---

# 6. Flow Aggregation

The `FlowTracker` groups packets into bidirectional network flows using:

- Source IP
- Destination IP
- Source port
- Destination port
- Protocol

Supported packet protocols include:

- TCP
- UDP
- Other IP protocols

For TCP traffic, the flow engine also tracks information such as:

- SYN
- SYN/ACK
- ACK
- FIN
- RST
- TCP timing information

Flows can be emitted when:

- A flow becomes inactive for the configured idle timeout
- A TCP flow closes
- The active emission interval is reached

Default configuration:

```text
FLOW_IDLE_TIMEOUT=15
FLOW_EMIT_SECONDS=3
```

---

# 7. Live Feature Engineering

The flow engine generates UNSW-NB15-style features including:

- `srcip`
- `dstip`
- `sport`
- `dsport`
- `proto`
- `service`
- `state`
- `dur`
- `spkts`
- `dpkts`
- `sbytes`
- `dbytes`
- `rate`
- `sttl`
- `dttl`
- `sload`
- `dload`
- `sloss`
- `dloss`
- `sinpkt`
- `dinpkt`
- `sjit`
- `djit`
- `swin`
- `stcpb`
- `dtcpb`
- `dwin`
- `tcprtt`
- `synack`
- `ackdat`
- `smean`
- `dmean`
- `trans_depth`
- `response_body_len`
- `ct_*` statistical fields
- FTP/HTTP-related counters
- `is_sm_ips_ports`

Some UNSW-NB15 fields cannot be reliably inferred from a short live observation. Those fields therefore use neutral/default values.

---

# 8. Important Live-Model Limitation

The model is trained on a structured network-flow dataset such as UNSW-NB15.

Live traffic is generated from packets and therefore has to be converted into an approximate version of the same feature space.

This creates a possible distribution mismatch:

```text
UNSW-NB15 training features
          ↓
      Random Forest
          ↑
          │
Approximate live features
          ↑
     Packet capture
```

As a result, ordinary traffic can sometimes be classified as an attack.

For example, internal Docker/container traffic may look unusual to a model trained primarily on benchmark network-flow data.

Therefore:

> A high live attack count should be investigated rather than automatically interpreted as evidence of a real attack.

This limitation is particularly important when demonstrating the system.

---

# 9. Traffic Analyzer

The Traffic Analyzer allows an unseen network-traffic CSV to be passed through the trained IDS.

It provides:

- Total traffic flows
- Normal predictions
- Attack predictions
- Attack probability
- Severity
- Row-level predictions
- Investigation filters
- Attack categories when available
- Top alert sources
- Confidence distribution
- Confusion matrix when ground truth is present
- Downloadable prediction CSV

The analyzer supports a separate test dataset from the training dataset.

Recommended workflow:

```text
Training CSV
    ↓
Train IDS
    ↓
Separate test/unseen CSV
    ↓
Traffic Analyzer
    ↓
Predictions
```

---

# 10. Adversarial Lab

The Adversarial Lab is designed to evaluate robustness of the differentiable neural IDS.

It supports:

- FGSM
- PGD
- Feature-space perturbations
- Attack → Normal evasion experiments
- Evasion history
- Comparison of original and perturbed predictions

The adversarial laboratory operates in **model feature space**.

It does **not** automatically generate real malicious packets.

Therefore, an adversarial result should be interpreted as:

> The model was successfully manipulated in feature space.

It should not be interpreted as proof that the same perturbation can be directly translated into valid network packets.

---

# 11. Machine Learning Pipeline

The training workflow is:

```text
Labelled CSV
     ↓
Target detection
     ↓
Binary labels
     ↓
Feature/target separation
     ↓
Numeric preprocessing
     ↓
Categorical encoding
     ↓
Scaling
     ↓
Random Forest
     ↓
PyTorch neural IDS
```

For the UNSW-NB15 binary experiment:

```text
label = 0 → Normal
label = 1 → Attack
```

The application automatically handles numerical and categorical columns using preprocessing, imputation, encoding and scaling.

The Random Forest uses:

```text
n_estimators = 200
random_state = 42
class_weight = balanced_subsample
n_jobs = -1
```

---

# 12. Docker Architecture

The application uses two main services:

```text
┌──────────────────────────────┐
│        Dashboard             │
│      Streamlit :8501         │
│                              │
│  Dashboard                   │
│  Live Sensor                 │
│  Train Model                 │
│  Traffic Analyzer            │
│  Adversarial Lab             │
└──────────────┬───────────────┘
               │
               │ shared data/models
               │
┌──────────────▼───────────────┐
│          Sensor              │
│                              │
│ tcpdump → PcapReader         │
│       → FlowTracker          │
│       → Random Forest        │
│       → live_alerts.jsonl    │
│       → live_state.json      │
└──────────────────────────────┘
```

---

# 13. Docker Services

## Dashboard

```text
Container: live-ids-dashboard
Port: 8501
```

Open:

```text
http://localhost:8501
```

## Sensor

```text
Container: live-ids-sensor
```

The sensor uses host networking and requires:

```text
NET_RAW
NET_ADMIN
```

for packet capture.

---

# 14. Starting the Project

From the project directory:

```bash
docker compose up -d --build
```

Check the services:

```bash
docker compose ps
```

Expected:

```text
live-ids-dashboard   Up
live-ids-sensor      Up
```

Open:

```text
http://localhost:8501
```

---

# 15. Stopping the Project

To stop and remove the running containers:

```bash
docker compose down
```

To temporarily stop the containers without removing them:

```bash
docker compose stop
```

Start them again:

```bash
docker compose up -d
```

If `docker compose up` is running in the foreground, press:

```text
Ctrl + C
```

---

# 16. Sensor Logs

View recent sensor logs:

```bash
docker compose logs --tail=100 sensor
```

A healthy startup should contain messages similar to:

```text
[sensor] Model loaded successfully.
[sensor] Capture interface: eth0
[sensor] Starting capture: /usr/bin/tcpdump -i eth0 ...
[tcpdump] tcpdump: listening on eth0
```

---

# 17. Sensor State

The sensor writes its current state to:

```text
data/live_state.json
```

Inspect it with:

```bash
docker compose exec sensor cat /app/data/live_state.json
```

Important fields include:

```text
packets_seen
flows_scored
attacks
normal
alerts
packet_errors
flow_errors
prediction_errors
interface
active_flows
last_packet_at
last_flow_at
sensor_status
last_error
```

A healthy sensor should have:

```text
packet_errors = 0
flow_errors = 0
prediction_errors = 0
```

---

# 18. Live Alert Data

Predicted flows are written to:

```text
data/live_alerts.jsonl
```

Check the number of records:

```bash
docker compose exec sensor sh -c "wc -l /app/data/live_alerts.jsonl"
```

View the latest records:

```bash
docker compose exec sensor sh -c "tail -5 /app/data/live_alerts.jsonl"
```

Each record can contain information such as:

```text
timestamp
srcip
dstip
sport
dsport
proto
state
prediction
class_name
attack_probability
severity
duration
spkts
dpkts
sbytes
dbytes
```

---

# 19. Testing Network Connectivity

The sensor container may not contain `curl`.

A simple Python connectivity test is:

```bash
docker compose exec sensor python -c "import urllib.request; print(urllib.request.urlopen('https://example.com').status)"
```

Expected:

```text
200
```

A TCP connectivity test:

```bash
docker compose exec sensor python -c "import socket; s=socket.create_connection(('8.8.8.8',53),5); print('connection successful'); s.close()"
```

---

# 20. Flow Engine Validation

The flow engine can be tested independently of live capture:

```bash
docker compose exec sensor python -c "from scapy.layers.inet import IP,TCP; from flow_engine import FlowTracker; p=IP(src='10.0.0.1',dst='10.0.0.2')/TCP(sport=12345,dport=80,flags='S'); t=FlowTracker(); t.add_packet(p); print('FLOW TEST OK:', t.active_count())"
```

Expected:

```text
FLOW TEST OK: 1
```

This confirms that packet → flow aggregation is functioning.

---

# 21. Important FlowTracker Fix

The live sensor previously reported:

```text
NameError: name 'self' is not defined
```

The cause was an incorrect declaration of `FlowTracker._key()` as a static method even though it accessed:

```python
self.flows
```

The method is now an instance method:

```python
def _key(self, packet):
```

This allows the flow tracker to correctly access its internal flow dictionary.

The corrected pipeline is:

```text
packet
  ↓
FlowTracker.add_packet()
  ↓
FlowTracker._key()
  ↓
self.flows
  ↓
Flow
```

---

# 22. Troubleshooting

## Sensor is not running

Check:

```bash
docker compose ps
```

Then:

```bash
docker compose logs --tail=100 sensor
```

Restart:

```bash
docker compose up -d --build
```

---

## `Flows Scored = 0`

Check:

```bash
docker compose exec sensor cat /app/data/live_state.json
```

Look especially at:

```text
packet_errors
flow_errors
prediction_errors
last_error
```

If `packet_errors` is increasing, inspect `last_error`.

---

## `Packets Seen` increases but `Flows Scored` stays at 0

This indicates that packet capture is functioning but flow creation/emission or ML processing needs investigation.

Check:

```bash
docker compose exec sensor cat /app/data/live_state.json
```

Then run the FlowTracker validation test described above.

---

## `Attacks` is unexpectedly high

Do not immediately conclude that the network is under attack.

Check:

1. Attack probability
2. Source/destination addresses
3. Protocol
4. Ports
5. Whether traffic is Docker/internal traffic
6. Whether the live feature distribution differs from the training dataset
7. Model performance on a labelled test dataset

Use the Traffic Analyzer with a known test dataset to evaluate the model independently.

---

# 23. macOS / Docker Networking

When running Docker Desktop on macOS, the sensor is running inside Docker's Linux environment.

The sensor therefore observes traffic available to its configured/container networking environment. It should not automatically be assumed that it has visibility into every packet on the physical Mac Wi-Fi interface.

For a presentation, describe the current setup accurately as:

> Dockerized live network-flow monitoring within the environment visible to the sensor.

If full host-level Wi-Fi monitoring is required, a host-side capture architecture may be necessary.

---

# 24. Data Files

The project uses:

```text
models/
├── preprocessor.joblib
├── random_forest.joblib
├── metadata.json
└── neural_ids.pt

data/
├── live_state.json
└── live_alerts.jsonl
```

The model files are loaded by `model_io.py`.

The live sensor reads the trained preprocessing pipeline, Random Forest and metadata before starting packet capture.

---

# 25. Recommended Demonstration

For a clean project demonstration:

### Step 1 — Start IDS

```bash
docker compose up -d
```

### Step 2 — Open dashboard

```text
http://localhost:8501
```

### Step 3 — Train model

Open:

```text
Train Model
```

Upload the labelled UNSW-NB15 training dataset.

### Step 4 — Verify model

Confirm that the model reports:

- Training rows
- Features
- Normal samples
- Attack samples

### Step 5 — Open Live Sensor

Confirm:

```text
Capture Engine Online
Packets Seen increasing
Active Flows
Flows Scored
```

### Step 6 — Investigate predictions

Review:

- Source IP
- Destination IP
- Source port
- Destination port
- Protocol
- Attack probability
- Severity

### Step 7 — Use Traffic Analyzer

Upload a separate unseen/test dataset.

Compare:

```text
Actual labels
        vs
Model predictions
```

### Step 8 — Use Adversarial Lab

Select a detected attack and run FGSM or PGD to demonstrate feature-space evasion.

---

# 26. Interpreting the Live Dashboard

A useful distinction is:

```text
Packet
   ≠
Flow
   ≠
Prediction
   ≠
Confirmed Attack
```

For example:

```text
161 packets
      ↓
106 scored flows
      ↓
49 model predictions = Attack
```

This does **not** mean:

```text
49 confirmed attacks occurred.
```

It means:

```text
49 observed flows were classified as Attack by the trained model.
```

This distinction should be maintained in project documentation and presentations.

---

# 27. Project Limitations

Current limitations include:

1. Live feature extraction approximates UNSW-NB15 features.
2. Some features cannot be accurately inferred from short live observations.
3. Internal Docker traffic may generate false positives.
4. Model predictions are not equivalent to confirmed malicious activity.
5. Docker Desktop networking on macOS limits direct visibility into host-level traffic.
6. The adversarial lab operates in feature space rather than generating guaranteed-valid malicious packets.
7. The Random Forest's confidence is model probability, not a formal probability that an attack actually occurred.
8. A benchmark-trained model may behave differently on real-world traffic.

---

# 28. Security and Ethical Use

Use the IDS only on networks and systems for which you have authorization.

The project is intended for:

- Education
- Research
- Defensive security
- Controlled laboratory experiments
- Model robustness testing

Do not use the system to monitor networks without appropriate authorization.

---

# 29. License

This project is licensed under the MIT License.

See the `LICENSE` file for the complete license text.

---

# 30. Project Summary

The final system provides an end-to-end experimental IDS workflow:

```text
┌────────────────────────────────────────────┐
│              LIVE ADVERSARIAL IDS          │
├────────────────────────────────────────────┤
│                                            │
│  Packet Capture                            │
│       ↓                                    │
│  Flow Aggregation                          │
│       ↓                                    │
│  Feature Engineering                       │
│       ↓                                    │
│  Random Forest Detection                   │
│       ↓                                    │
│  Normal / Attack                           │
│       ↓                                    │
│  Probability + Severity                    │
│       ↓                                    │
│  Live Threat Dashboard                     │
│                                            │
│  ───────────────────────────────────────   │
│                                            │
│  Traffic Analyzer                          │
│       ↓                                    │
│  Offline evaluation                        │
│                                            │
│  ───────────────────────────────────────   │
│                                            │
│  Adversarial Lab                           │
│       ↓                                    │
│  FGSM / PGD feature-space testing          │
│                                            │
└────────────────────────────────────────────┘
```

The project demonstrates both **operational intrusion detection** and **machine-learning robustness testing** in a single interactive environment.
