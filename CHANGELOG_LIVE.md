# Live IDS build — first implementation

## Added

- Dockerfile and Docker Compose architecture.
- Dedicated packet-capture sensor service.
- Bidirectional 5-tuple flow tracker.
- Live flow feature extraction for common UNSW-NB15 fields.
- Persistent model artifacts under `models/`.
- Live JSONL alert stream and sensor state file.
- Streamlit **Live Sensor** workspace page.
- Automatic live dashboard updates with `st.fragment(run_every=2)`.
- Linux-native host-network capture configuration.

## Retained

- Training from labelled CSV.
- Random Forest and PyTorch neural IDS.
- Traffic Analyzer for offline labelled/unseen CSV analysis.
- FGSM and PGD adversarial robustness lab.

## Next engineering steps

1. Retrain with a strictly live-compatible feature subset.
2. Add PCAP/offline capture mode using the same flow engine.
3. Add host-side capture bridge for macOS Docker Desktop.
4. Add persistent SQLite alert history.
5. Add flow-rate and attack-rate time-series charts.
6. Add sensor health / heartbeat monitoring.
