from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import torch


MODEL_DIR = Path(__file__).resolve().parent / "models"
PREPROCESSOR_FILE = MODEL_DIR / "preprocessor.joblib"
RF_FILE = MODEL_DIR / "random_forest.joblib"
METADATA_FILE = MODEL_DIR / "metadata.json"
NN_FILE = MODEL_DIR / "neural_ids.pt"


def _ensure_dir() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


def save_models(preprocessor: Any, rf: Any, nn_model: torch.nn.Module, metadata: dict) -> None:
    _ensure_dir()
    joblib.dump(preprocessor, PREPROCESSOR_FILE)
    joblib.dump(rf, RF_FILE)
    torch.save(nn_model.state_dict(), NN_FILE)
    METADATA_FILE.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_models() -> tuple[Any, Any, dict] | None:
    if not PREPROCESSOR_FILE.exists() or not RF_FILE.exists() or not METADATA_FILE.exists():
        return None
    preprocessor = joblib.load(PREPROCESSOR_FILE)
    rf = joblib.load(RF_FILE)
    metadata = json.loads(METADATA_FILE.read_text(encoding="utf-8"))
    return preprocessor, rf, metadata


def load_neural_state() -> dict | None:
    if not NN_FILE.exists():
        return None
    return torch.load(NN_FILE, map_location="cpu", weights_only=True)
