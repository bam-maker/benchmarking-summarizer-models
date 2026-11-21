"""Device selection and synchronized timing shared by all four backends."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def resolve_device(requested="auto"):
    import torch

    requested = requested.lower()
    if requested == "auto":
        requested = os.environ.get("SUMMARIZER_DEVICE", "auto").lower()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested not in {"cpu", "cuda"}:
        raise ValueError("Device must be Auto, CPU, or CUDA.")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable. Select CPU or Auto.")
    return requested


def synchronize(device):
    import torch

    if device == "cuda":
        torch.cuda.synchronize()
