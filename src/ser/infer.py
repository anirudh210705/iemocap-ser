"""Predict the emotion of a single WAV file using a saved checkpoint."""

from __future__ import annotations

import argparse

import torch
import torchaudio

from . import LABEL_NAMES
from .checkpoints import load_model_checkpoint_state
from .models import build_model
from .utils import device_from_name


@torch.no_grad()
def predict(checkpoint_path: str, audio_path: str, device_name: str = "auto") -> dict[str, float]:
    device = device_from_name(device_name)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = build_model(config).to(device)
    load_model_checkpoint_state(model, checkpoint)
    model.eval()
    waveform, source_rate = torchaudio.load(audio_path)
    waveform = waveform.mean(dim=0)
    sample_rate = int(config.get("sample_rate", 16_000))
    if source_rate != sample_rate:
        waveform = torchaudio.functional.resample(waveform, source_rate, sample_rate)
    waveform = waveform[: round(sample_rate * float(config.get("max_seconds", 12.0)))].unsqueeze(0).to(device)
    mask = torch.ones_like(waveform, dtype=torch.long)
    probabilities = model(waveform, mask).softmax(dim=-1).squeeze(0).cpu().tolist()
    return dict(zip(LABEL_NAMES, probabilities))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    probabilities = predict(args.checkpoint, args.audio, args.device)
    for label, probability in sorted(probabilities.items(), key=lambda item: item[1], reverse=True):
        print(f"{label:>7}: {probability:.4f}")


if __name__ == "__main__":
    main()
