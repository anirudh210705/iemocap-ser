"""Waveform augmentations used only for training samples."""

from __future__ import annotations

import random

import torch
import torchaudio.functional as AF


class WaveformAugment:
    def __init__(self, mode: str = "none", sample_rate: int = 16_000) -> None:
        # SpecAugment is applied inside the model, after spectrogram/feature
        # extraction. It is intentionally a no-op at waveform level.
        valid = {"none", "noise", "speed", "pitch", "combined", "specaugment"}
        if mode not in valid:
            raise ValueError(f"augmentation must be one of {sorted(valid)}")
        self.mode = mode
        self.sample_rate = sample_rate

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        mode = self.mode
        if mode == "combined":
            mode = random.choice(("noise", "speed", "pitch"))
        if mode == "noise":
            signal_rms = waveform.square().mean().sqrt().clamp_min(1e-6)
            snr_db = random.uniform(15.0, 30.0)
            noise_rms = signal_rms / (10 ** (snr_db / 20))
            waveform = waveform + torch.randn_like(waveform) * noise_rms
        elif mode == "speed":
            factor = random.choice((0.9, 0.95, 1.05, 1.1))
            target_rate = max(1, round(self.sample_rate * factor))
            waveform = AF.resample(AF.resample(waveform, self.sample_rate, target_rate), target_rate, self.sample_rate)
        elif mode == "pitch":
            waveform = AF.pitch_shift(waveform, self.sample_rate, random.choice((-2.0, -1.0, 1.0, 2.0)))
        return waveform.clamp(-1.0, 1.0)


def mixup_batch(inputs: torch.Tensor, labels: torch.Tensor, alpha: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if alpha <= 0:
        return inputs, labels, labels, 1.0
    lam = float(torch.distributions.Beta(alpha, alpha).sample())
    permutation = torch.randperm(inputs.size(0), device=inputs.device)
    return lam * inputs + (1.0 - lam) * inputs[permutation], labels, labels[permutation], lam
