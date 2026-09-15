"""PyTorch dataset and padding collator for IEMOCAP utterances."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset

from . import LABEL_TO_ID
from .augmentations import WaveformAugment


class IEMOCAPDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        data_root: str | Path,
        sample_rate: int = 16_000,
        max_seconds: float = 12.0,
        augmentation: str = "none",
    ) -> None:
        self.frame = pd.read_csv(csv_path)
        self.data_root = Path(data_root)
        self.sample_rate = sample_rate
        self.max_samples = round(sample_rate * max_seconds)
        self.augment = WaveformAugment(augmentation, sample_rate)
        unknown = set(self.frame["label"]) - set(LABEL_TO_ID)
        if unknown:
            raise ValueError(f"Unknown labels: {sorted(unknown)}")

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.frame.iloc[index]
        path = self.data_root / str(row.audio_path)
        waveform, source_rate = torchaudio.load(path)
        waveform = waveform.mean(dim=0)
        if source_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, source_rate, self.sample_rate)
        waveform = waveform[: self.max_samples]
        waveform = self.augment(waveform)
        return {
            "input_values": waveform,
            "label": LABEL_TO_ID[str(row.label)],
            "utterance_id": str(row.utterance_id),
        }


class AudioCollator:
    def __call__(self, batch: list[dict[str, object]]) -> dict[str, object]:
        lengths = torch.tensor([len(item["input_values"]) for item in batch], dtype=torch.long)
        maximum = int(lengths.max())
        values = torch.zeros(len(batch), maximum, dtype=torch.float32)
        mask = torch.zeros(len(batch), maximum, dtype=torch.long)
        for index, item in enumerate(batch):
            waveform = item["input_values"]
            values[index, : len(waveform)] = waveform
            mask[index, : len(waveform)] = 1
        return {
            "input_values": values,
            "attention_mask": mask,
            "labels": torch.tensor([item["label"] for item in batch], dtype=torch.long),
            "utterance_ids": [item["utterance_id"] for item in batch],
        }

