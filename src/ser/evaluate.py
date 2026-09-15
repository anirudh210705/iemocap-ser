"""Evaluate a saved checkpoint on a CSV split without modifying the model."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm.auto import tqdm

from .dataset import AudioCollator, IEMOCAPDataset
from .checkpoints import load_model_checkpoint_state
from .metrics import save_evaluation
from .models import build_model
from .utils import device_from_name, load_config


@torch.no_grad()
def evaluate(checkpoint_path: str, csv_path: str, output_dir: str, device_name: str = "auto") -> dict[str, object]:
    device = device_from_name(device_name)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    dataset = IEMOCAPDataset(
        csv_path, str(config["data_root"]), int(config.get("sample_rate", 16_000)),
        float(config.get("max_seconds", 12.0)), "none",
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=int(config.get("batch_size", 4)), shuffle=False,
        num_workers=int(config.get("num_workers", 2)), collate_fn=AudioCollator(),
    )
    model = build_model(config).to(device)
    load_model_checkpoint_state(model, checkpoint)
    model.eval()
    targets, predictions = [], []
    for batch in tqdm(loader, desc="Evaluating"):
        logits = model(batch["input_values"].to(device), batch["attention_mask"].to(device))
        targets.extend(batch["labels"].tolist())
        predictions.extend(logits.argmax(dim=-1).cpu().tolist())
    return save_evaluation(targets, predictions, Path(output_dir))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    metrics = evaluate(args.checkpoint, args.csv, args.output_dir, args.device)
    print(f"Macro-F1: {metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
