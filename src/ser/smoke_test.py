"""Run one real batch through forward/backward without an optimizer step."""

from __future__ import annotations

import argparse
import json

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from .dataset import AudioCollator, IEMOCAPDataset
from .models import build_model
from .utils import device_from_name, load_config, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--max-seconds", type=float, default=4.0)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))
    device = device_from_name(args.device)
    if device.type != "cuda":
        raise RuntimeError("GPU smoke test requires CUDA")
    dataset = IEMOCAPDataset(
        csv_path=str(config["train_csv"]),
        data_root=str(config["data_root"]),
        sample_rate=int(config.get("sample_rate", 16_000)),
        max_seconds=args.max_seconds,
        augmentation="none",
    )
    count = min(args.samples, len(dataset))
    batch = next(iter(DataLoader(Subset(dataset, range(count)), batch_size=count, collate_fn=AudioCollator())))
    model = build_model(config).to(device)
    model.train()
    inputs = batch["input_values"].to(device)
    masks = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    torch.cuda.reset_peak_memory_stats()
    logits = model(inputs, masks)
    loss = nn.CrossEntropyLoss()(logits.float(), labels)
    loss.backward()

    trainable_with_grad = sum(
        parameter.numel() for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    )
    frozen_with_grad = sum(
        parameter.numel() for parameter in model.parameters()
        if not parameter.requires_grad and parameter.grad is not None
    )
    result = {
        "device": torch.cuda.get_device_name(device),
        "input_shape": list(inputs.shape),
        "logit_shape": list(logits.shape),
        "loss": float(loss.detach()),
        "trainable_parameters_with_gradient": trainable_with_grad,
        "frozen_parameters_with_gradient": frozen_with_grad,
        "peak_gpu_memory_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
    }
    if logits.shape != (count, 4):
        raise RuntimeError(f"Expected {(count, 4)} logits, received {tuple(logits.shape)}")
    if trainable_with_grad == 0:
        raise RuntimeError("No trainable parameters received gradients")
    if frozen_with_grad != 0:
        raise RuntimeError("Frozen parameters unexpectedly received gradients")
    print(json.dumps(result, indent=2))
    print("SMOKE_TEST_PASSED")


if __name__ == "__main__":
    main()

