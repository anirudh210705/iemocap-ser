"""Train a configured SER model. Importing this module never starts training."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .augmentations import mixup_batch
from .checkpoints import load_model_checkpoint_state, model_state_for_checkpoint
from .dataset import AudioCollator, IEMOCAPDataset
from .metrics import classification_metrics
from .models import build_model
from .utils import device_from_name, load_config, set_seed


def make_loader(config: dict[str, object], split: str, shuffle: bool) -> DataLoader:
    csv_path = str(config[f"{split}_csv"])
    dataset = IEMOCAPDataset(
        csv_path=csv_path,
        data_root=str(config["data_root"]),
        sample_rate=int(config.get("sample_rate", 16_000)),
        max_seconds=float(config.get("max_seconds", 12.0)),
        augmentation=str(config.get("augmentation", "none")) if shuffle else "none",
    )
    return DataLoader(
        dataset, batch_size=int(config.get("batch_size", 4)), shuffle=shuffle,
        num_workers=int(config.get("num_workers", 2)), pin_memory=torch.cuda.is_available(),
        collate_fn=AudioCollator(), persistent_workers=int(config.get("num_workers", 2)) > 0,
    )


def class_weights(csv_path: str, device: torch.device) -> torch.Tensor:
    counts = Counter(pd.read_csv(csv_path)["label"])
    order = ("angry", "happy", "neutral", "sad")
    total = sum(counts.values())
    return torch.tensor([total / (len(order) * counts[label]) for label in order], dtype=torch.float32, device=device)


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, dict[str, object]]:
    model.eval()
    total_loss, targets, predictions = 0.0, [], []
    for batch in loader:
        inputs = batch["input_values"].to(device)
        masks = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        logits = model(inputs, masks)
        total_loss += float(criterion(logits, labels)) * len(labels)
        targets.extend(labels.cpu().tolist())
        predictions.extend(logits.argmax(dim=-1).cpu().tolist())
    return total_loss / len(loader.dataset), classification_metrics(targets, predictions)


def train(config: dict[str, object], device_name: str = "auto", resume: str | None = None) -> Path:
    seed = int(config.get("seed", 42))
    set_seed(seed)
    device = device_from_name(device_name)
    output_dir = Path(str(config["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    train_loader = make_loader(config, "train", True)
    val_loader = make_loader(config, "val", False)
    model = build_model(config).to(device)
    if hasattr(model, "parameter_summary"):
        print("Parameters:", json.dumps(model.parameter_summary()))
    weights = class_weights(str(config["train_csv"]), device) if config.get("class_weighted_loss", True) else None
    criterion = nn.CrossEntropyLoss(weight=weights)
    encoder_module = getattr(model, "audio_encoder", getattr(model, "encoder", None))
    encoder_parameters = [] if encoder_module is None else [p for p in encoder_module.parameters() if p.requires_grad]
    encoder_ids = {id(parameter) for parameter in encoder_parameters}
    head_parameters = [p for p in model.parameters() if p.requires_grad and id(p) not in encoder_ids]
    parameter_groups = []
    if encoder_parameters:
        parameter_groups.append({"params": encoder_parameters, "lr": float(config.get("encoder_learning_rate", config["learning_rate"]))})
    if head_parameters:
        parameter_groups.append({"params": head_parameters, "lr": float(config.get("head_learning_rate", config["learning_rate"]))})
    optimizer = AdamW(parameter_groups, weight_decay=float(config.get("weight_decay", 0)))
    accumulation = int(config.get("gradient_accumulation_steps", 1))
    amp_enabled = device.type == "cuda" and bool(config.get("amp", True))
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    from transformers import get_cosine_schedule_with_warmup

    optimizer_steps_per_epoch = max(1, (len(train_loader) + accumulation - 1) // accumulation)
    total_steps = optimizer_steps_per_epoch * int(config["epochs"])
    warmup_steps = int(total_steps * float(config.get("warmup_ratio", 0.1)))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    start_epoch, best_f1, bad_epochs = 0, -1.0, 0
    history: list[dict[str, object]] = []

    if resume:
        checkpoint = torch.load(resume, map_location=device, weights_only=False)
        load_model_checkpoint_state(model, checkpoint)
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        if "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = checkpoint["epoch"] + 1
        best_f1 = checkpoint["best_macro_f1"]
        history_path = output_dir / "history.json"
        if history_path.exists():
            history = json.loads(history_path.read_text(encoding="utf-8"))
        if "bad_epochs" in checkpoint:
            bad_epochs = int(checkpoint["bad_epochs"])
        elif history:
            best_index = max(range(len(history)), key=lambda index: float(history[index]["macro_f1"]))
            bad_epochs = len(history) - best_index - 1

    for epoch in range(start_epoch, int(config["epochs"])):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        for step, batch in enumerate(progress):
            inputs = batch["input_values"].to(device)
            masks = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            mixed, first, second, lam = mixup_batch(inputs, labels, float(config.get("mixup_alpha", 0.0)))
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                logits = model(mixed, masks)
                loss = (lam * criterion(logits, first) + (1 - lam) * criterion(logits, second)) / accumulation
            scaler.scale(loss).backward()
            if (step + 1) % accumulation == 0 or step + 1 == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    (parameter for parameter in model.parameters() if parameter.requires_grad),
                    float(config.get("max_grad_norm", 1.0)),
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            running_loss += float(loss) * accumulation * len(labels)
            progress.set_postfix(loss=f"{float(loss) * accumulation:.4f}")

        val_loss, metrics = validate(model, val_loader, criterion, device)
        record = {"epoch": epoch + 1, "train_loss": running_loss / len(train_loader.dataset), "val_loss": val_loss, **{k: v for k, v in metrics.items() if k != "per_class"}}
        history.append(record)
        (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        improved = float(metrics["macro_f1"]) > best_f1
        if improved:
            best_f1, bad_epochs = float(metrics["macro_f1"]), 0
        else:
            bad_epochs += 1
        saved_model_state, compact = model_state_for_checkpoint(model)
        state = {
            "epoch": epoch,
            "model": saved_model_state,
            "compact_model_state": compact,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_macro_f1": best_f1,
            "bad_epochs": bad_epochs,
            "config": config,
        }
        torch.save(state, output_dir / "last.pt")
        if improved:
            torch.save(state, output_dir / "best.pt")
        print(json.dumps(record, indent=2))
        if bad_epochs >= int(config.get("patience", 5)):
            break
    return output_dir / "best.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume")
    args = parser.parse_args()
    train(load_config(args.config), args.device, args.resume)


if __name__ == "__main__":
    main()
