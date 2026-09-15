"""Run and aggregate the five speaker-independent IEMOCAP folds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .evaluate import evaluate
from .train import train
from .utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--folds", nargs="*", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    base = load_config(args.config)
    root_output = Path(str(base["output_dir"])) / "loso"
    results = []
    for fold in args.folds:
        if fold not in range(1, 6):
            raise ValueError("Fold numbers must be between 1 and 5")
        split = Path("artifacts/splits/loso") / f"fold_{fold}"
        config = dict(base)
        config.update({
            "train_csv": str(split / "train.csv"),
            "val_csv": str(split / "val.csv"),
            "output_dir": str(root_output / f"fold_{fold}"),
        })
        test_csv = str(split / "test.csv")
        print(f"Fold {fold}: train={config['train_csv']} val={config['val_csv']} test={test_csv}")
        if args.dry_run:
            continue
        fold_output = Path(str(config["output_dir"]))
        resume = fold_output / "last.pt"
        checkpoint = train(config, args.device, str(resume) if resume.exists() else None)
        metrics = evaluate(str(checkpoint), test_csv, str(fold_output / "test_results"), args.device)
        results.append({"fold": fold, "macro_f1": float(metrics["macro_f1"]), "accuracy": float(metrics["accuracy"])})
        root_output.mkdir(parents=True, exist_ok=True)
        (root_output / "fold_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    if results:
        summary = {
            "folds": results,
            "macro_f1_mean": float(np.mean([row["macro_f1"] for row in results])),
            "macro_f1_std": float(np.std([row["macro_f1"] for row in results], ddof=1)) if len(results) > 1 else 0.0,
            "accuracy_mean": float(np.mean([row["accuracy"] for row in results])),
        }
        (root_output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
