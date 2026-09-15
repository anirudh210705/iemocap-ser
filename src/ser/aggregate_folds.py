"""Aggregate completed fold evaluation files without retraining models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def aggregate(root: Path, expected_folds: int = 5) -> dict[str, object]:
    rows: list[dict[str, float | int]] = []
    for fold in range(1, expected_folds + 1):
        path = root / f"fold_{fold}" / "test_results" / "metrics.json"
        if not path.exists():
            raise FileNotFoundError(f"Fold {fold} has no completed test metrics: {path}")
        metrics = json.loads(path.read_text(encoding="utf-8"))
        rows.append({
            "fold": fold,
            "accuracy": float(metrics["accuracy"]),
            "macro_f1": float(metrics["macro_f1"]),
            "weighted_f1": float(metrics["weighted_f1"]),
        })

    summary: dict[str, object] = {
        "folds": rows,
        "accuracy_mean": float(np.mean([row["accuracy"] for row in rows])),
        "accuracy_std": float(np.std([row["accuracy"] for row in rows], ddof=1)),
        "macro_f1_mean": float(np.mean([row["macro_f1"] for row in rows])),
        "macro_f1_std": float(np.std([row["macro_f1"] for row in rows], ddof=1)),
        "weighted_f1_mean": float(np.mean([row["weighted_f1"] for row in rows])),
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "fold_results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--expected-folds", type=int, default=5)
    args = parser.parse_args()
    print(json.dumps(aggregate(args.root, args.expected_folds), indent=2))


if __name__ == "__main__":
    main()
