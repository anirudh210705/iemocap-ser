"""Run configured experiments sequentially; use --dry-run to inspect them."""

from __future__ import annotations

import argparse
from pathlib import Path

from .train import train
from .utils import load_config


DEFAULT_CONFIGS = (
    "configs/cnn_baseline.json",
    "configs/wav2vec2_base.json",
    "configs/bloomz_1b7_main.json",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", nargs="*", default=list(DEFAULT_CONFIGS))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for config_path in args.configs:
        config = load_config(config_path)
        output = Path(str(config["output_dir"]))
        print(f"{config_path}: {config['model']} -> {output}")
        if not args.dry_run:
            resume = output / "last.pt"
            train(config, args.device, str(resume) if resume.exists() else None)


if __name__ == "__main__":
    main()

