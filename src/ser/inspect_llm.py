"""Inspect emotion verbalizers using only an LLM tokenizer (no model weights)."""

from __future__ import annotations

import argparse

from .llm_classifier import resolve_emotion_token_ids
from .utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    if "llm_name" not in config:
        raise ValueError("Configuration does not define llm_name")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(config["llm_name"]))
    mapping = resolve_emotion_token_ids(
        tokenizer,
        config.get("emotion_verbalizers"),
        config.get("emotion_token_overrides"),
    )
    print(f"LLM: {config['llm_name']}")
    for label, token_id in mapping.items():
        print(f"{label:>7} -> {token_id:>7} -> {tokenizer.convert_ids_to_tokens(token_id)!r}")


if __name__ == "__main__":
    main()

