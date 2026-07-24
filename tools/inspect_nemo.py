#!/usr/bin/env python3
"""Inspect a NeMo archive without importing NeMo or executing model code."""

from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path
from typing import Any

import torch
import yaml


def load_archive(archive: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    with tarfile.open(archive, "r") as tar:
        config_member = next(
            (m for m in tar.getmembers() if m.name.endswith("model_config.yaml")),
            None,
        )
        weights_member = next(
            (m for m in tar.getmembers() if m.name.endswith("model_weights.ckpt")),
            None,
        )
        if config_member is None or weights_member is None:
            raise RuntimeError("archive must contain model_config.yaml and model_weights.ckpt")
        config = yaml.safe_load(tar.extractfile(config_member)) or {}
        state = torch.load(tar.extractfile(weights_member), map_location="cpu", weights_only=True)
    return config, state


def shape(value: Any) -> list[int] | None:
    return list(value.shape) if hasattr(value, "shape") else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    config, state = load_archive(args.archive)
    report = {
        "archive": str(args.archive),
        "target": config.get("target"),
        "nemo_version": config.get("nemo_version"),
        "embedding_dim": config.get("embedding_dim"),
        "frame_stacking_factor": config.get("frame_stacking_factor"),
        "local_transformer_type": config.get("local_transformer_type"),
        "local_transformer_layers": config.get("local_transformer_n_layers"),
        "encoder_layers": config.get("encoder", {}).get("n_layers"),
        "decoder_layers": config.get("decoder", {}).get("n_layers"),
        "sample_rate": config.get("sample_rate", 22050),
        "speaker_map": config.get("speaker_map"),
        "text_tokenizers": sorted(config.get("text_tokenizers", {}).keys()),
        "tensor_count": len(state),
        "tensors": {name: {"shape": shape(value), "dtype": str(value.dtype)} for name, value in state.items()},
    }
    if args.as_json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        for key in (
            "archive", "target", "nemo_version", "embedding_dim",
            "frame_stacking_factor", "local_transformer_type",
            "local_transformer_layers", "encoder_layers", "decoder_layers",
            "sample_rate", "speaker_map", "tensor_count",
        ):
            print(f"{key}: {report[key]}")
        print("text_tokenizers:", ", ".join(report["text_tokenizers"]))
        print("first_tensors:")
        for name, metadata in list(report["tensors"].items())[:20]:
            print(f"  {name}: {metadata['shape']} {metadata['dtype']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
