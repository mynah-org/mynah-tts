#!/usr/bin/env python3
"""Export the exact Magpie tokenizer vocabulary for the native normalized-text path."""

from __future__ import annotations

import argparse
import tarfile
import tempfile
from pathlib import Path

import yaml
from nemo.collections.tts.models import MagpieTTSModel

from oracle_magpie import inference_codec_archive, patch_tarfile_for_python310


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--codec", type=Path, required=True)
    parser.add_argument("--byt5-tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", default="english_phoneme")
    args = parser.parse_args()

    patch_tarfile_for_python310()
    codec_archive = inference_codec_archive(args.codec)
    with tarfile.open(args.archive, "r") as tar:
        member = next(m for m in tar.getmembers() if m.name.endswith("model_config.yaml"))
        config = yaml.safe_load(tar.extractfile(member))
    config["codecmodel_path"] = str(codec_archive)
    for tokenizer in config.get("text_tokenizers", {}).values():
        if tokenizer.get("pretrained_model") == "google/byt5-small":
            tokenizer["pretrained_model"] = str(args.byt5_tokenizer)
    config_file = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    with config_file:
        yaml.safe_dump(config, config_file, sort_keys=False)
    config_path = Path(config_file.name)
    try:
        model = MagpieTTSModel.restore_from(
            str(args.archive), map_location="cpu", override_config_path=str(config_path)
        )
        tokenizer = model.tokenizer.tokenizers[args.tokenizer]
        offset = model.tokenizer.tokenizer_offsets[args.tokenizer]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as stream:
            stream.write(f"# tokenizer={args.tokenizer} offset={offset}\n")
            for index, token in enumerate(tokenizer.tokens):
                stream.write(f"{offset + index}\t{token}\n")
        print(f"wrote native tokenizer vocabulary: {args.output}")
    finally:
        config_path.unlink(missing_ok=True)
        codec_archive.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
