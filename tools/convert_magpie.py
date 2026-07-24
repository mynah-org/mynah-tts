#!/usr/bin/env python3
"""Convert Magpie and NanoCodec NeMo archives into a deterministic model pack.

This tool only reads tar members and tensors. It does not import NeMo and never
executes code from a checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml
from safetensors.torch import save_file


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def archive_members(path: Path) -> list[str]:
    with tarfile.open(path, "r") as tar:
        return [member.name for member in tar.getmembers() if member.isfile()]


def load_nemo(path: Path) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    with tarfile.open(path, "r") as tar:
        config_member = next(
            (m for m in tar.getmembers() if m.name.endswith("model_config.yaml")), None
        )
        weights_member = next(
            (m for m in tar.getmembers() if m.name.endswith("model_weights.ckpt")), None
        )
        if config_member is None or weights_member is None:
            raise RuntimeError(f"{path} is missing model_config.yaml or model_weights.ckpt")
        config = yaml.safe_load(tar.extractfile(config_member)) or {}
        state = torch.load(tar.extractfile(weights_member), map_location="cpu", weights_only=True)
    return config, state


def fuse_weight_norm(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Materialize PyTorch weight-norm parameters for a dependency-free runtime.

    NeMo stores parametrized convolution weights as ``original0`` (g) and
    ``original1`` (v).  The native runtime only needs the effective weight,
    so conversion computes ``g * v / ||v||`` once and drops the parametrizer.
    The default PyTorch weight-norm dimension is zero for both convolution and
    transposed-convolution modules.
    """
    materialized = dict(state)
    for key in list(state):
        suffix = ".parametrizations.weight.original0"
        if not key.endswith(suffix):
            continue
        prefix = key[: -len(suffix)]
        v_key = prefix + ".parametrizations.weight.original1"
        if v_key not in state:
            raise RuntimeError(f"weight norm pair is incomplete for {prefix}")
        g = state[key]
        v = state[v_key]
        reduce_dims = tuple(range(1, v.ndim))
        norm = torch.linalg.vector_norm(v, dim=reduce_dims, keepdim=True)
        materialized[prefix + ".weight"] = g * v / norm
        del materialized[key]
        del materialized[v_key]
    return materialized


def copy_assets(archive: Path, output: Path, config: dict[str, Any]) -> None:
    tokenizer_dir = output / "tokenizer"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            name = Path(member.name).name
            if name in {"model_config.yaml", "model_weights.ckpt"}:
                continue
            if not name.endswith((".dict", ".txt", ".heteronym", ".json")):
                continue
            destination = tokenizer_dir / name
            with tar.extractfile(member) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)
    speaker_reference = config.get("speaker_map", "")
    if isinstance(speaker_reference, str):
        speaker_name = Path(speaker_reference.split(":", 1)[-1]).name
        with tarfile.open(archive, "r") as tar:
            member = next((m for m in tar.getmembers() if Path(m.name).name == speaker_name), None)
            if member is not None:
                with tar.extractfile(member) as source, (output / "speakers.json").open("wb") as target:
                    shutil.copyfileobj(source, target)


def write_license_notice(output: Path) -> None:
    licenses = output / "LICENSES"
    licenses.mkdir(parents=True, exist_ok=True)
    (licenses / "MODEL_LICENSE.md").write_text(
        "# Model license notice\n\n"
        "Source model: NVIDIA MagpieTTS Multilingual 357M\n\n"
        "License: NVIDIA Open Model License\n\n"
        "License URL: https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license\n\n"
        "This notice accompanies the converted model pack. It does not change the\n"
        "license terms or grant rights to redistribute the checkpoint.\n"
    )


def write_manifest(output: Path, config: dict[str, Any], state: dict[str, torch.Tensor],
                   tts_archive: Path, codec_archive: Path | None, revision: str) -> None:
    audio_embedding = state["audio_embeddings.0.weight"]
    text_embedding = state["text_embedding.weight"]
    speakers = config.get("speaker_map")
    speaker_count = int(state.get("_baked_embedding_T", torch.tensor(0)).item()) if "_baked_embedding_T" in state else 0
    if (output / "speakers.json").exists():
        speaker_count = len(json.loads((output / "speakers.json").read_text()))
    codec_config: dict[str, Any] = {}
    if codec_archive is not None:
        codec_config["source"] = str(codec_archive)
        codec_config["sha256"] = sha256(codec_archive)
    manifest = {
        "format": 1,
        "engine": "magpie",
        "revision": revision,
        "dtype": "float32",
        "source": "nvidia/magpie_tts_multilingual_357m",
        "sample_rate": int(config.get("sample_rate", 22050)),
        "frame_rate": 22050.0 / 1024.0,
        "frame_stacking_factor": int(config["frame_stacking_factor"]),
        "codebook_count": 8,
        "codebook_size": 2016,
        "audio_vocab_size": int(audio_embedding.shape[0]),
        # NeMo SpecialAudioToken indices: BOS=codebook_size+0, EOS=codebook_size+1.
        "audio_bos_id": 2016,
        "audio_eos_id": 2017,
        "stacked_audio_streams": int(len([k for k in state if k.startswith("audio_embeddings.") and k.endswith(".weight")])),
        "hidden_dim": int(config["embedding_dim"]),
        "encoder_layers": int(config["encoder"]["n_layers"]),
        "decoder_layers": int(config["decoder"]["n_layers"]),
        "local_transformer_layers": int(config["local_transformer_n_layers"]),
        "local_transformer_type": config["local_transformer_type"],
        "text_vocab_size": int(text_embedding.shape[0]),
        "text_max_length": int(config["encoder"]["max_length_causal_mask"]),
        "speaker_count": speaker_count,
        "speaker_map": speakers,
        "languages": ["ar", "de", "en", "es", "fr", "hi", "it", "ja", "ko", "pt", "vi", "zh"],
        "requires_normalized_text": True,
        "tokenizers": sorted(config.get("text_tokenizers", {}).keys()),
        "language_to_tokenizer": {
            "ar": "arabic_MSA_chartokenizer",
            "de": "german_phoneme",
            "en": "english_phoneme",
            "es": "spanish_phoneme",
            "fr": "french_chartokenizer",
            "hi": "hindi_phoneme",
            "it": "italian_chartokenizer",
            "ja": "japanese_phoneme",
            "ko": "korean_chartokenizer",
            "pt": "portuguese_Brazilian_phoneme",
            "vi": "vietnamese_chartokenizer",
            "zh": "mandarin_phoneme",
        },
        "inference": {
            **config.get("inference_parameters", {}),
            "min_generated_frames": config.get("inference_parameters", {}).get(
                "min_generated_frames", 4
            ),
        },
        "codec": {
            "sample_rate": 22050,
            "samples_per_frame": 1024,
            "codebooks": 8,
            "levels": [8, 7, 6, 6],
            **codec_config,
        },
        "weights": {"tts": "tts.safetensors", "codec": "codec.safetensors"},
    }
    (output / "model.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def convert(args: argparse.Namespace) -> None:
    if args.tts_archive is None and not args.codec_only:
        raise SystemExit("--tts-archive is required unless --codec-only is used")
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    if args.tts_archive is not None:
        config, state = load_nemo(args.tts_archive)
        tts_state = fuse_weight_norm(state)
        save_file({name: tensor.contiguous() for name, tensor in tts_state.items()}, str(output / "tts.safetensors"))
        copy_assets(args.tts_archive, output, config)
        write_license_notice(output)
        if args.codec_archive is not None:
            _, codec_state = load_nemo(args.codec_archive)
            codec_state = fuse_weight_norm(codec_state)
            save_file({name: tensor.contiguous() for name, tensor in codec_state.items()}, str(output / "codec.safetensors"))
        elif not (output / "codec.safetensors").exists():
            raise SystemExit("--codec-archive is required for a complete Magpie pack")
        write_manifest(output, config, state, args.tts_archive, args.codec_archive, args.revision)
        source = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "converter": "tools/convert_magpie.py",
            "tts_archive": str(args.tts_archive),
            "tts_sha256": sha256(args.tts_archive),
            "codec_archive": str(args.codec_archive) if args.codec_archive else None,
            "codec_sha256": sha256(args.codec_archive) if args.codec_archive else None,
            "members": archive_members(args.tts_archive),
        }
        (output / "source.json").write_text(json.dumps(source, indent=2, sort_keys=True) + "\n")
        print(f"wrote Magpie model pack: {output}")
        return
    if args.codec_archive is None:
        raise SystemExit("--codec-archive is required")
    _, codec_state = load_nemo(args.codec_archive)
    codec_state = fuse_weight_norm(codec_state)
    save_file({name: tensor.contiguous() for name, tensor in codec_state.items()}, str(output / "codec.safetensors"))
    print(f"wrote codec tensors: {output / 'codec.safetensors'}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tts-archive", type=Path)
    parser.add_argument("--codec-archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", default="v2607")
    parser.add_argument("--codec-only", action="store_true")
    convert(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
