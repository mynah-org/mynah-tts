#!/usr/bin/env python3
"""Run the official NeMo Magpie path as a reference/oracle smoke test."""

from __future__ import annotations

import argparse
import copy
import io
import tarfile
import tempfile
from pathlib import Path

import soundfile as sf
import yaml
from nemo.collections.tts.models import MagpieTTSModel


def patch_tarfile_for_python310() -> None:
    """Bridge NeMo's Python 3.12 tarfile call on Python 3.10.

    The downloaded archive is a local, trusted NeMo checkpoint. The shim keeps
    NeMo's existing extraction path usable on Python 3.10, where `filter` is
    not yet an accepted keyword by TarFile.extract.
    """
    if "filter" in tarfile.TarFile.extract.__code__.co_varnames:
        return
    original_extract = tarfile.TarFile.extract

    def extract_compat(self, member, path="", set_attrs=True, *, filter=None):
        del filter
        return original_extract(self, member, path=path, set_attrs=set_attrs)

    tarfile.TarFile.extract = extract_compat


def local_config(archive: Path, codec: Path, byt5_tokenizer: Path) -> Path:
    with tarfile.open(archive, "r") as tar:
        member = next(m for m in tar.getmembers() if m.name.endswith("model_config.yaml"))
        config = yaml.safe_load(tar.extractfile(member))
    config["codecmodel_path"] = str(codec)
    for tokenizer in config.get("text_tokenizers", {}).values():
        if tokenizer.get("pretrained_model") == "google/byt5-small":
            tokenizer["pretrained_model"] = str(byt5_tokenizer)
    handle = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    with handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return Path(handle.name)


def inference_codec_archive(archive: Path) -> Path:
    """Remove the codec's training-only WavLM discriminator from a temp copy."""
    handle = tempfile.NamedTemporaryFile(suffix=".nemo", delete=False)
    handle.close()
    with tarfile.open(archive, "r") as source, tarfile.open(handle.name, "w") as target:
        for member in source.getmembers():
            if member.name.endswith("model_config.yaml"):
                config = yaml.safe_load(source.extractfile(member)) or {}
                config["discriminator"] = None
                config["discriminator_loss"] = None
                payload = yaml.safe_dump(config, sort_keys=False).encode()
                replacement = copy.copy(member)
                replacement.size = len(payload)
                target.addfile(replacement, io.BytesIO(payload))
            elif member.isfile():
                with source.extractfile(member) as stream:
                    target.addfile(member, stream)
            else:
                target.addfile(member)
    return Path(handle.name)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--codec", type=Path, required=True)
    parser.add_argument("--byt5-tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--text", default="Hello from the Magpie TTS oracle.")
    parser.add_argument("--language", default="en")
    parser.add_argument("--speaker", type=int, default=4)
    parser.add_argument("--cfg", action="store_true")
    args = parser.parse_args()

    patch_tarfile_for_python310()
    codec_archive = inference_codec_archive(args.codec)
    config_path = local_config(args.archive, codec_archive, args.byt5_tokenizer)
    try:
        model = MagpieTTSModel.restore_from(
            str(args.archive),
            map_location="cpu",
            override_config_path=str(config_path),
        )
        model.eval()
        audio, audio_len = model.do_tts(
            args.text,
            language=args.language,
            apply_TN=False,
            use_cfg=args.cfg,
            speaker_index=args.speaker,
        )
        length = int(audio_len.reshape(-1)[0].item())
        waveform = audio.reshape(-1)[:length].detach().cpu().numpy()
        sf.write(args.output, waveform, int(model.sample_rate), subtype="PCM_16")
        print(f"wrote NeMo oracle WAV: {args.output} ({length} samples at {model.sample_rate} Hz)")
    finally:
        config_path.unlink(missing_ok=True)
        codec_archive.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
