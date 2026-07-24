#!/usr/bin/env python3
"""Generate a short sample for each of the 12 supported languages.

Uses NeMo tokenizers to produce token IDs, then calls the native
mynah-tts binary to synthesize.  Writes one WAV per language into
the output directory.

Usage:
    .venv/bin/python tools/gen_matrix.py --model-dir models/magpie-v2607-pack \
        --archive models/magpie-v2607/magpie_tts_multilingual_357m.nemo \
        --codec models/nano-codec-22khz/nemo-nano-codec-22khz-1.89kbps-21.5fps.nemo \
        --byt5 models/byt5-small-tokenizer \
        --binary build/cpu/mynah-tts \
        --output build/gen-matrix
"""

from __future__ import annotations

import argparse
import copy
import io
import json
import subprocess
import tarfile
import tempfile
from pathlib import Path

import yaml

# Short, simple text per language (ASCII-safe where possible).
TEXTS = {
    "en": "hello world",
    "fr": "bonjour le monde",
    "it": "ciao mondo",
    "es": "hola mundo",
    "de": "hallo Welt",
    "pt": "olá mundo",
    "vi": "xin chào thế giới",
    "ko": "안녕하세요 세계",
    "ja": "こんにちは世界",
    "zh": "你好世界",
    "hi": "नमस्ते दुनिया",
    "ar": "مرحبا بالعالم",
}

SPEAKER = 4
MAX_STEPS = 30
SEED = 42

# Language → tokenizer name mapping (from NeMo LANGUAGE_TOKENIZER_MAP,
# adjusted to match the actual tokenizer names in the v2607 checkpoint).
LANG_TOKENIZER = {
    "en": "english_phoneme",
    "de": "german_phoneme",
    "es": "spanish_phoneme",
    "fr": "french_chartokenizer",
    "it": "italian_chartokenizer",
    "vi": "vietnamese_chartokenizer",
    "ko": "korean_chartokenizer",
    "zh": "mandarin_phoneme",
    "hi": "hindi_phoneme",
    "ja": "japanese_phoneme",
    "pt": "portuguese_Brazilian_phoneme",
    "ar": "arabic_MSA_chartokenizer",
}


def patch_tarfile() -> None:
    if "filter" in tarfile.TarFile.extract.__code__.co_varnames:
        return
    orig = tarfile.TarFile.extract
    def compat(self, member, path="", set_attrs=True, *, filter=None, **kwargs):
        return orig(self, member, path=path, set_attrs=set_attrs)
    tarfile.TarFile.extract = compat


def prepare_codec(codec_path: Path) -> Path:
    handle = tempfile.NamedTemporaryFile(suffix=".nemo", delete=False)
    handle.close()
    with tarfile.open(codec_path, "r") as src, tarfile.open(handle.name, "w") as dst:
        for m in src.getmembers():
            if m.name.endswith("model_config.yaml"):
                cfg = yaml.safe_load(src.extractfile(m)) or {}
                cfg["discriminator"] = None
                cfg["discriminator_loss"] = None
                payload = yaml.safe_dump(cfg, sort_keys=False).encode()
                r = copy.copy(m)
                r.size = len(payload)
                dst.addfile(r, io.BytesIO(payload))
            elif m.isfile():
                dst.addfile(m, src.extractfile(m))
            else:
                dst.addfile(m)
    return Path(handle.name)


def prepare_config(archive: Path, codec_path: Path, byt5: Path) -> Path:
    with tarfile.open(archive, "r") as tar:
        member = next(m for m in tar.getmembers() if m.name.endswith("model_config.yaml"))
        config = yaml.safe_load(tar.extractfile(member))
    config["codecmodel_path"] = str(codec_path)
    for tok in config.get("text_tokenizers", {}).values():
        if tok.get("pretrained_model") == "google/byt5-small":
            tok["pretrained_model"] = str(byt5)
    handle = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    with handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return Path(handle.name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--codec", type=Path, required=True)
    parser.add_argument("--byt5", type=Path, required=True)
    parser.add_argument("--binary", type=Path, default=Path("build/cpu/mynah-tts"))
    parser.add_argument("--output", type=Path, default=Path("build/gen-matrix"))
    parser.add_argument("--speaker", type=int, default=SPEAKER)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    patch_tarfile()

    # Lazy-import NeMo so the script can show --help without it.
    from nemo.collections.tts.models import MagpieTTSModel

    print("Loading NeMo model (for tokenization only) ...")
    codec_tmp = prepare_codec(args.codec)
    config_tmp = prepare_config(args.archive, codec_tmp, args.byt5)
    try:
        model = MagpieTTSModel.restore_from(
            str(args.archive), map_location="cpu",
            override_config_path=str(config_tmp),
        )
        model.eval()
    finally:
        config_tmp.unlink(missing_ok=True)
        codec_tmp.unlink(missing_ok=True)

    tokenizer = model.tokenizer
    available = list(tokenizer.tokenizers.keys())
    print(f"Available tokenizers: {available}")

    args.output.mkdir(parents=True, exist_ok=True)
    results = []

    for lang, text in TEXTS.items():
        tok_name = LANG_TOKENIZER.get(lang)
        if tok_name is None or tok_name not in tokenizer.tokenizers:
            print(f"  {lang}: SKIP (tokenizer '{tok_name}' not in model)")
            results.append({"lang": lang, "status": "skip", "reason": f"tokenizer {tok_name} not found"})
            continue
        sub_tok = tokenizer.tokenizers[tok_name]

        try:
            token_ids = sub_tok.encode(text)
        except Exception as exc:
            print(f"  {lang}: SKIP (encode error: {exc})")
            results.append({"lang": lang, "status": "skip", "reason": str(exc)})
            continue

        ids_str = ",".join(str(t) for t in token_ids)
        wav_path = args.output / f"{lang}.wav"

        cmd = [
            str(args.binary),
            "--synthesize", str(args.model_dir),
            "--tokens", ids_str,
            "--output", str(wav_path),
            "--speaker", str(args.speaker),
            "--max-steps", str(args.max_steps),
            "--seed", str(args.seed),
            "--temperature", "0",
        ]
        env_extra = {"MYNAH_THREADS": "1", "MYNAH_QUANT": "int8"}
        import os
        env = {**os.environ, **env_extra}

        print(f"  {lang}: {len(token_ids)} tokens → {wav_path.name} ...", end=" ", flush=True)
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120, env=env)
            if proc.returncode == 0:
                # Extract sample count from output
                for line in proc.stdout.splitlines():
                    if "wrote native WAV" in line:
                        print(line.strip())
                        break
                else:
                    print("OK")
                results.append({"lang": lang, "status": "ok", "tokens": len(token_ids),
                                "text": text, "wav": str(wav_path)})
            else:
                print(f"FAIL: {proc.stderr.strip()[:200]}")
                results.append({"lang": lang, "status": "fail", "error": proc.stderr.strip()[:500]})
        except subprocess.TimeoutExpired:
            print("TIMEOUT")
            results.append({"lang": lang, "status": "timeout"})

    # Write summary
    summary_path = args.output / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSummary: {summary_path}")
    ok = sum(1 for r in results if r["status"] == "ok")
    print(f"  {ok}/{len(TEXTS)} languages generated successfully")
    return 0 if ok == len(TEXTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
