#!/usr/bin/env python3
"""Dump per-stage reference tensors from the official PocketTTS implementation.

This is offline tooling only: it never runs as part of the C runtime. It exists
so `engine_pocket.c` can be checked stage by stage instead of being judged on
whether the final WAV sounds plausible.

The PocketTTS graph is one transformer step plus a one-step flow head plus a
causal SEANet decoder, so every stage below is a single tensor per frame. Stages
and tolerances are listed in `.work/pocket-tts-oracle.md`.

Usage:
    uv run --with pocket-tts --with numpy python tools/oracle_pocket.py \
        --language english --voice alba --out build/oracle-pocket
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch

# Module name prefixes we record, mapped to the oracle stage they belong to.
# Anything not listed is ignored, which keeps the dump bounded and readable.
WATCHED = (
    "flow_lm.conditioner.embed",
    "flow_lm.input_linear",
    "flow_lm.transformer.layers.0",
    "flow_lm.transformer.layers.5",
    "flow_lm.out_norm",
    "flow_lm.out_eos",
    "flow_lm.flow_net",
    "flow_lm.flow_net.final_layer",
    "flow_lm.flow_net.res_blocks.0",
    "mimi.quantizer",
    "mimi.upsample",
    "mimi.decoder_transformer",
    "mimi.decoder_transformer.transformer.layers.0",
    "mimi.decoder",
)

MAX_CALLS_PER_MODULE = 4  # steps 0..3; enough to catch prefill vs steady state


def to_numpy(x):
    """Tensor -> ndarray. Returns None for anything that is not array-like.

    Note the ndarray passthrough: `_save` recurses over containers and would
    otherwise re-enter with an already-converted array and drop it on the floor.
    """
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, torch.Tensor):
        return x.detach().to(torch.float32).cpu().numpy()
    if isinstance(x, (tuple, list)):
        return [to_numpy(v) for v in x]
    return None


class Recorder:
    """Collects module inputs and outputs, bounded per module."""

    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.counts: dict[str, int] = {}
        self.index: list[dict] = []
        self.handles = []

    def _save(self, name: str, call: int, kind: str, value, slot: int | None = None):
        arr = to_numpy(value)
        if arr is None:
            return
        if isinstance(arr, list):
            # A module returning a tuple/list (ProjectedTransformer does) or the
            # positional args of any module. Flatten one level, skipping the
            # entries that are not tensors at all.
            for i, sub in enumerate(arr):
                if sub is not None:
                    self._save(name, call, kind, sub, slot=i)
            return
        tag = f"{name}.{kind}{'' if slot is None else f'{slot}'}.call{call}"
        path = self.out_dir / f"{tag}.npy"
        np.save(path, arr)
        self.index.append(
            {
                "module": name,
                "call": call,
                "kind": kind if slot is None else f"{kind}{slot}",
                "file": path.name,
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                # cheap invariants a C implementation can check without the file
                "finite": bool(np.isfinite(arr).all()),
                "mean": float(np.nanmean(arr)) if arr.size else 0.0,
                "absmax": float(np.nanmax(np.abs(arr))) if arr.size else 0.0,
                "sha256": hashlib.sha256(np.ascontiguousarray(arr)).hexdigest()[:16],
            }
        )

    def hook(self, name: str):
        def fn(_module, args, output):
            call = self.counts.get(name, 0)
            if call >= MAX_CALLS_PER_MODULE:
                self.counts[name] = call + 1
                return
            self.counts[name] = call + 1
            self._save(name, call, "in", list(args))
            self._save(name, call, "out", output)

        return fn

    def attach(self, model: torch.nn.Module):
        available = dict(model.named_modules())
        missing = [w for w in WATCHED if w not in available]
        if missing:
            raise SystemExit(
                "upstream module names moved; not found: " + ", ".join(missing)
            )
        for name in WATCHED:
            self.handles.append(available[name].register_forward_hook(self.hook(name)))
        return len(self.handles)

    def detach(self):
        for h in self.handles:
            h.remove()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--language", default="english")
    ap.add_argument("--voice", default="alba")
    ap.add_argument(
        "--text",
        default="Hello world. I am Kyutai's Pocket TTS. I hope you'll like me.",
    )
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--temp", type=float, default=None, help="default: the model's own")
    ap.add_argument("--out", type=Path, default=Path("build/oracle-pocket"))
    ap.add_argument("--wav", type=Path, default=None, help="also write the waveform")
    args = ap.parse_args()

    from pocket_tts import TTSModel  # imported late so --help works without torch

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    model = TTSModel.load_model(language=args.language)
    if args.temp is not None:
        model.temp = args.temp

    # Tokenizer parity is checked separately from the graph: exact IDs, no tolerance.
    tokenizer = model.flow_lm.conditioner.tokenizer
    token_ids = tokenizer(args.text)[0].tolist()
    np.save(out_dir / "stage01.token_ids.npy", np.asarray(token_ids, dtype=np.int64))

    voice_state = model.get_state_for_audio_prompt(args.voice)
    # The voice *is* the KV cache; record its shape so the C loader can assert it.
    voice_desc = {
        k: {kk: list(vv.shape) for kk, vv in v.items() if isinstance(vv, torch.Tensor)}
        for k, v in voice_state.items()
    }

    recorder = Recorder(out_dir)
    hooked = recorder.attach(model)
    print(f"hooked {hooked} modules")

    torch.manual_seed(args.seed)  # re-seed so the noise is a function of --seed alone
    audio = model.generate_audio(voice_state, args.text)
    recorder.detach()

    audio_np = to_numpy(audio).reshape(-1)
    np.save(out_dir / "stage12.waveform.npy", audio_np)

    manifest = {
        "generator": "tools/oracle_pocket.py",
        "upstream": "kyutai-labs/pocket-tts",
        "language": args.language,
        "voice": args.voice,
        "text": args.text,
        "seed": args.seed,
        "temperature": float(model.temp),
        "sampler_decode_steps": int(model.sampler_decode_steps),
        "eos_threshold": float(model.eos_threshold),
        "sample_rate": int(model.sample_rate),
        "token_ids": token_ids,
        "voice_state_shapes": voice_desc,
        "waveform": {
            "samples": int(audio_np.size),
            "seconds": float(audio_np.size / model.sample_rate),
            "peak": float(np.max(np.abs(audio_np))) if audio_np.size else 0.0,
            "rms": float(np.sqrt(np.mean(audio_np**2))) if audio_np.size else 0.0,
            "finite": bool(np.isfinite(audio_np).all()),
        },
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "tensors": recorder.index,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    if args.wav is not None:
        import scipy.io.wavfile

        args.wav.parent.mkdir(parents=True, exist_ok=True)
        scipy.io.wavfile.write(str(args.wav), model.sample_rate, audio_np)

    print(f"wrote {len(recorder.index)} tensors + manifest to {out_dir}")
    print(f"  tokens={len(token_ids)}  frames~{audio_np.size / 1920:.0f}  "
          f"seconds={audio_np.size / model.sample_rate:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
