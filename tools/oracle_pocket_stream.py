#!/usr/bin/env python3
"""Measure what the PocketTTS codec decoder needs in order to stream.

`stream.c` has to decide how audio comes out frame by frame. There are two ways
to do it and this tool settles which, with numbers instead of assumptions:

  A. carry explicit convolution state across chunks, as upstream does
  B. re-decode a suffix with K frames of left context and throw the context away,
     which is what the Magpie/NanoCodec path in `graph.c` does today

For B the question is the smallest K that still reproduces the one-shot output.
That number is the PocketTTS equivalent of `STREAM_CONTEXT_FRAMES 32`, and
`.work/pocket-tts-oracle.md` (E2-3) says it must be measured, not guessed.

Usage:
    uv run --with pocket-tts --with numpy python tools/oracle_pocket_stream.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

CONTEXTS = (0, 1, 2, 4, 8, 16, 32, 64)


def collect_latents(model, voice_state, text: str) -> torch.Tensor:
    """Run a normal generation and capture the latents handed to the codec.

    `mimi.quantizer` sees `[B, C, T]` for every decode call, so concatenating its
    inputs rebuilds the whole sequence regardless of how the worker chunked it.
    """
    chunks: list[torch.Tensor] = []
    handle = model.mimi.quantizer.register_forward_hook(
        lambda _m, args, _o: chunks.append(args[0].detach().clone())
    )
    try:
        model.generate_audio(voice_state, text)
    finally:
        handle.remove()
    if not chunks:
        raise SystemExit("no codec calls captured; upstream decode path moved")
    return torch.cat(chunks, dim=-1).transpose(-1, -2)  # -> [B, T, C]


def codec_state_len(model, frames: int) -> int:
    """Codec state is sized in ENCODER frames, not latent frames.

    The decoder transformer runs after the upsample, so it sees
    `frames * (encoder_frame_rate / frame_rate)` positions — 16 per 80 ms frame
    here. Sizing it in latent frames silently allocates 16x too little and the
    KV write fails at the first chunk.
    """
    stride = int(round(model.mimi.encoder_frame_rate / model.mimi.frame_rate))
    return frames * stride + 64


def decode_once(model, latents: torch.Tensor) -> np.ndarray:
    from pocket_tts.modules.stateful_module import init_states

    state = init_states(model.mimi, batch_size=1,
                        sequence_length=codec_state_len(model, latents.shape[1]))
    with torch.no_grad():
        out = model.mimi.decode_from_latent(latents, state)
    return out.detach().cpu().numpy().reshape(-1)


def decode_streaming(model, latents: torch.Tensor, chunk: int) -> np.ndarray:
    """Option A: one state, carried across chunks.

    `increment_steps` after every call is NOT optional. The codec keeps an
    explicit position counter, separate from the convolution ring buffers, and it
    advances by `encoder_stride` per latent frame. Without it every chunk
    rewrites the decoder transformer's KV cache from position zero and the output
    degrades smoothly as the chunk gets smaller — no crash, no warning.
    """
    from pocket_tts.modules.stateful_module import increment_steps, init_states

    stride = int(round(model.mimi.encoder_frame_rate / model.mimi.frame_rate))
    state = init_states(model.mimi, batch_size=1,
                        sequence_length=codec_state_len(model, latents.shape[1]))
    pieces = []
    with torch.no_grad():
        for start in range(0, latents.shape[1], chunk):
            span = latents[:, start : start + chunk]
            piece = model.mimi.decode_from_latent(span, state)
            increment_steps(model.mimi, state, increment=stride * span.shape[1])
            pieces.append(piece.detach().cpu().numpy().reshape(-1))
    return np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)


def decode_with_context(model, latents: torch.Tensor, chunk: int, ctx: int) -> np.ndarray:
    """Option B: a fresh state per chunk, primed with `ctx` frames of left context."""
    from pocket_tts.modules.stateful_module import init_states

    frame = model.mimi.frame_size
    pieces = []
    with torch.no_grad():
        for start in range(0, latents.shape[1], chunk):
            lo = max(0, start - ctx)
            state = init_states(model.mimi, batch_size=1,
                                sequence_length=codec_state_len(model, chunk + ctx))
            # One call per chunk, so no increment_steps is needed here: the state
            # is thrown away afterwards. That is the whole point of option B.
            piece = model.mimi.decode_from_latent(latents[:, lo : start + chunk], state)
            audio = piece.detach().cpu().numpy().reshape(-1)
            pieces.append(audio[(start - lo) * frame :])
    return np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)


def compare(ref: np.ndarray, got: np.ndarray) -> dict:
    n = min(ref.size, got.size)
    a, b = ref[:n], got[:n]
    diff = np.abs(a - b)
    denom = float(np.sqrt(np.mean(a**2))) or 1.0
    return {
        "samples": int(n),
        "length_delta": int(got.size - ref.size),
        "identical": bool(np.array_equal(a, b)),
        "max_abs": float(diff.max()) if n else 0.0,
        "rel_rms": float(np.sqrt(np.mean(diff**2)) / denom) if n else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--language", default="english")
    ap.add_argument("--voice", default="alba")
    ap.add_argument("--text", default="Hello world. I am Kyutai's Pocket TTS. I hope you'll like me.")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--chunk", type=int, default=4, help="frames emitted per chunk")
    ap.add_argument("--out", type=Path, default=Path("build/oracle-pocket/streaming.json"))
    args = ap.parse_args()

    from pocket_tts import TTSModel

    torch.manual_seed(args.seed)
    model = TTSModel.load_model(language=args.language)
    voice_state = model.get_state_for_audio_prompt(args.voice)

    torch.manual_seed(args.seed)
    latents = collect_latents(model, voice_state, args.text)
    frames = latents.shape[1]
    stride = int(round(model.mimi.encoder_frame_rate / model.mimi.frame_rate))
    print(f"latents: {tuple(latents.shape)}  frames={frames}  "
          f"frame_size={model.mimi.frame_size}  encoder_stride={stride}")

    reference = decode_once(model, latents)
    print(f"one-shot decode: {reference.size} samples ({reference.size / model.sample_rate:.2f}s)")

    report = {
        "language": args.language,
        "voice": args.voice,
        "seed": args.seed,
        "frames": int(frames),
        "chunk_frames": args.chunk,
        "frame_size": int(model.mimi.frame_size),
        "reference_samples": int(reference.size),
    }

    report["carried_state"] = {}
    for chunk in (1, 2, 4, 16):
        res = compare(reference, decode_streaming(model, latents, chunk))
        report["carried_state"][str(chunk)] = res
        print(f"A) carried state, chunk={chunk:3d}: max_abs={res['max_abs']:.3e} "
              f"rel_rms={res['rel_rms']:.3e} identical={res['identical']}")

    report["left_context"] = {}
    for ctx in CONTEXTS:
        got = decode_with_context(model, latents, args.chunk, ctx)
        res = compare(reference, got)
        report["left_context"][str(ctx)] = res
        print(f"B) ctx={ctx:3d} frames: max_abs={res['max_abs']:.3e} "
              f"rel_rms={res['rel_rms']:.3e} identical={res['identical']}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
