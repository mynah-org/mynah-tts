#!/usr/bin/env python3
"""Generate a tiny deterministic Magpie-shaped model pack.

The `graph.c` split (E1) has to prove one thing: that it did not change the
numbers. Nothing in the repo can prove that today — `models/` is empty, and both
of `graph.c`'s self-tests return 0 without executing anything outside Accelerate,
so on Linux they are no-ops. `make stream-test` and `make server-test` both
refuse to run without a pack.

This produces a pack that exercises the whole graph — text encoder, decoder with
cross-attention, local transformer, NanoCodec, WAV — with random but *fixed*
weights. The audio is noise. That is fine: for "did the refactor change the
numbers", a deterministic pack is exactly as good as the real checkpoint, and it
costs a few seconds instead of a 2 GB download.

It is not a substitute for NeMo oracle parity, which is a different question and
already closed (`docs/oracle-parity.md`).

Shapes are not free parameters. These are literals in `src/graph.c` and the pack
must match them or the graph reads out of bounds without saying so:

  * heads = 12, so `hidden_dim` must be divisible by 12   (graph.c:739,1016,2970)
  * ffn = 4 * hidden, conv kernel = 3                      (graph.c:739)
  * frame_stacking_factor = 2, hard-failed otherwise       (graph.c:770)
  * local_transformer_layers <= 4                          (graph.c:834)
  * codec: FSQ levels {8,7,6,6} = 2016, latent 32,
    pre-conv 864 channels, 5 stages rates {8,8,4,2,2}      (graph.c:2653-2687)

Usage:
    uv run --with numpy --with safetensors python tools/make_fake_pack.py \
        --output models/fake-magpie
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

# --- literals mirrored from src/graph.c; see the docstring ------------------
HEADS = 12
FFN_MULT = 4
CONV_KERNEL = 3
STACKING = 2
CODEBOOKS = 8
CODEBOOK_SIZE = 2016          # = 8 * 7 * 6 * 6, the FSQ level product
FSQ_LEVELS = [8, 7, 6, 6]
LATENT_CHANNELS = 32
CODEC_PRE_CHANNELS = 864
CODEC_RATES = [8, 8, 4, 2, 2]  # product 1024 = samples per frame
RES_KERNELS = [3, 7, 11]
RES_DILATIONS = [1, 3, 5]
SAMPLE_RATE = 22050
SAMPLES_PER_FRAME = 1024


def maker(seed: int):
    """One generator, drawn in a fixed order, so the pack is reproducible."""
    rng = np.random.default_rng(seed)

    def weights(*shape: int, scale: float | None = None) -> np.ndarray:
        fan_in = shape[-1] if len(shape) > 1 else shape[0]
        s = scale if scale is not None else 1.0 / math.sqrt(fan_in)
        return (rng.standard_normal(shape) * s).astype(np.float32)

    def ones(*shape: int) -> np.ndarray:
        return np.ones(shape, dtype=np.float32)

    def zeros(*shape: int) -> np.ndarray:
        return np.zeros(shape, dtype=np.float32)

    return weights, ones, zeros


def transformer_block(t, prefix, layers, width, max_length, cross, w, one):
    """Shapes read off transformer_stack / self_attention / cross_attention /
    causal_conv_ffn in src/graph.c."""
    ffn = width * FFN_MULT
    t[f"{prefix}.position_embeddings.weight"] = w(max_length, width, scale=0.02)
    for layer in range(layers):
        p = f"{prefix}.layers.{layer}"
        t[f"{p}.norm_self.weight"] = one(width)
        t[f"{p}.self_attention.qkv_net.weight"] = w(3 * width, width)
        t[f"{p}.self_attention.o_net.weight"] = w(width, width)
        if cross:
            t[f"{p}.norm_xattn_query.weight"] = one(width)
            t[f"{p}.norm_xattn_memory.weight"] = one(width)
            t[f"{p}.cross_attention.q_net.weight"] = w(width, width)
            t[f"{p}.cross_attention.kv_net.weight"] = w(2 * width, width)
            t[f"{p}.cross_attention.o_net.weight"] = w(width, width)
        t[f"{p}.norm_pos_ff.weight"] = one(width)
        # conv weights are [out][in][kernel] row-major (graph.c:352-360)
        t[f"{p}.pos_ff.proj.conv.weight"] = w(ffn, width, CONV_KERNEL)
        t[f"{p}.pos_ff.o_net.conv.weight"] = w(width, ffn, CONV_KERNEL)
    t[f"{prefix}.norm_out.weight"] = one(width)


def build_tts(cfg, seed):
    w, one, zero = maker(seed)
    width = cfg["hidden_dim"]
    streams = STACKING * CODEBOOKS
    audio_vocab = cfg["audio_vocab_size"]
    t: dict[str, np.ndarray] = {}

    t["text_embedding.weight"] = w(cfg["text_vocab_size"], width, scale=0.02)
    transformer_block(t, "encoder", cfg["encoder_layers"], width,
                      cfg["text_max_length"], False, w, one)

    # baked_context_embedding is [speakers, context_length * width] (graph.c:3772)
    t["baked_context_embedding.weight"] = w(
        cfg["speaker_count"], cfg["context_length"] * width, scale=0.02)

    transformer_block(t, "decoder", cfg["decoder_layers"], width,
                      cfg["decoder_max_length"], True, w, one)

    for stream in range(streams):
        t[f"audio_embeddings.{stream}.weight"] = w(audio_vocab, width, scale=0.02)

    # Non-local path: one flat projection over every stream (graph.c:3866-3890)
    t["final_proj.weight"] = w(streams * audio_vocab, width)
    t["final_proj.bias"] = zero(streams * audio_vocab)

    transformer_block(t, "local_transformer", cfg["local_transformer_layers"],
                      width, streams + 1, False, w, one)
    # local_transformer has no cross-attention and no norm_out in the fast path,
    # but transformer_block writes norm_out harmlessly; keep it for symmetry.
    for stream in range(streams):
        t[f"local_transformer_out_projections.{stream}.weight"] = w(audio_vocab, width)
        t[f"local_transformer_out_projections.{stream}.bias"] = zero(audio_vocab)
    return t


def build_codec(seed):
    """decode_codec in src/graph.c drives these shapes; none are configurable."""
    w, one, zero = maker(seed)
    t: dict[str, np.ndarray] = {}

    t["audio_decoder.pre_conv.conv.weight"] = w(CODEC_PRE_CHANNELS, LATENT_CHANNELS, 7)
    t["audio_decoder.pre_conv.conv.bias"] = zero(CODEC_PRE_CHANNELS)

    channels = CODEC_PRE_CHANNELS
    for stage, rate in enumerate(CODEC_RATES):
        # half_snake reads alpha over channels / 2 (graph.c:2258)
        t[f"audio_decoder.activations.{stage}.activation.snake_act.alpha"] = one(channels // 2)
        nxt = channels // 2
        # transposed conv weight is [in][out/groups][kernel], groups = out (graph.c:2189)
        t[f"audio_decoder.up_sample_conv_layers.{stage}.conv.weight"] = w(channels, 1, rate * 2)
        t[f"audio_decoder.up_sample_conv_layers.{stage}.conv.bias"] = zero(nxt)
        for branch, kernel in enumerate(RES_KERNELS):
            for dilation in range(len(RES_DILATIONS)):
                p = f"audio_decoder.res_layers.{stage}.res_blocks.{branch}.res_blocks.{dilation}"
                t[f"{p}.input_activation.activation.snake_act.alpha"] = one(nxt // 2)
                t[f"{p}.input_conv.conv.weight"] = w(nxt, nxt, kernel)
                t[f"{p}.input_conv.conv.bias"] = zero(nxt)
                t[f"{p}.skip_activation.activation.snake_act.alpha"] = one(nxt // 2)
                t[f"{p}.skip_conv.conv.weight"] = w(nxt, nxt, kernel)
                t[f"{p}.skip_conv.conv.bias"] = zero(nxt)
        channels = nxt

    t["audio_decoder.post_activation.activation.snake_act.alpha"] = one(channels // 2)
    t["audio_decoder.post_conv.conv.weight"] = w(1, channels, 3)
    t["audio_decoder.post_conv.conv.bias"] = zero(1)
    return t


# `mynah_tokenizer_open` loads the vocabulary of EVERY language at startup and
# fails the whole pack if one is missing (tokenizer.c:457-540), so a usable pack
# needs all eight files even though the tests only exercise one language. The
# G2P dictionaries are optional: without them `encode_ipa` falls back to
# per-character graphemes, which is why the vocabularies below are just
# characters. `vocab_load_json` assigns ids by position and skips strings that
# are empty or 63+ bytes long.
VOCAB_NAMES = [
    "english_phoneme", "german_phoneme", "spanish_phoneme",
    "portuguese_Brazilian_phoneme", "hindi_phoneme",
    "arabic_MSA_chartokenizer", "japanese_phoneme", "mandarin_phoneme",
]


def write_tokenizer(out: Path, prefixes: tuple[str, ...] = ("", "#")) -> int:
    tokenizer_dir = out / "tokenizer"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    base = ["<pad>", "<unk>", " "]
    chars = [chr(c) for c in range(0x21, 0x7F)]            # printable ASCII
    ipa = list("ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧ")
    # German and Brazilian Portuguese prefix their graphemes (tokenizer.c:475-481),
    # so both the bare and the prefixed form have to resolve.
    tokens = base + chars + [p + c for p in prefixes if p for c in chars] + ipa
    payload = json.dumps(tokens, ensure_ascii=False, indent=0)
    for name in VOCAB_NAMES:
        (tokenizer_dir / f"{name}_vocab.json").write_text(payload, encoding="utf-8")
    return len(tokens)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", type=Path, default=Path("models/fake-magpie"))
    ap.add_argument("--seed", type=int, default=20260912)
    ap.add_argument("--hidden-dim", type=int, default=24, help="must be divisible by 12")
    ap.add_argument("--encoder-layers", type=int, default=1)
    ap.add_argument("--decoder-layers", type=int, default=1)
    ap.add_argument("--local-layers", type=int, default=1, help="at most 4")
    ap.add_argument("--max-decoder-steps", type=int, default=8)
    # ByT5 emits byte + 3 (tokenizer.c:encode_byt5), so the vocabulary has to
    # clear 258; test_stream.c uses speaker 4 and token ids up to 93.
    ap.add_argument("--text-vocab", type=int, default=384)
    ap.add_argument("--speakers", type=int, default=6)
    args = ap.parse_args()

    if args.hidden_dim % HEADS:
        raise SystemExit(f"--hidden-dim must be divisible by {HEADS} (graph.c hardcodes heads)")
    if args.local_layers > 4:
        raise SystemExit("--local-layers must be <= 4 (fixed [4] arrays at graph.c:834)")

    cfg = {
        "hidden_dim": args.hidden_dim,
        "encoder_layers": args.encoder_layers,
        "decoder_layers": args.decoder_layers,
        "local_transformer_layers": args.local_layers,
        "text_vocab_size": args.text_vocab,
        # test_server.sh posts a ~150-character sentence; ByT5 is one id per byte.
        "text_max_length": 512,
        "decoder_max_length": 64,
        "context_length": 4,
        "speaker_count": args.speakers,
        "audio_vocab_size": CODEBOOK_SIZE + 2,   # + AUDIO_BOS, AUDIO_EOS
        "max_decoder_steps": args.max_decoder_steps,
    }

    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    tts = build_tts(cfg, args.seed)
    codec = build_codec(args.seed + 1)
    save_file(tts, str(out / "tts.safetensors"))
    save_file(codec, str(out / "codec.safetensors"))

    manifest = {
        "format": 1,
        "engine": "magpie",
        "revision": "fake-deterministic",
        "dtype": "float32",
        "source": "tools/make_fake_pack.py (synthetic, not a real model)",
        "sample_rate": SAMPLE_RATE,
        "frame_rate": SAMPLE_RATE / SAMPLES_PER_FRAME,
        "frame_stacking_factor": STACKING,
        "codebook_count": CODEBOOKS,
        "codebook_size": CODEBOOK_SIZE,
        "audio_vocab_size": cfg["audio_vocab_size"],
        "audio_bos_id": CODEBOOK_SIZE,
        "audio_eos_id": CODEBOOK_SIZE + 1,
        "stacked_audio_streams": STACKING * CODEBOOKS,
        "hidden_dim": cfg["hidden_dim"],
        "encoder_layers": cfg["encoder_layers"],
        "decoder_layers": cfg["decoder_layers"],
        "local_transformer_layers": cfg["local_transformer_layers"],
        "local_transformer_type": "ar",
        "text_vocab_size": cfg["text_vocab_size"],
        "text_max_length": cfg["text_max_length"],
        "max_decoder_steps": cfg["max_decoder_steps"],
        "speaker_count": cfg["speaker_count"],
        # "it" selects the ByT5 byte-level path (tokenizer.c:469), which needs no
        # vocab file and no G2P dictionary — "en" would require both.
        "languages": ["it"],
        "language_to_tokenizer": {"it": "italian_chartokenizer"},
        "requires_normalized_text": True,
        # The loader's JSON parser is flat `strstr` scanning (mynah_tts.c:112-163),
        # so these have to be top-level keys, not nested under "inference".
        "temperature": 0.7,
        "topk": 80,
        "min_generated_frames": 4,
        "inference": {"min_generated_frames": 4},
        "codec": {
            "sample_rate": SAMPLE_RATE,
            "samples_per_frame": SAMPLES_PER_FRAME,
            "codebooks": CODEBOOKS,
            "levels": FSQ_LEVELS,
        },
        "weights": {"tts": "tts.safetensors", "codec": "codec.safetensors"},
    }
    (out / "model.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    # speakers.json is a flat {"Name": id} object (server/main.c:178). The names
    # the server test asks for must be present.
    names = ["Sofia", "Leo", "Mila", "Nico", "Iris", "Bruno", "Vera", "Enzo"]
    speakers = {names[i % len(names)]: i for i in range(cfg["speaker_count"])}
    (out / "speakers.json").write_text(json.dumps(speakers, indent=2) + "\n")
    vocab_size = write_tokenizer(out)
    if vocab_size > cfg["text_vocab_size"]:
        raise SystemExit(f"tokenizer vocabulary ({vocab_size}) exceeds text_vocab_size")

    (out / "source.json").write_text(json.dumps({
        "created_at": datetime.now(timezone.utc).isoformat(),
        "converter": "tools/make_fake_pack.py",
        "synthetic": True,
        "seed": args.seed,
        "note": "Random weights. Only useful as a fixed point for refactor goldens.",
    }, indent=2, sort_keys=True) + "\n")

    tts_bytes = sum(a.nbytes for a in tts.values())
    codec_bytes = sum(a.nbytes for a in codec.values())
    print(f"wrote synthetic pack: {out}")
    print(f"  tts  : {len(tts):4d} tensors, {tts_bytes / 1e6:8.2f} MB")
    print(f"  codec: {len(codec):4d} tensors, {codec_bytes / 1e6:8.2f} MB")
    print(f"  tokenizer: {len(VOCAB_NAMES)} vocabularies, {vocab_size} tokens each")
    print(f"  hidden={cfg['hidden_dim']} enc={cfg['encoder_layers']} "
          f"dec={cfg['decoder_layers']} local={cfg['local_transformer_layers']} "
          f"seed={args.seed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
