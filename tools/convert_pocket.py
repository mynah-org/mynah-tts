#!/usr/bin/env python3
"""Convert a Kyutai PocketTTS language checkpoint into a mynah model pack.

Offline tooling only: the C runtime never runs this, and this never imports
torch, transformers or `pocket_tts`.  It reads safetensors headers and raw
tensor bytes directly, so `numpy` plus the standard library is the whole
dependency list.  BF16 payloads are copied byte for byte; nothing round-trips
through a framework that would silently re-quantize them.

    uv run --with numpy python tools/convert_pocket.py \
        --language english --output models/pocket-english

Source layout (already downloaded, gated repo, see `--source`):

    <snapshot>/languages/<language>/model.safetensors      schema-derived tensors
                                                               (BF16, F32, or mixed)
    <snapshot>/languages/<language>/tokenizer.model        SentencePiece Unigram
    <snapshot>/languages/<language>/embeddings/<voice>.safetensors   KV cache, F32

Produced pack:

    model.json            engine "pocket", every dimension derived from a tensor
    tts.safetensors       flow_lm.* + mimi.*  (minus the cloning group with --no-cloning)
    tokenizer.model       copied verbatim from the language directory
    voices/<name>.safetensors   the voice KV caches, cast to F16
    speakers.json         name, frames, dataset, license, commercial_use
    source.json           repo, revision, sha256 of every source file
    LICENSES/MODEL_LICENSE.md   CC-BY-4.0 + attribution + modification statement
    LICENSES/VOICES.md          the per-voice table
    NOTICE                upstream prohibited-use, passed on as a notice

### Why every scalar sits at the top level of `model.json`

The runtime's manifest reader is a flat `strstr` scan (`src/mynah_tts.c`,
`json_value`): it looks for the *first* occurrence of `"key"` anywhere in the
file and parses whatever follows.  It has no notion of nesting, so a key inside
a nested object is indistinguishable from a top-level one — and worse, a nested
key that sorts earlier will shadow the real one.  `tools/make_fake_pack.py`
already documents this for Magpie.

Consequence, enforced by `check_flat_manifest()` below: every scalar the loader
could ever want is a top-level key, and every such key name occurs **exactly
once** in the serialized document.  Nested objects exist only for things a C
loader will not read with `json_value` (file lists, provenance).

### No magic numbers

Every dimension in `model.json` is derived from a tensor shape in the source
checkpoint, and cross-checked against at least one other tensor.  What genuinely
cannot be read out of the weights lives in the constants block below, each with
the note section it came from.  A shape that does not match its expectation is a
hard failure: a broken pack that loads is worse than no pack.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import struct
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Callable, Iterable, NoReturn

import numpy as np

# --------------------------------------------------------------------------
# Constants: the facts that are NOT readable from a tensor shape.
# Sources are `.work/pocket-tts-model-facts.md` (read out of the weights and the
# reference implementation) and `.work/licensing-and-voice-policy.md`.
# --------------------------------------------------------------------------

REPO_ID = "kyutai/pocket-tts"
REPO_URL = "https://huggingface.co/kyutai/pocket-tts"
CODE_URL = "https://github.com/kyutai-labs/pocket-tts"
PAPER_URL = "https://arxiv.org/abs/2509.06926"

# Mimi runs at 24 kHz mono. facts §1: "1920 samples = 80 ms @ 24 kHz mono".
# Not derivable from any tensor; frame_rate and frame_ms are computed from it
# together with the SEANet ratios, so a wrong value here fails EXPECTED_* below.
SAMPLE_RATE = 24000
AUDIO_CHANNELS = 1

# facts §1: decoder transformer "context 250" (Mimi's streaming attention
# window, in frames). Not a tensor shape.
DECODER_TRANSFORMER_CONTEXT = 250

# facts §1 / §11: DEFAULT_SAMPLER_DECODE_STEPS = 1, the head is LSD-distilled,
# CFG is distilled into the weights so there is no CFG at inference.
FLOW_DECODE_STEPS = 1
USES_CFG = False

# facts §1: `flow_lm.out_eos` is a single logit, thresholded at -4.0.
EOS_THRESHOLD = -4.0

# facts §1: default temperature 0.7; the English config overrides it to 0.3.
DEFAULT_TEMPERATURE = 0.7
ENGLISH_TEMPERATURE = 0.3

# facts §8: on long inputs the model skips parts of sentences; upstream
# mitigates with MAX_TOKEN_PER_CHUNK = 50. This is model behaviour to reproduce,
# not a runtime tuning knob.
MAX_TOKENS_PER_CHUNK = 50

# facts §11: two different LayerNorms with two different epsilons. The backbone
# uses torch.nn.LayerNorm(eps=1e-5) with weight and bias; the flow head uses a
# custom LayerNorm with eps=1e-6 and var(unbiased=False).
BACKBONE_LAYERNORM_EPS = 1e-5
FLOW_LAYERNORM_EPS = 1e-6

# facts §11: backbone FFN activation is F.gelu(x, approximate="tanh"), FFN is
# not gated; the flow head MLPs use SiLU.
BACKBONE_FFN_ACTIVATION = "gelu_tanh"
FLOW_MLP_ACTIVATION = "silu"

# facts §5: nn.Embedding(n_bins + 1, dim) — the last row is padding and the
# tokenizer never emits it. The tokenizer adds no BOS and no EOS of its own.
TOKENIZER_KIND = "sentencepiece_unigram"
TOKENIZER_ADDS_BOS = False
TOKENIZER_ADDS_EOS = False

# facts §12: `flow_lm.forward` replaces NaN positions with `bos_emb`, and
# `_expand_kv_cache` fills unused KV slots with NaN. A C implementation must
# either reproduce the sentinel or track validity explicitly.
NAN_IS_BOS_SENTINEL = True

# Expectations from the note, checked against what is actually derived. These
# are assertions, not inputs: if the derived value differs, the checkpoint is
# not the one this converter was written against and it says so.
EXPECTED_SAMPLES_PER_FRAME = 1920   # facts §1
EXPECTED_FRAME_RATE = 12.5          # facts §1
EXPECTED_SEANET_RATIOS = (6, 5, 4)  # facts §1
# The exact tensor count is not an architectural input: every backbone block
# contributes the same eight tensors.  Keep the non-backbone count as a guard so
# a missing/extra flow or Mimi tensor is still rejected, while a deeper valid
# backbone is accepted.  166 is the current generation (with
# bos_before_voice); 165 is the 2026-01 generation.
EXPECTED_NON_BACKBONE_TENSOR_COUNTS = {165, 166}
BACKBONE_TENSOR_NAMES = (
    "self_attn.in_proj.weight", "self_attn.out_proj.weight",
    "linear1.weight", "linear2.weight",
    "norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias",
)

# Per-voice provenance, from `.work/licensing-and-voice-policy.md`, which read it
# from https://huggingface.co/kyutai/tts-voices. jean (EARS) and cosette
# (Expresso) are CC BY-NC 4.0, so they are not usable commercially — in every
# language, since the same 26 names ship in every pack.
# A voice that is not in this table is marked "unknown" / commercial_use false.
# The converter does not guess: the newer LibriVox-style names look like CC0 but
# the name -> file mapping has not been verified, and the per-language defaults
# (estelle, giovanni, juergen, lola, rafael) have no published provenance at all.
VOICE_LICENSES: dict[str, tuple[str, str, bool]] = {
    "alba":    ("alba-mackenna",   "CC-BY-4.0",    True),
    "marius":  ("voice-donations", "CC0-1.0",      True),
    "javert":  ("voice-donations", "CC0-1.0",      True),
    "fantine": ("VCTK",            "CC-BY-4.0",    True),
    "eponine": ("VCTK",            "CC-BY-4.0",    True),
    "azelma":  ("VCTK",            "CC-BY-4.0",    True),
    "jean":    ("EARS",            "CC-BY-NC-4.0", False),
    "cosette": ("Expresso",        "CC-BY-NC-4.0", False),
}
UNKNOWN_VOICE = ("unknown", "unknown", False)

# Verbatim from the gated model card ("Prohibited use" / extra_gated_prompt).
# E6: this is a condition of *access*, not a license term. CC-BY-4.0 forbids
# imposing additional restrictions on recipients, so it travels as a notice.
PROHIBITED_USE = (
    "Use of our model must comply with all applicable laws and regulations and "
    "must not result in, involve, or facilitate any illegal, harmful, deceptive, "
    "fraudulent, or unauthorized activity. Prohibited uses include, without "
    "limitation, voice impersonation or cloning without explicit and lawful "
    "consent; misinformation, disinformation, or deception (including fake news, "
    "fraudulent calls, or presenting generated content as genuine recordings of "
    "real people or events); and the generation of unlawful, harmful, libelous, "
    "abusive, harassing, discriminatory, hateful, or privacy-invasive content. "
    "We disclaim all liability for any non-compliant use."
)

HF_REPO_DIR = "models--kyutai--pocket-tts"


def fail(message: str) -> NoReturn:
    raise SystemExit(f"convert_pocket: {message}")


# --------------------------------------------------------------------------
# Minimal safetensors reader/writer. No torch, and BF16 stays BF16.
# --------------------------------------------------------------------------

ST_ITEMSIZE = {
    "BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
    "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
    "I32": 4, "U32": 4, "F32": 4,
    "I64": 8, "U64": 8, "F64": 8,
}


@dataclass(frozen=True)
class TensorRef:
    """One tensor in a source safetensors file, as byte range plus shape."""

    name: str
    # "BF16", "F32", or "mixed" for the source checkpoint.  The 24L
    # English checkpoint intentionally stores flow/backbone in F32 and Mimi in
    # BF16; the pack target policy is handled independently by convert().
    dtype: str
    shape: tuple[int, ...]
    start: int
    end: int

    @property
    def nbytes(self) -> int:
        return self.end - self.start


def st_read_header(path: Path) -> tuple[dict[str, TensorRef], dict[str, str], int]:
    """Parse the safetensors header and validate every byte range."""
    try:
        with path.open("rb") as stream:
            raw_length = stream.read(8)
            if len(raw_length) != 8:
                fail(f"{path} is too short to be a safetensors file")
            header_length = struct.unpack("<Q", raw_length)[0]
            if header_length == 0 or header_length > 100 * 1024 * 1024:
                fail(f"{path} has an implausible header length {header_length}")
            header_bytes = stream.read(header_length)
            if len(header_bytes) != header_length:
                fail(f"{path} is truncated inside the header")
            file_size = path.stat().st_size
    except OSError as error:
        fail(f"cannot read {path}: {error}")
    try:
        header = json.loads(header_bytes)
    except json.JSONDecodeError as error:
        fail(f"{path} has an unparsable safetensors header: {error}")

    metadata = header.pop("__metadata__", {}) or {}
    if not isinstance(metadata, dict):
        fail(f"{path} has a non-object __metadata__")
    data_start = 8 + header_length
    refs: dict[str, TensorRef] = {}
    for name, entry in header.items():
        try:
            dtype = str(entry["dtype"])
            shape = tuple(int(v) for v in entry["shape"])
            start, end = (int(v) for v in entry["data_offsets"])
        except (KeyError, TypeError, ValueError) as error:
            fail(f"{path}: malformed header entry for {name!r}: {error}")
        if dtype not in ST_ITEMSIZE:
            fail(f"{path}: tensor {name!r} has unsupported dtype {dtype}")
        expected = ST_ITEMSIZE[dtype] * math.prod(shape) if shape else ST_ITEMSIZE[dtype]
        if end - start != expected:
            fail(f"{path}: tensor {name!r} claims {end - start} bytes but "
                 f"{dtype}{list(shape)} needs {expected}")
        if start < 0 or end < start or data_start + end > file_size:
            fail(f"{path}: tensor {name!r} byte range [{start}, {end}) is out of file")
        refs[name] = TensorRef(name, dtype, shape, data_start + start, data_start + end)
    return refs, {str(k): str(v) for k, v in metadata.items()}, data_start


def st_raw(stream: BinaryIO, ref: TensorRef) -> bytes:
    stream.seek(ref.start)
    blob = stream.read(ref.nbytes)
    if len(blob) != ref.nbytes:
        fail(f"short read for tensor {ref.name!r}")
    return blob


@dataclass
class OutTensor:
    """A tensor to write, sized up front so the header can be built lazily."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    nbytes: int
    emit: Callable[[], bytes]


def st_write(path: Path, tensors: list[OutTensor],
             metadata: dict[str, str] | None = None) -> int:
    """Write a safetensors file deterministically (tensors in the given order).

    Matches the reference serializer: 8-byte little-endian header length, a JSON
    header space-padded so the data section starts 8-byte aligned, then the
    payloads back to back.
    """
    header: dict[str, Any] = {}
    offset = 0
    for tensor in tensors:
        header[tensor.name] = {
            "dtype": tensor.dtype,
            "shape": list(tensor.shape),
            "data_offsets": [offset, offset + tensor.nbytes],
        }
        offset += tensor.nbytes
    if metadata:
        header["__metadata__"] = dict(sorted(metadata.items()))
    encoded = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
    encoded += b" " * ((-(8 + len(encoded))) % 8)
    with path.open("wb") as out:
        out.write(struct.pack("<Q", len(encoded)))
        out.write(encoded)
        for tensor in tensors:
            blob = tensor.emit()
            if len(blob) != tensor.nbytes:
                fail(f"tensor {tensor.name!r} produced {len(blob)} bytes, "
                     f"header says {tensor.nbytes}")
            out.write(blob)
    return path.stat().st_size


# --------------------------------------------------------------------------
# dtype conversions. safetensors is little-endian; refuse to run anywhere else.
# --------------------------------------------------------------------------

def bf16_to_f32(blob: bytes) -> bytes:
    words = np.frombuffer(blob, dtype="<u2").astype(np.uint32) << np.uint32(16)
    return words.astype("<u4").tobytes()


def f32_to_bf16(blob: bytes) -> bytes:
    words = np.frombuffer(blob, dtype="<u4")
    # round-half-to-even on the discarded low 16 bits
    bias = ((words >> np.uint32(16)) & np.uint32(1)) + np.uint32(0x7FFF)
    rounded = ((words + bias) >> np.uint32(16)).astype(np.uint16)
    # keep NaN a NaN: adding the bias can carry a NaN into an infinity
    nan = (words & np.uint32(0x7FFFFFFF)) > np.uint32(0x7F800000)
    rounded = np.where(nan, (words >> np.uint32(16)).astype(np.uint16) | np.uint16(0x40),
                       rounded)
    return rounded.astype("<u2").tobytes()


def f32_to_f16(blob: bytes, what: str) -> bytes:
    values = np.frombuffer(blob, dtype="<f4")
    if not np.isfinite(values).all():
        fail(f"{what} contains non-finite values; refusing to cast to F16")
    peak = float(np.abs(values).max()) if values.size else 0.0
    if peak > 65504.0:
        fail(f"{what} has peak magnitude {peak:g}, which overflows F16")
    return values.astype("<f2").tobytes()


def f16_max_abs_error(blob: bytes) -> tuple[float, float]:
    values = np.frombuffer(blob, dtype="<f4")
    if values.size == 0:
        return 0.0, 0.0
    error = np.abs(values.astype(np.float16).astype(np.float32) - values)
    peak = float(np.abs(values).max())
    return float(error.max()), (float(error.max()) / peak if peak > 0 else 0.0)


# --------------------------------------------------------------------------
# Source discovery
# --------------------------------------------------------------------------

def default_snapshot() -> Path:
    """Resolve the pinned snapshot through the Hugging Face cache's refs/main.

    refs/main is written by the downloader and names exactly one revision, so
    there is nothing to guess even when several snapshots are present.
    """
    roots = []
    if os.environ.get("HF_HUB_CACHE"):
        roots.append(Path(os.environ["HF_HUB_CACHE"]))
    if os.environ.get("HF_HOME"):
        roots.append(Path(os.environ["HF_HOME"]) / "hub")
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")

    tried = []
    for root in roots:
        repo = root / HF_REPO_DIR
        tried.append(str(repo))
        if not repo.is_dir():
            continue
        ref = repo / "refs" / "main"
        if ref.is_file():
            revision = ref.read_text(encoding="utf-8").strip()
            snapshot = repo / "snapshots" / revision
            if snapshot.is_dir():
                return snapshot
            fail(f"{ref} points at {revision}, which is not in {repo / 'snapshots'}")
        snapshots = sorted(p for p in (repo / "snapshots").glob("*") if p.is_dir())
        if len(snapshots) == 1:
            return snapshots[0]
        fail(f"{repo} has {len(snapshots)} snapshots and no refs/main; "
             f"pass --source explicitly")
    fail("cannot find the PocketTTS download. Looked for: " + ", ".join(tried) +
         ". Pass --source <snapshot directory>.")


def available_languages(snapshot: Path) -> list[str]:
    languages = snapshot / "languages"
    if not languages.is_dir():
        return []
    return sorted(p.name for p in languages.iterdir()
                  if p.is_dir() and (p / "model.safetensors").exists())


# --------------------------------------------------------------------------
# SentencePiece: count the pieces without importing sentencepiece.
# --------------------------------------------------------------------------

def sentencepiece_piece_count(path: Path) -> int:
    """Count top-level `pieces` entries in a SentencePiece ModelProto.

    ModelProto field 1 is `repeated SentencePiece pieces`. Walking the top level
    of the protobuf is enough to count them; nothing is decoded.
    """
    blob = path.read_bytes()
    index = 0
    count = 0

    def varint(at: int) -> tuple[int, int]:
        result = 0
        shift = 0
        while True:
            if at >= len(blob):
                fail(f"{path} is a truncated protobuf")
            byte = blob[at]
            at += 1
            result |= (byte & 0x7F) << shift
            if not byte & 0x80:
                return result, at
            shift += 7
            if shift > 63:
                fail(f"{path} has an overlong varint")

    while index < len(blob):
        key, index = varint(index)
        field, wire = key >> 3, key & 7
        if wire == 2:
            length, index = varint(index)
            if field == 1:
                count += 1
            index += length
        elif wire == 0:
            _, index = varint(index)
        elif wire == 5:
            index += 4
        elif wire == 1:
            index += 8
        else:
            fail(f"{path} is not a SentencePiece model (wire type {wire})")
        if index > len(blob):
            fail(f"{path} is a truncated protobuf")
    return count


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# --------------------------------------------------------------------------
# Tensor grouping
# --------------------------------------------------------------------------

# Tensors used only by the voice-cloning path (facts §12: `_encode_audio`).
# `--no-cloning` drops exactly this set and nothing else.
CLONING_PREFIXES = ("mimi.encoder", "mimi.downsample")
CLONING_NAMES = ("flow_lm.speaker_proj_weight",)
# Everything else the decode path needs. `mimi.quantizer` is a pass-through
# Conv1d 32 -> 512 (a DummyQuantizer, facts §1) and `mimi.upsample` is the
# 16x transposed conv in front of the SEANet decoder: both are core, not codec
# bookkeeping, and dropping them would break generation.
CORE_PREFIXES = ("flow_lm.", "mimi.decoder", "mimi.quantizer", "mimi.upsample")


def classify(name: str) -> str:
    if name in CLONING_NAMES or name.startswith(CLONING_PREFIXES):
        return "cloning"
    if name.startswith(CORE_PREFIXES):
        return "core"
    return "unknown"


# --------------------------------------------------------------------------
# Schema derivation: every number below comes from a tensor shape.
# --------------------------------------------------------------------------

def need(refs: dict[str, TensorRef], name: str) -> TensorRef:
    ref = refs.get(name)
    if ref is None:
        fail(f"checkpoint is missing the required tensor {name!r}")
    return ref


def expect_shape(refs: dict[str, TensorRef], name: str,
                 shape: tuple[int, ...]) -> TensorRef:
    ref = need(refs, name)
    if ref.shape != shape:
        fail(f"tensor {name!r} has shape {list(ref.shape)}, expected {list(shape)}")
    return ref


def layer_indices(refs: Iterable[str], pattern: str, what: str) -> int:
    compiled = re.compile(pattern)
    found = {int(m.group(1)) for name in refs if (m := compiled.match(name))}
    if not found:
        fail(f"checkpoint has no {what}")
    if found != set(range(len(found))):
        fail(f"{what} indices are not contiguous from 0: {sorted(found)}")
    return len(found)


@dataclass
class Schema:
    """Everything the pack declares, derived and cross-checked."""

    tensor_count: int
    dtype: str
    latent_dim: int
    hidden_dim: int
    attention_heads: int
    head_dim: int
    ffn_dim: int
    transformer_layers: int
    text_embedding_rows: int
    text_vocab_size: int
    text_padding_id: int
    flow_dim: int
    flow_res_blocks: int
    flow_time_conditions: int
    flow_time_freqs: int
    speaker_proj_input_dim: int
    has_bos_before_voice: bool
    codec_dim: int
    codec_ratios: tuple[int, ...]
    codec_upsample_stride: int
    codec_downsample_stride: int
    codec_transformer_layers: int
    codec_transformer_dim: int
    codec_transformer_ffn: int
    codec_transformer_heads: int
    seanet_n_filters: int
    seanet_kernel_size: int
    seanet_residual_kernel_size: int
    seanet_last_kernel_size: int
    seanet_n_residual_layers: int
    seanet_compress: int
    samples_per_frame: int
    generation: str


def derive_schema(refs: dict[str, TensorRef], voice_refs: dict[str, TensorRef],
                  voice_name: str) -> Schema:
    dtypes = {ref.dtype for ref in refs.values()}
    unsupported = dtypes - {"BF16", "F32"}
    if unsupported:
        fail(f"checkpoint uses unsupported dtypes {sorted(unsupported)}; "
             "PocketTTS expects BF16/F32 weights")
    dtype = next(iter(dtypes)) if len(dtypes) == 1 else "mixed"

    # --- latent and model width -------------------------------------------
    latent_dim = need(refs, "flow_lm.bos_emb").shape[0]
    input_linear = need(refs, "flow_lm.input_linear.weight")
    if len(input_linear.shape) != 2 or input_linear.shape[1] != latent_dim:
        fail(f"flow_lm.input_linear.weight is {list(input_linear.shape)}, expected "
             f"[hidden, {latent_dim}]")
    hidden_dim = input_linear.shape[0]
    expect_shape(refs, "flow_lm.emb_mean", (latent_dim,))
    expect_shape(refs, "flow_lm.emb_std", (latent_dim,))
    expect_shape(refs, "flow_lm.out_norm.weight", (hidden_dim,))
    expect_shape(refs, "flow_lm.out_norm.bias", (hidden_dim,))
    expect_shape(refs, "flow_lm.out_eos.weight", (1, hidden_dim))
    expect_shape(refs, "flow_lm.out_eos.bias", (1,))

    # --- backbone ---------------------------------------------------------
    layers = layer_indices(refs, r"^flow_lm\.transformer\.layers\.(\d+)\.",
                           "flow_lm.transformer layers")
    ffn_dim = need(refs, "flow_lm.transformer.layers.0.linear1.weight").shape[0]
    for layer in range(layers):
        prefix = f"flow_lm.transformer.layers.{layer}"
        expect_shape(refs, f"{prefix}.self_attn.in_proj.weight", (3 * hidden_dim, hidden_dim))
        expect_shape(refs, f"{prefix}.self_attn.out_proj.weight", (hidden_dim, hidden_dim))
        expect_shape(refs, f"{prefix}.linear1.weight", (ffn_dim, hidden_dim))
        expect_shape(refs, f"{prefix}.linear2.weight", (hidden_dim, ffn_dim))
        for norm in ("norm1", "norm2"):
            expect_shape(refs, f"{prefix}.{norm}.weight", (hidden_dim,))
            expect_shape(refs, f"{prefix}.{norm}.bias", (hidden_dim,))
        expected = {f"{prefix}.{suffix}" for suffix in BACKBONE_TENSOR_NAMES}
        actual = {name for name in refs if name.startswith(prefix + ".")}
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            fail(f"backbone layer {layer} tensor set is incompatible; "
                 f"missing={missing} extra={extra}")

    non_backbone_count = len(refs) - layers * len(BACKBONE_TENSOR_NAMES)
    if non_backbone_count not in EXPECTED_NON_BACKBONE_TENSOR_COUNTS:
        fail(f"checkpoint has {len(refs)} tensors for {layers} backbone layers: "
             f"{non_backbone_count} non-backbone tensors, expected one of "
             f"{sorted(EXPECTED_NON_BACKBONE_TENSOR_COUNTS)} for the verified "
             "Pocket graph")

    # --- heads: QKV is fused, so the only place the head split is visible is
    #     the voice KV cache, [2, 1, T, heads, head_dim]. facts §4.
    cache_name = "transformer.layers.0.self_attn/cache"
    cache = voice_refs.get(cache_name)
    if cache is None:
        fail(f"voice {voice_name!r} has no {cache_name!r}; the attention head "
             f"split cannot be derived without it")
    if len(cache.shape) != 5 or cache.shape[0] != 2 or cache.shape[1] != 1:
        fail(f"voice {voice_name!r} KV cache is {list(cache.shape)}, expected "
             f"[2, 1, frames, heads, head_dim]")
    attention_heads, head_dim = cache.shape[3], cache.shape[4]
    if attention_heads * head_dim != hidden_dim:
        fail(f"voice KV says heads*head_dim = {attention_heads}*{head_dim} = "
             f"{attention_heads * head_dim}, but hidden_dim is {hidden_dim}")

    # --- text conditioner --------------------------------------------------
    embed = need(refs, "flow_lm.conditioner.embed.weight")
    if len(embed.shape) != 2 or embed.shape[1] != hidden_dim:
        fail(f"flow_lm.conditioner.embed.weight is {list(embed.shape)}, expected "
             f"[vocab + 1, {hidden_dim}]")
    text_embedding_rows = embed.shape[0]
    text_vocab_size = text_embedding_rows - 1   # facts §5: last row is padding
    text_padding_id = text_embedding_rows - 1

    # --- flow head ---------------------------------------------------------
    input_proj = need(refs, "flow_lm.flow_net.input_proj.weight")
    if len(input_proj.shape) != 2 or input_proj.shape[1] != latent_dim:
        fail(f"flow_lm.flow_net.input_proj.weight is {list(input_proj.shape)}, "
             f"expected [flow_dim, {latent_dim}]")
    flow_dim = input_proj.shape[0]
    expect_shape(refs, "flow_lm.flow_net.input_proj.bias", (flow_dim,))
    expect_shape(refs, "flow_lm.flow_net.cond_embed.weight", (flow_dim, hidden_dim))
    expect_shape(refs, "flow_lm.flow_net.cond_embed.bias", (flow_dim,))
    flow_res_blocks = layer_indices(refs, r"^flow_lm\.flow_net\.res_blocks\.(\d+)\.",
                                    "flow_lm.flow_net.res_blocks")
    for block in range(flow_res_blocks):
        prefix = f"flow_lm.flow_net.res_blocks.{block}"
        # adaLN emits shift, scale, gate -> 3 * flow_dim (facts §11)
        expect_shape(refs, f"{prefix}.adaLN_modulation.1.weight", (3 * flow_dim, flow_dim))
        expect_shape(refs, f"{prefix}.adaLN_modulation.1.bias", (3 * flow_dim,))
        expect_shape(refs, f"{prefix}.in_ln.weight", (flow_dim,))
        expect_shape(refs, f"{prefix}.in_ln.bias", (flow_dim,))
        for linear in ("0", "2"):
            expect_shape(refs, f"{prefix}.mlp.{linear}.weight", (flow_dim, flow_dim))
            expect_shape(refs, f"{prefix}.mlp.{linear}.bias", (flow_dim,))
    # final layer emits shift, scale only -> 2 * flow_dim; norm_final has no
    # affine parameters, so no tensor for it must exist (facts §11).
    expect_shape(refs, "flow_lm.flow_net.final_layer.adaLN_modulation.1.weight",
                 (2 * flow_dim, flow_dim))
    expect_shape(refs, "flow_lm.flow_net.final_layer.adaLN_modulation.1.bias",
                 (2 * flow_dim,))
    expect_shape(refs, "flow_lm.flow_net.final_layer.linear.weight", (latent_dim, flow_dim))
    expect_shape(refs, "flow_lm.flow_net.final_layer.linear.bias", (latent_dim,))
    if any(n.startswith("flow_lm.flow_net.final_layer.norm_final") for n in refs):
        fail("flow_net.final_layer.norm_final has affine parameters in this "
             "checkpoint; the reference implementation has none (facts §11)")

    flow_time_conditions = layer_indices(
        refs, r"^flow_lm\.flow_net\.time_embed\.(\d+)\.", "flow_lm.flow_net.time_embed")
    flow_time_freqs = need(refs, "flow_lm.flow_net.time_embed.0.freqs").shape[0]
    for condition in range(flow_time_conditions):
        prefix = f"flow_lm.flow_net.time_embed.{condition}"
        expect_shape(refs, f"{prefix}.freqs", (flow_time_freqs,))
        # cat([cos, sin]) -> 2 * freqs (facts §11)
        expect_shape(refs, f"{prefix}.mlp.0.weight", (flow_dim, 2 * flow_time_freqs))
        expect_shape(refs, f"{prefix}.mlp.0.bias", (flow_dim,))
        expect_shape(refs, f"{prefix}.mlp.2.weight", (flow_dim, flow_dim))
        expect_shape(refs, f"{prefix}.mlp.2.bias", (flow_dim,))
        expect_shape(refs, f"{prefix}.mlp.3.alpha", (flow_dim,))

    # --- codec: SEANet decoder --------------------------------------------
    quantizer = need(refs, "mimi.quantizer.output_proj.weight")
    if len(quantizer.shape) != 3 or quantizer.shape[1:] != (latent_dim, 1):
        fail(f"mimi.quantizer.output_proj.weight is {list(quantizer.shape)}, "
             f"expected [codec_dim, {latent_dim}, 1]")
    codec_dim = quantizer.shape[0]

    upsample = need(refs, "mimi.upsample.convtr.convtr.weight")
    if len(upsample.shape) != 3 or upsample.shape[0] != codec_dim or upsample.shape[1] != 1:
        fail(f"mimi.upsample.convtr.convtr.weight is {list(upsample.shape)}, "
             f"expected [{codec_dim}, 1, kernel]")
    codec_upsample_stride = stride_from_kernel(upsample.shape[2], "mimi.upsample")

    # convtr kernel = 2 * stride throughout Mimi's SEANet, so the ratios are
    # readable from the decoder's transposed convolutions, in model order.
    convtr = sorted(
        ((int(m.group(1)), name) for name in refs
         if (m := re.match(r"^mimi\.decoder\.model\.(\d+)\.convtr\.weight$", name))))
    if not convtr:
        fail("checkpoint has no mimi.decoder.model.*.convtr.weight; cannot derive "
             "the SEANet ratios")
    codec_ratios = tuple(stride_from_kernel(refs[name].shape[2], name)
                         for _, name in convtr)

    samples_per_frame = codec_upsample_stride * math.prod(codec_ratios)

    codec_transformer_layers = layer_indices(
        refs, r"^mimi\.decoder_transformer\.transformer\.layers\.(\d+)\.",
        "mimi.decoder_transformer layers")
    codec_transformer_dim = need(
        refs, "mimi.decoder_transformer.transformer.layers.0.self_attn.out_proj.weight"
    ).shape[0]
    codec_transformer_ffn = need(
        refs, "mimi.decoder_transformer.transformer.layers.0.linear1.weight").shape[0]
    for layer in range(codec_transformer_layers):
        prefix = f"mimi.decoder_transformer.transformer.layers.{layer}"
        expect_shape(refs, f"{prefix}.self_attn.in_proj.weight",
                     (3 * codec_transformer_dim, codec_transformer_dim))
        expect_shape(refs, f"{prefix}.self_attn.out_proj.weight",
                     (codec_transformer_dim, codec_transformer_dim))
        expect_shape(refs, f"{prefix}.linear1.weight",
                     (codec_transformer_ffn, codec_transformer_dim))
        expect_shape(refs, f"{prefix}.linear2.weight",
                     (codec_transformer_dim, codec_transformer_ffn))
    # QKV is fused here too, so the head count is not in a shape. The only
    # defensible derivation is that Mimi's transformer uses the same head_dim as
    # the backbone (64, measured from the voice KV above). Flagged in model.json
    # as derived, and to be confirmed by the E2 oracle before C relies on it.
    if codec_transformer_dim % head_dim != 0:
        fail(f"codec transformer dim {codec_transformer_dim} is not a multiple of "
             f"the backbone head_dim {head_dim}; the head count cannot be derived")
    codec_transformer_heads = codec_transformer_dim // head_dim

    # --- cloning path / generation ----------------------------------------
    speaker_proj = need(refs, "flow_lm.speaker_proj_weight")
    if len(speaker_proj.shape) != 2 or speaker_proj.shape[0] != hidden_dim:
        fail(f"flow_lm.speaker_proj_weight is {list(speaker_proj.shape)}, expected "
             f"[{hidden_dim}, in]")
    speaker_proj_input_dim = speaker_proj.shape[1]
    downsample = need(refs, "mimi.downsample.conv.conv.weight")
    if len(downsample.shape) != 3 or downsample.shape[1] != codec_dim:
        fail(f"mimi.downsample.conv.conv.weight is {list(downsample.shape)}, "
             f"expected [out, {codec_dim}, kernel]")
    if downsample.shape[0] != speaker_proj_input_dim:
        fail(f"mimi.downsample output channels {downsample.shape[0]} disagree with "
             f"speaker_proj input dim {speaker_proj_input_dim}")
    codec_downsample_stride = stride_from_kernel(downsample.shape[2], "mimi.downsample")
    has_bos_before_voice = "flow_lm.bos_before_voice" in refs
    if has_bos_before_voice:
        expect_shape(refs, "flow_lm.bos_before_voice", (1, 1, hidden_dim))

    # facts §2: the two known generations differ in exactly these three tensors.
    if has_bos_before_voice and speaker_proj_input_dim == latent_dim:
        generation = "current"
    elif not has_bos_before_voice and speaker_proj_input_dim == codec_dim:
        generation = "2026-01"
    else:
        fail(f"unknown PocketTTS generation: bos_before_voice="
             f"{has_bos_before_voice}, speaker_proj input dim="
             f"{speaker_proj_input_dim} (expected {latent_dim} or {codec_dim})")

    # --- expectations from the note ---------------------------------------
    if codec_ratios != EXPECTED_SEANET_RATIOS:
        fail(f"derived SEANet ratios {list(codec_ratios)} differ from the recorded "
             f"{list(EXPECTED_SEANET_RATIOS)} (.work/pocket-tts-model-facts.md §1)")
    if samples_per_frame != EXPECTED_SAMPLES_PER_FRAME:
        fail(f"derived samples_per_frame {samples_per_frame} differs from the "
             f"recorded {EXPECTED_SAMPLES_PER_FRAME}")
    frame_rate = SAMPLE_RATE / samples_per_frame
    if abs(frame_rate - EXPECTED_FRAME_RATE) > 1e-9:
        fail(f"derived frame rate {frame_rate} differs from the recorded "
             f"{EXPECTED_FRAME_RATE}")
    # SEANet geometry. Needed by the decoder (E3) and by the cloning encoder
    # (E7), and absent from model.json until now, which forced both to hardcode
    # it. Every value is read off a tensor and cross-checked; nothing is copied
    # from the upstream YAML.
    first_enc = refs.get("mimi.encoder.model.0.conv.weight")
    if first_enc is None or len(first_enc.shape) != 3 or first_enc.shape[1] != 1:
        fail("mimi.encoder.model.0.conv.weight missing or not [n_filters, 1, kernel]")
    seanet_n_filters = first_enc.shape[0]
    seanet_kernel_size = first_enc.shape[2]

    # A residual unit is block.1 (narrow) then block.3 (back to width); the
    # compression ratio and the residual kernel both fall out of block.1.
    res_in = refs.get("mimi.encoder.model.1.block.1.conv.weight")
    res_out = refs.get("mimi.encoder.model.1.block.3.conv.weight")
    if res_in is None or res_out is None or len(res_in.shape) != 3:
        fail("mimi.encoder.model.1.block.{1,3}.conv.weight missing")
    if res_in.shape[1] != seanet_n_filters:
        fail(f"residual block input {res_in.shape[1]} != n_filters {seanet_n_filters}")
    if res_in.shape[0] == 0 or seanet_n_filters % res_in.shape[0] != 0:
        fail(f"n_filters {seanet_n_filters} is not a multiple of the residual "
             f"width {res_in.shape[0]}; compress is not an integer")
    seanet_compress = seanet_n_filters // res_in.shape[0]
    seanet_residual_kernel_size = res_in.shape[2]
    if res_out.shape[2] != 1:
        fail(f"residual output conv kernel is {res_out.shape[2]}, expected 1")

    # One residual unit per stage: blocks are numbered 1 and 3, so a second unit
    # would add 5 and 7. Count rather than assume.
    res_indices = sorted({int(m.group(1)) for name in refs
                          if (m := re.match(r"mimi\.encoder\.model\.1\.block\.(\d+)\.conv\.weight$", name))})
    if res_indices != [1, 3]:
        fail(f"unexpected residual block indices {res_indices}; "
             "n_residual_layers is not 1 and the derivation below is wrong")
    seanet_n_residual_layers = 1

    last_conv = refs.get("mimi.decoder.model.11.conv.weight")
    if last_conv is None or len(last_conv.shape) != 3 or last_conv.shape[0] != 1:
        fail("mimi.decoder.model.11.conv.weight missing or not [1, ch, kernel]")
    seanet_last_kernel_size = last_conv.shape[2]

    # dilation_base is deliberately NOT emitted: with one residual layer only
    # base**0 == 1 is ever used, so the checkpoint cannot witness it and any
    # value here would be a guess. Add it when a model with more layers appears.


    return Schema(
        tensor_count=len(refs), dtype=dtype,
        latent_dim=latent_dim, hidden_dim=hidden_dim,
        attention_heads=attention_heads, head_dim=head_dim,
        ffn_dim=ffn_dim, transformer_layers=layers,
        text_embedding_rows=text_embedding_rows, text_vocab_size=text_vocab_size,
        text_padding_id=text_padding_id,
        flow_dim=flow_dim, flow_res_blocks=flow_res_blocks,
        flow_time_conditions=flow_time_conditions, flow_time_freqs=flow_time_freqs,
        speaker_proj_input_dim=speaker_proj_input_dim,
        has_bos_before_voice=has_bos_before_voice,
        codec_dim=codec_dim, codec_ratios=codec_ratios,
        codec_upsample_stride=codec_upsample_stride,
        codec_downsample_stride=codec_downsample_stride,
        codec_transformer_layers=codec_transformer_layers,
        codec_transformer_dim=codec_transformer_dim,
        codec_transformer_ffn=codec_transformer_ffn,
        codec_transformer_heads=codec_transformer_heads,
        seanet_n_filters=seanet_n_filters,
        seanet_kernel_size=seanet_kernel_size,
        seanet_residual_kernel_size=seanet_residual_kernel_size,
        seanet_last_kernel_size=seanet_last_kernel_size,
        seanet_n_residual_layers=seanet_n_residual_layers,
        seanet_compress=seanet_compress,
        samples_per_frame=samples_per_frame, generation=generation)


def stride_from_kernel(kernel: int, what: str) -> int:
    """Mimi's SEANet uses kernel = 2 * stride for every (transposed) conv."""
    if kernel <= 0 or kernel % 2 != 0:
        fail(f"{what} has kernel {kernel}; SEANet strided convolutions use an "
             f"even kernel = 2 * stride")
    return kernel // 2


# --------------------------------------------------------------------------
# Voices
# --------------------------------------------------------------------------

@dataclass
class Voice:
    name: str
    source: Path
    frames: int
    dataset: str
    license: str
    commercial_use: bool
    source_bytes: int
    packed_bytes: int = 0
    f16_abs_error: float = 0.0
    f16_rel_error: float = 0.0


def read_voice(path: Path, schema: Schema | None) -> tuple[dict[str, TensorRef], int]:
    """Open a voice file and validate that it is a KV cache of the right shape."""
    refs, _, _ = st_read_header(path)
    caches = sorted(name for name in refs if name.endswith("/cache"))
    offsets = sorted(name for name in refs if name.endswith("/offset"))
    if not caches:
        fail(f"{path} has no '*/cache' tensor; this is not a PocketTTS voice state")
    if len(caches) != len(offsets):
        fail(f"{path} has {len(caches)} caches but {len(offsets)} offsets")
    frames = None
    for name in caches:
        ref = refs[name]
        if ref.dtype != "F32":
            fail(f"{path}: {name} is {ref.dtype}, expected F32")
        if len(ref.shape) != 5 or ref.shape[0] != 2 or ref.shape[1] != 1:
            fail(f"{path}: {name} is {list(ref.shape)}, expected "
                 f"[2, 1, frames, heads, head_dim]")
        if frames is None:
            frames = ref.shape[2]
        elif ref.shape[2] != frames:
            fail(f"{path}: {name} has {ref.shape[2]} frames, another layer has {frames}")
        if schema is not None and ref.shape[3:] != (schema.attention_heads, schema.head_dim):
            fail(f"{path}: {name} is {list(ref.shape)}, but the model has "
                 f"{schema.attention_heads} heads of {schema.head_dim}")
    if schema is not None:
        layers = layer_indices(caches, r"^transformer\.layers\.(\d+)\.self_attn/cache$",
                               f"{path} KV layers")
        if layers != schema.transformer_layers:
            fail(f"{path} has {layers} KV layers but the model has "
                 f"{schema.transformer_layers} transformer layers")
    with path.open("rb") as stream:
        for name in offsets:
            ref = refs[name]
            if ref.dtype != "I64" or ref.shape != (1,):
                fail(f"{path}: {name} is {ref.dtype}{list(ref.shape)}, expected I64[1]")
            value = int(np.frombuffer(st_raw(stream, ref), dtype="<i8")[0])
            if value != frames:
                fail(f"{path}: {name} is {value} but the cache holds {frames} frames; "
                     f"a partially filled voice cache is not supported")
    assert frames is not None
    return refs, frames


# --------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------

def flat_lookup(text: str, key: str) -> str | None:
    """Reimplementation of `json_value` in src/mynah_tts.c, for validation."""
    needle = f'"{key}"'
    at = text.find(needle)
    if at < 0:
        return None
    at += len(needle)
    while at < len(text) and text[at] in " \t\r\n:":
        at += 1
    return text[at:] if at < len(text) else None


def check_flat_manifest(text: str, manifest: dict[str, Any]) -> None:
    """Every top-level scalar must be findable, and findable only once.

    The runtime's reader takes the first `"key"` in the file, so a duplicate key
    name anywhere — including inside a nested object — silently redirects it.
    """
    for key, value in manifest.items():
        occurrences = text.count(f'"{key}"')
        if occurrences != 1:
            fail(f"model.json key {key!r} occurs {occurrences} times; the runtime's "
                 f"flat JSON reader would pick the first one")
        if isinstance(value, (dict, list)):
            continue
        tail = flat_lookup(text, key)
        if tail is None:
            fail(f"model.json key {key!r} is not reachable by the flat reader")
        if isinstance(value, bool):
            if not tail.startswith("true" if value else "false"):
                fail(f"model.json key {key!r} does not read back as {value}")
        elif isinstance(value, str):
            if not tail.startswith(f'"{value}"'):
                fail(f"model.json key {key!r} does not read back as {value!r}")
        elif isinstance(value, int):
            if int(re.match(r"-?\d+", tail).group(0)) != value:  # type: ignore[union-attr]
                fail(f"model.json key {key!r} does not read back as {value}")
        elif isinstance(value, float):
            matched = re.match(r"-?\d+(\.\d+)?([eE][-+]?\d+)?", tail)
            if matched is None or abs(float(matched.group(0)) - value) > 1e-12:
                fail(f"model.json key {key!r} does not read back as {value}")


def build_manifest(language: str, revision: str, schema: Schema, voices: list[Voice],
                   cloning: bool, dtype_label: str, voice_dtype: str,
                   tokenizer_pieces: int) -> dict[str, Any]:
    temperature = ENGLISH_TEMPERATURE if language.startswith("english") else DEFAULT_TEMPERATURE
    manifest: dict[str, Any] = {
        # --- identity ---
        "format": 1,
        "engine": "pocket",
        "language": language,
        "generation": schema.generation,
        "revision": revision,
        "source": REPO_ID,
        "dtype": dtype_label,

        # --- audio ---
        "sample_rate": SAMPLE_RATE,
        "audio_channels": AUDIO_CHANNELS,
        "samples_per_frame": schema.samples_per_frame,
        "frame_rate": SAMPLE_RATE / schema.samples_per_frame,
        "frame_ms": 1000.0 * schema.samples_per_frame / SAMPLE_RATE,

        # --- backbone ---
        "latent_dim": schema.latent_dim,
        "hidden_dim": schema.hidden_dim,
        "attention_heads": schema.attention_heads,
        "head_dim": schema.head_dim,
        "ffn_dim": schema.ffn_dim,
        "transformer_layers": schema.transformer_layers,
        "layernorm_eps": BACKBONE_LAYERNORM_EPS,
        "ffn_activation": BACKBONE_FFN_ACTIVATION,
        "qkv_fused": True,

        # --- text conditioner / tokenizer ---
        "text_vocab_size": schema.text_vocab_size,
        "text_embedding_rows": schema.text_embedding_rows,
        "text_padding_id": schema.text_padding_id,
        "tokenizer_kind": TOKENIZER_KIND,
        "tokenizer_file": "tokenizer.model",
        "tokenizer_pieces": tokenizer_pieces,
        "tokenizer_adds_bos": TOKENIZER_ADDS_BOS,
        "tokenizer_adds_eos": TOKENIZER_ADDS_EOS,
        "max_tokens_per_chunk": MAX_TOKENS_PER_CHUNK,

        # --- flow head ---
        "flow_dim": schema.flow_dim,
        "flow_res_blocks": schema.flow_res_blocks,
        "flow_time_conditions": schema.flow_time_conditions,
        "flow_time_freqs": schema.flow_time_freqs,
        "flow_layernorm_eps": FLOW_LAYERNORM_EPS,
        "flow_mlp_activation": FLOW_MLP_ACTIVATION,
        "flow_decode_steps": FLOW_DECODE_STEPS,

        # --- sampling ---
        "temperature": temperature,
        "eos_threshold": EOS_THRESHOLD,
        "uses_cfg": USES_CFG,
        "nan_is_bos": NAN_IS_BOS_SENTINEL,

        # --- codec ---
        "codec_dim": schema.codec_dim,
        "codec_ratios": list(schema.codec_ratios),
        "codec_upsample_stride": schema.codec_upsample_stride,
        "codec_downsample_stride": schema.codec_downsample_stride,
        "codec_transformer_layers": schema.codec_transformer_layers,
        "codec_transformer_dim": schema.codec_transformer_dim,
        "codec_transformer_ffn": schema.codec_transformer_ffn,
        # derived as codec_transformer_dim / head_dim; confirm against the oracle
        "codec_transformer_heads": schema.codec_transformer_heads,
        "seanet_n_filters": schema.seanet_n_filters,
        "seanet_kernel_size": schema.seanet_kernel_size,
        "seanet_residual_kernel_size": schema.seanet_residual_kernel_size,
        "seanet_last_kernel_size": schema.seanet_last_kernel_size,
        "seanet_n_residual_layers": schema.seanet_n_residual_layers,
        "seanet_compress": schema.seanet_compress,
        "codec_transformer_context": DECODER_TRANSFORMER_CONTEXT,

        # --- voices ---
        "speaker_count": len(voices),
        "speakers_file": "speakers.json",
        "voices_dir": "voices",
        "voice_dtype": {"F16": "float16", "F32": "float32"}[voice_dtype],
        "voice_is_kv_cache": True,

        # --- cloning ---
        "supports_cloning": cloning,
        "speaker_proj_input_dim": schema.speaker_proj_input_dim,
        "insert_bos_before_voice": schema.has_bos_before_voice,

        # --- files (not read by the flat parser) ---
        "weights": {"tts": "tts.safetensors"},
        "tensor_count": 0,   # filled in by the caller once the file is written
    }
    return manifest


# --------------------------------------------------------------------------
# License and notice documents
# --------------------------------------------------------------------------

def write_licenses(output: Path, language: str, revision: str, schema: Schema,
                   voices: list[Voice], dtype_label: str, voice_dtype: str,
                   cloning: bool) -> None:
    licenses = output / "LICENSES"
    licenses.mkdir(parents=True, exist_ok=True)

    modifications = [
        "selected a subset of the checkpoint's tensors and repacked them as "
        "`tts.safetensors`" + ("" if cloning else
                               " (voice-cloning tensors omitted: mimi.encoder*, "
                               "mimi.downsample*, flow_lm.speaker_proj_weight)"),
        "converted to the mynah model pack format (`model.json`, `speakers.json`, "
        "`source.json`)",
    ]
    source_dtype_label = {"BF16": "bfloat16", "F32": "float32"}.get(schema.dtype)
    source_preserved = (schema.dtype == "mixed" and dtype_label == "mixed")
    if not source_preserved and dtype_label != source_dtype_label:
        modifications.append(
            f"converted the checkpoint weight policy ({schema.dtype}) into the "
            f"pack's {dtype_label} representation")
    else:
        modifications.append(
            "model weight values are unchanged; they are copied byte for byte "
            f"with the source dtype policy ({dtype_label})")
    if voice_dtype == "F16":
        modifications.append(
            "quantized the predefined voice states (transformer KV caches) from "
            "float32 to float16")

    (licenses / "MODEL_LICENSE.md").write_text(
        "# Model license\n\n"
        f"Source model: **Kyutai Pocket TTS** (`{REPO_ID}`), language "
        f"`{language}`, revision `{revision}`.\n\n"
        f"- Model card: {REPO_URL}\n"
        f"- Reference implementation: {CODE_URL} (MIT)\n"
        f"- Paper: {PAPER_URL}\n\n"
        "## License\n\n"
        "The model weights are licensed under the **Creative Commons Attribution "
        "4.0 International** license (CC-BY-4.0).\n\n"
        "- License text: https://creativecommons.org/licenses/by/4.0/legalcode\n"
        "- Human-readable summary: https://creativecommons.org/licenses/by/4.0/\n\n"
        "## Attribution\n\n"
        "> Pocket TTS by Kyutai — Manu Orsini, Simon Rouard, Gabriel De Marmiesse,\n"
        "> Vaclav Volhejn, Neil Zeghidour, Alexandre Defossez.\n"
        f"> Licensed under CC-BY-4.0. Source: {REPO_URL}\n\n"
        "## Statement of modifications\n\n"
        "CC-BY-4.0 section 3(a)(1)(B) requires that modifications be indicated.\n"
        "This pack is a modified version of the original weights. The changes are:\n\n"
        + "".join(f"{index}. {item}\n" for index, item in enumerate(modifications, 1))
        + "\nNo weight value was retrained, fine-tuned or otherwise learned from "
        "new data.\n\n"
        "## Scope\n\n"
        "This license covers the **model weights and the voice states** in this\n"
        "pack. It does not cover the mynah-tts runtime, which is distributed\n"
        "separately under its own license. CC-BY-4.0 has no share-alike clause, so\n"
        "no obligation propagates to the runtime.\n\n"
        "Individual voices carry their own, stricter licenses. See "
        "[VOICES.md](VOICES.md).\n\n"
        "See `../NOTICE` for the upstream prohibited-use statement, which is passed\n"
        "on as a notice and not as a license restriction.\n",
        encoding="utf-8")

    rows = "".join(
        f"| `{voice.name}` | {voice.frames} | {voice.dataset} | {voice.license} | "
        f"{'yes' if voice.commercial_use else '**no**'} |\n" for voice in voices)
    non_commercial = [v for v in voices if not v.commercial_use]
    unknown = [v for v in voices if v.license == "unknown"]
    (licenses / "VOICES.md").write_text(
        "# Voice licenses\n\n"
        f"Pack language: `{language}`. The voice states are transformer KV caches\n"
        "computed from reference recordings; the license of the recording travels\n"
        "with the state.\n\n"
        "Per-dataset licensing is published at https://huggingface.co/kyutai/tts-voices\n\n"
        "| Voice | Frames | Source dataset | License | Commercial use |\n"
        "|---|---:|---|---|---|\n"
        + rows + "\n"
        + ("All voices in this pack permit commercial use.\n" if not non_commercial else
           "## Not usable commercially\n\n"
           + "".join(f"- `{v.name}` — {v.dataset}, {v.license}\n" for v in non_commercial)
           + "\nRebuild with `--commercial-only` to exclude them.\n")
        + ("" if not unknown else
           "\n## Unverified provenance\n\n"
           "The source recording for the following voices has not been verified\n"
           "against the upstream per-dataset table, so they are reported as\n"
           "`unknown` and **not** marked commercially usable. This is a gap in the\n"
           "documentation, not a statement that the license is restrictive:\n\n"
           + "".join(f"- `{v.name}`\n" for v in unknown)),
        encoding="utf-8")

    (output / "NOTICE").write_text(
        "NOTICE\n"
        "======\n\n"
        f"This model pack contains weights derived from {REPO_ID}, licensed under\n"
        "CC-BY-4.0. See LICENSES/MODEL_LICENSE.md for the license and the statement\n"
        "of modifications, and LICENSES/VOICES.md for per-voice licensing.\n\n"
        "Prohibited use (upstream statement)\n"
        "-----------------------------------\n\n"
        "The following is reproduced from the Kyutai Pocket TTS model card. It is a\n"
        "condition of access accepted by whoever downloaded the original weights,\n"
        "not a term of the CC-BY-4.0 license: CC-BY-4.0 forbids imposing additional\n"
        "restrictions on recipients, so this does not restrict your rights under\n"
        "that license. It is passed on because it is worth honouring.\n\n"
        + "\n".join(wrap_paragraph(PROHIBITED_USE, 78)) + "\n\n"
        "The clause with real force for a speech runtime is consent: do not clone or\n"
        "imitate a person's voice without their explicit and lawful consent.\n\n"
        "Transparency\n"
        "------------\n\n"
        "Audio produced by this pack is synthetic. Disclosing that may be a legal\n"
        "obligation where you operate (for example EU AI Act article 50).\n",
        encoding="utf-8")


def wrap_paragraph(text: str, width: int) -> list[str]:
    lines: list[str] = []
    current = ""
    for word in text.split():
        if current and len(current) + 1 + len(word) > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines


# --------------------------------------------------------------------------
# Pack verification: re-read what was written and check it against itself.
# --------------------------------------------------------------------------

def verify_pack(output: Path) -> list[str]:
    """Re-open the finished pack and check every declared field against a tensor.

    This is deliberately independent of the objects used to write the pack: it
    reads model.json and the safetensors headers from disk, the way a runtime
    would, so a field that drifted from the tensors fails here and not in C.
    """
    notes: list[str] = []
    text = (output / "model.json").read_text(encoding="utf-8")
    manifest = json.loads(text)
    check_flat_manifest(text, manifest)

    refs, _, _ = st_read_header(output / "tts.safetensors")
    if len(refs) != manifest["tensor_count"]:
        fail(f"model.json says {manifest['tensor_count']} tensors, "
             f"tts.safetensors has {len(refs)}")
    declared_dtype = manifest["dtype"]
    allowed_dtypes = {
        "bfloat16": {"BF16"},
        "float32": {"F32"},
        "mixed": {"BF16", "F32"},
    }.get(declared_dtype)
    if allowed_dtypes is None:
        fail(f"model.json has unsupported pack dtype {declared_dtype!r}")
    wrong = sorted(n for n, r in refs.items() if r.dtype not in allowed_dtypes)
    if wrong:
        fail(f"tts.safetensors has {len(wrong)} tensors outside {sorted(allowed_dtypes)}, "
             f"first is {wrong[0]}")

    hidden = manifest["hidden_dim"]
    latent = manifest["latent_dim"]
    flow = manifest["flow_dim"]
    checks: list[tuple[str, tuple[int, ...]]] = [
        ("flow_lm.bos_emb", (latent,)),
        ("flow_lm.emb_mean", (latent,)),
        ("flow_lm.emb_std", (latent,)),
        ("flow_lm.input_linear.weight", (hidden, latent)),
        ("flow_lm.out_norm.weight", (hidden,)),
        ("flow_lm.out_eos.weight", (1, hidden)),
        ("flow_lm.conditioner.embed.weight",
         (manifest["text_embedding_rows"], hidden)),
        ("flow_lm.flow_net.input_proj.weight", (flow, latent)),
        ("flow_lm.flow_net.cond_embed.weight", (flow, hidden)),
        ("flow_lm.flow_net.final_layer.linear.weight", (latent, flow)),
        ("mimi.quantizer.output_proj.weight", (manifest["codec_dim"], latent, 1)),
        ("mimi.upsample.convtr.convtr.weight",
         (manifest["codec_dim"], 1, 2 * manifest["codec_upsample_stride"])),
    ]
    for layer in range(manifest["transformer_layers"]):
        prefix = f"flow_lm.transformer.layers.{layer}"
        checks += [(f"{prefix}.self_attn.in_proj.weight", (3 * hidden, hidden)),
                   (f"{prefix}.linear1.weight", (manifest["ffn_dim"], hidden))]
    for block in range(manifest["flow_res_blocks"]):
        checks.append((f"flow_lm.flow_net.res_blocks.{block}.adaLN_modulation.1.weight",
                       (3 * flow, flow)))
    for condition in range(manifest["flow_time_conditions"]):
        checks.append((f"flow_lm.flow_net.time_embed.{condition}.freqs",
                       (manifest["flow_time_freqs"],)))
    for layer in range(manifest["codec_transformer_layers"]):
        prefix = f"mimi.decoder_transformer.transformer.layers.{layer}"
        checks += [(f"{prefix}.self_attn.out_proj.weight",
                    (manifest["codec_transformer_dim"],) * 2),
                   (f"{prefix}.linear1.weight", (manifest["codec_transformer_ffn"],
                                                 manifest["codec_transformer_dim"]))]
    for name, shape in checks:
        ref = refs.get(name)
        if ref is None:
            fail(f"tts.safetensors is missing {name}, which model.json implies")
        if ref.shape != shape:
            fail(f"tts.safetensors {name} is {list(ref.shape)}, model.json implies "
                 f"{list(shape)}")
    notes.append(f"{len(checks)} declared dimensions cross-checked against tensors")

    convtr = sorted(
        ((int(m.group(1)), name) for name in refs
         if (m := re.match(r"^mimi\.decoder\.model\.(\d+)\.convtr\.weight$", name))))
    ratios = [refs[name].shape[2] // 2 for _, name in convtr]
    if ratios != manifest["codec_ratios"]:
        fail(f"model.json codec_ratios {manifest['codec_ratios']} but the decoder "
             f"transposed convolutions say {ratios}")
    if math.prod(ratios) * manifest["codec_upsample_stride"] != manifest["samples_per_frame"]:
        fail("model.json samples_per_frame does not equal prod(codec_ratios) * "
             "codec_upsample_stride")
    if abs(manifest["sample_rate"] / manifest["samples_per_frame"]
           - manifest["frame_rate"]) > 1e-9:
        fail("model.json frame_rate is not sample_rate / samples_per_frame")

    cloning_present = any(classify(name) == "cloning" for name in refs)
    if cloning_present != manifest["supports_cloning"]:
        fail(f"model.json supports_cloning={manifest['supports_cloning']} but the "
             f"cloning tensors are {'present' if cloning_present else 'absent'}")
    if manifest["supports_cloning"]:
        expect_shape(refs, "flow_lm.speaker_proj_weight",
                     (hidden, manifest["speaker_proj_input_dim"]))
        expect_shape(refs, "mimi.downsample.conv.conv.weight",
                     (manifest["speaker_proj_input_dim"], manifest["codec_dim"],
                      2 * manifest["codec_downsample_stride"]))
    if ("flow_lm.bos_before_voice" in refs) != manifest["insert_bos_before_voice"]:
        fail("model.json insert_bos_before_voice disagrees with the tensors")

    # --- tokenizer ---
    tokenizer = output / manifest["tokenizer_file"]
    if not tokenizer.is_file():
        fail(f"pack is missing {manifest['tokenizer_file']}")
    pieces = sentencepiece_piece_count(tokenizer)
    if pieces != manifest["tokenizer_pieces"]:
        fail(f"model.json tokenizer_pieces={manifest['tokenizer_pieces']} but "
             f"{tokenizer.name} has {pieces}")
    if pieces != manifest["text_vocab_size"]:
        fail(f"tokenizer has {pieces} pieces but text_vocab_size is "
             f"{manifest['text_vocab_size']}")
    notes.append(f"tokenizer: {pieces} pieces == text_vocab_size")

    # --- voices ---
    speakers = json.loads((output / manifest["speakers_file"]).read_text(encoding="utf-8"))
    if len(speakers["voices"]) != manifest["speaker_count"]:
        fail("speakers.json voice count disagrees with model.json speaker_count")
    want_voice_dtype = {"float16": "F16", "float32": "F32"}[manifest["voice_dtype"]]
    for entry in speakers["voices"]:
        path = output / entry["file"]
        if not path.is_file():
            fail(f"speakers.json lists {entry['file']}, which is not in the pack")
        voice_refs, _, _ = st_read_header(path)
        caches = [r for n, r in voice_refs.items() if n.endswith("/cache")]
        if len(caches) != manifest["transformer_layers"]:
            fail(f"{path.name} has {len(caches)} KV layers, model has "
                 f"{manifest['transformer_layers']}")
        for ref in caches:
            if ref.dtype != want_voice_dtype:
                fail(f"{path.name}: {ref.name} is {ref.dtype}, model.json says "
                     f"{manifest['voice_dtype']}")
            if ref.shape != (2, 1, entry["frames"], manifest["attention_heads"],
                             manifest["head_dim"]):
                fail(f"{path.name}: {ref.name} is {list(ref.shape)}, expected "
                     f"[2, 1, {entry['frames']}, {manifest['attention_heads']}, "
                     f"{manifest['head_dim']}]")
        if manifest["commercial_only"] and not entry["commercial_use"]:
            fail(f"pack was built with --commercial-only but {entry['name']} is not "
                 f"marked commercial_use")
    notes.append(f"{len(speakers['voices'])} voice files match "
                 f"[2, 1, T, {manifest['attention_heads']}, {manifest['head_dim']}] "
                 f"{manifest['voice_dtype']}")

    # --- source manifest ---
    source = json.loads((output / "source.json").read_text(encoding="utf-8"))
    missing = [entry["path"] for entry in source["files"]
               if not Path(entry["path"]).is_file()]
    if missing:
        notes.append(f"{len(missing)} source files are no longer on disk "
                     f"(sha256 recorded anyway)")
    for name in ("NOTICE", "LICENSES/MODEL_LICENSE.md", "LICENSES/VOICES.md"):
        if not (output / name).is_file():
            fail(f"pack is missing {name}")
    return notes


# --------------------------------------------------------------------------
# Conversion
# --------------------------------------------------------------------------

def convert(args: argparse.Namespace) -> int:
    if sys.byteorder != "little":
        fail("safetensors payloads are little-endian; this converter needs a "
             "little-endian host")

    snapshot = args.source or default_snapshot()
    if not snapshot.is_dir():
        fail(f"{snapshot} is not a directory")
    revision = snapshot.name

    languages = available_languages(snapshot)
    language_dir = snapshot / "languages" / args.language
    if not (language_dir / "model.safetensors").is_file():
        fail(f"{args.language!r} is not in {snapshot / 'languages'}. "
             f"Available: {', '.join(languages) or '(none)'}")
    model_path = language_dir / "model.safetensors"
    tokenizer_path = language_dir / "tokenizer.model"
    if not tokenizer_path.is_file():
        fail(f"{language_dir} has no tokenizer.model")
    embeddings_dir = language_dir / "embeddings"
    if not embeddings_dir.is_dir():
        fail(f"{language_dir} has no embeddings/ directory; a pack needs voices")

    # --- voices: pick before deriving the schema, the head split lives in them
    available_voices = sorted(p.stem for p in embeddings_dir.glob("*.safetensors"))
    if not available_voices:
        fail(f"{embeddings_dir} contains no .safetensors voice files")
    if args.voices.strip() == "all":
        requested = available_voices
    else:
        requested = [name.strip() for name in args.voices.split(",") if name.strip()]
        unknown = [name for name in requested if name not in available_voices]
        if unknown:
            fail(f"unknown voice(s) {', '.join(unknown)}. Available: "
                 f"{', '.join(available_voices)}")
        requested = sorted(dict.fromkeys(requested))

    selected: list[Voice] = []
    skipped: list[str] = []
    for name in requested:
        dataset, license_name, commercial = VOICE_LICENSES.get(name, UNKNOWN_VOICE)
        if args.commercial_only and not commercial:
            skipped.append(f"{name} ({license_name})")
            continue
        path = embeddings_dir / f"{name}.safetensors"
        selected.append(Voice(name=name, source=path, frames=0, dataset=dataset,
                              license=license_name, commercial_use=commercial,
                              source_bytes=path.stat().st_size))
    if not selected:
        fail("no voices left after filtering; a pack with no voice cannot generate "
             "audio (a voice IS the prefill KV cache)")

    # --- read the checkpoint ----------------------------------------------
    refs, _, _ = st_read_header(model_path)
    probe_refs, probe_frames = read_voice(selected[0].source, None)
    schema = derive_schema(refs, probe_refs, selected[0].name)

    for voice in selected:
        _, voice.frames = read_voice(voice.source, schema)

    # --- tensor selection --------------------------------------------------
    groups: dict[str, list[str]] = {"core": [], "cloning": [], "unknown": []}
    for name in refs:
        groups[classify(name)].append(name)
    if groups["unknown"]:
        fail(f"checkpoint has {len(groups['unknown'])} tensors this converter does "
             f"not know how to classify, first is {sorted(groups['unknown'])[0]}. "
             f"Refusing to write a pack that silently drops weights.")
    cloning = not args.no_cloning
    if cloning and not groups["cloning"]:
        fail("the checkpoint has no voice-cloning tensors; pass --no-cloning")
    keep = sorted(groups["core"] + (groups["cloning"] if cloning else []))

    fixed_target = {"bf16": "BF16", "f32": "F32"}.get(args.dtype)
    if fixed_target is None and args.dtype != "source":
        fail(f"unknown dtype policy {args.dtype!r}")

    def output_dtype(ref: TensorRef) -> str:
        return ref.dtype if fixed_target is None else fixed_target

    def convert_blob(blob: bytes, source_dtype: str, target_dtype: str,
                     what: str) -> bytes:
        if source_dtype == target_dtype:
            return blob
        if source_dtype == "F32" and target_dtype == "BF16":
            return f32_to_bf16(blob)
        if source_dtype == "BF16" and target_dtype == "F32":
            return bf16_to_f32(blob)
        fail(f"cannot convert {what} from {source_dtype} to {target_dtype}")

    target_dtypes = {output_dtype(refs[name]) for name in keep}
    dtype_labels = {"BF16": "bfloat16", "F32": "float32"}
    dtype_label = (next(iter(dtype_labels[d] for d in target_dtypes))
                   if len(target_dtypes) == 1 else "mixed")
    dropped_pack_bytes = (sum(
        ST_ITEMSIZE[output_dtype(refs[name])] *
        (math.prod(refs[name].shape) if refs[name].shape else 1)
        for name in groups["cloning"])
        if not cloning else 0)

    output: Path = args.output
    output.mkdir(parents=True, exist_ok=True)
    (output / "LICENSES").mkdir(exist_ok=True)
    shutil.rmtree(output / "voices", ignore_errors=True)
    (output / "voices").mkdir()

    # --- tts.safetensors ---------------------------------------------------
    with model_path.open("rb") as source_stream:
        def emitter(ref: TensorRef) -> Callable[[], bytes]:
            def emit() -> bytes:
                blob = st_raw(source_stream, ref)
                return convert_blob(blob, ref.dtype, output_dtype(ref), ref.name)
            return emit

        out_tensors = [
            OutTensor(name=name, dtype=output_dtype(refs[name]), shape=refs[name].shape,
                      nbytes=ST_ITEMSIZE[output_dtype(refs[name])] *
                      (math.prod(refs[name].shape) if refs[name].shape else 1),
                      emit=emitter(refs[name]))
            for name in keep
        ]
        tts_bytes = st_write(output / "tts.safetensors", out_tensors)

    # --- tokenizer ---------------------------------------------------------
    shutil.copyfile(tokenizer_path, output / "tokenizer.model")
    tokenizer_pieces = sentencepiece_piece_count(output / "tokenizer.model")
    if tokenizer_pieces != schema.text_vocab_size:
        fail(f"tokenizer.model has {tokenizer_pieces} pieces but the conditioner "
             f"embedding implies a vocabulary of {schema.text_vocab_size} "
             f"(+1 padding row)")

    # --- voices ------------------------------------------------------------
    voice_dtype = "F16"
    voice_source_bytes = 0
    for voice in selected:
        voice_refs, _, _ = st_read_header(voice.source)
        with voice.source.open("rb") as stream:
            out_tensors = []
            worst_abs = worst_rel = 0.0
            for name in sorted(voice_refs):
                ref = voice_refs[name]
                blob = st_raw(stream, ref)
                if ref.dtype == "F32" and name.endswith("/cache"):
                    abs_error, rel_error = f16_max_abs_error(blob)
                    worst_abs = max(worst_abs, abs_error)
                    worst_rel = max(worst_rel, rel_error)
                    payload = f32_to_f16(blob, f"{voice.name}:{name}")
                    out_tensors.append(OutTensor(name, "F16", ref.shape,
                                                 len(payload), lambda p=payload: p))
                else:
                    out_tensors.append(OutTensor(name, ref.dtype, ref.shape,
                                                 len(blob), lambda b=blob: b))
            voice.packed_bytes = st_write(
                output / "voices" / f"{voice.name}.safetensors", out_tensors)
            voice.f16_abs_error = worst_abs
            voice.f16_rel_error = worst_rel
        voice_source_bytes += voice.source_bytes

    voice_packed_bytes = sum(v.packed_bytes for v in selected)

    # --- speakers.json -----------------------------------------------------
    model_sha = sha256_file(model_path)
    speakers = {
        # facts §12: a voice KV is bound to the weights that produced it. The
        # language and the source checkpoint digest are recorded so a runtime can
        # refuse a voice that crossed a model boundary.
        "language": args.language,
        "model_sha256": model_sha,
        "revision": revision,
        "voices": [
            {
                "name": voice.name,
                "file": f"voices/{voice.name}.safetensors",
                "frames": voice.frames,
                "seconds": round(voice.frames * schema.samples_per_frame / SAMPLE_RATE, 3),
                "source_dataset": voice.dataset,
                "license": voice.license,
                "commercial_use": voice.commercial_use,
            }
            for voice in selected
        ],
    }
    (output / "speakers.json").write_text(
        json.dumps(speakers, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # --- model.json --------------------------------------------------------
    manifest = build_manifest(args.language, revision, schema, selected, cloning,
                              dtype_label, voice_dtype, tokenizer_pieces)
    manifest["tensor_count"] = len(keep)
    manifest["commercial_only"] = bool(args.commercial_only)
    manifest_text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    check_flat_manifest(manifest_text, manifest)
    (output / "model.json").write_text(manifest_text, encoding="utf-8")

    # --- licenses ----------------------------------------------------------
    write_licenses(output, args.language, revision, schema, selected,
                   dtype_label, voice_dtype, cloning)

    # --- source.json (the only file with a timestamp) ----------------------
    source_files = [
        {"role": "weights", "path": str(model_path.resolve()),
         "name": f"languages/{args.language}/model.safetensors",
         "sha256": model_sha, "bytes": model_path.stat().st_size},
        {"role": "tokenizer", "path": str(tokenizer_path.resolve()),
         "name": f"languages/{args.language}/tokenizer.model",
         "sha256": sha256_file(tokenizer_path),
         "bytes": tokenizer_path.stat().st_size},
    ] + [
        {"role": "voice", "path": str(voice.source.resolve()),
         "name": f"languages/{args.language}/embeddings/{voice.name}.safetensors",
         "sha256": sha256_file(voice.source), "bytes": voice.source_bytes}
        for voice in selected
    ]
    readme = snapshot / "README.md"
    if readme.is_file():
        source_files.append({"role": "model_card", "path": str(readme.resolve()),
                             "name": "README.md", "sha256": sha256_file(readme),
                             "bytes": readme.stat().st_size})
    (output / "source.json").write_text(json.dumps({
        "created_at": datetime.now(timezone.utc).isoformat(),
        "converter": "tools/convert_pocket.py",
        "repo": REPO_ID,
        "repo_url": REPO_URL,
        "revision": revision,
        "snapshot": str(snapshot.resolve()),
        "language": args.language,
        "license": "CC-BY-4.0",
        "options": {
            "dtype": args.dtype,
            "cloning": cloning,
            "commercial_only": bool(args.commercial_only),
            "voices": args.voices,
            "voice_dtype": "f16",
        },
        "files": source_files,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # --- verify ------------------------------------------------------------
    notes = verify_pack(output)

    # --- report ------------------------------------------------------------
    print(f"wrote PocketTTS model pack: {output}")
    print(f"  source     : {REPO_ID} @ {revision}  language={args.language} "
          f"generation={schema.generation}")
    print(f"  tensors    : {len(keep)} of {schema.tensor_count} "
          f"({'with' if cloning else 'without'} cloning), {dtype_label}")
    print(f"  tts        : {tts_bytes / 1e6:9.2f} MB")
    if not cloning:
        print(f"  omitted    : {len(groups['cloning'])} cloning tensors, "
              f"{dropped_pack_bytes / 1e6:.2f} MB")
    print(f"  voices     : {len(selected)}  {voice_source_bytes / 1e6:.2f} MB F32 -> "
          f"{voice_packed_bytes / 1e6:.2f} MB F16 "
          f"(saved {(voice_source_bytes - voice_packed_bytes) / 1e6:.2f} MB, "
          f"{100.0 * (1 - voice_packed_bytes / voice_source_bytes):.1f}%)")
    worst = max(selected, key=lambda v: v.f16_abs_error)
    print(f"  F16 error  : max abs {worst.f16_abs_error:.3e} "
          f"(rel {worst.f16_rel_error:.3e}) on {worst.name}")
    if skipped:
        print(f"  excluded   : {len(skipped)} non-commercial/unverified voices: "
              f"{', '.join(skipped)}")
    non_commercial = [v.name for v in selected if not v.commercial_use]
    if non_commercial:
        print(f"  WARNING    : {len(non_commercial)} voice(s) are not cleared for "
              f"commercial use: {', '.join(non_commercial)}")
        print("               rebuild with --commercial-only to exclude them")
    print(f"  source dtypes: {schema.dtype}; pack dtype: {dtype_label}")
    print(f"  dims       : latent={schema.latent_dim} d_model={schema.hidden_dim} "
          f"heads={schema.attention_heads}x{schema.head_dim} ffn={schema.ffn_dim} "
          f"layers={schema.transformer_layers} flow={schema.flow_dim}x"
          f"{schema.flow_res_blocks}")
    print(f"  codec      : ratios={list(schema.codec_ratios)} "
          f"up/down={schema.codec_upsample_stride}/{schema.codec_downsample_stride} "
          f"{schema.samples_per_frame} samples/frame "
          f"{SAMPLE_RATE / schema.samples_per_frame} Hz @ {SAMPLE_RATE} Hz")
    total = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
    print(f"  pack total : {total / 1e6:9.2f} MB")
    for note in notes:
        print(f"  verified   : {note}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--language", required=True,
                        help="language directory inside the snapshot, e.g. english")
    parser.add_argument("--output", type=Path, required=True,
                        help="destination pack directory")
    parser.add_argument("--source", type=Path,
                        help="snapshot directory (default: the Hugging Face cache "
                             "entry named by refs/main)")
    parser.add_argument("--voices", default="all",
                        help='"all" (default) or a comma-separated subset')
    parser.add_argument("--no-cloning", action="store_true",
                        help="omit mimi.encoder*, mimi.downsample* and "
                             "flow_lm.speaker_proj_weight")
    parser.add_argument("--commercial-only", action="store_true",
                        help="keep only voices whose license clears commercial use; "
                             "voices with unverified provenance are excluded too")
    parser.add_argument("--dtype", choices=("bf16", "f32", "source"), default="bf16",
                        help="model weight policy: bf16 (default), f32, or source "
                             "to preserve each checkpoint tensor dtype")
    return convert(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
