#!/usr/bin/env python3
"""Is PocketTTS ternarizable? A measurement, not an opinion.

This is offline analysis tooling. It never runs as part of the C runtime, it
never writes into a model pack, and it does not change production inference
behaviour. It exists to answer one question cheaply, *before* anyone writes a
ternary GEMV kernel in C:

    does the trained PocketTTS checkpoint tolerate W1.58 weights, and if so,
    which tensors, by how much, and under which post-training method?

The expensive thing here is the C/SIMD runtime work that ternary would justify
(2-bit packing, SDOT/SMMLA/VNNI accumulation, per-row scale application, a new
qtype through `src/qmat.c` and every dispatch table). The cheap thing is a fake
-quantization sweep in Python. Do the cheap thing first, and be willing to come
back with a negative result.

Methods implemented (see `docs/ternary-feasibility.md` for what is exact, what
is adapted, and what is deliberately omitted):

    naive   symmetric per-tensor / per-row ternary with a searched threshold
    itf     PT^2-LLM Iterative Ternary Fitting (weight-domain, alternating)
    aga     ITF + PT^2-LLM Activation-aware Grid Alignment (output-domain)
    e2m     TWLA E2M-ATQ: Euclidean warm start -> manifold relocation
    gptq    AGA grid + GPTQ-style sequential error compensation
    kotms   TWLA-style Kronecker orthogonal tri-modal shaping, then AGA

Every method produces the same object: a ternary matrix T in {-1,0,+1}, a
per-row scale alpha and a per-row shift mu, so the storage accounting and the
kernel-shape analysis do not depend on which method won.

Usage, in the order the phases are meant to run:

    python3 tools/ternary_feasibility.py census   --model models/pocket-en
    uv run --with pocket-tts --with numpy --with scipy \
        python3 tools/ternary_feasibility.py calib --language english
    python3 tools/ternary_feasibility.py sweep    --methods naive,itf,aga,gptq
    uv run --with pocket-tts --with numpy --with scipy \
        python3 tools/ternary_feasibility.py cumulative --method gptq
    python3 tools/ternary_feasibility.py shapes
    python3 tools/ternary_feasibility.py report

`census`, `sweep`, `shapes` and `report` need only numpy and the converted
model pack. `calib` and `cumulative` need the upstream `pocket-tts` package,
because the only honest source of calibration activations and of end-to-end
audio is the reference implementation itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tests"))

DEFAULT_MODEL = REPO / "models" / "pocket-en"
DEFAULT_OUT = REPO / "build" / "ternary"

# --------------------------------------------------------------------------
# model pack: header, dtypes, tensors
# --------------------------------------------------------------------------

# safetensors dtype -> (numpy dtype to read as, bytes per element)
_ST_DTYPE = {
    "F64": (np.dtype("<f8"), 8),
    "F32": (np.dtype("<f4"), 4),
    "F16": (np.dtype("<f2"), 2),
    "BF16": (np.dtype("<u2"), 2),
    "I64": (np.dtype("<i8"), 8),
    "I32": (np.dtype("<i4"), 4),
    "I16": (np.dtype("<i2"), 2),
    "I8": (np.dtype("<i1"), 1),
    "U8": (np.dtype("<u1"), 1),
    "BOOL": (np.dtype("?"), 1),
}


def bf16_to_f32(raw: np.ndarray) -> np.ndarray:
    """bf16 bit pattern in uint16 -> float32. A shift, not a conversion table."""
    return (raw.astype(np.uint32) << 16).view(np.float32)


class Pack:
    """Read-only view of a converted model pack.

    Never mutates the file. `tensor()` returns a fresh float32 copy, so a
    caller that fake-quantizes cannot corrupt the checkpoint by accident --
    which is the whole safety contract of this tool.
    """

    def __init__(self, root: Path):
        self.root = Path(root)
        self.config = json.loads((self.root / "model.json").read_text())
        name = self.config.get("weights", {}).get("tts", "tts.safetensors")
        self.weights_path = self.root / name
        with self.weights_path.open("rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            self.header = json.loads(f.read(n))
            self._data_off = 8 + n
        self.metadata = self.header.pop("__metadata__", {})

    def __contains__(self, key: str) -> bool:
        return key in self.header

    def names(self) -> list[str]:
        return list(self.header)

    def spec(self, name: str) -> dict:
        return self.header[name]

    def shape(self, name: str) -> tuple[int, ...]:
        return tuple(self.header[name]["shape"])

    def dtype(self, name: str) -> str:
        return self.header[name]["dtype"]

    def numel(self, name: str) -> int:
        n = 1
        for d in self.shape(name):
            n *= d
        return int(n)

    def tensor(self, name: str) -> np.ndarray:
        spec = self.header[name]
        dt, _ = _ST_DTYPE[spec["dtype"]]
        lo, hi = spec["data_offsets"]
        with self.weights_path.open("rb") as f:
            f.seek(self._data_off + lo)
            raw = np.frombuffer(f.read(hi - lo), dtype=dt)
        if spec["dtype"] == "BF16":
            arr = bf16_to_f32(raw)
        else:
            arr = raw.astype(np.float32)
        return arr.reshape(spec["shape"]).copy()

    def sha256(self) -> str:
        h = hashlib.sha256()
        with self.weights_path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()


# --------------------------------------------------------------------------
# taxonomy: what each tensor is, and whether the streaming loop touches it
# --------------------------------------------------------------------------


@dataclass
class TensorInfo:
    name: str
    component: str          # backbone / flow_head / mimi_decoder / ...
    group: str              # attn_qkv / attn_out / ffn_up / ffn_down / ...
    role: str               # linear / conv1d / convtr1d / embedding / norm / param
    shape: tuple[int, ...]
    numel: int
    dtype: str
    streaming: bool         # touched once per generated frame at serve time
    layer: int | None = None
    # canonical 2-D form used by every quantizer: (rows, contract)
    rows: int = 0
    contract: int = 0
    eligible: bool = False   # is this a sane ternary candidate at all?
    reason: str = ""


def _layer_index(name: str) -> int | None:
    parts = name.split(".")
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
            return int(parts[i + 1])
        if p == "res_blocks" and i + 1 < len(parts) and parts[i + 1].isdigit():
            return int(parts[i + 1])
    return None


def classify(name: str, shape: tuple[int, ...], dtype: str) -> TensorInfo:
    """Map an upstream tensor name onto component / group / role.

    The names are the upstream PocketTTS module paths, preserved verbatim by
    `tools/convert_pocket.py`, so this table doubles as documentation of the
    graph. Anything that falls through lands in `other` and stays FP -- a new
    tensor must be classified deliberately, never ternarized by default.
    """
    numel = int(np.prod(shape)) if shape else 1
    nd = len(shape)
    layer = _layer_index(name)

    # role from the shape and the leaf name
    leaf = name.rsplit(".", 1)[-1]
    if leaf in ("bias", "scale", "alpha", "freqs", "emb_mean", "emb_std"):
        role = "param" if leaf != "bias" else "bias"
    elif nd == 1:
        role = "norm" if leaf == "weight" else "param"
    elif nd == 2:
        role = "embedding" if name.endswith("conditioner.embed.weight") else "linear"
    elif nd == 3:
        role = "convtr1d" if ".convtr" in name else "conv1d"
    else:
        role = "param"
    if name.endswith("bos_before_voice") or name.endswith("bos_emb"):
        role = "param"

    streaming = True
    component = "other"
    group = "other"

    if name.startswith("flow_lm.transformer."):
        component = "backbone"
        if "self_attn.in_proj" in name:
            group = "attn_qkv"
        elif "self_attn.out_proj" in name:
            group = "attn_out"
        elif "linear1" in name:
            group = "ffn_up"
        elif "linear2" in name:
            group = "ffn_down"
        elif "norm" in name:
            group = "norm"
    elif name.startswith("flow_lm.conditioner."):
        component = "conditioner"
        group = "text_embedding"
        streaming = False          # prefill only, once per request
    elif name.startswith("flow_lm.flow_net."):
        component = "flow_head"
        if "adaLN_modulation" in name:
            group = "flow_adaln"
        elif "res_blocks" in name and ".mlp." in name:
            group = "flow_mlp"
        elif "in_ln" in name:
            group = "norm"
        elif "time_embed" in name:
            group = "flow_time_embed"
            streaming = False      # t is fixed for flow_decode_steps=1
        elif "cond_embed" in name:
            group = "flow_cond"
        elif "input_proj" in name:
            group = "flow_in"
        elif "final_layer.linear" in name:
            group = "flow_out"
        elif "final_layer" in name:
            group = "flow_adaln"
    elif name.startswith("flow_lm."):
        component = "backbone_io"
        if "input_linear" in name:
            group = "latent_in"
        elif "speaker_proj" in name:
            group = "speaker"
        elif "out_eos" in name:
            group = "eos_head"
        elif "out_norm" in name:
            group = "norm"
        else:
            group = "param"
    elif name.startswith("mimi.decoder_transformer."):
        component = "mimi_dec_tr"
        if "self_attn.in_proj" in name:
            group = "attn_qkv"
        elif "self_attn.out_proj" in name:
            group = "attn_out"
        elif "linear1" in name:
            group = "ffn_up"
        elif "linear2" in name:
            group = "ffn_down"
        else:
            group = "norm"
    elif name.startswith("mimi.decoder."):
        component = "mimi_decoder"
        group = "convtr" if ".convtr" in name else "conv"
    elif name.startswith("mimi.encoder_transformer."):
        component = "mimi_enc_tr"
        group = "attn" if "self_attn" in name else "ffn"
        streaming = False          # voice cloning only
    elif name.startswith("mimi.encoder."):
        component = "mimi_encoder"
        group = "conv"
        streaming = False
    elif name.startswith("mimi.downsample"):
        component = "mimi_encoder"
        group = "downsample"
        streaming = False
    elif name.startswith("mimi.upsample"):
        component = "mimi_decoder"
        group = "upsample"
    elif name.startswith("mimi.quantizer"):
        component = "mimi_decoder"
        group = "latent_proj"

    info = TensorInfo(
        name=name, component=component, group=group, role=role,
        shape=shape, numel=numel, dtype=dtype, streaming=streaming, layer=layer,
    )
    _set_canonical(info)
    return info


def _set_canonical(info: TensorInfo) -> None:
    """Rows and contraction dim of the 2-D matrix the quantizers see.

    The contraction dim is the one the activation is summed over, because that
    is the axis the Gram matrix S = X^T X lives on. Getting this wrong makes
    every activation-aware method silently wrong, so it is one function.

        linear    (out, in)            -> rows=out,     contract=in
        conv1d    (out, in, k)         -> rows=out,     contract=in*k
        convtr1d  (in, out, k)         -> rows=out*k,   contract=in
        embedding (rows, dim)          -> a lookup, not a contraction
    """
    sh = info.shape
    if info.role == "linear":
        info.rows, info.contract = int(sh[0]), int(sh[1])
        info.eligible = True
    elif info.role == "conv1d":
        info.rows, info.contract = int(sh[0]), int(sh[1] * sh[2])
        info.eligible = True
    elif info.role == "convtr1d":
        info.rows, info.contract = int(sh[1] * sh[2]), int(sh[0])
        info.eligible = True
        if sh[1] == 1:
            # ConvTranspose1d weight is (in, out/groups, k). out/groups == 1 on
            # this graph means depthwise: every input channel has its own
            # filter and there is no contraction to share a scale over. The
            # canonical (out*k, in) form would be a lie, so refuse it here
            # rather than let an activation-aware method fit a Gram matrix over
            # an axis the operator never sums.
            info.eligible = False
            info.reason = "depthwise transposed conv: no shared contraction"
    elif info.role == "embedding":
        info.rows, info.contract = int(sh[0]), int(sh[1])
        info.eligible = True
        info.reason = "lookup table: no contraction, per-row scale is per-token"
    else:
        info.rows, info.contract = info.numel, 1
        info.eligible = False
        info.reason = f"{info.role}: too few parameters to pay for a scale"

    # A ternary row needs enough columns for one shared scale to mean anything,
    # and enough parameters that the 32-bit scale is not most of the storage.
    if info.eligible and info.contract < 32:
        info.eligible = False
        info.reason = f"contraction dim {info.contract} < 32: scale overhead dominates"
    if info.eligible and info.numel < 4096:
        info.eligible = False
        info.reason = f"{info.numel} parameters: not worth a kernel"


def canonical_matrix(w: np.ndarray, info: TensorInfo) -> np.ndarray:
    """Weight tensor -> (rows, contract) matrix, C-contiguous float32."""
    if info.role in ("linear", "embedding"):
        return np.ascontiguousarray(w, dtype=np.float32)
    if info.role == "conv1d":                      # (out, in, k) -> (out, in*k)
        return np.ascontiguousarray(w.reshape(w.shape[0], -1), dtype=np.float32)
    if info.role == "convtr1d":                    # (in, out, k) -> (out*k, in)
        return np.ascontiguousarray(
            np.transpose(w, (1, 2, 0)).reshape(-1, w.shape[0]), dtype=np.float32)
    raise ValueError(f"no canonical form for {info.name} ({info.role})")


def uncanonical(m: np.ndarray, info: TensorInfo) -> np.ndarray:
    """Inverse of `canonical_matrix`, so a fake-quantized matrix can go back."""
    sh = info.shape
    if info.role in ("linear", "embedding"):
        return m.reshape(sh)
    if info.role == "conv1d":
        return m.reshape(sh)
    if info.role == "convtr1d":
        return np.ascontiguousarray(
            np.transpose(m.reshape(sh[1], sh[2], sh[0]), (2, 0, 1)))
    raise ValueError(f"no inverse for {info.name} ({info.role})")


def inventory(pack: Pack) -> list[TensorInfo]:
    return [classify(n, pack.shape(n), pack.dtype(n)) for n in pack.names()]


# --------------------------------------------------------------------------
# the ternary object, and its honest storage cost
# --------------------------------------------------------------------------


@dataclass
class Ternary:
    """T in {-1,0,+1} with a per-row scale and shift. Ŵ = diag(a)·T + m·1^T."""
    t: np.ndarray            # int8, (rows, contract), values in [-levels, levels]
    alpha: np.ndarray        # float32, (rows,)
    mu: np.ndarray           # float32, (rows,)
    method: str
    iters: int = 0
    rotation: tuple[np.ndarray, np.ndarray] | None = None
    # 1 = ternary. The uniform baselines reuse this container so that int8 and
    # int4 are measured through exactly the same dequantize/metric path as the
    # ternary methods -- a baseline computed by a second code path is a
    # comparison between two code paths, not between two quantizers.
    levels: int = 1

    def dequant(self) -> np.ndarray:
        w = self.alpha[:, None] * self.t.astype(np.float32) + self.mu[:, None]
        if self.rotation is not None:
            r1, r2 = self.rotation
            w = kron_apply_right(w, r1, r2, transpose=True)
        return w

    @property
    def bits(self) -> float:
        return 2.0 if self.levels == 1 else math.ceil(math.log2(2 * self.levels + 1))

    def stats(self) -> dict:
        n = self.t.size
        return {
            "levels": self.levels,
            "zeros": float(np.count_nonzero(self.t == 0) / n),
            "plus": float(np.count_nonzero(self.t > 0) / n),
            "minus": float(np.count_nonzero(self.t < 0) / n),
            "alpha_mean": float(self.alpha.mean()),
            "alpha_min": float(self.alpha.min()),
            "alpha_max": float(self.alpha.max()),
            "mu_absmean": float(np.abs(self.mu).mean()),
        }


def ternary_bytes(rows: int, contract: int, *, shift: bool, scale_bits: int = 16,
                  rotation: tuple[int, int] | None = None,
                  bits_per_weight: float = 2.0) -> int:
    """Bytes actually on disk for one ternary matrix. No nominal 1.58.

    2 bits per weight packed 4-to-a-byte, plus one scale per row, plus one
    shift per row when the grid is asymmetric, plus the Kronecker rotation
    factors when the method needs them at runtime. A report that quotes 1.58
    and omits these is quoting a number nobody can ship.
    """
    n = rows * contract
    packed = int(math.ceil(n * bits_per_weight / 8.0))
    meta = rows * (scale_bits // 8)
    if shift:
        meta += rows * (scale_bits // 8)
    rot = 0
    if rotation is not None:
        n1, n2 = rotation
        rot = (n1 * n1 + n2 * n2) * 4
    return packed + meta + rot


# --------------------------------------------------------------------------
# quantizers
# --------------------------------------------------------------------------


def _solve_2x2(a11, a12, a22, b1, b2, *, eps=1e-12):
    """Per-row 2x2 symmetric solve by Cramer, with the singular case named.

    Singular means the row's ternary pattern carries no information the shift
    cannot also express (all-zero T, or T constant). Falling back to alpha=0
    there is correct: the best fit really is the constant mu.
    """
    det = a11 * a22 - a12 * a12
    bad = np.abs(det) < eps * np.maximum(np.abs(a11 * a22), 1.0)
    det_safe = np.where(bad, 1.0, det)
    alpha = (b1 * a22 - b2 * a12) / det_safe
    mu = (a11 * b2 - a12 * b1) / det_safe
    alpha = np.where(bad, 0.0, alpha)
    mu = np.where(bad, b2 / np.maximum(a22, eps), mu)
    return alpha.astype(np.float32), mu.astype(np.float32)


def _round_ternary(w: np.ndarray, alpha: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """Flexible rounding: nearest element of {-1,0,+1} to (w - mu)/alpha."""
    a = np.where(np.abs(alpha) < 1e-20, 1.0, alpha)[:, None]
    z = (w - mu[:, None]) / a
    return np.clip(np.rint(z), -1.0, 1.0).astype(np.int8)


def quant_naive(w: np.ndarray, *, per_row: bool = True,
                grid: Iterable[float] = None) -> Ternary:
    """Symmetric ternary, threshold searched rather than assumed.

    The literature's 0.7*mean|W| (TWN) and 0.75 (BitNet) are two points on a
    curve whose minimum depends on the weight distribution. Searching costs
    milliseconds and removes an arbitrary constant from the comparison, which
    matters because `naive` is the baseline every other method is judged
    against: a deliberately weak baseline would flatter the rest.
    """
    if grid is None:
        grid = np.arange(0.30, 1.31, 0.05)
    w = np.asarray(w, dtype=np.float32)
    rows = w.shape[0]
    mean_abs = np.abs(w).mean(axis=1) if per_row else np.full(rows, np.abs(w).mean())
    best_err = np.full(rows, np.inf, dtype=np.float64)
    best_t = np.zeros_like(w, dtype=np.int8)
    best_a = np.zeros(rows, dtype=np.float32)
    for k in grid:
        thr = (k * mean_abs)[:, None]
        t = (np.abs(w) > thr).astype(np.int8) * np.sign(w).astype(np.int8)
        num = np.sum(np.abs(w) * (t != 0), axis=1)
        den = np.maximum(np.count_nonzero(t, axis=1), 1)
        a = (num / den).astype(np.float32)
        err = np.sum((w - a[:, None] * t) ** 2, axis=1)
        better = err < best_err
        if better.any():
            best_err = np.where(better, err, best_err)
            best_t[better] = t[better]
            best_a[better] = a[better]
    if not per_row:                       # one scale for the whole tensor
        a = np.full(rows, float(best_a.mean()), dtype=np.float32)
        best_a = a
    return Ternary(best_t, best_a, np.zeros(rows, dtype=np.float32), "naive")


def quant_itf(w: np.ndarray, *, iters: int = 10, shift: bool = True,
              init: Ternary | None = None) -> Ternary:
    """PT^2-LLM Iterative Ternary Fitting, weight domain.

    Alternates the closed-form grid (alpha, mu) against flexible rounding of T,
    minimising ||W - (diag(alpha)T + mu 1^T)||_F^2. Exact w.r.t. the paper's
    equations; what we do *not* do here is the blockwise GPTQ interleaving --
    that lives in `quant_gptq`.
    """
    w = np.asarray(w, dtype=np.float32)
    rows, d = w.shape
    t = (init or quant_naive(w)).t.astype(np.float32)
    alpha = np.zeros(rows, dtype=np.float32)
    mu = np.zeros(rows, dtype=np.float32)
    used = 0
    for it in range(iters):
        s_tt = np.sum(t * t, axis=1)
        s_t = np.sum(t, axis=1)
        s_wt = np.sum(w * t, axis=1)
        s_w = np.sum(w, axis=1)
        if shift:
            alpha, mu = _solve_2x2(s_tt, s_t, np.full(rows, float(d)), s_wt, s_w)
        else:
            alpha = (s_wt / np.maximum(s_tt, 1e-12)).astype(np.float32)
            mu = np.zeros(rows, dtype=np.float32)
        t_new = _round_ternary(w, alpha, mu).astype(np.float32)
        used = it + 1
        if np.array_equal(t_new, t):
            t = t_new
            break
        t = t_new
    s_tt = np.sum(t * t, axis=1)
    s_t = np.sum(t, axis=1)
    if shift:
        alpha, mu = _solve_2x2(s_tt, s_t, np.full(rows, float(d)),
                               np.sum(w * t, axis=1), np.sum(w, axis=1))
    else:
        alpha = (np.sum(w * t, axis=1) / np.maximum(s_tt, 1e-12)).astype(np.float32)
        mu = np.zeros(rows, dtype=np.float32)
    return Ternary(t.astype(np.int8), alpha, mu, "itf", iters=used)


def align_grid(w: np.ndarray, t: np.ndarray, s: np.ndarray, *,
               shift: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Activation-aware grid alignment: refit (alpha, mu) under the S metric.

    Minimises Tr(E S E^T) with E = W - diag(alpha)T - mu 1^T and S = X^T X,
    T frozen. This is PT^2's AGA and TWLA's E2M-ATQ stage 2 -- the two papers
    describe the same per-row 2x2 system in different notation, which is worth
    knowing before anyone implements both.
    """
    w = np.asarray(w, dtype=np.float32)
    tf = t.astype(np.float32)
    s = np.asarray(s, dtype=np.float32)
    ts = tf @ s                         # (rows, d); S symmetric so this is (S t)^T
    u = s.sum(axis=1)                   # S·1
    a11 = np.einsum("rd,rd->r", ts, tf)             # t^T S t
    a12 = tf @ u                                    # t^T S 1
    a22 = np.full(w.shape[0], float(u.sum()))       # 1^T S 1
    b1 = np.einsum("rd,rd->r", ts, w)               # t^T S w
    b2 = w @ u                                      # 1^T S w
    if shift:
        return _solve_2x2(a11.astype(np.float64), a12.astype(np.float64),
                          a22.astype(np.float64), b1.astype(np.float64),
                          b2.astype(np.float64))
    alpha = (b1 / np.maximum(a11, 1e-12)).astype(np.float32)
    return alpha, np.zeros(w.shape[0], dtype=np.float32)


def quant_aga(w: np.ndarray, s: np.ndarray, *, iters: int = 10,
              shift: bool = True) -> Ternary:
    """ITF for the pattern, AGA for the grid. PT^2-LLM without SSR/GPTQ."""
    q = quant_itf(w, iters=iters, shift=shift)
    alpha, mu = align_grid(w, q.t, s, shift=shift)
    return Ternary(q.t, alpha, mu, "aga", iters=q.iters)


def quant_e2m(w: np.ndarray, s: np.ndarray, *, iters: int = 15) -> Ternary:
    """TWLA E2M-ATQ: Euclidean warm start, then manifold relocation.

    Adapted, not exact. The paper's stage 1 initialises mu to the per-row mean
    and applies a residual-mean correction inside the loop; stage 2 is the same
    per-row 2x2 system as AGA. We keep both and note the consequence in the
    report: with T frozen, E2M-ATQ and ITF+AGA differ only in how they arrive
    at T, so any gap between them is a statement about the warm start.
    """
    w = np.asarray(w, dtype=np.float32)
    rows, d = w.shape
    mu = w.mean(axis=1).astype(np.float32)
    resid = w - mu[:, None]
    t = quant_naive(resid).t.astype(np.float32)
    alpha = np.zeros(rows, dtype=np.float32)
    for _ in range(iters):
        s_tt = np.sum(t * t, axis=1)
        alpha = (np.sum(resid * t, axis=1) / np.maximum(s_tt, 1e-12)).astype(np.float32)
        err = resid - alpha[:, None] * t
        mu = mu + err.mean(axis=1).astype(np.float32)      # residual-mean correction
        resid = w - mu[:, None]
        t_new = _round_ternary(resid, alpha, np.zeros(rows, dtype=np.float32))
        if np.array_equal(t_new.astype(np.float32), t):
            t = t_new.astype(np.float32)
            break
        t = t_new.astype(np.float32)
    alpha, mu = align_grid(w, t.astype(np.int8), s, shift=True)
    return Ternary(t.astype(np.int8), alpha, mu, "e2m", iters=iters)


def quant_gptq(w: np.ndarray, s: np.ndarray, *, blocksize: int = 128,
               damp: float = 0.01, iters: int = 10, reorder: bool = True) -> Ternary:
    """AGA grid plus GPTQ-style sequential error compensation.

    The grid (alpha, mu) is fitted once by ITF+AGA and then held fixed, so the
    output object is the same shape as every other method's. What changes is
    T: columns are quantized left to right and the residual is pushed onto the
    not-yet-quantized columns through the inverse Hessian, which is the single
    biggest lever the ternary papers all end up using.

    `reorder` is our stand-in for PT^2's SSR: quantize the columns the metric
    cares about least *last*, so their error has nowhere left to go. SSR picks
    columns by structural similarity; we pick by diag(H), which is cheaper and
    is the ordering GPTQ's own act-order uses. Documented as adapted.
    """
    w = np.asarray(w, dtype=np.float64)
    rows, d = w.shape
    h = np.asarray(s, dtype=np.float64).copy()
    diag = np.diag(h).copy()
    dead = diag <= 0
    if dead.any():
        h[dead, dead] = 1.0
        w[:, dead] = 0.0
    perm = np.arange(d)
    if reorder:
        perm = np.argsort(-np.diag(h))
        h = h[perm][:, perm]
        w = w[:, perm]
    h = h + np.eye(d) * (damp * np.mean(np.diag(h)))
    try:
        hinv = np.linalg.cholesky(np.linalg.inv(h)).T.copy()
    except np.linalg.LinAlgError:
        h = h + np.eye(d) * (0.1 * np.mean(np.diag(h)))
        hinv = np.linalg.cholesky(np.linalg.inv(h)).T.copy()

    base = quant_aga(w.astype(np.float32),
                     s[perm][:, perm].astype(np.float32) if reorder else s,
                     iters=iters)
    alpha = base.alpha.astype(np.float64)
    mu = base.mu.astype(np.float64)
    a_safe = np.where(np.abs(alpha) < 1e-20, 1.0, alpha)

    q = np.zeros((rows, d), dtype=np.int8)
    err_block = np.zeros((rows, blocksize), dtype=np.float64)
    wc = w.copy()
    for i0 in range(0, d, blocksize):
        i1 = min(i0 + blocksize, d)
        nb = i1 - i0
        for j in range(nb):
            col = i0 + j
            wj = wc[:, col]
            d_jj = hinv[col, col]
            z = (wj - mu) / a_safe
            tj = np.clip(np.rint(z), -1.0, 1.0)
            qj = alpha * tj + mu
            q[:, col] = tj.astype(np.int8)
            e = (wj - qj) / d_jj
            if j + 1 < nb:
                wc[:, col + 1:i1] -= np.outer(e, hinv[col, col + 1:i1])
            err_block[:, j] = e
        if i1 < d:
            wc[:, i1:] -= err_block[:, :nb] @ hinv[i0:i1, i1:]
    if reorder:
        inv = np.empty_like(perm)
        inv[perm] = np.arange(d)
        q = q[:, inv]
    alpha32 = alpha.astype(np.float32)
    mu32 = mu.astype(np.float32)
    alpha32, mu32 = align_grid(np.asarray(w if not reorder else w[:, inv],
                                          dtype=np.float32), q, s, shift=True)
    return Ternary(q, alpha32, mu32, "gptq", iters=iters)


# --------------------------------------------------------------------------
# KOTMS-style rotation (adapted: Kronecker + Cayley, tri-modal surrogate)
# --------------------------------------------------------------------------


def kron_factors(d: int) -> tuple[int, int]:
    """Split d into the most square pair of factors. Cost of R is n1^2+n2^2."""
    best = (1, d)
    for n1 in range(1, int(math.isqrt(d)) + 1):
        if d % n1 == 0:
            best = (n1, d // n1)
    return best


def kron_apply_right(w: np.ndarray, r1: np.ndarray, r2: np.ndarray,
                     *, transpose: bool = False) -> np.ndarray:
    """W @ (R1 kron R2), or W @ (R1 kron R2)^T, without forming the d x d matrix.

    vec-trick: for one row v of length n1*n2 viewed as V (n1 x n2),
    v @ (R1 kron R2) = vec(R1^T V R2). This is why the Kronecker structure is
    the whole point -- the dense rotation would cost as much as the layer.
    """
    n1, n2 = r1.shape[0], r2.shape[0]
    rows = w.shape[0]
    v = w.reshape(rows, n1, n2)
    a = r1.T if not transpose else r1
    b = r2 if not transpose else r2.T
    out = np.einsum("ij,rjk,kl->ril", a, v, b, optimize=True)
    return out.reshape(rows, n1 * n2)


def _cayley(a: np.ndarray) -> np.ndarray:
    n = a.shape[0]
    skew = a - a.T
    eye = np.eye(n)
    return np.linalg.solve(eye + skew, eye - skew)


def fit_kotms(w: np.ndarray, *, steps: int = 60, lr: float = 0.02,
              seed: int = 1234) -> tuple[np.ndarray, np.ndarray]:
    """Fit R = R1 kron R2 so that W R looks tri-modal, per TWLA's shaping loss.

    Adapted: TWLA optimises the TriGMM likelihood with Adam over Cayley
    parameters and shares the rotation with the activations. We do the same in
    numpy with a finite-difference-free surrogate -- we maximise the negative
    log-likelihood of a fixed symmetric three-component mixture by gradient
    descent on the skew parameters, using an analytic gradient through the
    vec-trick. Omitted: the shared activation rotation is *not* re-derived from
    calibration outliers, and the zero-mass regulariser uses a fixed beta.
    """
    rng = np.random.default_rng(seed)
    rows, d = w.shape
    n1, n2 = kron_factors(d)
    if n1 == 1:
        return np.eye(1, dtype=np.float32), np.eye(d, dtype=np.float32)
    a1 = rng.normal(0.0, 1e-3, (n1, n1)).astype(np.float64)
    a2 = rng.normal(0.0, 1e-3, (n2, n2)).astype(np.float64)
    wd = np.asarray(w, dtype=np.float64)

    def shaped_loss(z):
        sigma = np.std(z, axis=1, keepdims=True) * 0.5 + 1e-8
        c = np.mean(np.abs(z), axis=1, keepdims=True) * 1.5 + 1e-8
        comps = np.stack([
            np.exp(-0.5 * ((z - c) / sigma) ** 2),
            np.exp(-0.5 * (z / sigma) ** 2),
            np.exp(-0.5 * ((z + c) / sigma) ** 2),
        ])
        mix = comps.mean(axis=0) + 1e-12
        return -np.mean(np.log(mix))

    best = (shaped_loss(wd), a1.copy(), a2.copy())
    for _ in range(steps):
        g1 = rng.normal(0.0, 1.0, (n1, n1))
        g2 = rng.normal(0.0, 1.0, (n2, n2))
        for sign in (+1.0, -1.0):
            c1 = a1 + sign * lr * g1
            c2 = a2 + sign * lr * g2
            z = kron_apply_right(wd, _cayley(c1), _cayley(c2))
            loss = shaped_loss(z)
            if loss < best[0]:
                best = (loss, c1.copy(), c2.copy())
        a1, a2 = best[1], best[2]
    return (_cayley(best[1]).astype(np.float32), _cayley(best[2]).astype(np.float32))


def quant_kotms(w: np.ndarray, s: np.ndarray, *, iters: int = 10,
                steps: int = 60) -> Ternary:
    """Rotate into a tri-modal basis, ternarize there, keep R for the runtime."""
    r1, r2 = fit_kotms(w, steps=steps)
    wr = kron_apply_right(np.asarray(w, dtype=np.float64), r1, r2).astype(np.float32)
    # S transforms as R^T S R; with Kronecker R we apply it on both sides.
    sr = kron_apply_right(kron_apply_right(np.asarray(s, np.float64), r1, r2).T,
                          r1, r2).astype(np.float32)
    q = quant_aga(wr, sr, iters=iters)
    q.method = "kotms"
    q.rotation = (r1, r2)
    return q


def quant_uniform(w: np.ndarray, bits: int) -> Ternary:
    """Per-row symmetric absmax, which is exactly what `src/qmat.c` ships.

    Not a strawman and not an improvement: the point of having it here is that
    the ternary numbers get compared against the quantizer already in the
    binary, measured through the same dequantize and the same metrics, rather
    than against a remembered figure from another run.
    """
    w = np.asarray(w, dtype=np.float32)
    levels = (1 << (bits - 1)) - 1
    scale = np.max(np.abs(w), axis=1) / levels
    scale = np.where(scale <= 0, 1.0, scale).astype(np.float32)
    t = np.clip(np.rint(w / scale[:, None]), -levels, levels).astype(np.int8)
    return Ternary(t, scale, np.zeros(w.shape[0], dtype=np.float32),
                   f"int{bits}", levels=levels)


METHODS = {
    "int8": lambda w, s, **kw: quant_uniform(w, 8),
    "int4": lambda w, s, **kw: quant_uniform(w, 4),
    "naive": lambda w, s, **kw: quant_naive(w),
    "naive-tensor": lambda w, s, **kw: quant_naive(w, per_row=False),
    "itf": lambda w, s, **kw: quant_itf(w, **kw),
    "aga": lambda w, s, **kw: quant_aga(w, s, **kw),
    "e2m": lambda w, s, **kw: quant_e2m(w, s),
    "gptq": lambda w, s, **kw: quant_gptq(w, s, **kw),
    "kotms": lambda w, s, **kw: quant_kotms(w, s, **kw),
}

for _m in ("naive", "itf", "aga", "e2m", "gptq", "kotms"):
    # `-gc` = the same method, then rescaled to reproduce the output norm.
    METHODS[f"{_m}-gc"] = METHODS[_m]
del _m

ACTIVATION_AWARE = {"aga", "e2m", "gptq", "kotms",
                    "aga-gc", "e2m-gc", "gptq-gc", "kotms-gc",
                    "naive-gc", "itf-gc"}


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------


def weight_metrics(w: np.ndarray, wq: np.ndarray) -> dict:
    w64 = np.asarray(w, dtype=np.float64)
    q64 = np.asarray(wq, dtype=np.float64)
    diff = q64 - w64
    denom = float(np.sqrt(np.sum(w64 ** 2)))
    num = float(np.sqrt(np.sum(diff ** 2)))
    wf, qf = w64.ravel(), q64.ravel()
    nw, nq = np.linalg.norm(wf), np.linalg.norm(qf)
    return {
        "mse": float(np.mean(diff ** 2)),
        "nmse": float(np.mean(diff ** 2) / max(np.mean(w64 ** 2), 1e-30)),
        "rel_fro": num / denom if denom > 0 else 0.0,
        "cos": float(wf @ qf / (nw * nq)) if nw > 0 and nq > 0 else 1.0,
        "max_abs": float(np.max(np.abs(diff))),
        "kurtosis": float(np.mean(((w64 - w64.mean()) / (w64.std() + 1e-30)) ** 4)),
        "outlier_frac": float(np.mean(np.abs(w64) > 4.0 * np.abs(w64).mean())),
    }


def output_metrics(w: np.ndarray, wq: np.ndarray, x: np.ndarray) -> dict:
    """Error of the layer's actual product on real calibration activations.

    x is (n, contract): the canonical activation rows that multiply the
    canonical weight matrix. The product is what the downstream graph sees, and
    it is the number that decides the question -- weight MSE is a proxy that
    has been wrong before.
    """
    xf = np.asarray(x, dtype=np.float32)
    ref = xf @ np.asarray(w, dtype=np.float32).T
    got = xf @ np.asarray(wq, dtype=np.float32).T
    d = (got - ref).astype(np.float64)
    r = ref.astype(np.float64)
    denom = float(np.sqrt(np.sum(r ** 2)))
    a, b = got.ravel().astype(np.float64), r.ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return {
        "out_mse": float(np.mean(d ** 2)),
        "out_nmse": float(np.mean(d ** 2) / max(np.mean(r ** 2), 1e-30)),
        "out_rel": float(np.sqrt(np.sum(d ** 2)) / denom) if denom > 0 else 0.0,
        "out_cos": float(a @ b / (na * nb)) if na > 0 and nb > 0 else 1.0,
        "out_max": float(np.max(np.abs(d))),
    }


# --------------------------------------------------------------------------
# per-frame call rates, for the kernel-readiness table
# --------------------------------------------------------------------------


def frame_rates(cfg: dict) -> dict[str, float]:
    """Hz at which each stage of the graph runs, derived from model.json.

    Nothing here is hardcoded to PocketTTS' current numbers: the codec ratios,
    the upsample stride and the frame rate all come from the pack, because the
    Italian pack and any future checkpoint are allowed to differ.
    """
    fr = float(cfg["frame_rate"])
    up = float(cfg.get("codec_upsample_stride", 16))
    ratios = list(cfg.get("codec_ratios", [6, 5, 4]))
    rates = {"frame": fr, "latent": fr, "codec_tr": fr * up}
    cur = fr * up
    for i, r in enumerate(ratios):
        cur = cur * r
        rates[f"codec_stage{i}"] = cur
    rates["waveform"] = cur
    return rates


def module_rate(info: TensorInfo, rates: dict[str, float]) -> float:
    """Calls per second for the tensor's module. Divide by frame_rate for
    calls per generated frame."""
    n = info.name
    if info.component in ("backbone", "backbone_io", "flow_head"):
        return rates["frame"]
    if info.component == "conditioner":
        return 0.0                      # prefill only
    if info.component == "mimi_dec_tr":
        return rates["codec_tr"]
    if info.component in ("mimi_encoder", "mimi_enc_tr"):
        return 0.0                      # cloning only
    if info.component == "mimi_decoder":
        if "quantizer" in n:
            return rates["latent"]
        if "upsample" in n:
            return rates["latent"]
        if ".model.0." in n:
            return rates["codec_tr"]
        if ".model.2." in n:
            return rates["codec_tr"]
        if ".model.3." in n:
            return rates.get("codec_stage0", 0.0)
        if ".model.5." in n:
            return rates.get("codec_stage0", 0.0)
        if ".model.6." in n:
            return rates.get("codec_stage1", 0.0)
        if ".model.8." in n:
            return rates.get("codec_stage1", 0.0)
        if ".model.9." in n:
            return rates.get("codec_stage2", 0.0)
        if ".model.11." in n:
            return rates.get("codec_stage2", 0.0)
    return 0.0


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float))


def _fmt(v, spec="{:.3e}"):
    if v is None:
        return "-"
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return "inf"
    return spec.format(v) if isinstance(v, float) else str(v)


# --------------------------------------------------------------------------
# phase 0: census
# --------------------------------------------------------------------------


def cmd_census(args) -> int:
    pack = Pack(args.model)
    inv = inventory(pack)
    total = sum(t.numel for t in inv)
    rates = frame_rates(pack.config)

    rows = []
    for t in inv:
        calls = module_rate(t, rates)
        rows.append({
            **{k: v for k, v in asdict(t).items() if k != "shape"},
            "shape": list(t.shape),
            "pct_params": 100.0 * t.numel / total,
            "bytes_fp32": t.numel * 4,
            "bytes_fp16": t.numel * 2,
            "bytes_int8": t.numel + t.rows * 4,
            "bytes_ternary": ternary_bytes(t.rows, t.contract, shift=True)
            if t.eligible else t.numel * 2,
            "calls_per_second": calls,
            "calls_per_frame": calls / rates["frame"] if rates["frame"] else 0.0,
        })

    by_component: dict[str, dict] = {}
    for r in rows:
        c = by_component.setdefault(r["component"], {
            "params": 0, "tensors": 0, "eligible_params": 0, "streaming_params": 0,
            "bytes_fp16": 0, "bytes_ternary": 0})
        c["params"] += r["numel"]
        c["tensors"] += 1
        c["bytes_fp16"] += r["bytes_fp16"]
        c["bytes_ternary"] += r["bytes_ternary"]
        if r["eligible"]:
            c["eligible_params"] += r["numel"]
        if r["streaming"]:
            c["streaming_params"] += r["numel"]

    by_group: dict[str, dict] = {}
    for r in rows:
        key = f"{r['component']}/{r['group']}"
        g = by_group.setdefault(key, {"params": 0, "tensors": 0, "eligible": 0})
        g["params"] += r["numel"]
        g["tensors"] += 1
        g["eligible"] += r["numel"] if r["eligible"] else 0

    # which subset of tensors reaches 50/70/80/90% of total weight storage
    ordered = sorted([r for r in rows if r["eligible"]],
                     key=lambda r: -r["numel"])
    cum, coverage = 0, {}
    for target in (0.5, 0.7, 0.8, 0.9):
        cum, n = 0, 0
        for r in ordered:
            cum += r["numel"]
            n += 1
            if cum / total >= target:
                break
        coverage[f"{int(target * 100)}%"] = {
            "tensors": n, "params": cum, "pct": 100.0 * cum / total,
            "smallest_included": ordered[n - 1]["name"] if n else None,
        }

    out = {
        "generator": "tools/ternary_feasibility.py census",
        "model": str(args.model),
        "model_sha256": pack.sha256() if args.checksum else None,
        "revision": pack.config.get("revision"),
        "source": pack.config.get("source"),
        "dtype_on_disk": pack.config.get("dtype"),
        "total_params": total,
        "total_tensors": len(rows),
        "eligible_params": sum(r["numel"] for r in rows if r["eligible"]),
        "streaming_params": sum(r["numel"] for r in rows if r["streaming"]),
        "eligible_streaming_params": sum(
            r["numel"] for r in rows if r["eligible"] and r["streaming"]),
        "frame_rates_hz": rates,
        "by_component": by_component,
        "by_group": by_group,
        "storage_coverage": coverage,
        "tensors": rows,
    }
    _write_json(Path(args.out) / "census.json", out)

    print(f"{total:,} parameters in {len(rows)} tensors  "
          f"({pack.config.get('dtype')} on disk, "
          f"{total * 2 / 1e6:.1f} MB)")
    print(f"{'component':<22}{'tensors':>8}{'params':>14}{'  %':>7}"
          f"{'eligible':>12}{'streaming':>12}")
    for c, v in sorted(by_component.items(), key=lambda kv: -kv[1]["params"]):
        print(f"{c:<22}{v['tensors']:>8}{v['params']:>14,}"
              f"{100.0 * v['params'] / total:>7.1f}"
              f"{v['eligible_params']:>12,}{v['streaming_params']:>12,}")
    print()
    for k, v in coverage.items():
        print(f"  {k:>4} of all weights = {v['tensors']:>3} tensors "
              f"({v['pct']:.1f}%), down to {v['smallest_included']}")
    print(f"\nwrote {Path(args.out) / 'census.json'}")
    return 0


# --------------------------------------------------------------------------
# phase 1/2 support: calibration capture from the reference implementation
# --------------------------------------------------------------------------


CALIB_TEXTS = [
    "Your card was declined.",
    "We found three matches.",
    "The room is ready for you.",
    "We are unable to change the booking within twelve hours of travel.",
    "The alarm will sound briefly while the system runs its daily test.",
    "Good question. I will have to find out and call you back.",
    "Yes, that is right. The second one replaced the first.",
    "The best time to see the site is early, before the coaches arrive, and the "
    "light is better then as well. Allow two hours for the main circuit and "
    "another hour if you want the upper terrace.",
]

EVAL_TEXTS = [
    ("short", "Your card was declined."),
    ("short", "The room is ready for you."),
    ("medium", "We are unable to change the booking within twelve hours of travel."),
    ("conversational", "Good question. I will have to find out and call you back."),
    ("long", "Before we go any further I should explain how the assessment works, "
     "because it is not quite what most people expect. A surveyor visits the "
     "property and spends roughly an hour looking at the structure, the roof "
     "and the drainage."),
    # Three long utterances, not one. The long class is where EOS is decided,
    # and a single 16-second sample cannot tell a systematic termination
    # failure from one unlucky trajectory.
    ("long", "The best time to see the site is early, before the coaches arrive, "
     "and the light is better then as well. Allow two hours for the main circuit "
     "and another hour if you want the upper terrace, which involves a steep "
     "climb and is not accessible by wheelchair."),
    ("long", "The route has changed since you last travelled with us, so I will "
     "run through it carefully. The service now leaves from the lower platform "
     "rather than the main concourse, and the first stop is the retail park "
     "instead of the hospital. Journey time is about eight minutes longer in "
     "the morning peak but slightly quicker in the evening."),
]


class _Collector:
    """Accumulates the exact Gram matrix and a reservoir of raw rows.

    Two different consumers, two different structures. The activation-aware
    grid solve needs S = X^T X over *all* the activations the layer ever sees,
    because a rank-deficient S makes GPTQ's Hessian inverse meaningless. The
    output-error metric needs actual rows, but only a few hundred of them.
    Collecting the sample and pretending S = sample^T sample would quietly
    break the first consumer, so we pay for both.
    """

    def __init__(self, contract: int, reservoir: int, seed: int):
        self.d = contract
        self.gram = np.zeros((contract, contract), dtype=np.float64)
        self.rows_seen = 0
        self.reservoir = reservoir
        self.sample = np.zeros((reservoir, contract), dtype=np.float32)
        self.filled = 0
        self.rng = np.random.default_rng(seed)
        self.calls = 0

    def add(self, x: np.ndarray) -> None:
        x = np.ascontiguousarray(x, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != self.d:
            return
        self.calls += 1
        self.gram += (x.T @ x).astype(np.float64)
        for i in range(x.shape[0]):
            self.rows_seen += 1
            if self.filled < self.reservoir:
                self.sample[self.filled] = x[i]
                self.filled += 1
            else:
                j = self.rng.integers(0, self.rows_seen)
                if j < self.reservoir:
                    self.sample[j] = x[i]


def _canon_input(module, x, kind: str) -> np.ndarray | None:
    """Module input -> the (n, contract) rows that multiply the canonical W."""
    import torch
    import torch.nn.functional as F
    if not isinstance(x, torch.Tensor):
        return None
    x = x.detach().to(torch.float32)
    if kind == "linear":
        return x.reshape(-1, x.shape[-1]).cpu().numpy()
    if kind == "convtr1d":
        # y[o,t'] = sum_i x[i,t] W[i,o,k]: the contraction is over in-channels,
        # so the rows are the time steps of the *input*, untouched by k.
        return x.transpose(1, 2).reshape(-1, x.shape[1]).cpu().numpy()
    if kind == "conv1d":
        conv = module
        pad = conv.padding[0] if isinstance(conv.padding, tuple) else conv.padding
        if isinstance(pad, str):
            pad = 0
        xp = F.pad(x, (pad, pad)) if pad else x
        k = conv.kernel_size[0]
        stride = conv.stride[0]
        dil = conv.dilation[0]
        span = (k - 1) * dil + 1
        if xp.shape[-1] < span:
            return None
        patches = xp.unfold(-1, span, stride)              # (B, C, T', span)
        if dil > 1:
            patches = patches[..., ::dil]
        # (B, C, T', k) -> (B*T', C*k) to match W.reshape(out, in*k)
        patches = patches.permute(0, 2, 1, 3).reshape(-1, xp.shape[1] * k)
        return patches.cpu().numpy()
    return None


def _load_upstream(language: str, seed: int, temperature: float | None = None):
    """Load the reference model, optionally with the sampler temperature pinned.

    `temperature=0` is the measurement mode and it is not cosmetic. At the
    shipped 0.3, two FP runs of the same sentence with different seeds correlate
    at only **0.648** in log-mel: the sampler's own variability is larger than
    anything a quantizer does, so an end-to-end metric against an FP reference
    measures the dice and not the weights. At 0 the two runs are **bit
    identical**, the floor is exactly zero, and every difference that remains is
    attributable to the weights.

    The cost is that 0 is not how the model ships, so the audio is a little
    flatter than production. Measure at 0; listen at 0.3.
    """
    import torch
    from pocket_tts import TTSModel
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    model = TTSModel.load_model(language=language)
    model.eval()
    if temperature is not None:
        model.temp = float(temperature)
    return model


def _eligible_modules(model, inv: list[TensorInfo]) -> dict[str, tuple]:
    """tensor name -> (module, kind, TensorInfo) for everything we can hook."""
    import torch
    mods = dict(model.named_modules())
    out = {}
    for t in inv:
        if not t.eligible or t.role == "embedding":
            continue
        mod_name = t.name[:-len(".weight")] if t.name.endswith(".weight") else None
        if mod_name is None or mod_name not in mods:
            continue
        m = mods[mod_name]
        if isinstance(m, torch.nn.Linear):
            kind = "linear"
        elif isinstance(m, torch.nn.ConvTranspose1d):
            kind = "convtr1d"
        elif isinstance(m, torch.nn.Conv1d):
            kind = "conv1d"
        else:
            continue
        if kind != t.role:
            raise SystemExit(f"role mismatch for {t.name}: pack={t.role} upstream={kind}")
        if getattr(m, "groups", 1) != 1:
            print(f"  skip {t.name}: groups={m.groups}, contraction is not shared")
            continue
        out[t.name] = (m, kind, t)
    return out


def cmd_calib(args) -> int:
    import torch
    pack = Pack(args.model)
    inv = inventory(pack)
    model = _load_upstream(args.language, args.seed, args.temperature)
    targets = _eligible_modules(model, inv)
    if not targets:
        raise SystemExit("no hookable modules found; upstream module names moved")

    out_dir = Path(args.out) / "calib"
    out_dir.mkdir(parents=True, exist_ok=True)
    collectors = {n: _Collector(t.contract, args.reservoir, args.seed + i)
                  for i, (n, (_, _, t)) in enumerate(targets.items())}
    mem = sum(c.d * c.d * 8 + c.reservoir * c.d * 4 for c in collectors.values())
    print(f"hooking {len(targets)} modules, {mem / 1e9:.2f} GB of Gram accumulators")

    handles = []
    for name, (m, kind, _) in targets.items():
        def make(nm, kd):
            def fn(module, inputs, _output):
                if not inputs:
                    return
                x = _canon_input(module, inputs[0], kd)
                if x is not None:
                    collectors[nm].add(x)
            return fn
        handles.append(m.register_forward_pre_hook(
            lambda mod, inp, nm=name, kd=kind: make(nm, kd)(mod, inp, None)))

    texts = CALIB_TEXTS[:args.examples]
    voice_state = model.get_state_for_audio_prompt(args.voice)
    t0 = time.time()
    with torch.no_grad():
        for i, text in enumerate(texts):
            torch.manual_seed(args.seed + i)
            model.generate_audio(voice_state, text)
            print(f"  [{i + 1}/{len(texts)}] {time.time() - t0:6.1f}s  {text[:56]}")
    for h in handles:
        h.remove()

    manifest = {
        "generator": "tools/ternary_feasibility.py calib",
        "language": args.language, "voice": args.voice, "seed": args.seed,
        "examples": len(texts), "reservoir": args.reservoir,
        "texts": texts, "modules": {},
    }
    for name, c in collectors.items():
        if c.calls == 0 or c.rows_seen == 0:
            print(f"  WARN {name}: never called")
            continue
        safe = name.replace("/", "_")
        np.save(out_dir / f"{safe}.gram.npy", (c.gram / c.rows_seen).astype(np.float32))
        np.save(out_dir / f"{safe}.x.npy", c.sample[:c.filled].astype(np.float16))
        manifest["modules"][name] = {
            "contract": c.d, "calls": c.calls, "rows_seen": c.rows_seen,
            "sampled": int(c.filled),
            "gram_trace": float(np.trace(c.gram) / c.rows_seen),
            "act_absmax": float(np.abs(c.sample[:c.filled]).max()) if c.filled else 0.0,
            "act_rms": float(np.sqrt(np.mean(c.sample[:c.filled] ** 2)))
            if c.filled else 0.0,
        }
    _write_json(Path(args.out) / "calib.json", manifest)
    print(f"captured {len(manifest['modules'])} modules in {time.time() - t0:.1f}s "
          f"-> {out_dir}")
    return 0


def load_calib(out: Path, name: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    safe = name.replace("/", "_")
    g = out / "calib" / f"{safe}.gram.npy"
    x = out / "calib" / f"{safe}.x.npy"
    gram = np.load(g) if g.exists() else None
    xs = np.load(x).astype(np.float32) if x.exists() else None
    return gram, xs


# --------------------------------------------------------------------------
# quantization cache: T, alpha, mu, rotation -- never a dequantized copy
# --------------------------------------------------------------------------


def gain_correct(q: Ternary, w: np.ndarray, x: np.ndarray) -> float:
    """Rescale the grid so the layer reproduces the output NORM, not just the MSE.

    A least-squares ternary fit is a projection: it minimises ||Wx - Ŵx||, and
    the minimiser is biased low in norm by exactly the fraction of variance it
    cannot explain. One layer loses ~2%. A codec decoder is eighteen layers in
    series, and 0.977^18 = 0.65 -- which is precisely the 38% level drop the
    rendered audio showed before this existed.

    Dividing the grid by the measured gain trades a little MSE for an unbiased
    norm. That is the right trade in a deterministic cascade, where the bias
    compounds and the noise does not. It is NOT obviously right inside the AR
    loop, where the state is fed back; measured separately.
    """
    ref = x @ np.asarray(w, dtype=np.float32).T
    got = x @ q.dequant().T
    nr, ng = float(np.linalg.norm(ref)), float(np.linalg.norm(got))
    if ng <= 0 or nr <= 0:
        return 1.0
    g = ng / nr
    q.alpha = (q.alpha / g).astype(np.float32)
    q.mu = (q.mu / g).astype(np.float32)
    return g


def quantize_cached(pack: Pack, info: TensorInfo, method: str, out: Path,
                    *, iters: int = 10, force: bool = False) -> tuple[Ternary, np.ndarray]:
    """Ternarize one tensor, memoised on disk. Returns (Ternary, W_canonical).

    The cache stores the ternary object, not the dequantized matrix: 2 bits a
    weight instead of 32, so all six methods together cost less than one FP32
    copy of the checkpoint. It also means the cache is the thing a future C
    packer would read, rather than a debug artifact.
    """
    cdir = out / "quant" / method
    cdir.mkdir(parents=True, exist_ok=True)
    safe = info.name.replace("/", "_")
    path = cdir / f"{safe}.npz"
    w = canonical_matrix(pack.tensor(info.name), info)
    if path.exists() and not force:
        z = np.load(path, allow_pickle=False)
        rot = (z["r1"], z["r2"]) if "r1" in z.files else None
        return Ternary(z["t"], z["alpha"], z["mu"], method,
                       iters=int(z["iters"]), rotation=rot,
                       levels=int(z["levels"]) if "levels" in z.files else 1), w
    gram, _ = load_calib(out, info.name)
    if method in ACTIVATION_AWARE and gram is None and info.role != "embedding":
        raise SystemExit(
            f"{method} needs calibration activations for {info.name}; run `calib` first")
    if gram is None:
        # Identity Gram degrades every activation-aware method to its
        # weight-domain twin, which is the right answer for a lookup table:
        # there is no X to be aware of. The sweep records `calibrated: false`
        # so no reader mistakes one for the other.
        gram = np.eye(info.contract, dtype=np.float32)
    base = method[:-3] if method.endswith("-gc") else method
    simple = ("naive", "naive-tensor", "e2m", "int8", "int4")
    q = METHODS[base](w, gram) if base in simple \
        else METHODS[base](w, gram, iters=iters)
    if method.endswith("-gc"):
        _, xs = load_calib(out, info.name)
        if xs is None or not xs.size:
            raise SystemExit(f"-gc needs calibration rows for {info.name}")
        gain_correct(q, w, xs)
        q.method = method
    payload = {"t": q.t, "alpha": q.alpha, "mu": q.mu, "iters": np.int32(q.iters),
               "levels": np.int32(q.levels)}
    if q.rotation is not None:
        payload["r1"], payload["r2"] = q.rotation
    np.savez(path, **payload)
    return q, w


# --------------------------------------------------------------------------
# phase 2: per-layer sensitivity
# --------------------------------------------------------------------------


def cmd_sweep(args) -> int:
    pack = Pack(args.model)
    inv = inventory(pack)
    out = Path(args.out)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    for m in methods:
        if m not in METHODS:
            raise SystemExit(f"unknown method {m}; have {sorted(METHODS)}")

    calib_manifest = {}
    cj = out / "calib.json"
    if cj.exists():
        calib_manifest = json.loads(cj.read_text()).get("modules", {})

    targets = [t for t in inv if t.eligible]
    if args.only:
        pats = [p.strip() for p in args.only.split(",")]
        targets = [t for t in targets if any(p in t.name for p in pats)]
    if args.max_params:
        targets = [t for t in targets if t.numel <= args.max_params]

    results = []
    t0 = time.time()
    for i, info in enumerate(targets):
        row = {
            "name": info.name, "component": info.component, "group": info.group,
            "role": info.role, "shape": list(info.shape), "numel": info.numel,
            "rows": info.rows, "contract": info.contract,
            "streaming": info.streaming,
            "calibrated": info.name in calib_manifest,
            "methods": {},
        }
        _, xs = load_calib(out, info.name)
        for m in methods:
            if m in ACTIVATION_AWARE and info.name not in calib_manifest:
                continue
            ts = time.time()
            try:
                q, w = quantize_cached(pack, info, m, out, iters=args.iters,
                                       force=args.force)
            except MemoryError:
                row["methods"][m] = {"error": "out of memory"}
                continue
            wq = q.dequant()
            e = weight_metrics(w, wq)
            e.update(q.stats())
            if xs is not None and xs.size:
                e.update(output_metrics(w, wq, xs))
            e["seconds"] = time.time() - ts
            e["bytes_ternary"] = ternary_bytes(
                info.rows, info.contract, shift=q.levels == 1,
                bits_per_weight=q.bits,
                rotation=kron_factors(info.contract) if q.rotation is not None else None)
            row["methods"][m] = e
        results.append(row)
        best = min((v.get("out_nmse", v.get("nmse", math.inf))
                    for v in row["methods"].values() if "error" not in v),
                   default=math.inf)
        print(f"[{i + 1:>3}/{len(targets)}] {info.name:<62} "
              f"best nmse {best:.4f}  {time.time() - t0:6.1f}s")

    _write_json(out / "sweep.json", {
        "generator": "tools/ternary_feasibility.py sweep",
        "model": str(args.model), "methods": methods, "iters": args.iters,
        "calibrated_modules": len(calib_manifest),
        "layers": results,
    })
    print(f"\nwrote {out / 'sweep.json'} ({len(results)} layers, "
          f"{time.time() - t0:.1f}s)")
    return 0


# --------------------------------------------------------------------------
# phase 3: cumulative ternarization + end-to-end audio
# --------------------------------------------------------------------------


def sensitivity_order(sweep: dict, method: str, *,
                      streaming_only: bool = False) -> list[dict]:
    """Least sensitive first, by output NMSE under `method`.

    Falls back to weight NMSE for a layer the calibration run never reached,
    and says so in the row, because silently mixing two different rankings is
    how a Pareto curve stops meaning anything.
    """
    rows = []
    for L in sweep["layers"]:
        m = L["methods"].get(method)
        if m is None or "error" in m:
            continue
        if streaming_only and not L["streaming"]:
            continue
        score = m.get("out_nmse")
        rows.append({
            "name": L["name"], "numel": L["numel"], "component": L["component"],
            "group": L["group"], "streaming": L["streaming"],
            "score": score if score is not None else m["nmse"],
            "score_kind": "out_nmse" if score is not None else "weight_nmse",
            "bytes_ternary": m["bytes_ternary"], "bytes_fp16": L["numel"] * 2,
        })
    rows.sort(key=lambda r: r["score"])
    return rows


def _apply_fake_quant(model, pack: Pack, inv_by_name: dict, names,
                      method, out: Path, iters: int):
    """`method` is either one name for every tensor, or a dict name -> method."""
    """Swap selected weights for their fake-quantized versions, in place.

    Returns an undo closure. The upstream model is a throwaway object in this
    process; the checkpoint on disk is never touched.
    """
    import torch
    sd = dict(model.named_parameters())
    undo = []
    applied = []
    for n in names:
        info = inv_by_name[n]
        if n not in sd:
            raise SystemExit(f"upstream has no parameter {n}")
        m = method[n] if isinstance(method, dict) else method
        if m is None:                      # fp16 tier: leave the weight alone
            continue
        q, w = quantize_cached(pack, info, m, out, iters=iters)
        wq = uncanonical(q.dequant(), info)
        p = sd[n]
        undo.append((p, p.data.clone()))
        with torch.no_grad():
            p.data.copy_(torch.from_numpy(np.ascontiguousarray(wq)).to(p.dtype))
        applied.append(n)

    def restore():
        with torch.no_grad():
            for p, old in undo:
                p.data.copy_(old)
    return restore, applied


def _audio_metrics(ref: np.ndarray, cand: np.ndarray, sr: int) -> dict:
    from parity_pocket import log_mel, pearson
    r = np.asarray(ref, dtype=np.float64).ravel()
    c = np.asarray(cand, dtype=np.float64).ravel()
    n = min(r.size, c.size)
    out = {
        "ref_seconds": r.size / sr, "cand_seconds": c.size / sr,
        "duration_ratio": (c.size / r.size) if r.size else float("inf"),
        "cand_peak": float(np.max(np.abs(c))) if c.size else 0.0,
        "cand_rms": float(np.sqrt(np.mean(c ** 2))) if c.size else 0.0,
        "ref_rms": float(np.sqrt(np.mean(r ** 2))) if r.size else 0.0,
        "finite": bool(np.isfinite(c).all()),
    }
    if n == 0:
        out.update(mel_corr=0.0, mel_l1=float("inf"), wave_rel=float("inf"))
        return out
    mr, mc = log_mel(r[:n], sr), log_mel(c[:n], sr)
    k = min(mr.shape[0], mc.shape[0])
    out["mel_corr"] = float(pearson(mc[:k], mr[:k]))
    out["mel_l1"] = float(np.mean(np.abs(mc[:k] - mr[:k])))
    denom = float(np.sqrt(np.sum(r[:n] ** 2)))
    out["wave_rel"] = float(np.sqrt(np.sum((c[:n] - r[:n]) ** 2)) / denom) \
        if denom > 0 else float("inf")
    # A collapse is not a small error: name it rather than letting a reader
    # infer it from a correlation that happens to be low.
    # The bounds are not a guess. FP-vs-FP at the shipped temperature, seven
    # texts, different seed, spans duration ratio 0.90-1.12; that is how much
    # two equally good takes of the same sentence differ. The original [0.6,
    # 1.6] was four times too loose and reported a configuration whose long
    # utterances ran +49% and +51% over as clean. Anything outside the measured
    # FP spread with margin is damage, not variation.
    out["collapsed"] = bool(
        out["cand_rms"] < 0.02 * max(out["ref_rms"], 1e-9)
        or out["duration_ratio"] > 1.25 or out["duration_ratio"] < 0.80
        or not out["finite"])
    return out


def _generate(model, voice_state, text: str, seed: int) -> np.ndarray:
    import torch
    torch.manual_seed(seed)
    with torch.no_grad():
        audio = model.generate_audio(voice_state, text)
    return audio.detach().to(torch.float32).cpu().numpy().reshape(-1)


def _write_wav(path: Path, sr: int, x: np.ndarray) -> None:
    import wave
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(np.asarray(x, dtype=np.float64), -1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def cmd_cumulative(args) -> int:
    pack = Pack(args.model)
    inv = inventory(pack)
    inv_by_name = {t.name: t for t in inv}
    out = Path(args.out)
    sweep = json.loads((out / "sweep.json").read_text())
    order = sensitivity_order(sweep, args.method, streaming_only=args.streaming_only)
    if args.include:
        pats = [x.strip() for x in args.include.split(",") if x.strip()]
        order = [r for r in order if any(x in r["name"] for x in pats)]
    if args.exclude:
        pats = [x.strip() for x in args.exclude.split(",") if x.strip()]
        order = [r for r in order if not any(x in r["name"] for x in pats)]
    if not order:
        raise SystemExit(f"no swept layers for method {args.method}")

    eligible_bytes = sum(r["bytes_fp16"] for r in order)
    total_params = sum(t.numel for t in inv)
    fracs = [float(x) for x in args.fractions.split(",")]

    model = _load_upstream(args.language, args.seed, args.temperature)
    voice_state = model.get_state_for_audio_prompt(args.voice)
    sr = int(model.sample_rate)
    wav_dir = out / "audio" / args.method
    texts = EVAL_TEXTS[:args.examples]

    print(f"reference pass ({len(texts)} utterances)")
    refs = []
    for i, (kind, text) in enumerate(texts):
        a = _generate(model, voice_state, text, args.seed + i)
        refs.append(a)
        _write_wav(wav_dir / f"fp32_{i:02d}_{kind}.wav", sr, a)
        print(f"  ref [{i}] {a.size / sr:5.2f}s  {text[:52]}")

    # The floor every later number is judged against. Same model, same text,
    # a different sampler seed: this is what "no change at all" measures, and
    # any ternary point at or above it is indistinguishable from re-rolling the
    # dice. Measuring it in the same process as the curve is deliberate -- a
    # floor taken on another day, on another machine, is not a floor.
    floor = []
    for i, (kind, text) in enumerate(texts):
        a = _generate(model, voice_state, text, args.seed + i + 10_000)
        m = _audio_metrics(refs[i], a, sr)
        m["kind"] = kind
        floor.append(m)
        if args.save_audio:
            _write_wav(wav_dir / f"fp32_reseed_{i:02d}_{kind}.wav", sr, a)
    floor_agg = {
        "mel_corr_mean": float(np.mean([p["mel_corr"] for p in floor])),
        "mel_corr_min": float(np.min([p["mel_corr"] for p in floor])),
        "mel_l1_mean": float(np.mean([p["mel_l1"] for p in floor])),
        "duration_ratio_mean": float(np.mean([p["duration_ratio"] for p in floor])),
        "wave_rel_mean": float(np.mean([p["wave_rel"] for p in floor])),
        "utterances": floor,
    }
    print(f"  FP-vs-FP seed floor: mel_corr {floor_agg['mel_corr_mean']:.4f} "
          f"(min {floor_agg['mel_corr_min']:.4f}), "
          f"mel_l1 {floor_agg['mel_l1_mean']:.4f}, "
          f"wave_rel {floor_agg['wave_rel_mean']:.3f}")

    if args.layout_budgets:
        lay_points = []
        for b in [float(x) for x in args.layout_budgets.split(",")]:
            lay = build_layout(sweep, b, ternary_method=args.method,
                               total_params=total_params)
            names = [n for n, a in lay["assign"].items() if a["method"] is not None]
            mmap = {n: lay["assign"][n]["method"] for n in names}
            restore, applied = _apply_fake_quant(
                model, pack, inv_by_name, names, mmap, out, args.iters)
            per = []
            for i, (kind, text) in enumerate(texts):
                a = _generate(model, voice_state, text, args.seed + i)
                m = _audio_metrics(refs[i], a, sr)
                m["kind"] = kind
                per.append(m)
                if args.save_audio:
                    _write_wav(wav_dir / f"layout_{b}_{i:02d}_{kind}.wav", sr, a)
            restore()
            agg = {k: lay[k] for k in ("budget_out_nmse", "params_by_tier",
                                       "pct_ternary", "checkpoint_mb",
                                       "effective_bits_per_weight")}
            agg.update({
                "layers_quantized": len(applied),
                "protected": [q["name"] for q in lay["protected"]],
                "mel_corr_mean": float(np.mean([q["mel_corr"] for q in per])),
                "mel_corr_min": float(np.min([q["mel_corr"] for q in per])),
                "mel_l1_mean": float(np.mean([q["mel_l1"] for q in per])),
                "duration_ratio_mean": float(np.mean([q["duration_ratio"] for q in per])),
                "collapsed": int(sum(1 for q in per if q["collapsed"])),
                "utterances": per,
            })
            lay_points.append(agg)
            print(f"  budget {b:<6} {agg['pct_ternary']:5.1f}% ternary  "
                  f"{agg['effective_bits_per_weight']:4.2f} bits/w  "
                  f"{agg['checkpoint_mb']:6.1f} MB  "
                  f"mel_corr {agg['mel_corr_mean']:.4f} "
                  f"(min {agg['mel_corr_min']:.4f})  "
                  f"collapsed {agg['collapsed']}/{len(per)}")
        _write_json(out / f"layout-eval-{args.method}.json", {
            "generator": "tools/ternary_feasibility.py cumulative --layout-budgets",
            "method": args.method, "temperature": args.temperature,
            "fp_seed_floor": floor_agg,
            "texts": [{"kind": k, "text": t} for k, t in texts],
            "audio_dir": str(wav_dir), "points": lay_points,
        })
        print(f"\nwrote {out / f'layout-eval-{args.method}.json'}")
        return 0

    points = []
    for frac in fracs:
        budget = frac * eligible_bytes
        chosen, acc = [], 0
        for r in order:
            if acc + r["bytes_fp16"] > budget and chosen:
                break
            chosen.append(r)
            acc += r["bytes_fp16"]
        names = [r["name"] for r in chosen]
        restore, applied = _apply_fake_quant(
            model, pack, inv_by_name, names, args.method, out, args.iters)
        tern_params = sum(inv_by_name[n].numel for n in applied)
        tern_bytes = sum(r["bytes_ternary"] for r in chosen)
        rest_bytes = (total_params - tern_params) * 2
        eff_bits = 8.0 * (tern_bytes + rest_bytes) / total_params
        per = []
        for i, (kind, text) in enumerate(texts):
            a = _generate(model, voice_state, text, args.seed + i)
            m = _audio_metrics(refs[i], a, sr)
            m["kind"] = kind
            per.append(m)
            tag = f"{args.method}_{int(frac * 100):03d}pct_{i:02d}_{kind}"
            if args.save_audio:
                _write_wav(wav_dir / f"{tag}.wav", sr, a)
        restore()
        agg = {
            "fraction": frac, "layers": len(applied),
            "ternary_params": tern_params,
            "pct_of_model": 100.0 * tern_params / total_params,
            "checkpoint_bytes": tern_bytes + rest_bytes,
            "checkpoint_mb": (tern_bytes + rest_bytes) / 1e6,
            "effective_bits_per_weight": eff_bits,
            "mel_corr_mean": float(np.mean([p["mel_corr"] for p in per])),
            "mel_corr_min": float(np.min([p["mel_corr"] for p in per])),
            "mel_l1_mean": float(np.mean([p["mel_l1"] for p in per])),
            "duration_ratio_mean": float(np.mean([p["duration_ratio"] for p in per])),
            "collapsed": int(sum(1 for p in per if p["collapsed"])),
            "worst_layer_added": chosen[-1]["name"] if chosen else None,
            "worst_score_added": chosen[-1]["score"] if chosen else None,
            "utterances": per,
        }
        points.append(agg)
        print(f"  {int(frac * 100):>3}%  {len(applied):>3} layers  "
              f"{tern_params / 1e6:5.1f}M params  "
              f"{eff_bits:4.2f} bits/w  "
              f"mel_corr {agg['mel_corr_mean']:.4f} (min {agg['mel_corr_min']:.4f})  "
              f"collapsed {agg['collapsed']}/{len(per)}")

    _write_json(out / f"cumulative-{args.method}.json", {
        "generator": "tools/ternary_feasibility.py cumulative",
        "method": args.method, "language": args.language, "voice": args.voice,
        "seed": args.seed, "streaming_only": args.streaming_only,
        "temperature": args.temperature,
        "total_params": total_params, "eligible_bytes_fp16": eligible_bytes,
        "texts": [{"kind": k, "text": t} for k, t in texts],
        "fp_seed_floor": floor_agg,
        "audio_dir": str(wav_dir),
        "order": order, "points": points,
    })
    print(f"\nwrote {out / f'cumulative-{args.method}.json'}; audio in {wav_dir}")
    return 0


# --------------------------------------------------------------------------
# self-test: the quantizer math, without the model
# --------------------------------------------------------------------------


def cmd_selftest(args) -> int:
    """Check the closed forms against brute force on synthetic data.

    Every estimator here is a closed-form solve that was transcribed from a
    paper. A transcription error in the 2x2 system would not crash and would not
    obviously corrupt audio -- it would quietly make one method look worse than
    another, which is exactly the kind of failure that gets written into a
    report as a finding. So the solves are checked against `lstsq`, and the
    orderings the report claims (AGA beats ITF on output error, GPTQ beats AGA)
    are asserted on data where the answer is known.
    """
    rng = np.random.default_rng(7)
    fails = []

    def check(name, ok, detail=""):
        print(f"  {'ok  ' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
        if not ok:
            fails.append(name)

    # 1. the per-row 2x2 solve == least squares on [t, 1]
    r, d = 8, 64
    w = rng.normal(size=(r, d)).astype(np.float32)
    t = rng.integers(-1, 2, size=(r, d)).astype(np.float32)
    a11 = np.sum(t * t, 1); a12 = np.sum(t, 1)
    a22 = np.full(r, float(d)); b1 = np.sum(w * t, 1); b2 = np.sum(w, 1)
    al, mu = _solve_2x2(a11, a12, a22, b1, b2)
    worst = 0.0
    for i in range(r):
        A = np.stack([t[i], np.ones(d)], 1)
        ref = np.linalg.lstsq(A, w[i], rcond=None)[0]
        worst = max(worst, abs(ref[0] - al[i]), abs(ref[1] - mu[i]))
    check("_solve_2x2 == lstsq", worst < 1e-4, f"max dev {worst:.2e}")

    # 2. the activation-aware solve == lstsq in the S metric
    x = rng.normal(size=(256, d)).astype(np.float32)
    S = (x.T @ x).astype(np.float32)
    al2, mu2 = align_grid(w, t.astype(np.int8), S)
    worst = 0.0
    for i in range(r):
        A = np.stack([t[i], np.ones(d)], 1)
        L = np.linalg.cholesky(S.astype(np.float64) + np.eye(d) * 1e-6)
        ref = np.linalg.lstsq(L.T @ A, L.T @ w[i], rcond=None)[0]
        worst = max(worst, abs(ref[0] - al2[i]), abs(ref[1] - mu2[i]))
    check("align_grid == lstsq in the S metric", worst < 1e-2, f"max dev {worst:.2e}")

    # 3. the orderings the report is built on
    w = rng.normal(size=(32, 128)).astype(np.float32)
    x = rng.normal(size=(512, 128)).astype(np.float32) * rng.uniform(0.2, 3.0, 128)
    S = (x.T @ x).astype(np.float32)
    e = {}
    for m in ("naive", "itf", "aga", "gptq"):
        q = METHODS[m](w, S, iters=10) if m in ("itf", "aga", "gptq") \
            else METHODS[m](w, S)
        e[m] = output_metrics(w, q.dequant(), x)["out_nmse"]
    check("ITF <= naive on weight error",
          weight_metrics(w, quant_itf(w).dequant())["nmse"]
          <= weight_metrics(w, quant_naive(w).dequant())["nmse"] + 1e-9)
    check("AGA < ITF on output error", e["aga"] < e["itf"],
          f"{e['aga']:.4f} vs {e['itf']:.4f}")
    check("GPTQ < AGA on output error", e["gptq"] < e["aga"],
          f"{e['gptq']:.4f} vs {e['aga']:.4f}")

    # 4. canonical round-trip for every role, including the transposed conv
    #    whose axis order is the one thing easy to get wrong
    for role, shape in (("linear", (12, 20)), ("conv1d", (8, 6, 3)),
                        ("convtr1d", (6, 8, 4))):
        info = TensorInfo(name="t", component="x", group="x", role=role,
                          shape=shape, numel=int(np.prod(shape)), dtype="F32",
                          streaming=True)
        _set_canonical(info)
        a = rng.normal(size=shape).astype(np.float32)
        m = canonical_matrix(a, info)
        check(f"canonical round-trip {role} {shape} -> {m.shape}",
              m.shape == (info.rows, info.contract)
              and np.array_equal(uncanonical(m, info), a))

    # 5. storage accounting is bytes, not a nominal bit count
    b = ternary_bytes(1024, 1024, shift=True)
    check("ternary_bytes counts the scale and the shift",
          b == (1024 * 1024 + 3) // 4 + 1024 * 2 + 1024 * 2,
          f"{8 * b / (1024 * 1024):.3f} bits/weight, not 1.58")

    # 6. gain correction actually makes the gain one
    q = quant_gptq(w, S)
    g = gain_correct(q, w, x)
    after = float(np.linalg.norm(x @ q.dequant().T) / np.linalg.norm(x @ w.T))
    check("gain_correct -> unit gain", abs(after - 1.0) < 1e-3,
          f"{g:.4f} -> {after:.4f}")

    print(f"\n{'FAILED: ' + ', '.join(fails) if fails else 'all checks passed'}")
    return 1 if fails else 0

# --------------------------------------------------------------------------
# listening set: the rule that actually decides
# --------------------------------------------------------------------------


def cmd_listen(args) -> int:
    """Render the same sentences under several configurations, for a human.

    Deliberately at the **shipped** temperature, not the measurement one. Every
    number in this tool was taken at temp 0 so that FP-vs-FP is bit identical
    and differences are attributable; none of that is how the model is served.
    The last gate is someone listening to audio produced the way production
    produces it, and a metric that disagrees with that loses.
    """
    pack = Pack(args.model)
    inv = inventory(pack)
    inv_by_name = {t.name: t for t in inv}
    out = Path(args.out)
    sweep = json.loads((out / "sweep.json").read_text())

    configs = []
    for spec in args.config:
        parts = spec.split(":")
        if len(parts) == 2:
            name, method, pats = parts[0], parts[1], None
        elif len(parts) == 3:
            name, method, pats = parts[0], parts[1], parts[2]
        else:
            raise SystemExit(f"--config wants NAME:METHOD[:PATTERNS], got {spec}")
        configs.append((name, method, pats))

    model = _load_upstream(args.language, args.seed, args.temperature)
    voice_state = model.get_state_for_audio_prompt(args.voice)
    sr = int(model.sample_rate)
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    texts = EVAL_TEXTS[:args.examples]

    lines = [f"PocketTTS ternary A/B — {time.strftime('%Y-%m-%d')}",
             f"voice={args.voice} language={args.language} seed={args.seed} "
             f"temperature={args.temperature} (the shipped value)",
             "",
             "Configurations:"]
    refs = []
    print(f"reference (fp32, temp {args.temperature})")
    for i, (kind, text) in enumerate(texts):
        a = _generate(model, voice_state, text, args.seed + i)
        refs.append(a)
        _write_wav(dest / f"00_reference_{i:02d}_{kind}.wav", sr, a)
    lines.append("  00_reference      the checkpoint as shipped, no quantization")

    summary = []
    for ci, (name, method, pats) in enumerate(configs, start=1):
        order = sensitivity_order(sweep, method)
        if pats:
            want = [x for x in pats.split("|") if x]
            order = [r for r in order if any(x in r["name"] for x in want)]
        names = [r["name"] for r in order]
        restore, applied = _apply_fake_quant(
            model, pack, inv_by_name, names, method, out, args.iters)
        nparams = sum(inv_by_name[n].numel for n in applied)
        per = []
        for i, (kind, text) in enumerate(texts):
            a = _generate(model, voice_state, text, args.seed + i)
            per.append(_audio_metrics(refs[i], a, sr))
            _write_wav(dest / f"{ci:02d}_{name}_{i:02d}_{kind}.wav", sr, a)
        restore()
        bad = sum(1 for q in per if q["collapsed"])
        summary.append((name, method, len(applied), nparams, bad, per))
        lines.append(f"  {ci:02d}_{name:<14} {method}, {len(applied)} layers, "
                     f"{nparams / 1e6:.1f}M params"
                     + (f", {bad}/{len(per)} failed to terminate" if bad else ""))
        print(f"  {ci:02d} {name:<16} {len(applied):>3} layers "
              f"{nparams / 1e6:5.1f}M  {bad}/{len(per)} non-terminating")

    lines += ["", "Sentences (index -> text):", ""]
    for i, (kind, text) in enumerate(texts):
        lines.append(f"  {i:02d} [{kind}]  {text}")
    lines += ["", "How to listen:", "",
              "  afplay 00_reference_04_long.wav",
              "  afplay 01_*_04_long.wav",
              "",
              "The long sentences are the ones that matter: every configuration",
              "handles a short sentence. Termination is what separates them.", ""]
    (dest / "FRASI.txt").write_text("\n".join(lines))
    print(f"\nwrote {len(configs) + 1} configurations x {len(texts)} utterances "
          f"to {dest}")
    return 0

# --------------------------------------------------------------------------
# phase 4: the protected set and the mixed-precision layout
# --------------------------------------------------------------------------


# Cheapest first. A tensor takes the first representation whose measured output
# error is inside the budget; "fp16" is the escape hatch and is what "protected"
# means in the report.
PRECISION_LADDER = [("ternary", 2.0), ("int4", 4.0), ("int8", 8.0), ("fp16", 16.0)]


def build_layout(sweep: dict, budget: float, *, ternary_method: str,
                 total_params: int) -> dict:
    """Assign each eligible tensor the cheapest precision under an error budget.

    The budget is on **output** NMSE against real activations, per tensor. That
    is a local criterion and it is not the same as end-to-end quality -- errors
    compose, and the flow head feeds a sampler. So a layout produced here is a
    hypothesis, and it is only worth anything once `cumulative --layout` has run
    it end to end. Saying that here rather than in a footnote because the
    temptation to quote the bit budget without the audio is the whole trap.
    """
    assign, protected, per_tier = {}, [], {k: 0 for k, _ in PRECISION_LADDER}
    tern_bytes = 0
    for L in sweep["layers"]:
        cand = None
        for tier, bits in PRECISION_LADDER:
            if tier == "fp16":
                cand = ("fp16", 16.0, L["numel"] * 2, None)
                break
            m = L["methods"].get(ternary_method if tier == "ternary" else tier)
            if m is None or "error" in m:
                continue
            err = m.get("out_nmse")
            if err is None or err > budget:
                continue
            cand = (tier, bits, m["bytes_ternary"],
                    ternary_method if tier == "ternary" else tier)
            break
        tier, bits, nbytes, method = cand
        assign[L["name"]] = {"tier": tier, "method": method, "bytes": nbytes,
                             "out_nmse": (L["methods"].get(method) or {}).get("out_nmse"),
                             "numel": L["numel"], "streaming": L["streaming"],
                             "component": L["component"], "group": L["group"]}
        per_tier[tier] += L["numel"]
        tern_bytes += nbytes
        if tier != "ternary":
            protected.append({"name": L["name"], "tier": tier,
                              "component": L["component"], "group": L["group"],
                              "numel": L["numel"],
                              "ternary_out_nmse": (L["methods"].get(ternary_method)
                                                   or {}).get("out_nmse")})
    covered = sum(a["numel"] for a in assign.values())
    rest_bytes = (total_params - covered) * 2          # everything unswept stays fp16
    total_bytes = tern_bytes + rest_bytes
    # Two budgets, because there are two questions. Checkpoint size is about
    # every weight on disk, including the Mimi encoder that only voice cloning
    # ever reads. The bandwidth argument in `.work/backbone-bandwidth.md` is
    # about the weights an AR step actually pulls through the cache and nothing
    # else. Quoting one as the other is how a 3.8 bits/weight headline gets
    # attached to a region that is really at 2.4.
    stream_params = sum(a["numel"] for a in assign.values() if a["streaming"])
    stream_bytes = sum(a["bytes"] for a in assign.values() if a["streaming"])
    stream_tern = sum(a["numel"] for a in assign.values()
                      if a["streaming"] and a["tier"] == "ternary")
    return {
        "streaming_params": stream_params,
        "streaming_bytes": stream_bytes,
        "streaming_mb": stream_bytes / 1e6,
        "streaming_pct_ternary": 100.0 * stream_tern / max(stream_params, 1),
        "effective_bits_streaming": 8.0 * stream_bytes / max(stream_params, 1),
        "uncalibrated_fp16_params": sum(
            a["numel"] for a in assign.values()
            if a["tier"] == "fp16" and a["out_nmse"] is None),
        "budget_out_nmse": budget,
        "ternary_method": ternary_method,
        "params_by_tier": per_tier,
        "unswept_params_fp16": total_params - covered,
        "pct_ternary": 100.0 * per_tier["ternary"] / total_params,
        "checkpoint_bytes": total_bytes,
        "checkpoint_mb": total_bytes / 1e6,
        "effective_bits_per_weight": 8.0 * total_bytes / total_params,
        "protected": protected,
        "assign": assign,
    }


def cmd_layout(args) -> int:
    pack = Pack(args.model)
    inv = inventory(pack)
    total = sum(t.numel for t in inv)
    out = Path(args.out)
    sweep = json.loads((out / "sweep.json").read_text())
    budgets = [float(b) for b in args.budgets.split(",")]
    layouts = [build_layout(sweep, b, ternary_method=args.method,
                            total_params=total) for b in budgets]
    _write_json(out / "layout.json", {
        "generator": "tools/ternary_feasibility.py layout",
        "total_params": total, "fp16_baseline_mb": total * 2 / 1e6,
        "layouts": layouts,
    })
    print(f"fp16 baseline: {total * 2 / 1e6:.1f} MB ({total:,} params)\n")
    print(f"{'budget':>8}{'%tern':>7}{'%int4':>7}{'%int8':>7}{'%fp16':>7}"
          f"{'bits/w':>8}{'ckpt MB':>9}  ||{'strm bits/w':>13}{'strm MB':>9}"
          f"{'%tern':>7}  protected (top)")
    for L in layouts:
        t = L["params_by_tier"]
        top = ", ".join(q["group"] for q in L["protected"][:3]) or "-"
        print(f"{L['budget_out_nmse']:>8.3f}"
              f"{100.0 * t['ternary'] / total:>7.1f}"
              f"{100.0 * t['int4'] / total:>7.1f}"
              f"{100.0 * t['int8'] / total:>7.1f}"
              f"{100.0 * (t['fp16'] + L['unswept_params_fp16']) / total:>7.1f}"
              f"{L['effective_bits_per_weight']:>8.2f}"
              f"{L['checkpoint_mb']:>9.1f}  ||"
              f"{L['effective_bits_streaming']:>13.2f}"
              f"{L['streaming_mb']:>9.1f}"
              f"{L['streaming_pct_ternary']:>7.1f}  {top}")
    print(f"\nwrote {out / 'layout.json'}")
    return 0

# --------------------------------------------------------------------------
# phase 5: what a future C kernel would actually have to do
# --------------------------------------------------------------------------


ISA_NOTES = {
    "gemv_tall": "NEON SDOT / x86 AVX2 VPMADDUBSW; VNNI VPDPBUSD where present. "
                 "No outer-product unit helps at B=1: one activation vector.",
    "gemm_batched": "ARM I8MM SMMLA (2x2 int8 outer product) or AVX-512 VNNI; "
                    "AMX INT8 only once B*rows justifies a tile reload.",
    "conv_im2col": "im2col into the GEMV/GEMM path; kernel width folds into K.",
    "depthwise": "no shared contraction: bandwidth-bound, ternary buys packing "
                 "but no MAC width.",
}


def cmd_shapes(args) -> int:
    pack = Pack(args.model)
    inv = inventory(pack)
    rates = frame_rates(pack.config)
    out = Path(args.out)
    sweep = None
    if (out / "sweep.json").exists():
        sweep = {L["name"]: L for L in json.loads((out / "sweep.json").read_text())["layers"]}

    tern_total = sum(ternary_bytes(t.rows, t.contract, shift=True)
                     for t in inv if t.eligible and t.streaming)
    rows = []
    for t in inv:
        if not t.eligible or not t.streaming:
            continue
        calls_s = module_rate(t, rates)
        calls_f = calls_s / rates["frame"] if rates["frame"] else 0.0
        tb = ternary_bytes(t.rows, t.contract, shift=True)
        if t.role == "linear":
            op = "gemv_tall"
        elif t.role == "conv1d":
            op = "conv_im2col"
        else:
            op = "depthwise" if t.shape[1] == 1 else "conv_im2col"
        macs_frame = t.rows * t.contract * calls_f
        entry = {
            "name": t.name, "component": t.component, "group": t.group,
            "M_rows": t.rows, "K_contract": t.contract,
            "shape": list(t.shape), "role": t.role,
            "calls_per_frame": calls_f, "calls_per_second": calls_s,
            "macs_per_frame": macs_frame,
            "bytes_ternary": tb,
            "pct_ternary_bytes": 100.0 * tb / tern_total if tern_total else 0.0,
            "weight_traffic_bytes_per_frame": tb * (1 if calls_f >= 1 else calls_f),
            "op": op, "isa": ISA_NOTES[op],
        }
        if sweep and t.name in sweep:
            m = sweep[t.name]["methods"]
            entry["best_method"] = min(
                ((k, v.get("out_nmse", v.get("nmse"))) for k, v in m.items()
                 if "error" not in v), key=lambda kv: kv[1], default=(None, None))[0]
        rows.append(entry)

    rows.sort(key=lambda r: -r["macs_per_frame"])
    total_macs = sum(r["macs_per_frame"] for r in rows)
    for r in rows:
        r["pct_macs"] = 100.0 * r["macs_per_frame"] / total_macs if total_macs else 0.0

    # group the per-tensor rows into the shapes a kernel author would actually
    # special-case: identical (M, K, op) repeated across layers is ONE kernel.
    shapes: dict[tuple, dict] = {}
    for r in rows:
        key = (r["M_rows"], r["K_contract"], r["op"])
        s = shapes.setdefault(key, {
            "M": r["M_rows"], "K": r["K_contract"], "op": r["op"],
            "instances": 0, "macs_per_frame": 0.0, "bytes_ternary": 0,
            "calls_per_frame": 0.0, "examples": [], "isa": r["isa"]})
        s["instances"] += 1
        s["macs_per_frame"] += r["macs_per_frame"]
        s["bytes_ternary"] += r["bytes_ternary"]
        s["calls_per_frame"] += r["calls_per_frame"]
        if len(s["examples"]) < 2:
            s["examples"].append(r["name"])
    ranked = sorted(shapes.values(), key=lambda s: -s["macs_per_frame"])
    for s in ranked:
        s["pct_macs"] = 100.0 * s["macs_per_frame"] / total_macs if total_macs else 0.0
        s["pct_bytes"] = 100.0 * s["bytes_ternary"] / tern_total if tern_total else 0.0

    _write_json(out / "shapes.json", {
        "generator": "tools/ternary_feasibility.py shapes",
        "frame_rates_hz": rates,
        "streaming_ternary_bytes": tern_total,
        "total_macs_per_frame": total_macs,
        "tensors": rows, "shapes": ranked,
    })
    print(f"streaming ternarizable weight bytes: {tern_total / 1e6:.2f} MB")
    print(f"{'M':>6}{'K':>7}  {'op':<14}{'inst':>5}{'calls/f':>9}"
          f"{'MMAC/f':>9}{'% MAC':>7}{'% bytes':>9}  example")
    for s in ranked[:14]:
        print(f"{s['M']:>6}{s['K']:>7}  {s['op']:<14}{s['instances']:>5}"
              f"{s['calls_per_frame']:>9.1f}{s['macs_per_frame'] / 1e6:>9.2f}"
              f"{s['pct_macs']:>7.1f}{s['pct_bytes']:>9.1f}  {s['examples'][0]}")
    print(f"\nwrote {out / 'shapes.json'}")
    return 0


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def cmd_report(args) -> int:
    out = Path(args.out)
    census = json.loads((out / "census.json").read_text())
    sweep = json.loads((out / "sweep.json").read_text()) if (out / "sweep.json").exists() else None
    shapes = json.loads((out / "shapes.json").read_text()) if (out / "shapes.json").exists() else None
    cumul = {}
    for p in sorted(out.glob("cumulative-*.json")):
        cumul[p.stem.split("-", 1)[1]] = json.loads(p.read_text())

    doc = [f"<!-- generated by tools/ternary_feasibility.py report on "
           f"{time.strftime('%Y-%m-%d')}; do not hand-edit the tables -->"]
    doc.append("\n## A. Parameter census\n")
    doc.append(f"`{census['source']}` rev `{census['revision']}`, "
               f"{census['total_params']:,} parameters in "
               f"{census['total_tensors']} tensors, {census['dtype_on_disk']} on disk.\n")
    rows = []
    tot = census["total_params"]
    for c, v in sorted(census["by_component"].items(), key=lambda kv: -kv[1]["params"]):
        rows.append([c, v["tensors"], f"{v['params']:,}",
                     f"{100.0 * v['params'] / tot:.1f}%",
                     f"{v['eligible_params']:,}",
                     "yes" if v["streaming_params"] else "no"])
    doc.append(_md_table(
        ["component", "tensors", "params", "% of model", "ternary-eligible",
         "in streaming path"], rows))

    if sweep:
        doc.append("\n## B. Per-layer sensitivity\n")
        methods = sweep["methods"]
        groups: dict[str, dict] = {}
        for L in sweep["layers"]:
            key = f"{L['component']}/{L['group']}"
            g = groups.setdefault(key, {"n": 0, "params": 0,
                                        **{m: [] for m in methods}})
            g["n"] += 1
            g["params"] += L["numel"]
            for m in methods:
                v = L["methods"].get(m)
                if v and "error" not in v:
                    g[m].append(v.get("out_nmse", v.get("nmse")))
        rows = []
        for key, g in sorted(groups.items(), key=lambda kv: -kv[1]["params"]):
            cells = []
            for m in methods:
                cells.append(f"{np.mean(g[m]):.4f}" if g[m] else "-")
            rows.append([key, g["n"], f"{g['params']:,}"] + cells)
        doc.append(_md_table(["group", "tensors", "params"] +
                             [f"{m} NMSE" for m in methods], rows))

    for method, c in cumul.items():
        doc.append(f"\n## C. Cumulative ternarization — {method}\n")
        rows = []
        for p in c["points"]:
            rows.append([f"{int(p['fraction'] * 100)}%", p["layers"],
                         f"{p['ternary_params'] / 1e6:.1f}M",
                         f"{p['pct_of_model']:.1f}%",
                         f"{p['effective_bits_per_weight']:.2f}",
                         f"{p['checkpoint_mb']:.1f}",
                         f"{p['mel_corr_mean']:.4f}",
                         f"{p['mel_corr_min']:.4f}",
                         f"{p['duration_ratio_mean']:.2f}",
                         p["collapsed"]])
        doc.append(_md_table(
            ["budget", "layers", "ternary params", "% of model",
             "eff. bits/weight", "ckpt MB", "mel corr mean", "mel corr min",
             "dur ratio", "collapsed"], rows))

    if shapes:
        doc.append("\n## D. Kernel shapes that would dominate a C backend\n")
        rows = []
        for s in shapes["shapes"][:10]:
            rows.append([f"{s['M']}x{s['K']}", s["op"], s["instances"],
                         f"{s['calls_per_frame']:.1f}",
                         f"{s['macs_per_frame'] / 1e6:.2f}",
                         f"{s['pct_macs']:.1f}%", f"{s['pct_bytes']:.1f}%",
                         s["examples"][0]])
        doc.append(_md_table(
            ["M x K", "op", "instances", "calls/frame", "MMAC/frame",
             "% MACs", "% ternary bytes", "example"], rows))

    path = Path(args.report)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_text() if path.exists() else ""
    marker = "<!-- BEGIN GENERATED TABLES -->"
    body = marker + "\n" + "\n".join(doc) + "\n<!-- END GENERATED TABLES -->"
    if marker in existing:
        head = existing.split(marker)[0]
        tail = existing.split("<!-- END GENERATED TABLES -->")[-1]
        path.write_text(head + body + tail)
    else:
        path.write_text(existing + ("\n" if existing else "") + body + "\n")
    print(f"wrote tables into {path}")
    return 0


# --------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=1234)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("census", help="phase 0: parameter census")
    p.add_argument("--checksum", action="store_true")
    p.set_defaults(fn=cmd_census)

    p = sub.add_parser("calib", help="capture calibration activations (needs pocket-tts)")
    p.add_argument("--language", default="english")
    p.add_argument("--voice", default="alba")
    p.add_argument("--examples", type=int, default=6)
    p.add_argument("--reservoir", type=int, default=512)
    p.add_argument("--temperature", type=float, default=None,
                   help="pin the sampler temperature; default is the model's own")
    p.set_defaults(fn=cmd_calib)

    p = sub.add_parser("sweep", help="phase 2: per-layer sensitivity")
    p.add_argument("--methods", default="naive,itf,aga,gptq")
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--only", default=None, help="comma-separated substrings")
    p.add_argument("--max-params", type=int, default=0)
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_sweep)

    p = sub.add_parser("cumulative", help="phase 3: progressive ternarization")
    p.add_argument("--method", default="gptq")
    p.add_argument("--language", default="english")
    p.add_argument("--voice", default="alba")
    p.add_argument("--fractions", default="0.1,0.25,0.5,0.7,0.8,0.9,1.0")
    p.add_argument("--examples", type=int, default=5)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--streaming-only", action="store_true")
    p.add_argument("--include", default=None,
                   help="restrict to layers whose name contains one of these "
                        "substrings; with --fractions 1.0 this probes one group")
    p.add_argument("--exclude", default=None)
    p.add_argument("--temperature", type=float, default=0.0,
                   help="0 (default) makes FP-vs-FP bit identical, so the metric "
                        "measures the weights and not the sampler")
    p.add_argument("--layout-budgets", default=None,
                   help="evaluate mixed-precision layouts at these per-tensor "
                        "output-NMSE budgets instead of ternary-only prefixes")
    p.add_argument("--no-save-audio", dest="save_audio", action="store_false",
                   default=True, help="skip the WAVs; keep only the metrics")
    p.set_defaults(fn=cmd_cumulative)

    p = sub.add_parser("layout", help="phase 4: protected set + mixed precision")
    p.add_argument("--method", default="gptq")
    p.add_argument("--budgets", default="0.001,0.005,0.01,0.02,0.05,0.1,0.2")
    p.set_defaults(fn=cmd_layout)

    p = sub.add_parser("listen", help="render an A/B set at the shipped temperature")
    p.add_argument("--config", action="append", default=[],
                   metavar="NAME:METHOD[:PAT|PAT]",
                   help="repeatable; PATTERNS restrict which layers are quantized")
    p.add_argument("--dest", type=Path, required=True)
    p.add_argument("--language", default="english")
    p.add_argument("--voice", default="alba")
    p.add_argument("--examples", type=int, default=7)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--temperature", type=float, default=0.3)
    p.set_defaults(fn=cmd_listen)

    p = sub.add_parser("self-test", help="check the quantizer math, no model needed")
    p.set_defaults(fn=cmd_selftest)

    p = sub.add_parser("shapes", help="phase 5: kernel readiness, no kernels")
    p.set_defaults(fn=cmd_shapes)

    p = sub.add_parser("report", help="render markdown tables")
    p.add_argument("--report", type=Path,
                   default=REPO / "docs" / "ternary-feasibility.md")
    p.set_defaults(fn=cmd_report)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
