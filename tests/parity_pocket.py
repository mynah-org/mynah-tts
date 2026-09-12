#!/usr/bin/env python3
"""Compare a PocketTTS dump produced by the C runtime against the Python oracle.

This is the other half of `tools/oracle_pocket.py`: the oracle says what the
reference implementation computes, this says whether `engine_pocket.c` computes
the same thing, stage by stage, with the per-stage tolerances written down in
`.work/pocket-tts-oracle.md`:

    1  tokenizer            token IDs                       exact
    2  LUT conditioner      text_embeddings                 abs 1e-5
    3  voice prefix         KV cache as loaded              exact
    4  backbone prefill     post-out_norm hidden            abs 1e-4 / rel 1e-3
    5  backbone step        hidden per AR step              abs 1e-4 / rel 1e-3
    6  EOS                  out_eos logit + crossing step   exact step index
    7  flow head            latent out, fixed noise         abs 1e-4
    8  latent denorm        emb_mean/emb_std applied        abs 1e-5
    9  Mimi input           output_proj + upsample          abs 1e-4
    10 decoder transformer  2-layer output                  abs 1e-4
    11 SEANet               per-stage conv output           abs 1e-3
    12 waveform             full WAV      duration exact, peak/RMS 1%, mel>0.99

Two design decisions worth stating, because both come straight from the failure
this epic exists to prevent (see "Bug found and fixed in the tool itself" in the
note — a dump that looked fine and was quietly missing 84 of 132 tensors):

  * A tensor present in one dump and not the other is a FAILURE, never a
    skipped line. An incomplete candidate must not be able to score 100%.
  * A candidate generated with different parameters (seed, temperature, decode
    steps, EOS threshold, sample rate, language, voice, text) is refused
    outright. Those runs are not comparable and a tolerance cannot make them so.

Exit codes are three-valued on purpose, so CI can tell "not written yet" from
"written and wrong":

    0   every compared stage is within tolerance
    1   parity failure, missing/extra tensor, or mismatched run parameters
    2   nothing to compare: the candidate (or the oracle) dump is not there

By default every stage is compared and the full table is printed before exiting
non-zero, because a table with all the failures in it is what you can act on;
`--fail-fast` stops at the first failing tensor instead.

Dependencies: numpy only. The mel correlation for stage 12 is a hand-written
STFT plus a triangular HTK mel filterbank, precisely so this test does not drag
in scipy/librosa to run in CI.

Usage:
    uv run --with numpy python tests/parity_pocket.py \
        --oracle build/oracle-pocket --candidate build/parity-pocket
    uv run --with numpy python tests/parity_pocket.py --stage seanet --verbose
    uv run --with numpy python tests/parity_pocket.py --json build/parity.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_ABSENT = 2

# Generation parameters that must match for a comparison to mean anything.
# Not negotiable and not tolerated: a diff between two different runs is noise
# with a plausible shape.
REQUIRED_PARAMS = (
    "language",
    "voice",
    "text",
    "seed",
    "temperature",
    "sampler_decode_steps",
    "eos_threshold",
    "sample_rate",
)


class Tol:
    """Per-stage tolerance. `exact` wins over everything else when set."""

    def __init__(self, abs_tol=None, rel_tol=None, exact=False, note=""):
        self.abs_tol = abs_tol
        self.rel_tol = rel_tol
        self.exact = exact
        self.note = note

    def describe(self) -> str:
        if self.exact:
            return "exact"
        parts = []
        if self.abs_tol is not None:
            parts.append(f"abs {self.abs_tol:g}")
        if self.rel_tol is not None:
            parts.append(f"rel {self.rel_tol:g}")
        return " / ".join(parts) if parts else "n/a"


# Stage table from `.work/pocket-tts-oracle.md`. Keys sort into report order.
STAGES: dict[str, Tol] = {
    "01_tokenizer": Tol(exact=True, note="token IDs"),
    "02_conditioner": Tol(abs_tol=1e-5, note="LUT text embeddings"),
    "03_voice_prefix": Tol(exact=True, note="voice KV cache as loaded"),
    "04_backbone_prefill": Tol(abs_tol=1e-4, rel_tol=1e-3, note="prefill hidden"),
    "05_backbone_step": Tol(abs_tol=1e-4, rel_tol=1e-3, note="AR step hidden"),
    "06_eos": Tol(abs_tol=1e-4, rel_tol=1e-3, note="logit; crossing step exact"),
    "07_flow_head": Tol(abs_tol=1e-4, note="one LSD step, fixed noise"),
    "08_latent_denorm": Tol(abs_tol=1e-5, note="emb_mean / emb_std"),
    "09_mimi_input": Tol(abs_tol=1e-4, note="output_proj + upsample"),
    "10_decoder_transformer": Tol(abs_tol=1e-4, note="2-layer, 16 pos/frame"),
    "11_seanet": Tol(abs_tol=1e-3, note="causal conv stack"),
    "12_waveform": Tol(note="duration exact, peak/RMS 1%, mel corr > 0.99"),
    "99_unclassified": Tol(abs_tol=1e-4, rel_tol=1e-3, note="not in the stage table"),
}

# Waveform-specific thresholds (stage 12).
WAVE_LEVEL_TOL = 0.01  # peak and RMS, relative
WAVE_MEL_CORR_MIN = 0.99

# Filename -> stage. Ordered, first match wins; `call` disambiguates 4 vs 5.
CALL_RE = re.compile(r"\.call(\d+)\.npy$")


def classify(fname: str, call: int | None) -> str:
    """Map a dumped `.npy` name to its oracle stage."""
    n = fname
    if n.startswith("stage01") or "token_ids" in n:
        return "01_tokenizer"
    if n.startswith("stage12") or "waveform" in n:
        return "12_waveform"
    if n.startswith("stage03") or n.startswith("voice"):
        return "03_voice_prefix"
    if n.startswith("stage08") or "denorm" in n:
        return "08_latent_denorm"
    if n.startswith("flow_lm.conditioner"):
        return "02_conditioner"
    if n.startswith("flow_lm.out_eos"):
        return "06_eos"
    if n.startswith("flow_lm.flow_net"):
        return "07_flow_head"
    if n.startswith(("flow_lm.input_linear", "flow_lm.transformer", "flow_lm.out_norm")):
        # The prefill is call 0 (text + voice prefix); every later call is one
        # AR step. They have different tolerances in spirit even though the
        # numbers coincide, and separating them makes the table readable.
        return "04_backbone_prefill" if call == 0 else "05_backbone_step"
    if n.startswith("mimi.decoder_transformer"):
        return "10_decoder_transformer"
    if n.startswith(("mimi.quantizer", "mimi.upsample")):
        return "09_mimi_input"
    if n.startswith("mimi.decoder") or "seanet" in n:
        return "11_seanet"
    return "99_unclassified"


# --------------------------------------------------------------------------
# numeric helpers
# --------------------------------------------------------------------------


def metrics(a: np.ndarray, b: np.ndarray) -> dict:
    """Absolute / relative error and correlation between two same-shape arrays.

    `rel` is a relative RMS (‖a-b‖ / ‖b‖), the same quantity
    `tools/oracle_pocket_stream.py` reports, not an elementwise ratio: an
    elementwise ratio explodes on the near-zero entries every hidden state has.
    """
    af = np.asarray(a, dtype=np.float64).ravel()
    bf = np.asarray(b, dtype=np.float64).ravel()
    out = {
        "size": int(bf.size),
        "finite_candidate": bool(np.isfinite(af).all()) if af.size else True,
        "finite_oracle": bool(np.isfinite(bf).all()) if bf.size else True,
    }
    if bf.size == 0:
        out.update(max_abs=0.0, rel=0.0, corr=1.0, n_diff=0, worst_index=None)
        return out
    diff = af - bf
    finite = np.isfinite(diff)
    max_abs = float(np.max(np.abs(diff[finite]))) if finite.any() else float("inf")
    denom = float(np.sqrt(np.sum(bf[np.isfinite(bf)] ** 2)))
    num = float(np.sqrt(np.sum(diff[finite] ** 2))) if finite.any() else float("inf")
    rel = num / denom if denom > 0 else (0.0 if num == 0 else float("inf"))
    out["max_abs"] = max_abs
    out["rel"] = rel
    out["n_diff"] = int(np.count_nonzero(diff != 0))
    idx = int(np.argmax(np.abs(np.where(finite, diff, 0.0)))) if finite.any() else 0
    out["worst_index"] = idx
    out["worst_oracle"] = float(bf[idx])
    out["worst_candidate"] = float(af[idx])
    out["corr"] = pearson(af, bf)
    return out


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation, with the degenerate cases spelled out."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 2:
        return 1.0 if np.array_equal(a, b) else float("nan")
    a, b = a[m], b[m]
    a = a - a.mean()
    b = b - b.mean()
    na, nb = float(np.sqrt(a @ a)), float(np.sqrt(b @ b))
    if na == 0.0 and nb == 0.0:
        return 1.0  # two constants; the abs check already decided if they match
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return float((a @ b) / (na * nb))


def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
    """Triangular HTK-style mel filterbank, [n_mels, n_fft//2+1]."""
    edges = mel_to_hz(np.linspace(hz_to_mel(0.0), hz_to_mel(sr / 2.0), n_mels + 2))
    freqs = np.linspace(0.0, sr / 2.0, n_fft // 2 + 1)
    fb = np.zeros((n_mels, freqs.size), dtype=np.float64)
    for i in range(n_mels):
        lo, ctr, hi = edges[i], edges[i + 1], edges[i + 2]
        left = (freqs - lo) / max(ctr - lo, 1e-9)
        right = (hi - freqs) / max(hi - ctr, 1e-9)
        fb[i] = np.clip(np.minimum(left, right), 0.0, None)
    return fb


def log_mel(x: np.ndarray, sr: int, n_fft: int = 1024, hop: int = 256,
            n_mels: int = 80) -> np.ndarray:
    """Hand-rolled log-mel spectrogram (numpy only, no scipy/librosa)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size < n_fft:
        x = np.pad(x, (0, n_fft - x.size))
    win = np.hanning(n_fft + 1)[:n_fft]  # periodic Hann
    frames = np.lib.stride_tricks.sliding_window_view(x, n_fft)[::hop]
    spec = np.abs(np.fft.rfft(frames * win, axis=-1)) ** 2
    mel = spec @ mel_filterbank(sr, n_fft, n_mels).T
    return np.log10(np.maximum(mel, 1e-10))


# --------------------------------------------------------------------------
# dump loading
# --------------------------------------------------------------------------


class Dump:
    def __init__(self, root: Path, label: str):
        self.root = root
        self.label = label
        self.manifest: dict = {}
        self.files: dict[str, Path] = {}
        self.integrity: list[str] = []

    @property
    def exists(self) -> bool:
        return self.root.is_dir() and (self.root / "manifest.json").is_file()

    def load(self) -> None:
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        self.files = {p.name: p for p in sorted(self.root.glob("*.npy"))}
        # A manifest that advertises a tensor it did not write is exactly the
        # "looks complete, is not" failure mode. Catch it here, not by luck.
        for entry in self.manifest.get("tensors", []):
            name = entry.get("file")
            if name and name not in self.files:
                self.integrity.append(
                    f"{self.label}: manifest lists {name} but the file is missing"
                )

    def array(self, name: str) -> np.ndarray:
        return np.load(self.files[name], allow_pickle=False)


# --------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------


def check_params(oracle: Dump, cand: Dump) -> list[str]:
    """Refuse to compare two runs that were not generated the same way."""
    problems = []
    for key in REQUIRED_PARAMS:
        if key not in oracle.manifest:
            problems.append(f"oracle manifest has no '{key}' (regenerate the oracle)")
            continue
        if key not in cand.manifest:
            problems.append(
                f"candidate manifest has no '{key}'; it must record the same "
                f"generation parameters as the oracle (oracle {key}="
                f"{oracle.manifest[key]!r})"
            )
            continue
        o, c = oracle.manifest[key], cand.manifest[key]
        same = (
            math.isclose(float(o), float(c), rel_tol=0.0, abs_tol=1e-9)
            if isinstance(o, (int, float)) and isinstance(c, (int, float))
            and not isinstance(o, bool) and not isinstance(c, bool)
            else o == c
        )
        if not same:
            problems.append(f"{key}: oracle={o!r} candidate={c!r}")
    return problems


def check_voice_shapes(oracle: Dump, cand: Dump) -> dict | None:
    """Stage 3 without dedicated tensors: the KV cache shapes must agree.

    The voice *is* the prefix cache; if the C loader reshaped it, nothing
    downstream is comparable, and silence here would hide that.
    """
    o = oracle.manifest.get("voice_state_shapes")
    if o is None:
        return None
    c = cand.manifest.get("voice_state_shapes")
    if c is None:
        return {
            "name": "voice_state_shapes (manifest)",
            "stage": "03_voice_prefix",
            "ok": False,
            "detail": "candidate manifest has no 'voice_state_shapes'; the C "
                      "dump must record the loaded voice KV cache shapes",
        }
    ok = o == c
    return {
        "name": "voice_state_shapes (manifest)",
        "stage": "03_voice_prefix",
        "ok": ok,
        "detail": "" if ok else f"oracle={json.dumps(o)} candidate={json.dumps(c)}",
    }


def check_token_ids(oracle: Dump, cand: Dump) -> dict | None:
    o = oracle.manifest.get("token_ids")
    if o is None:
        return None
    c = cand.manifest.get("token_ids")
    if c is None:
        return {
            "name": "token_ids (manifest)",
            "stage": "01_tokenizer",
            "ok": False,
            "detail": "candidate manifest has no 'token_ids'",
        }
    ok = list(o) == list(c)
    detail = ""
    if not ok:
        detail = f"len oracle={len(o)} candidate={len(c)}; " + first_int_diff(o, c)
    return {"name": "token_ids (manifest)", "stage": "01_tokenizer",
            "ok": ok, "detail": detail}


def first_int_diff(o, c) -> str:
    for i in range(min(len(o), len(c))):
        if o[i] != c[i]:
            return f"first difference at index {i}: oracle={o[i]} candidate={c[i]}"
    return "one is a prefix of the other"


def eos_step(dump: Dump) -> tuple[int | None, str]:
    """Step index where out_eos crosses the threshold, or None.

    Prefers an explicit `eos_step` in the manifest; otherwise derives it from
    the dumped `flow_lm.out_eos.out.callN` logits. The oracle only keeps the
    first few calls, so `None` legitimately means "no crossing in what was
    dumped" — which is still a value both sides must agree on.
    """
    if "eos_step" in dump.manifest:
        v = dump.manifest["eos_step"]
        return (None if v is None else int(v)), "manifest"
    thr = float(dump.manifest.get("eos_threshold", -4.0))
    steps = []
    for name in dump.files:
        m = CALL_RE.search(name)
        if m and name.startswith("flow_lm.out_eos.out"):
            steps.append((int(m.group(1)), name))
    for step, name in sorted(steps):
        if float(np.max(dump.array(name))) > thr:
            return step, f"derived from {len(steps)} dumped logits"
    return None, f"derived from {len(steps)} dumped logits (no crossing)"


def check_eos_step(oracle: Dump, cand: Dump) -> dict:
    o, osrc = eos_step(oracle)
    c, csrc = eos_step(cand)
    ok = o == c
    return {
        "name": "eos_step (exact)",
        "stage": "06_eos",
        "ok": ok,
        "detail": f"oracle={o} ({osrc}) candidate={c} ({csrc})"
        + ("" if ok else "  <-- must be the same step index"),
    }


def check_latent_denorm(oracle: Dump, cand: Dump) -> list[dict]:
    """Stage 8 is the delta between flow_net.out and mimi.quantizer.in0.

    The note says so explicitly: the denormalization has no module of its own,
    so the only way to see emb_mean/emb_std in isolation is the difference
    between two captured tensors. Comparing that difference catches wrong
    constants even when the upstream latent already drifted.
    """
    rows = []
    tol = STAGES["08_latent_denorm"]
    for name in sorted(oracle.files):
        m = CALL_RE.search(name)
        if not (m and name.startswith("flow_lm.flow_net.out.")):
            continue
        call = m.group(1)
        qname = f"mimi.quantizer.in0.call{call}.npy"
        needed = (name, qname)
        if not all(n in oracle.files and n in cand.files for n in needed):
            continue
        try:
            od = oracle.array(qname).ravel() - oracle.array(name).ravel()
            cd = cand.array(qname).ravel() - cand.array(name).ravel()
        except ValueError:
            continue
        if od.shape != cd.shape:
            rows.append({
                "name": f"latent_denorm delta call{call}",
                "stage": "08_latent_denorm", "ok": False,
                "detail": f"shape oracle={od.shape} candidate={cd.shape}",
            })
            continue
        mt = metrics(cd, od)
        ok = mt["max_abs"] <= tol.abs_tol
        rows.append({
            "name": f"latent_denorm delta call{call}",
            "stage": "08_latent_denorm",
            "ok": ok,
            "shape": list(od.shape),
            "max_abs": mt["max_abs"], "rel": mt["rel"], "corr": mt["corr"],
            "detail": "" if ok else
            f"max_abs={mt['max_abs']:.3e} > {tol.abs_tol:g} "
            f"(emb_mean/emb_std applied differently?)",
        })
    return rows


def check_waveform(oracle: Dump, cand: Dump, name: str) -> list[dict]:
    """Stage 12: duration exact, peak/RMS within 1%, mel correlation > 0.99."""
    sr = int(oracle.manifest.get("sample_rate", 24000))
    o = np.asarray(oracle.array(name), dtype=np.float64).ravel()
    c = np.asarray(cand.array(name), dtype=np.float64).ravel()
    rows = []

    dur_ok = o.size == c.size
    rows.append({
        "name": f"{name} duration", "stage": "12_waveform", "ok": dur_ok,
        "shape": [int(o.size)],
        "detail": f"oracle={o.size} samples ({o.size / sr:.3f}s) "
                  f"candidate={c.size} samples ({c.size / sr:.3f}s)"
                  + ("" if dur_ok else "  <-- duration must be exact"),
    })

    fin_ok = bool(np.isfinite(c).all())
    rows.append({
        "name": f"{name} finite", "stage": "12_waveform", "ok": fin_ok,
        "detail": "" if fin_ok else
        f"{int((~np.isfinite(c)).sum())} non-finite samples in the candidate",
    })

    def level(tag, fo, fc):
        ok = abs(fc - fo) <= WAVE_LEVEL_TOL * max(abs(fo), 1e-12)
        rows.append({
            "name": f"{name} {tag}", "stage": "12_waveform", "ok": ok,
            "max_abs": abs(fc - fo),
            "rel": abs(fc - fo) / max(abs(fo), 1e-12),
            "detail": f"oracle={fo:.6f} candidate={fc:.6f} "
                      f"({abs(fc - fo) / max(abs(fo), 1e-12) * 100:.3f}%"
                      f", tol {WAVE_LEVEL_TOL * 100:g}%)",
        })

    peak_o = float(np.max(np.abs(o))) if o.size else 0.0
    peak_c = float(np.max(np.abs(c))) if c.size else 0.0
    level("peak", peak_o, peak_c)
    rms_o = float(np.sqrt(np.mean(o ** 2))) if o.size else 0.0
    rms_c = float(np.sqrt(np.mean(c ** 2))) if c.size else 0.0
    level("rms", rms_o, rms_c)

    n = min(o.size, c.size)
    if n == 0 or not fin_ok:
        corr = float("nan")
    else:
        corr = pearson(log_mel(c[:n], sr), log_mel(o[:n], sr))
    corr_ok = bool(np.isfinite(corr)) and corr > WAVE_MEL_CORR_MIN
    rows.append({
        "name": f"{name} mel corr", "stage": "12_waveform", "ok": corr_ok,
        "corr": None if not np.isfinite(corr) else corr,
        "detail": f"log-mel corr={corr:.6f} (80 mels, n_fft 1024, hop 256, "
                  f"{sr} Hz), threshold > {WAVE_MEL_CORR_MIN}"
                  + ("" if n == o.size == c.size else
                     f"  [computed on the first {n} samples]"),
    })

    if n and fin_ok:
        mt = metrics(c[:n], o[:n])
        rows.append({
            "name": f"{name} samples", "stage": "12_waveform", "ok": True,
            "shape": [n], "max_abs": mt["max_abs"], "rel": mt["rel"],
            "corr": mt["corr"],
            "detail": "informational: the sample-domain delta carries no "
                      "tolerance of its own (different FP orderings are allowed)",
        })
    return rows


def compare_tensor(oracle: Dump, cand: Dump, name: str) -> dict:
    call_m = CALL_RE.search(name)
    call = int(call_m.group(1)) if call_m else None
    stage = classify(name, call)
    tol = STAGES[stage]
    row = {"name": name, "stage": stage, "call": call, "tolerance": tol.describe()}

    o = oracle.array(name)
    c = cand.array(name)
    row["shape"] = list(o.shape)
    if tuple(o.shape) != tuple(c.shape):
        row["ok"] = False
        row["detail"] = f"shape oracle={list(o.shape)} candidate={list(c.shape)}"
        return row
    if str(o.dtype) != str(c.dtype):
        row["dtype_note"] = f"oracle={o.dtype} candidate={c.dtype}"

    mt = metrics(c, o)
    row.update({k: mt[k] for k in ("max_abs", "rel", "corr", "n_diff", "size")})
    row["finite"] = mt["finite_candidate"]

    if not mt["finite_candidate"]:
        row["ok"] = False
        row["detail"] = "candidate contains non-finite values"
        return row

    if tol.exact:
        ok = bool(np.array_equal(np.asarray(o), np.asarray(c)))
        row["ok"] = ok
        if not ok:
            row["detail"] = (
                f"{mt['n_diff']}/{mt['size']} elements differ; worst at flat index "
                f"{mt['worst_index']}: oracle={mt['worst_oracle']!r} "
                f"candidate={mt['worst_candidate']!r}  <-- this stage is exact"
            )
        return row

    ok = False
    if tol.abs_tol is not None and mt["max_abs"] <= tol.abs_tol:
        ok = True
    if not ok and tol.rel_tol is not None and mt["rel"] <= tol.rel_tol:
        ok = True
    row["ok"] = ok
    if not ok:
        row["detail"] = (
            f"max_abs={mt['max_abs']:.3e} rel={mt['rel']:.3e} corr={mt['corr']:.6f} "
            f"(tolerance {tol.describe()}); worst at flat index {mt['worst_index']}: "
            f"oracle={mt['worst_oracle']:.9g} candidate={mt['worst_candidate']:.9g}"
        )
    return row


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------


def fmt(v, spec="{:.3e}"):
    if v is None:
        return "-"
    if isinstance(v, float) and not math.isfinite(v):
        return "nan" if math.isnan(v) else ("inf" if v > 0 else "-inf")
    return spec.format(v)


def print_table(rows: list[dict], verbose: bool) -> None:
    hdr = f"{'stage':<22} {'shape':<20} {'max_abs':>10} {'rel':>10} {'corr':>8}  verdict"
    print(hdr)
    print("-" * len(hdr))
    if verbose:
        for r in rows:
            print(f"{r['stage']:<22} {str(r.get('shape', '-')):<20} "
                  f"{fmt(r.get('max_abs')):>10} {fmt(r.get('rel')):>10} "
                  f"{fmt(r.get('corr'), '{:.5f}'):>8}  "
                  f"{'PASS' if r['ok'] else 'FAIL'}  {r['name']}")
        print("-" * len(hdr))

    for stage in sorted({r["stage"] for r in rows}):
        srows = [r for r in rows if r["stage"] == stage]
        bad = [r for r in srows if not r["ok"]]
        scored = [r for r in srows if isinstance(r.get("max_abs"), float)]
        # Prefer a row that also carries a shape, so the column is informative.
        worst = max((r for r in scored if r.get("shape")),
                    key=lambda r: r["max_abs"], default=None)
        if worst is None:
            worst = max(scored, key=lambda r: r["max_abs"], default=None)
        corrs = [r["corr"] for r in srows
                 if isinstance(r.get("corr"), float) and math.isfinite(r["corr"])]
        rels = [r["rel"] for r in srows
                if isinstance(r.get("rel"), float) and math.isfinite(r["rel"])]
        verdict = "PASS" if not bad else f"FAIL ({len(bad)}/{len(srows)})"
        shape = str(worst.get("shape", "-")) if worst else "-"
        print(f"{stage:<22} {shape:<20} "
              f"{fmt(worst['max_abs']) if worst else '-':>10} "
              f"{fmt(max(rels)) if rels else '-':>10} "
              f"{fmt(min(corrs), '{:.5f}') if corrs else '-':>8}  "
              f"{verdict:<12} {len(srows)} tensors, tol {STAGES[stage].describe()}")


def print_expectations(oracle: Dump, cand_root: Path) -> None:
    print(f"candidate dump not found: {cand_root}")
    print()
    print("Nothing was compared. This is exit 2, not a parity failure: the C")
    print("side has not written a dump here yet.")
    print()
    print("Expected in that directory:")
    print("  manifest.json   with at least these keys, identical to the oracle's:")
    for key in REQUIRED_PARAMS:
        val = oracle.manifest.get(key, "<oracle missing>")
        if isinstance(val, str) and len(val) > 46:
            val = val[:43] + "..."
        print(f"                    {key:<22} = {val!r}")
    print("                  plus 'token_ids' and 'voice_state_shapes', and")
    print("                  optionally 'eos_step' (else it is derived from the")
    print("                  dumped out_eos logits).")
    n = len(oracle.files)
    print(f"  {n} *.npy files, one per oracle tensor, SAME FILE NAMES, e.g.:")
    for name in list(oracle.files)[:6]:
        print(f"                    {name}")
    if n > 6:
        print(f"                    ... and {n - 6} more (see the oracle directory)")
    print()
    print("Every one of them is required: a tensor present in the oracle and")
    print("missing from the candidate is reported as a FAILURE, not skipped.")
    print()
    print("To produce the oracle side:")
    print("    make oracle-pocket")
    print("To self-check this tool (oracle against itself, must be all zero):")
    print(f"    uv run --with numpy python tests/parity_pocket.py "
          f"--candidate {oracle.root}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--oracle", type=Path, default=Path("build/oracle-pocket"),
                    help="reference dump from tools/oracle_pocket.py")
    ap.add_argument("--candidate", type=Path, default=Path("build/parity-pocket"),
                    help="dump produced by the C runtime")
    ap.add_argument("--stage", default=None,
                    help="only compare stages matching this (id, name or substring: "
                         "'7', 'stage07', 'flow', 'seanet', 'waveform')")
    ap.add_argument("--fail-fast", action="store_true",
                    help="stop at the first failing tensor instead of reporting all")
    ap.add_argument("--verbose", "-v", action="store_true",
                    help="one line per tensor, not just per stage")
    ap.add_argument("--json", dest="json_out", type=Path, default=None,
                    help="write the machine-readable report here")
    args = ap.parse_args()

    oracle = Dump(args.oracle, "oracle")
    cand = Dump(args.candidate, "candidate")

    if not oracle.exists:
        print(f"oracle dump not found: {args.oracle}", file=sys.stderr)
        print("run `make oracle-pocket` (or tools/oracle_pocket.py --out <dir>) first",
              file=sys.stderr)
        return EXIT_ABSENT
    oracle.load()

    if not cand.exists:
        print_expectations(oracle, args.candidate)
        return EXIT_ABSENT
    cand.load()

    report = {
        "oracle": str(args.oracle),
        "candidate": str(args.candidate),
        "stage_filter": args.stage,
        "params": {k: oracle.manifest.get(k) for k in REQUIRED_PARAMS},
        "tolerances": {k: v.describe() for k, v in STAGES.items()},
    }

    param_problems = check_params(oracle, cand)
    if param_problems:
        print("REFUSED: the two dumps were not generated the same way.")
        print("A per-stage diff between different runs is meaningless, so this is")
        print("rejected rather than tolerated.")
        for p in param_problems:
            print(f"  - {p}")
        report["ok"] = False
        report["param_problems"] = param_problems
        if args.json_out:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(json.dumps(report, indent=2))
        return EXIT_FAIL

    for msg in oracle.integrity + cand.integrity:
        print(f"integrity: {msg}")

    # Stage filter.
    def wanted(stage: str) -> bool:
        if not args.stage:
            return True
        q = args.stage.lower().strip()
        return q in stage.lower() or q.lstrip("0") == stage.split("_")[0].lstrip("0") \
            or q.replace("stage", "").lstrip("0") == stage.split("_")[0].lstrip("0")

    rows: list[dict] = []
    failed = False

    def add(row) -> bool:
        """Record a row; return True if we should stop (fail-fast)."""
        nonlocal failed
        if not wanted(row["stage"]):
            return False
        rows.append(row)
        if not row["ok"]:
            failed = True
            return args.fail_fast
        return False

    # Files present in one dump and not the other: a failure, by design.
    only_oracle = sorted(set(oracle.files) - set(cand.files))
    only_cand = sorted(set(cand.files) - set(oracle.files))
    stop = False
    for name in only_oracle:
        call_m = CALL_RE.search(name)
        stop = add({
            "name": name, "ok": False,
            "stage": classify(name, int(call_m.group(1)) if call_m else None),
            "detail": "missing from the candidate dump (incomplete dump)",
        }) or stop
    for name in only_cand:
        call_m = CALL_RE.search(name)
        stop = add({
            "name": name, "ok": False,
            "stage": classify(name, int(call_m.group(1)) if call_m else None),
            "detail": "present in the candidate but not in the oracle "
                      "(stale file, or the oracle needs regenerating)",
        }) or stop

    if not stop:
        for row in filter(None, (check_token_ids(oracle, cand),
                                 check_voice_shapes(oracle, cand),
                                 check_eos_step(oracle, cand))):
            if add(row):
                stop = True
                break

    if not stop:
        for name in sorted(set(oracle.files) & set(cand.files)):
            call_m = CALL_RE.search(name)
            stage = classify(name, int(call_m.group(1)) if call_m else None)
            if not wanted(stage):
                continue
            if stage == "12_waveform":
                for row in check_waveform(oracle, cand, name):
                    if add(row):
                        stop = True
                        break
            elif add(compare_tensor(oracle, cand, name)):
                stop = True
            if stop:
                break

    if not stop:
        for row in check_latent_denorm(oracle, cand):
            if add(row):
                break

    if oracle.integrity or cand.integrity:
        failed = True

    # ---- report -----------------------------------------------------------
    print()
    print(f"oracle    {args.oracle}   ({len(oracle.files)} tensors)")
    print(f"candidate {args.candidate}   ({len(cand.files)} tensors)")
    p = oracle.manifest
    print(f"run       {p.get('language')}/{p.get('voice')} seed={p.get('seed')} "
          f"temp={p.get('temperature')} steps={p.get('sampler_decode_steps')} "
          f"eos_thr={p.get('eos_threshold')} sr={p.get('sample_rate')}")
    if args.stage:
        print(f"filter    --stage {args.stage}")
    print()
    if not rows:
        print("nothing compared: the --stage filter matched no tensor")
        print("known stages: " + ", ".join(STAGES))
        report["ok"] = False
        report["rows"] = []
        if args.json_out:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(json.dumps(report, indent=2))
        return EXIT_FAIL

    print_table(rows, args.verbose)

    bad = [r for r in rows if not r["ok"]]
    print()
    if bad:
        print(f"FAILURES ({len(bad)} of {len(rows)} checks):")
        for r in bad:
            print(f"  [{r['stage']}] {r['name']}")
            if r.get("detail"):
                print(f"      {r['detail']}")
        if args.fail_fast:
            print("  (--fail-fast: stopped at the first failure)")
    else:
        print(f"OK: {len(rows)} checks within tolerance")

    report["ok"] = not failed
    report["rows"] = rows
    report["only_in_oracle"] = only_oracle
    report["only_in_candidate"] = only_cand
    report["integrity"] = oracle.integrity + cand.integrity
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, default=str))
        print(f"json report -> {args.json_out}")

    return EXIT_FAIL if failed else EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
