#!/usr/bin/env python3
"""Map noisy / metallic Pocket TTS outputs in a set of captured WAVs.

WER only says whether the words are intelligible: a clip that sounds metallic
or hissy but reads correctly scores 0%. This tool measures the audio itself,
per clip, and groups the outliers by voice, sentence kind, length, seed and
opening word, so one can tell a voice trait (every clip of a voice is dull)
from a sporadic generation artefact (a few clips of an otherwise clean voice).

Per clip, on the speech frames (energy within 25 dB of the clip's 95th
percentile):

  hf_db     energy 4-11 kHz over 80 Hz-4 kHz, dB (median over frames);
            high = hiss / broadband "metallic" bursts
  harm      median normalised autocorrelation peak (60-400 Hz lag range) of
            the 1 kHz low-passed signal; low = harmonics smeared, raspy voice
  cpp       median cepstral peak prominence; low = breathy / noisy phonation
  floor_db  median level of the non-speech frames relative to the speech peak;
            high = audible background noise in the pauses
  lead_s    silence before the first speech frame

A clip is an outlier when it sits far from the other clips of the same voice
(and model set): robust z-score (median/MAD) of hf_db, -harm, -cpp or floor_db
above --z. Voice-level traits show up in the per-voice medians instead.

Input is the manifest.csv written by tools/gpu/bundle.sh next to the wav/
folder (columns file, voice, kind, seed, text, wer, flags, measured):

    uv run --with numpy --with scipy python tools/pocket_audio_noise.py \
        --set 24L-A=DIR_A --set 24L-B=DIR_B --set 6L-A=DIR_C --out OUT \
        [--z 3.5] [--spectrograms 24]   # PNGs need matplotlib too

Offline tooling only. It writes OUT/noise.csv (one row per clip) and
OUT/noise-summary.txt; with --spectrograms N also OUT/spec/*.png for the N
worst clips, for listening side by side.
"""
import argparse
import csv
import os
import sys
import wave

import numpy as np
from scipy.signal import butter, sosfilt

FEATS = ("hf_db", "harm", "cpp", "floor_db")
BAD_SIGN = {"hf_db": 1.0, "harm": -1.0, "cpp": -1.0, "floor_db": 1.0}


def load(path):
    with wave.open(path) as w:
        if w.getsampwidth() != 2 or w.getnchannels() != 1:
            raise ValueError("%s: expected 16-bit mono" % path)
        sr = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float64) / 32768.0
    return x, sr


def features(path):
    x, sr = load(path)
    n, hop = int(0.04 * sr), int(0.01 * sr)
    if len(x) < 4 * n:
        return None
    win = np.hanning(n)
    nfft = 4096
    freqs = np.fft.rfftfreq(nfft, 1.0 / sr)
    lo_band = (freqs >= 80) & (freqs < 4000)
    hi_band = (freqs >= 4000) & (freqs < min(11000, sr / 2))
    lp = sosfilt(butter(4, 1000, "low", fs=sr, output="sos"), x)
    lag0, lag1 = int(sr / 400), int(sr / 60)
    q0, q1 = int(sr / 330), int(sr / 60)
    qs = np.arange(q0, q1 + 1)

    starts = np.arange(0, len(x) - n, hop)
    frames = np.stack([x[s:s + n] for s in starts])
    energy = 10 * np.log10((frames ** 2).mean(1) + 1e-12)
    top = np.percentile(energy, 95)
    speech = energy > top - 25
    quiet = energy < top - 40

    spec = np.abs(np.fft.rfft(frames * win, nfft, axis=1)) ** 2 + 1e-18
    hf = 10 * np.log10(spec[:, hi_band].sum(1) / spec[:, lo_band].sum(1))

    lpf = np.stack([lp[s:s + n] for s in starts]) * win
    ac = np.fft.irfft(np.abs(np.fft.rfft(lpf, 2 * n, axis=1)) ** 2, axis=1)
    harm = ac[:, lag0:lag1 + 1].max(1) / (ac[:, 0] + 1e-12)

    ceps = np.fft.irfft(10 * np.log10(spec), axis=1)[:, :q1 + 1]
    seg = ceps[:, q0:q1 + 1]
    k = seg.argmax(1)
    slope, icpt = np.polyfit(qs, seg.T, 1)
    cpp = seg[np.arange(len(seg)), k] - (slope * qs[k] + icpt)

    first = int(np.argmax(energy > top - 30))
    return {
        "dur_s": len(x) / sr,
        "hf_db": float(np.median(hf[speech])),
        "harm": float(np.median(harm[speech])),
        "cpp": float(np.median(cpp[speech])),
        "floor_db": float(np.median(energy[quiet]) - top) if quiet.sum() >= 5 else float("nan"),
        "lead_s": first * hop / sr,
    }


def robust_z(values):
    v = np.asarray(values, float)
    ok = ~np.isnan(v)
    if ok.sum() < 5:
        return np.zeros_like(v)
    med = np.median(v[ok])
    mad = np.median(np.abs(v[ok] - med)) * 1.4826 or 1e-9
    z = (v - med) / mad
    z[~ok] = 0.0
    return z


def length_class(dur):
    return "<2s" if dur < 2 else "2-5s" if dur < 5 else "5-15s" if dur < 15 else ">=15s"


def first_word(text):
    w = (text or "").strip().split()
    return w[0].strip(",.!?;:\"'").lower() if w else ""


def table(title, rows, key, out):
    groups = {}
    for r in rows:
        groups.setdefault(key(r), []).append(r)
    out.append("\n%s\n  %-22s %6s %8s %6s   %7s %6s %6s %8s" % (
        title, "group", "clips", "outliers", "%", "hf_db", "harm", "cpp", "floor_db"))
    for g in sorted(groups, key=lambda g: (-np.mean([r["outlier"] for r in groups[g]]), str(g))):
        rs = groups[g]
        bad = sum(r["outlier"] for r in rs)
        med = [np.nanmedian([r[f] for r in rs]) for f in FEATS]
        out.append("  %-22s %6d %8d %5.1f%%   %7.1f %6.3f %6.3f %8.1f" % (
            (str(g)[:22], len(rs), bad, 100.0 * bad / len(rs)) + tuple(med)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", action="append", required=True, metavar="NAME=DIR",
                    help="a folder holding manifest.csv and wav/ (repeatable)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--z", type=float, default=3.5, help="robust z-score above which a clip is an outlier")
    ap.add_argument("--spectrograms", type=int, default=0, metavar="N")
    args = ap.parse_args()

    rows = []
    for spec in args.set:
        name, _, d = spec.partition("=")
        with open(os.path.join(d, "manifest.csv"), newline="") as f:
            for m in csv.DictReader(f):
                path = os.path.join(d, "wav", m["file"])
                if not os.path.isfile(path):
                    continue
                ft = features(path)
                if ft is None:
                    continue
                rows.append(dict(set=name, model=name.split("-")[0], file=m["file"], path=path,
                                 voice=m.get("voice", ""), kind=m.get("kind", ""), seed=m.get("seed", ""),
                                 text=m.get("text", ""), wer=float(m.get("wer") or "nan"),
                                 asr_flags=m.get("flags", ""), measured=m.get("measured", ""), **ft))
        print("%s: %d clips" % (name, sum(r["set"] == name for r in rows)), file=sys.stderr)
    if not rows:
        sys.exit("no clips found")

    # Outliers relative to the same voice on the same model: a dull voice is
    # a trait, not an artefact of one clip.
    for key in sorted({(r["model"], r["voice"]) for r in rows}):
        grp = [r for r in rows if (r["model"], r["voice"]) == key]
        zs = {f: robust_z([r[f] for r in grp]) * BAD_SIGN[f] for f in FEATS}
        for i, r in enumerate(grp):
            r["score"] = float(max(zs[f][i] for f in FEATS))
            r["why"] = ";".join(f for f in FEATS if zs[f][i] > args.z)
            r["outlier"] = int(r["score"] > args.z)
    for r in rows:
        r["len"] = length_class(r["dur_s"])
        r["first"] = first_word(r["text"])

    os.makedirs(args.out, exist_ok=True)
    cols = ["set", "file", "voice", "kind", "len", "dur_s", "seed", "first", "outlier", "score", "why",
            "hf_db", "harm", "cpp", "floor_db", "lead_s", "wer", "asr_flags", "measured", "text"]
    rows.sort(key=lambda r: -r["score"])
    with open(os.path.join(args.out, "noise.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})

    out = ["Pocket audio noise map: %d clips, outlier = robust z > %.1f within (model, voice)" % (len(rows), args.z)]
    nb = sum(r["outlier"] for r in rows)
    out.append("outliers: %d (%.1f%%)" % (nb, 100.0 * nb / len(rows)))
    table("By model / voice (medians show the voice's own timbre)", rows, lambda r: "%s %s" % (r["model"], r["voice"]), out)
    table("By set", rows, lambda r: r["set"], out)
    table("By sentence kind", rows, lambda r: r["kind"], out)
    table("By clip length", rows, lambda r: r["len"], out)
    table("By voice x length", rows, lambda r: "%s %s" % (r["voice"], r["len"]), out)
    table("By ASR flag (WER-flagged vs not)", rows, lambda r: "wer-flagged" if "wer" in r["asr_flags"] else "clean-wer", out)
    table("By warm-up vs measured", rows, lambda r: "measured" if r["measured"] == "True" else "warm-up", out)
    firsts = {}
    for r in rows:
        firsts.setdefault(r["first"], []).append(r)
    common = [r for r in rows if len(firsts[r["first"]]) >= 8]
    table("By opening word (words with >= 8 clips)", common, lambda r: r["first"], out)
    seeds = {}
    for r in rows:
        seeds.setdefault((r["model"], r["seed"]), []).append(r["outlier"])
    rep = [(k, v) for k, v in seeds.items() if len(v) > 1]
    if rep:
        both = sum(1 for _, v in rep if all(v))
        some = sum(1 for _, v in rep if any(v))
        out.append("\nSeeds seen more than once: %d; outlier in all of them %d, in some %d" % (len(rep), both, some))
    out.append("\nWorst clips:")
    for r in rows[:40]:
        out.append("  %5.1f %-6s %-26s %-7s %-14s %-5s %5.2fs wer %.2f why=%s  %s" % (
            r["score"], r["set"], r["file"], r["voice"], r["kind"], r["len"], r["dur_s"],
            r["wer"], r["why"] or "-", r["text"][:50]))
    txt = "\n".join(out) + "\n"
    with open(os.path.join(args.out, "noise-summary.txt"), "w") as f:
        f.write(txt)
    print(txt)

    if args.spectrograms:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        sd = os.path.join(args.out, "spec")
        os.makedirs(sd, exist_ok=True)
        for i, r in enumerate(rows[:args.spectrograms]):
            x, sr = load(r["path"])
            fig, ax = plt.subplots(figsize=(10, 2.6))
            ax.specgram(x, NFFT=1024, Fs=sr, noverlap=768, vmin=-130, cmap="magma")
            ax.set_ylim(0, min(12000, sr / 2))
            ax.set_title("%s %s %s z=%.1f %s" % (r["set"], r["file"], r["voice"], r["score"], r["why"]), fontsize=9)
            fig.tight_layout()
            fig.savefig(os.path.join(sd, "%02d-%s-%s.png" % (i, r["set"], r["file"][:-4])), dpi=70)
            plt.close(fig)


if __name__ == "__main__":
    main()
