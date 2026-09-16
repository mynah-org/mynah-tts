#!/usr/bin/env python3
"""codec_int8_quality.py -- the quality gate for the codec conv stack's int8 path.

WHY A SEPARATE GATE.  ``src/convq8.c``'s self-test proves the KERNEL: its int8
arithmetic against an f64 oracle, on synthetic data, with absolute bounds.  It
cannot prove the thing that decides whether the path may ship, which is what
happens to real audio when real weights -- with real outliers, which uniform
random data does not have -- go through it, and whether the generated FRAME
COUNT moves.

THREE MEASUREMENTS, in the order they can fail:

1. SAMPLE COUNT.  The codec is downstream of the sampler, so int8 in the conv
   stack must not change how many frames were generated or where EOS landed.
   If this moves, nothing else matters: the change has reached the AR loop and
   the whole premise ("error that cannot feed back") is false.  Checked to the
   byte, not to a tolerance.
2. WAVEFORM CORRELATION / SNR against the same request with the conv stack in
   f32 -- the same binary, one environment variable apart, so nothing else can
   differ.
3. LOG-MEL CORRELATION, because two waveforms can decorrelate on phase while
   being perceptually identical, and the opposite is what would worry us.

THE BOUNDS ARE CALIBRATED, not chosen.  Two reference points:

* The int8 this project ALREADY SHIPS in the same region.
  `codec_transformer:int8` moves this utterance by 25.3 dB SNR, 0.99852
  waveform correlation and 0.99953 log-mel correlation, and that was accepted.
  Note the shape of it: the transformer's int8 produces a DIFFERENT but
  spectrally identical waveform (waveform correlation falls, log-mel does
  not), while the conv stack's adds a low-level broadband residual (waveform
  correlation stays, log-mel falls).  They are not comparable on one number,
  which is why this gate measures both.
* The configuration that was REJECTED on quality.  Admitting the 32-channel
  stage measured 0.9785 log-mel here -- an eightfold increase in spectral
  error over the 0.9976 the shipped gate produces -- for 6 ms of 44.  The
  log-mel bound is set between those two, so it fails that configuration.

Observed over three texts x three seeds on `models/pocket-en`: waveform
correlation 0.999886-0.999920, log-mel 0.996234-0.997161, SNR 36.4-37.9 dB,
and the sample count identical in every pair.
"""
import argparse
import math
import os
import struct
import subprocess
import sys
import tempfile
import wave

import numpy as np


def read_wav(path):
    with wave.open(path) as w:
        n = w.getnframes()
        rate = w.getframerate()
        data = struct.unpack("<%dh" % n, w.readframes(n))
    return np.asarray(data, dtype=np.float64), rate


def log_mel(x, rate, n_fft=1024, hop=256, n_mels=64):
    """A small mel filterbank; absolute values do not matter, only the
    correlation between two of them, so the normalisation is irrelevant."""
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
    window = np.hanning(n_fft)
    frames = 1 + (len(x) - n_fft) // hop
    spec = np.empty((frames, n_fft // 2 + 1))
    for i in range(frames):
        seg = x[i * hop:i * hop + n_fft] * window
        spec[i] = np.abs(np.fft.rfft(seg))
    def hz2mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)
    def mel2hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)
    edges = mel2hz(np.linspace(hz2mel(50.0), hz2mel(rate / 2.0), n_mels + 2))
    bins = np.floor((n_fft + 1) * edges / rate).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(n_mels):
        lo, mid, hi = bins[m], bins[m + 1], bins[m + 2]
        for k in range(lo, mid):
            if mid > lo:
                fb[m, k] = (k - lo) / (mid - lo)
        for k in range(mid, hi):
            if hi > mid:
                fb[m, k] = (hi - k) / (hi - mid)
    return np.log(spec @ fb.T + 1e-6)


def corr(a, b):
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    den = math.sqrt(float(a @ a) * float(b @ b))
    return float(a @ b) / den if den else 0.0


def synth(binary, model, text, seed, out, q8):
    env = dict(os.environ)
    env["MYNAH_CODEC_CONV_Q8"] = "1" if q8 else "0"
    env.setdefault("MYNAH_THREADS", "2")
    cmd = [binary, "--synthesize", model, "--text", text, "--lang", "en",
           "--seed", str(seed), "--output", out]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("synthesis failed (%d): %s" % (r.returncode, r.stderr[-800:]))


TEXTS = [
    "the quick brown fox jumps over the lazy dog and keeps running",
    "she sells sea shells by the sea shore on a bright windy morning",
    "numbers like nineteen eighty four and two thousand twenty six",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default="build/cpu/mynah-tts")
    ap.add_argument("--model", required=True)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--min-wave-corr", type=float, default=0.9995)
    ap.add_argument("--min-mel-corr", type=float, default=0.995)
    ap.add_argument("--min-snr-db", type=float, default=34.0)
    args = ap.parse_args()

    worst_wave, worst_mel, worst_snr = 1.0, 1.0, 1e9
    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        for ti, text in enumerate(TEXTS):
            for seed in range(1, args.seeds + 1):
                a_path = os.path.join(tmp, "f32.wav")
                b_path = os.path.join(tmp, "q8.wav")
                synth(args.binary, args.model, text, seed, a_path, q8=False)
                synth(args.binary, args.model, text, seed, b_path, q8=True)
                a, rate = read_wav(a_path)
                b, _ = read_wav(b_path)
                tag = "text %d seed %d" % (ti, seed)
                if len(a) != len(b):
                    failures.append("%s: sample count moved, %d -> %d -- the "
                                    "conv stack reached the frame count"
                                    % (tag, len(a), len(b)))
                    continue
                n = len(a)
                err = math.sqrt(float(((a - b) ** 2).sum()) / n)
                rms = math.sqrt(float((a ** 2).sum()) / n)
                snr = 20.0 * math.log10(rms / err) if err > 0 else 99.0
                wc = corr(a, b)
                mc = corr(log_mel(a, rate), log_mel(b, rate))
                worst_wave = min(worst_wave, wc)
                worst_mel = min(worst_mel, mc)
                worst_snr = min(worst_snr, snr)
                print("  %-16s n=%6d  wave corr %.6f  mel corr %.6f  SNR %5.1f dB"
                      % (tag, n, wc, mc, snr))
                if wc < args.min_wave_corr:
                    failures.append("%s: waveform correlation %.6f < %.6f"
                                    % (tag, wc, args.min_wave_corr))
                if mc < args.min_mel_corr:
                    failures.append("%s: log-mel correlation %.6f < %.6f"
                                    % (tag, mc, args.min_mel_corr))
                if snr < args.min_snr_db:
                    failures.append("%s: SNR %.1f dB < %.1f dB"
                                    % (tag, snr, args.min_snr_db))
    print("worst: wave %.6f, mel %.6f, SNR %.1f dB"
          % (worst_wave, worst_mel, worst_snr))
    if failures:
        for f in failures:
            print("FAIL: " + f, file=sys.stderr)
        return 1
    print("codec int8 conv quality gate: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
