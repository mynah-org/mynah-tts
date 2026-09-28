#!/usr/bin/env python3
"""Per-window report of a pocket_ladder.py soak (one <tag>-c<C>.jsonl).

Cuts the measured requests into N equal time windows by send time and prints,
per window, what a qualification needs to see: throughput, streaming RTF and
TTFA percentiles, stalls, failures, and whether any of them drift. A soak
whose last windows are worse than its first has qualified nothing.
"""

import argparse
import json
import math
import sys


def pct(v, p):
    if not v:
        return float("nan")
    s = sorted(v)
    k = (len(s) - 1) * p / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    return s[lo] if lo == hi else s[lo] + (s[hi] - s[lo]) * (k - lo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("--windows", type=int, default=10)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.jsonl)]
    meas = [r for r in rows if r.get("measured")]
    if not meas:
        print("no measured requests", file=sys.stderr)
        return 1
    t0 = min(r["t_send"] for r in meas)
    t1 = max(r.get("t_done", r["t_send"]) for r in meas)
    span = max(t1 - t0, 1e-6)
    width = span / a.windows
    print("window   n   ok  fail  audio-s/s  rtf_p50  rtf_p95  ttfa_p95_ms  gap_p95_ms  stall250  stall500")
    firsts, lasts = [], []
    for w in range(a.windows):
        lo, hi = t0 + w * width, t0 + (w + 1) * width
        rs = [r for r in meas if lo <= r["t_send"] < hi or (w == a.windows - 1 and r["t_send"] == hi)]
        ok = [r for r in rs if r.get("ok")]
        audio = sum(r["audio_s"] for r in ok)
        rtf = [r["rtf_stream"] for r in ok if r.get("rtf_stream") is not None]
        ttfa = [r["ttfa_s"] * 1000 for r in ok if r.get("ttfa_s") is not None]
        gap = [r["max_gap_s"] * 1000 for r in ok if r.get("max_gap_s") is not None]
        line = (w, len(rs), len(ok), len(rs) - len(ok), audio / width, pct(rtf, 50),
                pct(rtf, 95), pct(ttfa, 95), pct(gap, 95),
                sum(r.get("stalls_250", 0) for r in ok), sum(r.get("stalls_500", 0) for r in ok))
        print("%6d %4d %4d %5d %10.2f %8.3f %8.3f %12.0f %11.0f %9d %9d" % line)
        (firsts if w < a.windows // 2 else lasts).append(line)
    total_ok = [r for r in meas if r.get("ok")]
    print("\nwhole soak: %d requests, %d ok, %d failed, %.2f audio-s/s, stream RTF p95 %.3f, TTFA p95 %.0f ms, stalls@250 %d, stalls@500 %d"
          % (len(meas), len(total_ok), len(meas) - len(total_ok),
             sum(r["audio_s"] for r in total_ok) / span,
             pct([r["rtf_stream"] for r in total_ok if r.get("rtf_stream") is not None], 95),
             pct([r["ttfa_s"] * 1000 for r in total_ok if r.get("ttfa_s") is not None], 95),
             sum(r.get("stalls_250", 0) for r in total_ok), sum(r.get("stalls_500", 0) for r in total_ok)))
    if firsts and lasts:
        f = sum(x[6] for x in firsts) / len(firsts)
        l = sum(x[6] for x in lasts) / len(lasts)
        print("drift: stream RTF p95 first half %.3f -> second half %.3f (%+.1f%%)" % (f, l, 100 * (l - f) / max(f, 1e-9)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
