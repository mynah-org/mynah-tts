#!/usr/bin/env python3
"""Locate the difference between two audio payloads, in samples.

`cmp -s` answers "are these the same bytes?"; when the answer is no it is not
actionable. A concurrency failure needs a locus: WHICH sample first diverged,
whether one side is simply shorter, and whether the divergence is a rounding
crumb or a different utterance. That is what this prints.

Accepts a RIFF/WAVE file or raw little-endian 16-bit PCM on either side, and
strips the header itself, so a WAV can be compared against the PCM a streaming
route produced without the caller doing `tail -c +45`.

    python3 tools/pcm_diff.py a.wav b.pcm [--rate 24000] [--label-a x] [--label-b y]
    python3 tools/pcm_diff.py --self-test

Exit status: 0 identical, 1 different, 2 unusable input. Offline tooling, no
third-party dependency: it has to run wherever the test script runs.
"""

from __future__ import annotations

import argparse
import array
import math
import os
import struct
import sys
import tempfile

_INT16 = 2


def _wav_data_span(raw: bytes, path: str) -> tuple[int, int]:
    """Byte range of the `data` chunk, walking the chunk list.

    Not a hardcoded 44: a WAV with a LIST/fact chunk before `data` would make
    the constant silently misalign every sample by a couple of bytes, which is
    exactly the kind of "difference" that would send someone hunting a
    concurrency bug that does not exist.
    """
    if len(raw) < 12 or raw[0:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise ValueError(f"{path}: not a RIFF/WAVE file")
    pos = 12
    while pos + 8 <= len(raw):
        cid = raw[pos : pos + 4]
        (size,) = struct.unpack("<I", raw[pos + 4 : pos + 8])
        body = pos + 8
        if cid == b"data":
            end = body + size
            # A streamed-then-finalized WAV can carry a size field that the
            # writer never went back to patch; trust the file, not the field.
            if size == 0 or end > len(raw):
                end = len(raw)
            return body, end
        pos = body + size + (size & 1)
    raise ValueError(f"{path}: RIFF file with no data chunk")


def load_pcm(path: str) -> tuple[bytes, str]:
    """Returns (pcm bytes, description of how it was interpreted)."""
    with open(path, "rb") as handle:
        raw = handle.read()
    if raw[0:4] == b"RIFF":
        start, end = _wav_data_span(raw, path)
        return raw[start:end], f"wav (header {start} B)"
    return raw, "raw pcm"


def samples(pcm: bytes) -> array.array:
    usable = len(pcm) - (len(pcm) % _INT16)
    a = array.array("h")
    a.frombytes(pcm[:usable])
    if sys.byteorder == "big":
        a.byteswap()
    return a


def peak_rms(a: array.array) -> tuple[int, float]:
    if not a:
        return 0, 0.0
    peak = max(max(a), -min(a))
    acc = 0
    for v in a:
        acc += v * v
    return peak, math.sqrt(acc / len(a))


def describe(path: str, label: str, pcm: bytes, kind: str, rate: int) -> str:
    a = samples(pcm)
    peak, rms = peak_rms(a)
    secs = (len(a) / rate) if rate > 0 else 0.0
    return (
        f"  {label}: {path}\n"
        f"    {kind}, {len(pcm)} B = {len(a)} samples"
        + (f" = {secs:.4f} s @ {rate} Hz" if rate > 0 else "")
        + f"\n    peak {peak}  rms {rms:.3f}"
    )


def diff(
    path_a: str,
    path_b: str,
    label_a: str = "A",
    label_b: str = "B",
    rate: int = 0,
    out=sys.stdout,
) -> int:
    pcm_a, kind_a = load_pcm(path_a)
    pcm_b, kind_b = load_pcm(path_b)
    a = samples(pcm_a)
    b = samples(pcm_b)

    if pcm_a == pcm_b:
        print(f"identical: {len(pcm_a)} B / {len(a)} samples", file=out)
        return 0

    print("DIFFERENT", file=out)
    print(describe(path_a, label_a, pcm_a, kind_a, rate), file=out)
    print(describe(path_b, label_b, pcm_b, kind_b, rate), file=out)

    n = min(len(a), len(b))
    first = None
    worst = 0
    worst_at = -1
    ndiff = 0
    for i in range(n):
        d = a[i] - b[i]
        if d:
            if first is None:
                first = i
            ndiff += 1
            if abs(d) > worst:
                worst = abs(d)
                worst_at = i

    if len(pcm_a) != len(pcm_b):
        print(
            f"    length: {label_a} - {label_b} = "
            f"{len(pcm_a) - len(pcm_b)} B "
            f"({len(a) - len(b)} samples)",
            file=out,
        )
    if first is None:
        print(
            f"    the shorter side is a prefix of the longer: all {n} common "
            "samples agree, the difference is length only",
            file=out,
        )
    else:
        where = f" = {first / rate:.4f} s" if rate > 0 else ""
        print(
            f"    first differing sample: index {first}{where} "
            f"(byte {first * _INT16} of the pcm payload): "
            f"{label_a}={a[first]} {label_b}={b[first]}",
            file=out,
        )
        print(
            f"    {ndiff} of {n} common samples differ "
            f"({100.0 * ndiff / n:.2f}%), largest |delta| {worst} "
            f"at sample {worst_at}",
            file=out,
        )
    return 1


# ----------------------------------------------------------------- self-test
#
# The comparator is only ever exercised on failure, which is the worst place
# for a bug to hide: a broken differ would turn every real mismatch into a
# confusing non-answer, and a differ that returned 0 unconditionally would turn
# the gate that calls it into decoration. So it checks itself before it is
# trusted, and the test script runs this first.


def _write(path: str, values, header: bool = False) -> None:
    a = array.array("h", values)
    if sys.byteorder == "big":
        a.byteswap()
    pcm = a.tobytes()
    with open(path, "wb") as handle:
        if header:
            handle.write(b"RIFF")
            handle.write(struct.pack("<I", 36 + len(pcm)))
            handle.write(b"WAVEfmt ")
            handle.write(struct.pack("<IHHIIHH", 16, 1, 1, 24000, 48000, 2, 16))
            handle.write(b"data")
            handle.write(struct.pack("<I", len(pcm)))
        handle.write(pcm)


def self_test() -> int:
    import io

    failures = []

    def check(name: str, cond: bool) -> None:
        if not cond:
            failures.append(name)

    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "a.pcm")
        q = os.path.join(tmp, "b.pcm")
        w = os.path.join(tmp, "a.wav")
        base = [0, 1000, -1000, 32767, -32768, 5, 6, 7]

        _write(p, base)
        _write(q, base)
        _write(w, base, header=True)

        sink = io.StringIO()
        check("identical pcm returns 0", diff(p, q, out=sink) == 0)
        check(
            "wav compares equal to the pcm it wraps",
            diff(w, p, out=sink) == 0,
        )

        # One sample changed, at a known index.
        changed = list(base)
        changed[3] = 32766
        _write(q, changed)
        sink = io.StringIO()
        check("one changed sample returns 1", diff(p, q, out=sink) == 1)
        text = sink.getvalue()
        check("reports the right index", "index 3 " in text)
        check("reports the byte offset", "byte 6 of the pcm payload" in text)
        check("reports both values", "A=32767 B=32766" in text)
        check("reports the delta", "largest |delta| 1" in text)
        check("counts the differing samples", "1 of 8 common samples" in text)

        # Truncation: a prefix, which is what a dropped tail looks like.
        _write(q, base[:5])
        sink = io.StringIO()
        check("truncation returns 1", diff(p, q, out=sink) == 1)
        text = sink.getvalue()
        check("reports the length delta", "= 6 B (3 samples)" in text)
        check("names a pure prefix", "prefix of the longer" in text)

        # Peak/RMS are reported, and are not both zero for real audio.
        _write(q, [0] * 8)
        sink = io.StringIO()
        check("silence differs from signal", diff(p, q, out=sink) == 1)
        text = sink.getvalue()
        check("reports a nonzero peak for A", "peak 32768" in text)
        check("reports a zero peak for B", "peak 0  rms 0.000" in text)

        # And the guard against the failure mode this whole exercise is about:
        # a comparator that cannot say no.
        sink = io.StringIO()
        check("differing files never return 0", diff(p, q, out=sink) != 0)

        # --extract must reproduce exactly the payload the differ compares,
        # because the test script compares extracted payloads with cmp(1) and
        # would otherwise be comparing something else than this tool reports.
        ext = os.path.join(tmp, "out.pcm")
        _write(p, base)
        argv = sys.argv
        try:
            sys.argv = ["pcm_diff.py", w, "--extract", ext]
            check("--extract exits 0", main() == 0)
        finally:
            sys.argv = argv
        with open(ext, "rb") as handle:
            got = handle.read()
        check("--extract strips the wav header", got == load_pcm(p)[0])

    if failures:
        for name in failures:
            print(f"self-test FAILED: {name}", file=sys.stderr)
        return 1
    print("pcm_diff self-test: PASS")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("a", nargs="?")
    ap.add_argument("b", nargs="?")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--rate", type=int, default=0, help="sample rate, for seconds")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument(
        "--extract",
        metavar="OUT",
        help="write A's pcm payload (header stripped) to OUT and exit",
    )
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    if args.extract:
        if not args.a:
            ap.error("--extract needs an input file")
        try:
            pcm, _ = load_pcm(args.a)
        except (OSError, ValueError) as exc:
            print(f"pcm_diff: {exc}", file=sys.stderr)
            return 2
        with open(args.extract, "wb") as handle:
            handle.write(pcm)
        return 0
    if not args.a or not args.b:
        ap.error("need two files, or --self-test")
    try:
        return diff(args.a, args.b, args.label_a, args.label_b, args.rate)
    except (OSError, ValueError) as exc:
        print(f"pcm_diff: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
