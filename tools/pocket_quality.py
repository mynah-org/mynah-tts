#!/usr/bin/env python3
"""Quick serving quality gate: concurrent distinct requests -> WAV -> WER.

Sends every sentence of a fixed set to a running mynah-tts server at a chosen
concurrency, with staggered starts so the scheduler's gangs keep changing,
stores each stream as a WAV and transcribes it with faster-whisper. A request
that received another request's audio, or garbage from a corrupted codec
state, shows up as a high per-utterance WER.

    python tools/pocket_quality.py --port 18080 --concurrency 64 --out /root/evidence/q-tag

Captured-audio mode (no server, no re-synthesis): transcribe the WAVs that
tools/pocket_ladder.py --save-audio wrote while the server was under load,
listed in its per-request JSONL, and flag suspicious audio automatically:

    python tools/pocket_quality.py --from-jsonl /root/evidence/T/T-c128.jsonl \
        [--out DIR] [--asr-workers 8 --asr-threads 4] [--no-asr]

It writes DIR/quality.csv (one row per WAV), DIR/quality-summary.txt and
DIR/quality-summary.json (WER overall, per kind and per voice, utterances
above 30% WER, flag counts). DIR defaults to <jsonl dir>/quality.

Offline tooling only; needs `faster-whisper` and `jiwer` in the environment
(`--no-asr` needs neither: audio flags only).
"""
import argparse
import array
import concurrent.futures
import csv
import http.client
import json
import math
import multiprocessing
import os
import random
import re
import sys
import time
import wave

SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Please call Stella and ask her to bring these things with her from the store.",
    "The museum opens at nine in the morning and closes late on Fridays.",
    "Our train was delayed because of a signal failure near the river bridge.",
    "She planted tomatoes, basil and a row of sunflowers along the fence.",
    "The committee will announce the winners at the end of the evening.",
    "A gentle breeze carried the smell of fresh bread through the open window.",
    "He forgot his umbrella on the bus and walked home in the rain.",
    "The new library has a quiet reading room on the top floor.",
    "Remember to water the plants while I am away next week.",
    "The children built a sandcastle with towers, walls and a small moat.",
    "Most of the snow had melted by the time the sun came out.",
    "The doctor said that a short walk every day would help a lot.",
    "They painted the kitchen a warm yellow colour last summer.",
    "The orchestra rehearsed the final movement three times before the concert.",
    "Could you send me the report before the meeting tomorrow afternoon?",
    "The old bridge was closed for repairs during most of the spring.",
    "My neighbour keeps bees and sells honey at the weekend market.",
    "The pilot welcomed everyone on board and described the weather ahead.",
    "Fresh coffee and a good book make a perfect rainy morning.",
    "The hikers reached the summit just before the clouds rolled in.",
    "Every student received a map, a compass and a bottle of water.",
    "The bakery on the corner sells the best cinnamon rolls in town.",
    "We watched the fireworks from the hill behind the old church.",
    "The engineer explained how the new bridge would carry heavy traffic.",
    "A small cat slept in the sunny window of the bookshop.",
    "The festival brought musicians and dancers from many different countries.",
    "Turn left at the second traffic light and park behind the school.",
    "The garden was full of butterflies after the summer rain.",
    "He practised the piano every evening until the melody felt natural.",
    "The ship left the harbour at dawn with a full load of timber.",
    "Our team finished the project two days ahead of schedule.",
]
VOICES = ["alba", "marius", "javert", "jean"]


_ONES = ("zero one two three four five six seven eight nine ten eleven twelve "
         "thirteen fourteen fifteen sixteen seventeen eighteen nineteen").split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
_ORD = {"one": "first", "two": "second", "three": "third", "five": "fifth",
        "eight": "eighth", "nine": "ninth", "twelve": "twelfth"}
# Spellings the ASR may pick either way; mapped to one form on both sides.
_SAME = {"alright": "all right", "ok": "okay", "colour": "color",
         "colours": "colors", "harbour": "harbor", "neighbour": "neighbor",
         "neighbours": "neighbors", "favourite": "favorite", "humour": "humor",
         "theatre": "theater", "centre": "center", "practised": "practiced",
         "cancelled": "canceled", "travelling": "traveling", "grey": "gray",
         "apologise": "apologize", "towards": "toward", "percent": "per cent"}


def _words(n):
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else " " + _ONES[n % 10])
    for div, name in ((10 ** 9, "billion"), (10 ** 6, "million"),
                      (1000, "thousand"), (100, "hundred")):
        if n >= div:
            rest = n % div
            return _words(n // div) + " " + name + ("" if rest == 0 else " " + _words(rest))
    return str(n)


def _number(m):
    digits, suffix = m.group(1).replace(",", ""), (m.group(2) or "").lower()
    if len(digits) > 12:
        return m.group(0)
    w = _words(int(digits))
    if suffix in ("st", "nd", "rd", "th"):
        head, _, last = w.rpartition(" ")
        last = _ORD.get(last, last[:-1] + "ieth" if last.endswith("y") else last + "th")
        w = (head + " " + last).strip()
    return " " + w + " "


def norm(s):
    """Lower-case words for WER. The ASR writes numbers as digits and a
    corpus may spell them out, so digits become words on both sides, and
    'hundred and five' reads the same as 'hundred five'."""
    s = s.lower().replace("-", " ").replace("%", " per cent ")
    s = re.sub(r"(\d[\d,]*)(st|nd|rd|th)?\b", _number, s)
    s = re.sub(r"[^a-z0-9' ]+", " ", s)
    s = re.sub(r"\b(hundred|thousand|million|billion) and\b", r"\1", s)
    return " ".join(_SAME.get(w, w) for w in s.split())


def one(port, idx, text, voice, seed, delay, out):
    time.sleep(delay)
    body = {"model": "pocket", "input": text, "voice": voice,
            "response_format": "pcm", "stream": True, "seed": seed}
    t0 = time.monotonic()
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=300)
    conn.request("POST", "/v1/audio/speech", body=json.dumps(body),
                 headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    if resp.status != 200:
        return {"idx": idx, "error": "http %d" % resp.status}
    pcm = resp.read()
    conn.close()
    path = os.path.join(out, "u%03d.wav" % idx)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(pcm)
    return {"idx": idx, "text": text, "voice": voice, "seed": seed, "wav": path,
            "seconds": len(pcm) / 48000.0, "wall": time.monotonic() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18080)
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--stagger", type=float, default=2.0,
                    help="max random start delay per request, seconds")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--whisper", default="small.en")
    ap.add_argument("--out", default="")
    ap.add_argument("--phase", choices=["all", "synth", "asr"], default="all",
                    help="synth only, ASR only on a previous --out, or both")
    ap.add_argument("--asr-threads", type=int, default=None,
                    help="CPU threads per ASR model (default: min(32, cores); "
                         "with --from-jsonl: 4 per worker)")
    ap.add_argument("--from-jsonl", default="",
                    help="transcribe the WAVs listed in a pocket_ladder per-request "
                         "JSONL (audio captured under load) instead of synthesising")
    ap.add_argument("--asr-workers", type=int, default=8,
                    help="--from-jsonl: ASR processes, each with --asr-threads")
    ap.add_argument("--no-asr", action="store_true",
                    help="--from-jsonl: audio flags only, no transcription")
    ap.add_argument("--measured-only", action="store_true",
                    help="--from-jsonl: skip the warmup requests")
    ap.add_argument("--silence-db", type=float, default=-50.0,
                    help="frame RMS below this dBFS counts as silence")
    ap.add_argument("--abrupt-db", type=float, default=-35.0,
                    help="last 50 ms RMS above this dBFS flags an abrupt end")
    ap.add_argument("--bad-wer", type=float, default=0.30)
    a = ap.parse_args()
    if a.from_jsonl:
        if a.asr_threads is None:
            a.asr_threads = 4
        if not a.out:
            a.out = os.path.join(os.path.dirname(os.path.abspath(a.from_jsonl)), "quality")
        os.makedirs(a.out, exist_ok=True)
        return captured(a)
    if a.asr_threads is None:
        a.asr_threads = min(32, os.cpu_count() or 4)
    if not a.out:
        ap.error("--out is required unless --from-jsonl is given")
    os.makedirs(a.out, exist_ok=True)
    rng = random.Random(a.seed)
    jobs = []
    for r in range(a.rounds):
        for i, text in enumerate(SENTENCES):
            idx = r * len(SENTENCES) + i
            jobs.append((idx, text, VOICES[idx % len(VOICES)], 1000 + idx,
                         rng.uniform(0.0, a.stagger) + r * 0.5))
    rows_path = os.path.join(a.out, "quality.jsonl")
    if a.phase == "asr":
        with open(rows_path) as f:
            rows = [json.loads(line) for line in f]
    else:
        rows = synthesize(a, jobs)
        with open(rows_path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    if a.phase == "synth":
        print("SYNTH utts=%d failed=%d" % (len(rows), sum(1 for r in rows if "error" in r)))
        return
    transcribe(a, rows, rows_path)


def synthesize(a, jobs):
    rows = []
    with concurrent.futures.ThreadPoolExecutor(a.concurrency) as ex:
        futs = [ex.submit(one, a.port, *j, a.out) for j in jobs]
        for f in futs:
            try:
                rows.append(f.result())
            except Exception as e:  # a dropped stream is a failed utterance
                rows.append({"error": "%s: %s" % (type(e).__name__, e)})
    return rows


def transcribe(a, rows, rows_path):
    failed = [r for r in rows if "error" in r]

    from faster_whisper import WhisperModel
    import jiwer
    model = WhisperModel(a.whisper, device="cpu", compute_type="int8",
                         cpu_threads=a.asr_threads)
    refs, hyps, bad = [], [], []
    for r in rows:
        if "error" in r:
            continue
        segs, _ = model.transcribe(r["wav"], language="en", beam_size=1)
        hyp = " ".join(s.text for s in segs)
        r["hyp"] = hyp
        r["wer"] = jiwer.wer(norm(r["text"]), norm(hyp) or "<empty>")
        refs.append(norm(r["text"]))
        hyps.append(norm(hyp) or "<empty>")
        if r["wer"] > 0.3:
            bad.append(r)
    with open(rows_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    wer = jiwer.wer(refs, hyps) if refs else 1.0
    print("QUALITY utts=%d failed=%d WER=%.2f%% utt>30%%=%d audio=%.1fs" % (
        len(rows), len(failed), 100.0 * wer, len(bad),
        sum(r.get("seconds", 0.0) for r in rows)))
    for r in bad:
        print("  BAD u%03d wer=%.2f ref=%r hyp=%r" % (r["idx"], r["wer"], r["text"], r["hyp"]))


# ---------------------------------------------------------------------------
# Captured-audio mode: WER and audio flags on the WAVs streamed under load.

WPS_RANGE = (1.2, 5.0)
LONG_SILENCE_S = 1.5
CLIP_FRAC = 0.001
FRAME_S = 0.02
TAIL_S = 0.05

try:
    import numpy as _np
except ImportError:  # stdlib fallback for --no-asr on a bare host
    _np = None

_ASR = {}


def _db(rms):
    return 20.0 * math.log10(rms / 32768.0) if rms > 0 else -200.0


def audio_stats(path, silence_db, abrupt_db):
    """Duration, silences, clipping and the tail level of one PCM16 WAV."""
    with wave.open(path, "rb") as w:
        rate, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if ch != 1 or width != 2:
        raise ValueError("expected mono PCM16, got %d ch x %d B" % (ch, width))
    st = {"rate": rate, "wav_samples": len(raw) // 2}
    n = st["wav_samples"]
    st["duration_s"] = n / float(rate)
    if n == 0:
        return st
    flen = max(1, int(rate * FRAME_S))
    tail = max(1, int(rate * TAIL_S))
    if _np is not None:
        x = _np.frombuffer(raw, dtype="<i2").astype(_np.float64)
        st["clip_frac"] = float(_np.count_nonzero(_np.abs(x) >= 32767)) / n
        st["peak_dbfs"] = _db(float(_np.max(_np.abs(x))))
        st["rms_dbfs"] = _db(float(_np.sqrt(_np.mean(x * x))))
        st["tail_dbfs"] = _db(float(_np.sqrt(_np.mean(x[-tail:] ** 2))))
        nf = n // flen
        fr = x[:nf * flen].reshape(nf, flen) if nf else x.reshape(1, -1)
        frame_db = [_db(v) for v in _np.sqrt(_np.mean(fr * fr, axis=1)).tolist()]
    else:
        x = array.array("h")
        x.frombytes(raw[:n * 2])
        if sys.byteorder != "little":
            x.byteswap()
        st["clip_frac"] = sum(1 for v in x if v >= 32767 or v <= -32767) / float(n)
        st["peak_dbfs"] = _db(float(max(abs(v) for v in x)))
        st["rms_dbfs"] = _db(math.sqrt(sum(v * v for v in x) / float(n)))
        st["tail_dbfs"] = _db(math.sqrt(sum(v * v for v in x[-tail:]) / float(min(tail, n))))
        nf = max(1, n // flen)
        frame_db = [_db(math.sqrt(sum(v * v for v in x[i * flen:(i + 1) * flen]) /
                                  float(len(x[i * flen:(i + 1) * flen]))))
                    for i in range(nf)]
    voiced = [i for i, d in enumerate(frame_db) if d >= silence_db]
    if not voiced:
        st["lead_sil_s"] = st["trail_sil_s"] = st["duration_s"]
        st["max_gap_s"] = 0.0
        st["silent"] = True
        return st
    st["lead_sil_s"] = voiced[0] * FRAME_S
    st["trail_sil_s"] = st["duration_s"] - (voiced[-1] + 1) * FRAME_S
    gaps = [b - a - 1 for a, b in zip(voiced, voiced[1:])]
    st["max_gap_s"] = max(gaps, default=0) * FRAME_S
    return st


def flags_for(r, st, a):
    f = []
    if st.get("missing"):
        return ["missing"]
    if st["wav_samples"] == 0:
        return ["empty"]
    if st.get("silent"):
        f.append("silent")
    if r.get("samples") is not None and r["samples"] != st["wav_samples"]:
        f.append("samples_mismatch")
    words = len(norm(r.get("text", "")).split())
    if words:
        wps = words / st["duration_s"]
        st["wps"] = wps
        if wps < WPS_RANGE[0]:
            f.append("slow_wps")
        elif wps > WPS_RANGE[1]:
            f.append("fast_wps")
    if st.get("lead_sil_s", 0) > LONG_SILENCE_S:
        f.append("lead_silence")
    if st.get("trail_sil_s", 0) > LONG_SILENCE_S:
        f.append("trail_silence")
    if st.get("max_gap_s", 0) > LONG_SILENCE_S:
        f.append("internal_silence")
    if st.get("clip_frac", 0) > CLIP_FRAC:
        f.append("clipping")
    if st.get("tail_dbfs", -200) > a.abrupt_db:
        f.append("abrupt_end")
    return f


def _asr_init(whisper, threads, use_asr):
    if use_asr:
        from faster_whisper import WhisperModel
        _ASR["model"] = WhisperModel(whisper, device="cpu", compute_type="int8",
                                     cpu_threads=threads)


def _analyse(job):
    r, a_dict = job
    out = {"id": r.get("id")}
    path = r["wav"]
    if not os.path.isfile(path):
        out["stats"] = {"missing": True}
        return out
    try:
        out["stats"] = audio_stats(path, a_dict["silence_db"], a_dict["abrupt_db"])
    except Exception as e:
        out["stats"] = {"missing": True}
        out["error"] = "%s: %s" % (type(e).__name__, e)
        return out
    model = _ASR.get("model")
    if model is not None and out["stats"]["wav_samples"] > 0:
        try:
            segs, _ = model.transcribe(path, language="en", beam_size=1)
            out["hyp"] = " ".join(s.text.strip() for s in segs)
        except Exception as e:  # one bad file must not kill a 30-min run
            out["error"] = "asr %s: %s" % (type(e).__name__, e)
    return out


CSV_COLS = ["id", "measured", "corpus_id", "kind", "voice", "seed", "duration_s",
            "words", "wps", "lead_sil_s", "trail_sil_s", "max_gap_s", "clip_frac",
            "peak_dbfs", "rms_dbfs", "tail_dbfs", "ttfa_s", "rtf_stream",
            "stalls_250", "wer", "flags", "text", "hyp", "wav"]


def captured(a):
    with open(a.from_jsonl) as f:
        recs = [json.loads(line) for line in f if line.strip()]
    todo = [r for r in recs if r.get("wav") and r.get("ok")
            and (r.get("measured") or not a.measured_only)]
    not_ok = sum(1 for r in recs if not r.get("ok"))
    if not todo:
        print("no records with a captured wav in %s" % a.from_jsonl, file=sys.stderr)
        return 1
    use_asr = not a.no_asr
    if use_asr:
        import jiwer  # fail before the long run, not after it
    a_dict = {"silence_db": a.silence_db, "abrupt_db": a.abrupt_db}
    workers = max(1, a.asr_workers)
    t0 = time.monotonic()
    results = {}
    ctx = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
            workers, mp_context=ctx, initializer=_asr_init,
            initargs=(a.whisper, a.asr_threads, use_asr)) as ex:
        for i, res in enumerate(ex.map(_analyse, [(r, a_dict) for r in todo],
                                       chunksize=4 if use_asr else 32), 1):
            results[i - 1] = res
            if i % 500 == 0:
                print("  %d/%d analysed, %.0f s" % (i, len(todo), time.monotonic() - t0),
                      file=sys.stderr, flush=True)
    rows = []
    for i, r in enumerate(todo):
        res = results[i]
        st = res["stats"]
        row = {k: r.get(k) for k in ("id", "measured", "corpus_id", "kind", "voice",
                                     "seed", "ttfa_s", "rtf_stream", "stalls_250",
                                     "text", "wav")}
        fl = flags_for(r, st, a)
        if res.get("error"):
            fl.append("unreadable")
        row.update({k: st.get(k) for k in ("duration_s", "wps", "lead_sil_s",
                                          "trail_sil_s", "max_gap_s", "clip_frac",
                                          "peak_dbfs", "rms_dbfs", "tail_dbfs")})
        row["words"] = len(norm(r.get("text", "")).split())
        if use_asr:
            ref = norm(r.get("text", ""))
            hyp = norm(res.get("hyp", "")) or "<empty>"
            row["hyp"] = res.get("hyp", "")
            row["_ref"], row["_hyp"] = ref, hyp
            row["wer"] = jiwer.wer(ref, hyp) if ref else None
            if row["wer"] is not None and row["wer"] > a.bad_wer:
                fl.append("wer")
        row["flags"] = ";".join(fl)
        rows.append(row)

    csv_path = os.path.join(a.out, "quality.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: ("%.4f" % v if isinstance(v, float) else v)
                        for k, v in row.items()})

    def wer_of(sel):
        sel = [x for x in sel if x.get("wer") is not None]
        if not sel:
            return None
        return jiwer.wer([x["_ref"] for x in sel], [x["_hyp"] for x in sel])

    flag_counts = {}
    for row in rows:
        for fl in filter(None, row["flags"].split(";")):
            flag_counts[fl] = flag_counts.get(fl, 0) + 1
    summ = {
        "source": os.path.abspath(a.from_jsonl), "records": len(recs),
        "records_not_ok": not_ok, "analysed": len(rows),
        "audio_s": sum(x.get("duration_s") or 0.0 for x in rows),
        "asr": ({"model": a.whisper, "compute": "int8 cpu", "beam": 1,
                 "workers": workers, "threads": a.asr_threads} if use_asr else None),
        "flag_counts": flag_counts,
        "flagged": sum(1 for x in rows if x["flags"]),
        "thresholds": {"wps": WPS_RANGE, "silence_s": LONG_SILENCE_S,
                       "silence_db": a.silence_db, "clip_frac": CLIP_FRAC,
                       "abrupt_db": a.abrupt_db, "bad_wer": a.bad_wer},
        "elapsed_s": time.monotonic() - t0,
    }
    if use_asr:
        summ["wer"] = wer_of(rows)
        summ["wer_by_kind"] = {k: wer_of([x for x in rows if x.get("kind") == k])
                               for k in sorted({x.get("kind") or "?" for x in rows})}
        summ["wer_by_voice"] = {v: wer_of([x for x in rows if x.get("voice") == v])
                                for v in sorted({x.get("voice") or "?" for x in rows})}
        summ["utt_over_bad_wer"] = sum(1 for x in rows if "wer" in x["flags"].split(";"))
    with open(os.path.join(a.out, "quality-summary.json"), "w") as f:
        json.dump(summ, f, indent=1)

    def pc(v):
        return "-" if v is None else "%.2f%%" % (100.0 * v)

    lines = ["QUALITY captured utts=%d not_ok=%d audio=%.0fs WER=%s utt>%d%%=%s flagged=%d"
             % (len(rows), not_ok, summ["audio_s"], pc(summ.get("wer")),
                int(a.bad_wer * 100), summ.get("utt_over_bad_wer", "-"), summ["flagged"])]
    if use_asr:
        lines.append("WER by kind:  " + "  ".join("%s %s" % (k, pc(v))
                                                  for k, v in summ["wer_by_kind"].items()))
        lines.append("WER by voice: " + "  ".join("%s %s" % (k, pc(v))
                                                  for k, v in summ["wer_by_voice"].items()))
    lines.append("flags: " + (", ".join("%s=%d" % kv for kv in sorted(flag_counts.items()))
                              or "none"))
    for row in rows:
        if use_asr and row.get("wer") is not None and row["wer"] > a.bad_wer:
            lines.append("  BAD req %s %s/%s wer=%.2f ref=%r hyp=%r wav=%s" % (
                row["id"], row.get("kind"), row.get("corpus_id"), row["wer"],
                row.get("text"), row.get("hyp"), row.get("wav")))
    for row in rows:
        other = [x for x in row["flags"].split(";") if x and x != "wer"]
        if other:
            lines.append("  FLAG req %s %s/%s %s dur=%.2fs wps=%s wav=%s" % (
                row["id"], row.get("kind"), row.get("corpus_id"), ",".join(other),
                row.get("duration_s") or 0.0,
                "-" if row.get("wps") is None else "%.2f" % row["wps"], row.get("wav")))
    text = "\n".join(lines) + "\n"
    with open(os.path.join(a.out, "quality-summary.txt"), "w") as f:
        f.write(text)
    sys.stdout.write(text)
    print("wrote %s and quality-summary.{txt,json} in %.0f s" % (csv_path, summ["elapsed_s"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
