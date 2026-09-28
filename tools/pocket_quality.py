#!/usr/bin/env python3
"""Quick serving quality gate: concurrent distinct requests -> WAV -> WER.

Sends every sentence of a fixed set to a running mynah-tts server at a chosen
concurrency, with staggered starts so the scheduler's gangs keep changing,
stores each stream as a WAV and transcribes it with faster-whisper. A request
that received another request's audio, or garbage from a corrupted codec
state, shows up as a high per-utterance WER.

    python tools/pocket_quality.py --port 18080 --concurrency 64 --out /root/evidence/q-tag

Offline tooling only; needs `faster-whisper` and `jiwer` in the environment.
"""
import argparse
import concurrent.futures
import http.client
import json
import os
import random
import re
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


def norm(s):
    s = s.lower().replace("-", " ")
    s = re.sub(r"[^a-z0-9' ]+", " ", s)
    return " ".join(s.split())


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
    ap.add_argument("--out", required=True)
    ap.add_argument("--phase", choices=["all", "synth", "asr"], default="all",
                    help="synth only, ASR only on a previous --out, or both")
    ap.add_argument("--asr-threads", type=int, default=min(32, os.cpu_count() or 4))
    a = ap.parse_args()
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


if __name__ == "__main__":
    main()
