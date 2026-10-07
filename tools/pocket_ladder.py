#!/usr/bin/env python3
"""Streaming capacity ladder for the Pocket HTTP server.

Stdlib only, so it runs on a bare GPU host. Every request is recorded with the
client-side arrival time of each PCM chunk, which is what streaming capacity
is about: TTFA, per-request streaming RTF and playback stalls are derived from
those timestamps, never from server counters.

Two load shapes:
  closed  : C workers, each sends its next request as soon as the previous
            one finished (offered concurrency = C).
  poisson : open-loop arrivals at --rate requests/s, capped at --max-open
            outstanding requests (anything above the cap is a client-side
            rejection and is reported as such).

Each level has a warmup window whose requests are discarded, then a
measurement window. A request belongs to the window in which it was sent.

Client capacity: one process holds one GIL, and at hundreds of streams it
reads thousands of chunks per second. --client-procs N (closed mode) spreads
the workers over N processes; every summary reports client_cpu_pct, the load
generator's own CPU. Run the same level at N=1 and N=4: if throughput moves,
the client was the ceiling.

Output: <out>/<tag>-c<C>.jsonl (one record per request) and a summary JSON
line per level on stdout and in <out>/<tag>-summary.jsonl.

Qualification options (all off by default; without them nothing changes):
  --corpus PATH     JSONL of {"id","kind","text"} used instead of the built-in
                    texts; kinds are drawn with --mix weights, and the
                    utterance and seed are a pure function of (--seed, request id).
  --voices a,b,c    one voice drawn per request, by seed (default: --voice).
  --save-audio DIR  keep the PCM exactly as the server streamed it under load
                    and write DIR/<tag>-c<C>-<reqid>.wav after the stream ends
                    (on a writer thread, off the request's timed path). The
                    per-request record gains the corpus id, text, voice, seed,
                    wav path and sample count. --save-every N keeps every Nth
                    request id; --save-max-mb stops saving past a disk cap.
"""

import argparse
import hashlib
import http.client
import json
import math
import multiprocessing
import os
import queue
import random
import subprocess
import sys
import threading
import time
import wave

TEXTS = {
    "short": [
        "Yes, that works for me.",
        "Thanks, I will call you back later.",
        "Sure, one moment please.",
        "No problem at all.",
    ],
    "conversational": [
        "Hi, thanks for calling. I can help you with your booking today, "
        "could you tell me your reference number?",
        "I checked your account and the payment went through this morning, "
        "so you should see it within a day or two.",
        "That sounds great. Let me confirm the details before we move on to "
        "the next step.",
    ],
    "medium": [
        "The train leaves the central station at a quarter past eight and "
        "arrives just before noon. If you prefer a quieter trip, the later "
        "service has more free seats, although it stops at every town along "
        "the coast.",
        "Our support team is available every day from nine in the morning to "
        "six in the evening. Outside those hours you can leave a message, and "
        "someone will get back to you as soon as possible on the next working "
        "day.",
    ],
    "long": [
        "When the old lighthouse was finally restored, the whole village came "
        "down to the harbour to watch the lamp being lit for the first time in "
        "forty years. Children climbed onto the sea wall, fishermen left their "
        "nets half mended, and the baker handed out warm bread to anyone who "
        "passed his door. As the light swept across the water, a few of the "
        "older residents admitted that they had never expected to see it "
        "shine again, and for a long moment nobody said anything at all.",
    ],
}

MIX = {"short": 0.25, "conversational": 0.40, "medium": 0.25, "long": 0.10}
CORPUS_MIX = "short=.25,conversational=.35,medium=.28,long=.12"


def builtin_pool():
    """The built-in texts as the (id, text) pool a corpus produces."""
    return {k: [("%s-%d" % (k, i), t) for i, t in enumerate(v)]
            for k, v in TEXTS.items()}


def load_corpus(path):
    pool, ids = {}, set()
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            r = json.loads(line)
            if r["id"] in ids:
                raise SystemExit("%s:%d: duplicate id %s" % (path, n, r["id"]))
            ids.add(r["id"])
            pool.setdefault(r["kind"], []).append((r["id"], r["text"]))
    if not pool:
        raise SystemExit("%s: empty corpus" % path)
    return pool


def parse_mix(spec, pool):
    mix = {}
    for part in spec.split(","):
        k, w = part.split("=")
        mix[k.strip()] = float(w)
    missing = [k for k, w in mix.items() if w > 0 and k not in pool]
    if missing:
        raise SystemExit("--mix kinds not in the corpus: %s" % ",".join(missing))
    total = sum(mix.values())
    if total <= 0:
        raise SystemExit("--mix weights sum to zero")
    return {k: w / total for k, w in mix.items() if w > 0}


def pick_text(rng, mix=MIX, pool=None):
    """(kind, id, text). Draws from rng exactly as the original built-in
    picker did, so a default run sends the same requests as before."""
    if pool is None:
        pool = builtin_pool()
    r = rng.random()
    acc = 0.0
    for kind, w in mix.items():
        acc += w
        if r <= acc:
            cid, text = rng.choice(pool[kind])
            return kind, cid, text
    kind = list(mix)[-1]
    cid, text = pool[kind][0]
    return kind, cid, text


def pct(values, p):
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * p / 100.0
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def stalls(chunks, prebuffer_s):
    """Simulate a player that starts after prebuffer_s of audio (or at end of
    stream) and stalls whenever the playhead reaches the received audio.
    chunks: list of (t_arrival, audio_seconds_in_chunk)."""
    if not chunks:
        return 0, 0.0
    total = sum(a for _, a in chunks)
    need = min(prebuffer_s, total)
    cum = 0.0
    start = None
    for t, a in chunks:
        cum += a
        if cum + 1e-9 >= need:
            start = t
            break
    count = 0
    stalled = 0.0
    cum = 0.0
    for t, a in chunks:
        if t > start:
            played = t - start - stalled
            if played > cum + 1e-6:
                count += 1
                stalled += played - cum
        cum += a
    return count, stalled


class Recorder:
    def __init__(self):
        self.lock = threading.Lock()
        self.records = []
        self.active = 0
        self.max_active = 0

    def enter(self):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)

    def leave(self, rec):
        with self.lock:
            self.active -= 1
            self.records.append(rec)


def one_request(host, port, body, bytes_per_s, timeout, keep=None):
    """keep: a list that receives each raw PCM chunk as it arrives (a
    reference append, no copy and no I/O), or None when not saving."""
    rec = {"t_send": time.monotonic()}
    chunks = []
    try:
        conn = http.client.HTTPConnection(host, port, timeout=timeout)
        conn.request("POST", "/v1/audio/speech", body=json.dumps(body),
                     headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        rec["t_headers"] = time.monotonic()
        rec["status"] = resp.status
        if resp.status != 200:
            rec["error"] = resp.read(512).decode("utf-8", "replace")
            conn.close()
            return rec, chunks
        while True:
            data = resp.read1(65536)
            if not data:
                break
            chunks.append((time.monotonic(), len(data)))
            if keep is not None:
                keep.append(data)
        conn.close()
    except Exception as e:  # timeouts and disconnects are data, not crashes
        rec["error"] = "%s: %s" % (type(e).__name__, e)
    rec["t_done"] = time.monotonic()
    return rec, chunks


def finish_record(rec, chunks, bytes_per_s, kind):
    rec["kind"] = kind
    nbytes = sum(n for _, n in chunks)
    audio = nbytes / bytes_per_s
    rec["bytes"] = nbytes
    rec["audio_s"] = audio
    rec["chunks"] = len(chunks)
    t0 = rec["t_send"]
    if chunks:
        rec["ttfa_s"] = chunks[0][0] - t0
        rec["ttfb_s"] = rec.get("t_headers", chunks[0][0]) - t0
        gen = rec["t_done"] - t0
        rec["rtf_e2e"] = gen / audio if audio > 0 else None
        first_a = chunks[0][1] / bytes_per_s
        rest = audio - first_a
        rec["rtf_stream"] = ((chunks[-1][0] - chunks[0][0]) / rest
                             if rest > 0.05 else None)
        gaps = [b[0] - a[0] for a, b in zip(chunks, chunks[1:])]
        rec["max_gap_s"] = max(gaps) if gaps else 0.0
        audio_chunks = [(t, n / bytes_per_s) for t, n in chunks]
        # The smallest delay after the first audio at which a player can start
        # and never run dry: chunk i (carrying audio from position cum_i) must
        # have arrived by start + cum_i.  safe_start = TTFA + that delay.
        cum, late = 0.0, 0.0
        for t, a in audio_chunks:
            late = max(late, (t - chunks[0][0]) - cum)
            cum += a
        rec["required_prebuffer_s"] = late
        rec["safe_start_s"] = rec["ttfa_s"] + late
        for pb in (0.25, 0.5):
            c, d = stalls(audio_chunks, pb)
            rec["stalls_%d" % int(pb * 1000)] = c
            rec["stall_s_%d" % int(pb * 1000)] = d
    rec["ok"] = rec.get("status") == 200 and "error" not in rec and nbytes > 0
    return rec


class AudioSaver:
    """Writes captured streams as WAV on one background thread, so a request
    worker never waits on the disk. The PCM is written exactly as received
    (an odd trailing byte, never expected, is dropped and flagged)."""

    def __init__(self, a):
        self.dir = a.save_audio
        self.every = max(1, a.save_every)
        self.cap = int(a.save_max_mb * 1048576)
        self.rate = a.sample_rate
        self.bytes = 0
        self.saved = 0
        self.skipped_cap = 0
        self.errors = 0
        self.closed = False
        self.lock = threading.Lock()
        self.q = queue.Queue()
        os.makedirs(self.dir, exist_ok=True)
        self.th = threading.Thread(target=self._run, daemon=True)
        self.th.start()

    def wants(self, reqid):
        return reqid % self.every == 0

    def submit(self, path, parts, nbytes):
        """Queue one finished stream; False once the disk cap is reached."""
        with self.lock:
            if self.closed or self.bytes + nbytes + 44 > self.cap:
                self.skipped_cap += 1
                return False
            self.bytes += nbytes + 44
            self.saved += 1
        self.q.put((path, parts))
        return True

    def _run(self):
        while True:
            item = self.q.get()
            if item is None:
                return
            path, parts = item
            try:
                pcm = b"".join(parts)
                with wave.open(path, "wb") as w:
                    w.setnchannels(1)
                    w.setsampwidth(2)
                    w.setframerate(self.rate)
                    w.writeframes(pcm[:len(pcm) & ~1])
            except Exception as e:
                with self.lock:
                    self.errors += 1
                print("save-audio: %s: %s" % (path, e), file=sys.stderr)

    def close(self):
        with self.lock:
            self.closed = True  # a straggler finishing later is not saved
        self.q.put(None)
        self.th.join()


def gpu_sampler(stop, out):
    q = ("utilization.gpu,memory.used,power.draw,clocks.sm,clocks.mem,"
         "temperature.gpu")
    while not stop.is_set():
        try:
            line = subprocess.run(
                ["nvidia-smi", "--query-gpu=" + q,
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5).stdout.strip()
            v = [float(x) for x in line.split(",")]
            out.append({"t": time.monotonic(), "util": v[0], "mem_mib": v[1],
                        "power_w": v[2], "sm_mhz": v[3], "mem_mhz": v[4],
                        "temp_c": v[5]})
        except Exception:
            pass
        stop.wait(1.0)


def proc_sampler(stop, pid, out):
    if not pid:
        return
    hz = os.sysconf("SC_CLK_TCK")
    last = None
    while not stop.is_set():
        try:
            with open("/proc/%d/stat" % pid) as f:
                st = f.read().rsplit(")", 1)[1].split()
            cpu = (int(st[11]) + int(st[12])) / hz
            with open("/proc/%d/status" % pid) as f:
                rss = next(int(l.split()[1]) for l in f if l.startswith("VmRSS"))
            now = time.monotonic()
            if last:
                out.append({"t": now,
                            "cpu_pct": 100.0 * (cpu - last[1]) / (now - last[0]),
                            "rss_mib": rss / 1024.0})
            last = (now, cpu)
        except Exception:
            pass
        stop.wait(1.0)


def fetch(host, port, path):
    try:
        conn = http.client.HTTPConnection(host, port, timeout=10)
        conn.request("GET", path)
        r = conn.getresponse()
        data = r.read().decode("utf-8", "replace")
        conn.close()
        return data
    except Exception as e:
        return "error: %s" % e


def parse_metrics(text):
    out = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 2:
            try:
                out[parts[0]] = float(parts[1])
            except ValueError:
                pass
    return out


def body_for(a, seed, i):
    """The request with id i: a pure function of (--seed, i), so a request id
    carries the same text, voice and seed whichever client process sends it."""
    r = random.Random(seed * 1000003 + i)
    kind, cid, text = pick_text(r, a.mix_w, a.pool)
    # Drawn, not i % len: --save-every N on ids would otherwise keep
    # only the voices whose index shares a factor with N.
    voice = r.choice(a.voice_list) if a.voice_list else a.voice
    return kind, cid, {"model": "pocket", "input": text, "voice": voice,
                       "response_format": "pcm", "stream": True,
                       "seed": (seed * 7919 + i) % 1000000}


def send_one(a, seed, i):
    """One request without audio capture, finished into its record."""
    kind, cid, body = body_for(a, seed, i)
    r, chunks = one_request(a.host, a.port, body, a.bytes_per_s, a.timeout)
    r["id"] = i
    finish_record(r, chunks, a.bytes_per_s, kind)
    if a.annotate:
        r["corpus_id"] = cid
        r["text"] = body["input"]
        r["voice"] = body["voice"]
        r["seed"] = body["seed"]
    return r


def closed_shard(a, conc, seed, shard, nshards, t_end, out_q):
    """A client process of --client-procs: `conc` closed-loop workers sending
    request ids shard+1, shard+1+nshards, ... until t_end (time.monotonic is
    one clock for every process of the host). Puts (records, max_active,
    cpu_s) on out_q. Each process has its own interpreter and GIL, which is
    the point: one process reading thousands of chunks per second on
    hundreds of threads can become the ceiling it is meant to measure."""
    rec = Recorder()
    seq = [0]
    seq_lock = threading.Lock()
    cpu0 = sum(os.times()[:2])

    def worker():
        while time.monotonic() < t_end:
            with seq_lock:
                i = seq[0] * nshards + shard + 1
                seq[0] += 1
            rec.enter()
            rec.leave(send_one(a, seed, i))

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(conc)]
    for th in threads:
        th.start()
    for th in threads:
        th.join(timeout=max(1.0, t_end - time.monotonic() + a.timeout))
    with rec.lock:
        records = list(rec.records)
    out_q.put((records, rec.max_active, sum(os.times()[:2]) - cpu0))


def run_level(a, conc, seed):
    rng = random.Random(seed)
    rec = Recorder()
    saver = AudioSaver(a) if a.save_audio else None
    t_start = time.monotonic()
    t_meas = t_start + a.warmup
    t_end = t_meas + a.duration
    req_seq = [0]
    seq_lock = threading.Lock()
    cpu0 = sum(os.times()[:2])
    shards = []

    def next_id():
        with seq_lock:
            req_seq[0] += 1
            return req_seq[0]

    def do_one(i):
        kind, cid, body = body_for(a, seed, i)
        keep = [] if saver is not None and saver.wants(i) else None
        rec.enter()
        r, chunks = one_request(a.host, a.port, body, a.bytes_per_s, a.timeout,
                                keep)
        r["id"] = i
        finish_record(r, chunks, a.bytes_per_s, kind)
        if a.annotate:
            r["corpus_id"] = cid
            r["text"] = body["input"]
            r["voice"] = body["voice"]
            r["seed"] = body["seed"]
        if keep is not None and r["bytes"] > 0:
            r["samples"] = r["bytes"] // 2
            if r["bytes"] & 1:
                r["odd_byte"] = True
            path = os.path.join(a.save_audio, "%s-c%d-%06d.wav" % (a.tag, conc, i))
            if saver.submit(path, keep, r["bytes"]):
                r["wav"] = os.path.abspath(path)
            else:
                r["wav_skipped"] = "save-max-mb"
        rec.leave(r)

    threads = []
    rejected_client = [0]
    if a.mode == "closed" and a.client_procs > 1:
        ctx = multiprocessing.get_context("spawn")
        out_q = ctx.Queue()
        n = min(a.client_procs, conc)
        for k in range(n):
            share = conc // n + (1 if k < conc % n else 0)
            p = ctx.Process(target=closed_shard,
                            args=(a, share, seed, k, n, t_end, out_q), daemon=True)
            p.start()
            shards.append(p)
    elif a.mode == "closed":
        def worker():
            while time.monotonic() < t_end:
                do_one(next_id())
        for _ in range(conc):
            th = threading.Thread(target=worker, daemon=True)
            th.start()
            threads.append(th)
    else:
        rate = a.rate if a.rate else conc / a.mean_request_s

        def arrivals():
            t = time.monotonic()
            while True:
                t += rng.expovariate(rate)
                if t >= t_end:
                    break
                d = t - time.monotonic()
                if d > 0:
                    time.sleep(d)
                if rec.active >= a.max_open:
                    rejected_client[0] += 1
                    continue
                th = threading.Thread(target=do_one, args=(next_id(),),
                                      daemon=True)
                th.start()
                threads.append(th)
        arr = threading.Thread(target=arrivals, daemon=True)
        arr.start()
        arr.join()

    gpu, proc = [], []
    stop = threading.Event()
    samplers = [threading.Thread(target=gpu_sampler, args=(stop, gpu),
                                 daemon=True),
                threading.Thread(target=proc_sampler,
                                 args=(stop, a.server_pid, proc), daemon=True)]
    for s in samplers:
        s.start()
    m0 = None
    while time.monotonic() < t_meas:
        time.sleep(0.1)
    m0 = parse_metrics(fetch(a.host, a.port, "/metrics"))
    for th in threads:
        th.join(timeout=max(1.0, t_end - time.monotonic() + a.timeout))
    client_cpu_s = sum(os.times()[:2]) - cpu0
    for _ in shards:
        try:
            recs, max_active, cpu_s = out_q.get(
                timeout=max(1.0, t_end - time.monotonic() + a.timeout + 30.0))
        except queue.Empty:
            print("client shard did not report", file=sys.stderr)
            continue
        rec.records.extend(recs)
        rec.max_active += max_active
        client_cpu_s += cpu_s
    for p in shards:
        p.join(timeout=10.0)
    m1 = parse_metrics(fetch(a.host, a.port, "/metrics"))
    stop.set()
    t_last = time.monotonic()
    if saver is not None:
        saver.close()

    meas = [r for r in rec.records if r["t_send"] >= t_meas]
    ok = [r for r in meas if r["ok"]]
    with open(os.path.join(a.out, "%s-c%d.jsonl" % (a.tag, conc)), "w") as f:
        for r in rec.records:
            r2 = dict(r)
            r2["measured"] = r["t_send"] >= t_meas
            for k in ("t_send", "t_headers", "t_done"):
                if k in r2:
                    r2[k] = r2[k] - t_start
            f.write(json.dumps(r2) + "\n")

    def stats(key, rows):
        v = [r[key] for r in rows if r.get(key) is not None]
        return {"p50": pct(v, 50), "p95": pct(v, 95), "p99": pct(v, 99),
                "max": max(v) if v else None}

    window = max(1e-6, max((r["t_done"] for r in meas), default=t_meas) - t_meas)
    gpu_m = [g for g in gpu if g["t"] >= t_meas]
    proc_m = [p for p in proc if p["t"] >= t_meas]
    delta = {k: m1[k] - m0.get(k, 0.0) for k in m1
             if isinstance(m1.get(k), float) and k in m0 and m1[k] != m0[k]}
    summary = {
        "tag": a.tag, "mode": a.mode, "offered": conc, "seed": seed,
        "warmup_s": a.warmup, "duration_s": a.duration,
        "sent": len(meas), "completed": len(ok),
        "failed": sum(1 for r in meas if not r["ok"]),
        # A stream that dies after its headers still ends a clean chunked
        # response, so the client cannot see it; the server's counter can.
        "server_jobs_failed": delta.get("mynah_server_jobs_failed_total", 0.0),
        "rejected_http": sum(1 for r in meas if r.get("status") in (429, 503)),
        "timeouts": sum(1 for r in meas if "timeout" in r.get("error", "").lower()),
        "rejected_client": rejected_client[0],
        "max_client_active": rec.max_active,
        "audio_s": sum(r["audio_s"] for r in ok),
        "audio_s_per_s": sum(r["audio_s"] for r in ok) / window,
        "ttfa": stats("ttfa_s", ok), "ttfb": stats("ttfb_s", ok),
        "rtf_stream": stats("rtf_stream", ok), "rtf_e2e": stats("rtf_e2e", ok),
        "max_gap": stats("max_gap_s", ok),
        "required_prebuffer": stats("required_prebuffer_s", ok),
        "safe_start": stats("safe_start_s", ok),
        "stalls_250": sum(r.get("stalls_250", 0) for r in ok),
        "stalls_500": sum(r.get("stalls_500", 0) for r in ok),
        "req_with_stall_250": sum(1 for r in ok if r.get("stalls_250")),
        "req_with_stall_500": sum(1 for r in ok if r.get("stalls_500")),
        "gpu_util_mean": (sum(g["util"] for g in gpu_m) / len(gpu_m)) if gpu_m else None,
        "gpu_mem_max_mib": max((g["mem_mib"] for g in gpu_m), default=None),
        "gpu_power_mean_w": (sum(g["power_w"] for g in gpu_m) / len(gpu_m)) if gpu_m else None,
        "gpu_sm_mhz_min": min((g["sm_mhz"] for g in gpu_m), default=None),
        "server_cpu_pct_mean": (sum(p["cpu_pct"] for p in proc_m) / len(proc_m)) if proc_m else None,
        # The load generator's own CPU over the level (warmup included), all
        # client processes summed: near 100 x client_procs it is the client,
        # not the server, that sets the ceiling.
        "client_procs": max(1, len(shards)),
        "client_cpu_pct": 100.0 * client_cpu_s / max(1e-6, t_last - t_start),
        "server_rss_max_mib": max((p["rss_mib"] for p in proc_m), default=None),
        "metrics_delta": delta,
        "wall_s": t_last - t_start,
    }
    if a.annotate:
        summary["corpus"] = os.path.abspath(a.corpus) if a.corpus else None
        summary["corpus_sha256"] = a.corpus_sha256
        summary["mix"] = a.mix_w
        summary["voices"] = a.voice_list or [a.voice]
    if saver is not None:
        summary["save_audio"] = os.path.abspath(a.save_audio)
        summary["save_every"] = saver.every
        summary["saved_wavs"] = saver.saved
        summary["saved_mb"] = saver.bytes / 1048576.0
        summary["save_skipped_cap"] = saver.skipped_cap
        summary["save_errors"] = saver.errors
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18080)
    ap.add_argument("--levels", default="1,2,4,8")
    ap.add_argument("--mode", choices=("closed", "poisson"), default="closed")
    ap.add_argument("--rate", type=float, default=0.0,
                    help="poisson arrivals/s (default: C / --mean-request-s)")
    ap.add_argument("--mean-request-s", type=float, default=6.0)
    ap.add_argument("--max-open", type=int, default=256)
    ap.add_argument("--warmup", type=float, default=20.0)
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--voice", default="alba")
    ap.add_argument("--sample-rate", type=int, default=24000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--server-pid", type=int, default=0)
    ap.add_argument("--out", default="ladder-out")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--stop-rtf-p95", type=float, default=3.0,
                    help="stop escalating once streaming RTF p95 exceeds this")
    ap.add_argument("--corpus", default="",
                    help="JSONL corpus {id,kind,text} instead of the built-in texts")
    ap.add_argument("--mix", default="",
                    help="kind weights (default with --corpus: %s)" % CORPUS_MIX)
    ap.add_argument("--voices", default="",
                    help="comma list of voices, one drawn per request (by seed)")
    ap.add_argument("--save-audio", default="",
                    help="directory for WAVs of the audio as streamed")
    ap.add_argument("--save-every", type=int, default=1,
                    help="save every Nth request id (default 1: all)")
    ap.add_argument("--save-max-mb", type=float, default=4000.0,
                    help="stop saving once this many MB were written")
    ap.add_argument("--client-procs", type=int, default=1,
                    help="closed mode: spread the C workers over N client "
                         "processes (one GIL each); default 1")
    a = ap.parse_args()
    if a.client_procs > 1 and (a.mode != "closed" or a.save_audio):
        ap.error("--client-procs > 1 needs --mode closed and no --save-audio")
    a.bytes_per_s = a.sample_rate * 2
    a.voice_list = [v.strip() for v in a.voices.split(",") if v.strip()]
    a.corpus_sha256 = None
    if a.corpus:
        with open(a.corpus, "rb") as f:
            a.corpus_sha256 = hashlib.sha256(f.read()).hexdigest()
        a.pool = load_corpus(a.corpus)
        a.mix_w = parse_mix(a.mix or CORPUS_MIX, a.pool)
    else:
        a.pool = builtin_pool()
        a.mix_w = parse_mix(a.mix, a.pool) if a.mix else MIX
    a.annotate = bool(a.corpus or a.save_audio or a.voice_list or a.mix)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "%s-health.json" % a.tag), "w") as f:
        f.write(fetch(a.host, a.port, "/health"))
    for conc in [int(x) for x in a.levels.split(",")]:
        s = run_level(a, conc, a.seed)
        line = json.dumps(s)
        print(line, flush=True)
        with open(os.path.join(a.out, "%s-summary.jsonl" % a.tag), "a") as f:
            f.write(line + "\n")
        p95 = s["rtf_stream"]["p95"]
        if s["completed"] == 0 or (p95 is not None and p95 > a.stop_rtf_p95):
            print("stopping escalation at C%d" % conc, file=sys.stderr)
            break


if __name__ == "__main__":
    main()
