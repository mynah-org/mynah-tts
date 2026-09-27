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

Output: <out>/<tag>-c<C>.jsonl (one record per request) and a summary JSON
line per level on stdout and in <out>/<tag>-summary.jsonl.
"""

import argparse
import http.client
import json
import math
import os
import random
import subprocess
import sys
import threading
import time

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


def pick_text(rng):
    r = rng.random()
    acc = 0.0
    for kind, w in MIX.items():
        acc += w
        if r <= acc:
            return kind, rng.choice(TEXTS[kind])
    return "long", TEXTS["long"][0]


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


def one_request(host, port, body, bytes_per_s, timeout):
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
        for pb in (0.25, 0.5):
            c, d = stalls(audio_chunks, pb)
            rec["stalls_%d" % int(pb * 1000)] = c
            rec["stall_s_%d" % int(pb * 1000)] = d
    rec["ok"] = rec.get("status") == 200 and "error" not in rec and nbytes > 0
    return rec


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


def run_level(a, conc, seed):
    rng = random.Random(seed)
    rec = Recorder()
    stop_load = threading.Event()
    t_start = time.monotonic()
    t_meas = t_start + a.warmup
    t_end = t_meas + a.duration
    req_seq = [0]
    seq_lock = threading.Lock()

    def body_for(i):
        r = random.Random(seed * 1000003 + i)
        kind, text = pick_text(r)
        return kind, {"model": "pocket", "input": text, "voice": a.voice,
                      "response_format": "pcm", "stream": True,
                      "seed": (seed * 7919 + i) % 1000000}

    def next_id():
        with seq_lock:
            req_seq[0] += 1
            return req_seq[0]

    def do_one(i):
        kind, body = body_for(i)
        rec.enter()
        r, chunks = one_request(a.host, a.port, body, a.bytes_per_s, a.timeout)
        r["id"] = i
        rec.leave(finish_record(r, chunks, a.bytes_per_s, kind))

    threads = []
    rejected_client = [0]
    if a.mode == "closed":
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
    m1 = parse_metrics(fetch(a.host, a.port, "/metrics"))
    stop.set()
    t_last = time.monotonic()

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
        "stalls_250": sum(r.get("stalls_250", 0) for r in ok),
        "stalls_500": sum(r.get("stalls_500", 0) for r in ok),
        "req_with_stall_250": sum(1 for r in ok if r.get("stalls_250")),
        "req_with_stall_500": sum(1 for r in ok if r.get("stalls_500")),
        "gpu_util_mean": (sum(g["util"] for g in gpu_m) / len(gpu_m)) if gpu_m else None,
        "gpu_mem_max_mib": max((g["mem_mib"] for g in gpu_m), default=None),
        "gpu_power_mean_w": (sum(g["power_w"] for g in gpu_m) / len(gpu_m)) if gpu_m else None,
        "gpu_sm_mhz_min": min((g["sm_mhz"] for g in gpu_m), default=None),
        "server_cpu_pct_mean": (sum(p["cpu_pct"] for p in proc_m) / len(proc_m)) if proc_m else None,
        "server_rss_max_mib": max((p["rss_mib"] for p in proc_m), default=None),
        "metrics_delta": delta,
        "wall_s": t_last - t_start,
    }
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
    a = ap.parse_args()
    a.bytes_per_s = a.sample_rate * 2
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
