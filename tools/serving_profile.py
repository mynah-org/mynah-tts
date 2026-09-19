#!/usr/bin/env python3
"""serving_profile.py -- concurrency profile of the mynah-tts HTTP server, with verdicts.

    # SCREEN: start a server on a scratch port, screen C1, C2, C4, tear it down
    python3 tools/serving_profile.py --mode wave --model models/pocket-en \
        --levels 1,2,4 --waves 3

    # QUALIFY: five measured minutes per level, after thirty discarded seconds
    python3 tools/serving_profile.py --mode soak --model models/pocket-en \
        --levels 4 --soak-seconds 300 --warmup-seconds 30 --window-seconds 60

    # attach to a server someone else is running
    python3 tools/serving_profile.py --url http://127.0.0.1:8973 --levels 1,2,4

WAVE and SOAK are different measurements and this tool refuses to conflate them
---------------------------------------------------------------------------------
**WAVE** fires all C requests at t=0, waits for them, repeats ``--waves`` times.  It
measures a burst arriving at an idle server.  It is a SCREEN: it is cheap, it
disqualifies fast, and it MAY NOT PROMOTE a configuration.  Every wave report says so.

**SOAK** keeps C requests in flight continuously for ``--soak-seconds`` after
``--warmup-seconds`` of discarded traffic, and cuts the measured span into windows so a
metric that WALKS is visible as a trend rather than averaged into a respectable mean.
A soak whose STREAM_RTF or prebuffer drifts is downgraded however good its aggregate.
Only a soak may promote.

The gap is not cosmetic: in the reference a configuration passed the wave screen at
STREAM_RTF 0.919 and failed a 30-minute soak at 1.004 with 596 rejects and 111 broken
pipes.  "C16 is the hard-capacity boundary, not a product point."

What it answers
---------------
Not "how fast is the server" but "at which concurrency can a real player still play".
For every concurrency level it reports:

  TTFB            time to the first byte of the response (the header).
  TTFA            time to the first byte of AUDIO.  Different from TTFB by construction:
                  this server sends the streaming header from the writer thread before
                  synthesis starts, so TTFB can be milliseconds while TTFA is seconds.
                  Reporting one for the other is the classic way to publish a latency
                  that no listener experiences.
  STREAM_RTF      (t_done - t_first_audio) / audio delivered after the first chunk.
                  A mean rate.  It is CAPACITY, and it is NOT a proof of continuity:
                  see tests/playback_sim.py and qwen-tts
                  .work/professional-streaming-architecture.md (E1/E2, "the cadence law").
  prebuffer       how long a player must hold back after first audio to never starve.
  safe_play_start the earliest instant it could press play at all (TTFA + prebuffer).
  stall rate      share of requests in which a player with a B-millisecond jitter buffer
                  runs dry at least once.
  max gap         the worst inter-arrival gap -- the raw cadence behind the above.
  aggregate       audio seconds served per wall second, and the inverse.

and then a VERDICT per level -- GOOD / MARGINAL / NOT STREAMABLE / INCONCLUSIVE -- with
every threshold printed next to the value that was compared against it, so the verdict
can be falsified by reading the table instead of trusting it.

Which metric gates
------------------
STREAM_RTF is CAPACITY.  It is mandatory (< 1) and preferred (<= 0.90), and it is never
allowed to promote on its own.  **required_prebuffer p95 and stall_rate@250 are what
qualify.**  Every quantum sweep on record moved those two while STREAM_RTF sat still --
in the reference the smallest quantum FAILED on STREAM_RTF while the largest PASSED on
it and stalled half the time.  Aggregate rate alone selects the wrong serving point.

The refusals (.work/engineering-method.md section 4)
----------------------------------------------------
* **Coalesced reads.** A mark is stamped when read() returns; a late reader finds
  several chunks queued and returns them microseconds apart, so N emissions become N
  marks in one instant.  Above ``--coalesced-refuse`` the cadence percentiles describe
  the CLIENT, and this tool exits 3 instead of printing them.  The reference declared a
  run with 33-37% coalesced reads not quotable.
* **Dispatch resolution.** ``--compare`` refuses to difference two runs whose engine,
  ISA, quantization, thread count, backend, build flags or route differ.  A sweep that
  changes two things has measured their sum.
* Neither refusal is a warning.  Both exit non-zero.

Engine-agnostic by construction
-------------------------------
Nothing here knows Magpie or PocketTTS.  Sample rate, channels and sample width come from
what the server announced on the wire (``X-Sample-Rate`` / ``X-Channels`` /
``X-Bits-Per-Sample`` on the streaming route, the WAV ``fmt`` chunk on the batch route).
If the server announces nothing, the profile REFUSES to run rather than assuming a rate:
a wrong rate turns every second of audio into a wrong RTF and a wrong verdict.

Measurement honesty
-------------------
* Every aggregate prints n, p50, p95, mean and sd.  A p50 with no spread hides bimodality,
  which is exactly the shape a serialized server produces.
* Requests launched and requests completed are reported separately.  If a level completed
  fewer than it launched, the level is NOT STREAMABLE and the report says so instead of
  quietly averaging the survivors.
* Timings are client-observed: a mark is stamped when a recv() hands the harness a
  complete HTTP chunk.  The time each recv spent blocked is recorded, and reads that
  returned in under a millisecond (already-queued data) are counted and reported, because
  those marks are late and make cadence numbers an upper bound on server lateness.
* A known limit of this server: streaming synthesis is serialized by a global mutex, one
  stream at a time per process.  C2+ in streaming mode is therefore expected to look bad.
  The point of this tool is to quantify that before it is removed, not to route around it.

Only the standard library is used.  Raw sockets, not an HTTP client library: the framing
is parsed here so that a chunk mark is stamped at the recv that completed it.
"""

import argparse
import json
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests"))
import playback_sim as PB  # noqa: E402


# --------------------------------------------------------------------------------------
# defaults
# --------------------------------------------------------------------------------------
# THE DEFAULT BANK IS A FILE, and the five lines below are only what is left when
# that file is missing.
#
# What those five lines were qualifying: five sentences, in ITALIAN, fed to
# whatever pack the run named -- which for models/pocket-en is an English pack.
# A C100 measured that way is not false, but it is qualified on a workload
# nobody will ever serve: five texts repeating, in the wrong language, with a
# duration spread that comes out of how an English tokenizer digests Italian.
#
# tests/load_texts_en_v2.txt is 277 texts in five classes with a version and a
# sha in its own header, so a number can name the corpus that produced it. It
# comes from the qwen-tts project (MIT) and is kept byte-identical there and
# here, which is what lets a number cross between them.
DEFAULT_BANK_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tests", "load_texts_en_v2.txt")

# The fallback: a neutral bank, short/medium/long, so a level is not qualified on one length.
# Text length is only one of the two duration knobs; --max-steps is the other, and the
# synthetic pack needs it (its model.json caps generation at 8 steps ~ 0.37 s of audio,
# which is too short for any cadence to exist).
DEFAULT_BANK = [
    ("short", "Buongiorno, come stai?"),
    ("short", "Il treno parte alle nove."),
    ("medium", "La riproduzione continua dipende dalla cadenza di arrivo, non dalla media."),
    ("medium", "Un flusso puo essere piu veloce del tempo reale e far comunque inceppare il player."),
    ("long", "Questa frase e deliberatamente piu lunga delle altre perche un profilo "
             "di concorrenza qualificato su una sola lunghezza non dice nulla su come "
             "il server si comporta quando le richieste hanno durate diverse fra loro."),
]

# The streaming envelope lives in tests/playback_sim.py (PB.ENVELOPE) so that the
# harness and the self-test cannot drift apart.  These are the command-line defaults,
# and every one of them is a flag.
DEFAULTS = {
    "rtf_hard": PB.ENVELOPE["rtf_hard"],
    "rtf_pref": PB.ENVELOPE["rtf_pref"],
    "ttfb_pref_ms": PB.ENVELOPE["ttfb_pref_ms"],
    "ttfa_pref_ms": PB.ENVELOPE["ttfa_pref_ms"],
    "prebuffer_pref_ms": PB.ENVELOPE["prebuffer_pref_ms"],
    "safe_start_pref_ms": PB.ENVELOPE["safe_start_pref_ms"],
    "stall_gate_ms": PB.ENVELOPE["stall_pref_ms"],           # 250 -- the qualifying gate
    "stall_mandatory_ms": PB.ENVELOPE["stall_mandatory_ms"],  # 500 -- the hard gate
    "coalesced_refuse": PB.COALESCED_REFUSE_SHARE,
    "coalesced_warn": PB.COALESCED_WARN_SHARE,
    "drift_rtf_tol": 0.05,
    "drift_prebuffer_tol_ms": 150.0,
    "drift_min_windows": 3,
}


# --------------------------------------------------------------------------------------
# HTTP over a raw socket -- so the per-chunk marks are the client's true receive instants
# --------------------------------------------------------------------------------------
class HttpError(Exception):
    pass


def _recv_some(sock, out_marks_time):
    """recv once, returning (data, t_after, blocked_s)."""
    t0 = time.perf_counter()
    data = sock.recv(65536)
    t1 = time.perf_counter()
    return data, t1, t1 - t0


def http_post_stream(host, port, path, payload, timeout):
    """POST ``payload`` (bytes) and read the response, stamping every chunk arrival.

    Returns a dict with ``status``, ``headers``, ``t_send``, ``ttfb_s`` (first response
    byte), ``marks`` [(t_rel, nbytes, blocked_s)], ``t_done_s``, ``body_bytes`` and
    ``chunked``.  Times are relative to the instant the request was written.

    The HTTP framing is decoded here on purpose.  A library reader hands back "some
    bytes"; the question this tool asks is when the SERVER's chunk became available, so a
    mark is stamped exactly when a chunk's payload is complete, and the recv that
    completed it reports how long it was blocked (a sub-millisecond block means the data
    was already queued and the mark is late).
    """
    req = (("POST %s HTTP/1.1\r\n"
            "Host: %s:%d\r\n"
            "Content-Type: application/json\r\n"
            "Content-Length: %d\r\n"
            "Connection: close\r\n\r\n") % (path, host, port, len(payload))).encode()
    sock = socket.create_connection((host, port), timeout)
    try:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.settimeout(timeout)
        t_send = time.perf_counter()
        sock.sendall(req + payload)

        buf = b""
        ttfb = None
        head_end = -1
        # --- headers -------------------------------------------------------------------
        while head_end < 0:
            data, t, _blocked = _recv_some(sock, t_send)
            if not data:
                raise HttpError("connection closed before the response header")
            if ttfb is None:
                ttfb = t - t_send
            buf += data
            head_end = buf.find(b"\r\n\r\n")
        head = buf[:head_end].decode("latin-1")
        buf = buf[head_end + 4:]
        lines = head.split("\r\n")
        m = re.match(r"HTTP/1\.[01] (\d{3})", lines[0])
        if not m:
            raise HttpError("malformed status line: %r" % lines[0][:80])
        status = int(m.group(1))
        headers = {}
        for ln in lines[1:]:
            if ":" in ln:
                k, v = ln.split(":", 1)
                headers[k.strip().lower()] = v.strip()
        chunked = headers.get("transfer-encoding", "").lower() == "chunked"
        clen = int(headers["content-length"]) if "content-length" in headers else None

        marks = []
        body = bytearray()
        done = False

        def consume_chunked(t, blocked):
            """Pull every complete chunk out of ``buf``; the first one stamped at this
            recv owns its blocked time, the rest were already queued behind it."""
            nonlocal buf, done
            first = True
            while True:
                nl = buf.find(b"\r\n")
                if nl < 0:
                    return
                size_line = buf[:nl].split(b";")[0].strip()
                try:
                    size = int(size_line, 16)
                except ValueError:
                    raise HttpError("bad chunk size %r" % size_line[:32])
                if len(buf) < nl + 2 + size + 2:
                    return                       # payload (or its CRLF) not here yet
                payload_ = bytes(buf[nl + 2:nl + 2 + size])
                buf = buf[nl + 2 + size + 2:]
                if size == 0:
                    done = True
                    return
                body.extend(payload_)
                marks.append((t - t_send, size, blocked if first else 0.0))
                first = False

        if chunked:
            consume_chunked(time.perf_counter(), 0.0)   # anything already in the header recv
            while not done:
                data, t, blocked = _recv_some(sock, t_send)
                if not data:
                    break                        # truncated stream: reported by the caller
                buf += data
                consume_chunked(t, blocked)
        else:
            # Content-Length (or close-delimited): the whole body is one arrival.  A batch
            # response has no cadence by definition, and the record will say so.
            t_first_body = None
            if buf:
                t_first_body = time.perf_counter()
                body.extend(buf)
            while clen is None or len(body) < clen:
                data, t, _blocked = _recv_some(sock, t_send)
                if not data:
                    break
                if t_first_body is None:
                    t_first_body = t
                body.extend(data)
            if body:
                marks.append(((t_first_body or time.perf_counter()) - t_send,
                              len(body), None))
            done = True

        t_done = time.perf_counter() - t_send
        return {"status": status, "headers": headers, "ttfb_s": ttfb, "marks": marks,
                "t_done_s": t_done, "body": bytes(body), "chunked": chunked,
                "complete": done, "t_send_abs": t_send}
    finally:
        try:
            sock.close()
        except OSError:
            pass


def http_get_json(host, port, path, timeout=5.0):
    req = (("GET %s HTTP/1.1\r\nHost: %s:%d\r\nConnection: close\r\n\r\n")
           % (path, host, port)).encode()
    with socket.create_connection((host, port), timeout) as sock:
        sock.settimeout(timeout)
        sock.sendall(req)
        buf = b""
        while True:
            data = sock.recv(65536)
            if not data:
                break
            buf += data
    head_end = buf.find(b"\r\n\r\n")
    if head_end < 0:
        raise HttpError("no header in the reply to %s" % path)
    body = buf[head_end + 4:]
    head = buf[:head_end].decode("latin-1")
    if head.startswith("HTTP/1.1 ") and not head.startswith("HTTP/1.1 200"):
        raise HttpError("%s -> %s" % (path, head.split("\r\n")[0]))
    if "transfer-encoding: chunked" in head.lower():
        out, i = b"", 0
        while i < len(body):
            nl = body.find(b"\r\n", i)
            if nl < 0:
                break
            size = int(body[i:nl].split(b";")[0].strip(), 16)
            if size == 0:
                break
            out += body[nl + 2:nl + 2 + size]
            i = nl + 2 + size + 2
        body = out
    return json.loads(body.decode("utf-8"))


# --------------------------------------------------------------------------------------
# audio format discovery -- from the wire, never from a constant
# --------------------------------------------------------------------------------------
def format_from_headers(headers):
    if "x-sample-rate" not in headers:
        return None
    try:
        return PB.AudioFormat(int(headers["x-sample-rate"]),
                              int(headers.get("x-channels", 1)),
                              int(headers.get("x-bits-per-sample", 16)))
    except (ValueError, KeyError):
        return None


def format_from_wav(body):
    """Parse the RIFF ``fmt `` chunk.  Returns (AudioFormat, pcm_offset) or None."""
    if len(body) < 44 or body[:4] != b"RIFF" or body[8:12] != b"WAVE":
        return None
    i = 12
    fmt = None
    while i + 8 <= len(body):
        cid = body[i:i + 4]
        size = int.from_bytes(body[i + 4:i + 8], "little")
        payload = body[i + 8:i + 8 + size]
        if cid == b"fmt " and size >= 16:
            channels = int.from_bytes(payload[2:4], "little")
            rate = int.from_bytes(payload[4:8], "little")
            bits = int.from_bytes(payload[14:16], "little")
            fmt = PB.AudioFormat(rate, channels, bits)
        elif cid == b"data" and fmt is not None:
            return fmt, i + 8, size
        i += 8 + size + (size & 1)
    return None


# --------------------------------------------------------------------------------------
# the load generator
# --------------------------------------------------------------------------------------
class Profile:
    def __init__(self, args, bank, health):
        self.args = args
        self.bank = bank
        self.health = health
        self.fmt = None                 # discovered on the first successful response
        self.fmt_source = None
        self.fmt_lock = threading.Lock()
        if args.sample_rate:
            self.fmt = PB.AudioFormat(args.sample_rate, args.channels, args.bits)
            self.fmt_source = "--sample-rate override"

    # -- one request ------------------------------------------------------------------
    def body_for(self, index):
        cls, text = self.bank[index % len(self.bank)]
        req = {"input": text, "stream": bool(self.args.stream),
               "seed": self.args.seed + (index if self.args.vary_seed else 0)}
        if self.args.voice:
            req["voice"] = self.args.voice
        if self.args.language:
            req["language"] = self.args.language
        if self.args.max_steps:
            req["max_steps"] = self.args.max_steps
        if self.args.stream:
            req["response_format"] = "pcm"
        if self.args.extra_json:
            req.update(json.loads(self.args.extra_json))
        return cls, text, json.dumps(req).encode("utf-8")

    def run_one(self, level, wave, index):
        cls, text, payload = self.body_for(index)
        rec = {"level": level, "wave": wave, "index": index, "class": cls,
               "text_chars": len(text), "ok": False, "error": None}
        try:
            r = http_post_stream(self.args.host, self.args.port, self.args.route,
                                 payload, self.args.timeout)
        except (OSError, HttpError) as e:
            rec["error"] = "%s: %s" % (type(e).__name__, e)
            return rec
        rec["status"] = r["status"]
        rec["t_send_abs"] = r["t_send_abs"]
        rec["t_done_abs"] = r["t_send_abs"] + r["t_done_s"]
        rec["ttfb_s"] = r["ttfb_s"]
        rec["t_done_s"] = r["t_done_s"]
        if r["status"] != 200:
            rec["error"] = "HTTP %d: %s" % (r["status"], r["body"][:160].decode("latin-1", "replace"))
            return rec
        if not r["complete"]:
            rec["error"] = "truncated response (%d bytes, no terminating chunk)" % len(r["body"])
            return rec

        marks = r["marks"]
        fmt = format_from_headers(r["headers"])
        source = "X-Sample-Rate header"
        if fmt is None:
            w = format_from_wav(r["body"])
            if w is not None:
                fmt, data_off, data_len = w
                source = "WAV fmt chunk"
                # The WAV header is not audio: re-stamp the single arrival with the PCM
                # length only, so RTF is not inflated by 44 bytes of metadata.
                marks = [(marks[0][0], data_len, marks[0][2])] if marks else []
        if fmt is None:
            with self.fmt_lock:
                fmt, source = self.fmt, self.fmt_source
        if fmt is None:
            rec["error"] = ("the server announced no audio format (no X-Sample-Rate "
                            "header, no WAV header): refusing to guess a sample rate")
            return rec
        with self.fmt_lock:
            if self.fmt is None:
                self.fmt, self.fmt_source = fmt, source
            elif self.fmt != fmt:
                rec["error"] = "audio format changed mid-profile: %r vs %r" % (self.fmt, fmt)
                return rec

        # Every playback metric -- TTFB, TTFA, STREAM_RTF, prebuffer, safe_play_start,
        # the stall rates, max_gap and the coalesced share -- is defined in
        # tests/playback_sim.py and computed there.  Nothing is recomputed here: a second
        # definition of a metric is a second product.
        k = PB.timeline_kpis(marks, r["t_done_s"], fmt, self.args.buffers,
                             ttfb_s=r["ttfb_s"])
        rec.update(k)
        rec["ok"] = True
        if self.args.marks:
            rec["marks"] = marks
        return rec

    # -- SOAK: fixed concurrency for a duration ----------------------------------------
    def run_soak(self, level):
        """``level`` workers issuing requests back to back for ``--soak-seconds``.

        This is NOT a wave.  A wave is C requests fired together and then silence, which
        measures a burst arriving at an idle server.  A soak keeps C requests in flight
        continuously, so the server is never allowed to catch up between them, caches
        settle, memory grows if it is going to grow, and thermal and scheduler effects
        have time to appear.  The first ``--warmup-seconds`` are RUN but DISCARDED, and
        the remainder is cut into windows so that a metric which walks over time is
        visible as a trend instead of being averaged into a respectable mean.

        Only a soak may promote a configuration.  The reference had one pass a wave
        screen at 0.919 and fail a 30-minute soak at 1.004 with 596 rejects.
        """
        records = []
        counter = [0]
        counter_lock = threading.Lock()
        t0 = time.perf_counter()
        t_warm_end = t0 + self.args.warmup_seconds
        t_end = t_warm_end + self.args.soak_seconds
        barrier = threading.Barrier(level)

        def worker(slot):
            barrier.wait()
            while time.perf_counter() < t_end:
                with counter_lock:
                    idx = counter[0]
                    counter[0] += 1
                t_send = time.perf_counter()
                rec = self.run_one(level, -1, idx if self.args.vary_text else slot)
                rec["t_send_rel"] = t_send - t0
                # A request is warm-up if it STARTED during warm-up.  Judging by
                # completion would let a long request started under load be discarded
                # because it happened to finish late.
                rec["warmup"] = t_send < t_warm_end
                rec["slot"] = slot
                records.append(rec)

        threads = [threading.Thread(target=worker, args=(i,), daemon=True)
                   for i in range(level)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(self.args.timeout + self.args.warmup_seconds
                   + self.args.soak_seconds + 60.0)
        wall = time.perf_counter() - t0
        measured = [r for r in records if not r.get("warmup")]
        return records, measured, wall, t_warm_end - t0

    def soak_windows(self, measured):
        """Cut the measured span into ``--window-seconds`` windows of OK records."""
        if not measured:
            return []
        w = self.args.window_seconds
        if w <= 0:
            return []
        base = min(r["t_send_rel"] for r in measured)
        buckets = {}
        for r in measured:
            if not r.get("ok"):
                continue
            buckets.setdefault(int((r["t_send_rel"] - base) // w), []).append(r)
        return [(i, buckets[i]) for i in sorted(buckets)]

    # -- one concurrency level ---------------------------------------------------------
    def run_level(self, level):
        records = []
        launched = 0
        t_level0 = time.perf_counter()
        for wave in range(self.args.waves):
            barrier = threading.Barrier(level)
            out = [None] * level
            base = wave * level if self.args.vary_text else 0

            def worker(slot):
                barrier.wait()
                out[slot] = self.run_one(level, wave, base + slot)

            threads = [threading.Thread(target=worker, args=(i,), daemon=True)
                       for i in range(level)]
            for t in threads:
                t.start()
            launched += level
            for t in threads:
                t.join(self.args.timeout + 30.0)
            for i, r in enumerate(out):
                records.append(r if r is not None else
                               {"level": level, "wave": wave, "index": base + i,
                                "ok": False, "error": "worker did not return"})
            if self.args.wave_gap > 0 and wave + 1 < self.args.waves:
                time.sleep(self.args.wave_gap)
        wall = time.perf_counter() - t_level0
        return records, launched, wall


# --------------------------------------------------------------------------------------
# aggregation and verdict
# --------------------------------------------------------------------------------------
def level_report(level, records, launched, wall, args, mode="wave", windows=None,
                 warmup_s=0.0, warmup_n=0, bank_len=0):
    ok = [r for r in records if r.get("ok")]
    bad = [r for r in records if not r.get("ok")]
    rep = {"level": level, "launched": launched, "completed": len(ok),
           "failed": len(bad), "wall_s": wall, "mode": mode,
           "warmup_s": warmup_s, "warmup_discarded": warmup_n,
           "errors": [r.get("error") for r in bad][:10]}

    rep["summary"] = PB.summarize(ok, args.buffers) if ok else {"n_records": 0,
                                                                "n_cadence": 0}
    rep["ttfb_ms"] = PB.spread([r.get("ttfb_s", float("nan")) * 1000.0 for r in ok])
    rep["ttfa_ms"] = PB.spread([r.get("ttfa_s", float("nan")) * 1000.0 for r in ok])
    rep["header_to_audio_ms"] = PB.spread([r.get("header_to_audio_s", float("nan")) * 1000.0
                                           for r in ok])
    rep["prebuffer_ms"] = PB.spread([r.get("required_prebuffer_s", float("nan")) * 1000.0
                                     for r in ok])
    rep["safe_play_start_ms"] = PB.spread([r.get("safe_play_start_s", float("nan")) * 1000.0
                                           for r in ok])
    rep["max_gap_ms"] = PB.spread([r.get("max_gap_s", float("nan")) * 1000.0 for r in ok])
    rep["stream_rtf"] = PB.spread([r.get("stream_rtf", float("nan")) for r in ok])
    rep["audio_s"] = PB.spread([r.get("delivered_s", float("nan")) for r in ok])
    rep["chunks"] = PB.spread([float(r.get("chunks", 0)) for r in ok])
    rep["request_wall_s"] = PB.spread([r.get("t_done_s", float("nan")) for r in ok])

    audio_total = sum(r.get("delivered_s", 0.0) for r in ok)
    rep["audio_total_s"] = audio_total
    # Aggregate over the level's own wall clock: how much audio the server actually
    # produced per second of wall while it was under this much load.
    rep["throughput_audio_s_per_s"] = audio_total / wall if wall > 0 else float("nan")
    rep["aggregate_rtf"] = wall / audio_total if audio_total > 0 else float("nan")
    for b in args.buffers:
        rep["stall_rate@%d" % b] = rep["summary"].get("stall_rate@%d" % b, float("nan"))
    rep["coalesced_chunk_share"] = rep["summary"].get("coalesced_chunk_share", float("nan"))
    mix = {}
    for r in records:
        mix[r.get("class", "?")] = mix.get(r.get("class", "?"), 0) + 1
    rep["mix"] = mix
    # How much of the bank was actually SPOKEN.  The request index is a monotonic
    # counter, so counting distinct indices only ever returns the request count --
    # it says nothing about text diversity, which is the thing a bank exists to
    # provide.  The text is bank[index % len(bank)], so that is what gets counted.
    idx = [r.get("index") for r in records if r.get("index") is not None]
    rep["request_slots"] = len(set(idx))
    rep["bank_size"] = bank_len
    rep["distinct_texts"] = (len({i % bank_len for i in idx}) if bank_len
                             else rep["request_slots"])
    rep["reuse_per_text"] = (len(idx) / rep["distinct_texts"]
                             if rep["distinct_texts"] else float("nan"))

    # -- the refusal: may these cadence percentiles be quoted at all? -------------------
    status, share, reasons = PB.quotable(rep["summary"], args.coalesced_refuse,
                                         args.coalesced_warn)
    rep["quotable"] = status
    rep["quotable_reasons"] = reasons
    rep["coalesced_chunk_share"] = share

    # -- the soak drift gate: a screen has none, and that is the difference -------------
    rep["windows"] = []
    rep["drift"] = None
    if mode == "soak" and windows:
        for i, recs in windows:
            ws = PB.summarize(recs, args.buffers)
            rep["windows"].append({
                "window": i, "n": len(recs),
                "stream_rtf_p95": ws.get("stream_rtf", {}).get("p95", float("nan")),
                "prebuffer_p95_ms": ws.get("required_prebuffer_s", {})
                                      .get("p95", float("nan")) * 1000.0,
                "ttfa_p95_ms": ws.get("ttfa_s", {}).get("p95", float("nan")) * 1000.0,
                "stall_rate@%d" % args.stall_gate_ms:
                    ws.get("stall_rate@%d" % args.stall_gate_ms, float("nan")),
                "summary": ws})
        wsums = [w["summary"] for w in rep["windows"]]
        rep["drift"] = {
            "stream_rtf": PB.drift_gate(wsums, "stream_rtf", "p95",
                                        args.drift_rtf_tol, args.drift_min_windows),
            "required_prebuffer_s": PB.drift_gate(wsums, "required_prebuffer_s", "p95",
                                                  args.drift_prebuffer_tol_ms / 1000.0,
                                                  args.drift_min_windows),
        }

    env = {"rtf_hard": args.rtf_hard, "rtf_pref": args.rtf_pref,
           "ttfb_pref_ms": args.ttfb_pref_ms, "ttfa_pref_ms": args.ttfa_pref_ms,
           "prebuffer_pref_ms": args.prebuffer_pref_ms,
           "safe_start_pref_ms": args.safe_start_pref_ms,
           "stall_pref_ms": args.stall_gate_ms,
           "stall_mandatory_ms": args.stall_mandatory_ms}
    rep["verdict"], rep["gates"] = PB.qualify(rep["summary"], rep["completed"],
                                              rep["launched"], env)

    # A drifting soak has not qualified anything, whatever its aggregate says.
    if rep["drift"]:
        drifted = [k for k, d in rep["drift"].items() if d["pass"] is False]
        if drifted and rep["verdict"] in ("GOOD", "MARGINAL"):
            rep["verdict"] = "NOT STREAMABLE"
            rep["verdict_note"] = ("the soak drifted on %s: the aggregate is not a "
                                   "steady state" % ", ".join(drifted))
        elif any(d["pass"] is None for d in rep["drift"].values()) and \
                rep["verdict"] == "GOOD":
            rep["verdict"] = "INCONCLUSIVE"
            rep["verdict_note"] = ("too few soak windows to test drift; a soak that "
                                   "cannot show a trend has not qualified anything")

    # A run whose cadence cannot be quoted cannot produce a cadence verdict either.
    if rep["quotable"] == "NOT QUOTABLE":
        rep["verdict_before_refusal"] = rep["verdict"]
        rep["verdict"] = "NOT QUOTABLE"
        rep["verdict_note"] = "; ".join(rep["quotable_reasons"])

    # A WAVE is a screen.  It may disqualify; it may never promote.
    rep["may_promote"] = (mode == "soak")
    return rep


# The verdict, the gates, the refusals and the drift gate all live in
# tests/playback_sim.py -- see PB.qualify / PB.quotable / PB.comparable / PB.drift_gate.
# They are NOT reimplemented here: a second copy of an envelope is a second product,
# and the two would disagree the first time one of them was tuned.


# --------------------------------------------------------------------------------------
# printing
# --------------------------------------------------------------------------------------
def fnum(v, w=7, p=2):
    return ("%*s" % (w, "n/a")) if v != v else ("%*.*f" % (w, p, v))


MODE_BANNER = {
    "wave": ("WAVE  --  SCREEN ONLY, MAY NOT PROMOTE",
             "%d synchronised wave(s) per level, all C requests fired at t=0, then "
             "silence.",
             "A wave measures a burst arriving at an idle server.  It is cheap and it "
             "disqualifies fast,",
             "but it may NEVER promote a configuration: the reference had one pass a "
             "wave screen at 0.919",
             "and fail a 30-minute soak at 1.004 with 596 rejects and 111 broken "
             "pipes."),
    "soak": ("SOAK  --  QUALIFICATION, THE ONLY MODE THAT MAY PROMOTE",
             "%d s at fixed concurrency after %d s of discarded warm-up, cut into %d s "
             "windows.",
             "C requests stay in flight continuously, so the server never catches up "
             "between them and a",
             "metric that walks over time shows as a trend instead of a respectable "
             "mean.  A soak that",
             "drifts has qualified nothing."),
}


def print_table(reports, args, meta):
    W = 116
    mode = args.mode
    title, shape, *why = MODE_BANNER[mode]
    shape = (shape % (args.waves,) if mode == "wave" else
             shape % (args.soak_seconds, args.warmup_seconds, args.window_seconds))
    print("=" * W)
    print("SERVING PROFILE  %s" % meta["started"])
    print("  MODE       %s" % title)
    print("             %s" % shape)
    for line in why:
        print("             %s" % line)
    print("  server     %s  (%s)" % (meta["target"], meta["server_mode"]))
    print("  model      %s   engine %s   sample_rate %s Hz announced by /health"
          % (meta["health"].get("model"), meta["health"].get("engine"),
             meta["health"].get("sample_rate")))
    print("  audio      %s  (source: %s)" % (meta.get("format"), meta.get("format_source")))
    print("  route      %s   sink %s   bank %d texts   max_steps %s"
          % (args.route, "stream" if args.stream else "batch",
             len(meta["bank"]), args.max_steps or "-"))
    print("  dispatch   %s" % meta.get("dispatch_desc", "<unrecorded>"))
    if getattr(args, "profile_doc", None):
        print(profile_banner(args.profile_doc, args.profile_path))
    print("  host       %s" % meta["host_desc"])
    print("  PLATFORM   %s" % meta.get("platform_caveat", ""))
    print("=" * W)
    print()
    # No column is ever labelled a bare "RTF".  STREAM_RTF is a per-request rate over the
    # streamed span; WALL/AUD is the level's aggregate wall cost per second of audio.
    # They are different numbers and have been confused before.
    cols = [("C", 3), ("done/lnc", 9), ("TTFB50", 7), ("TTFB95", 7), ("TTFA50", 7),
            ("TTFA95", 7), ("SRTF50", 7), ("SRTF95", 7), ("SRTFsd", 7), ("preb50", 7),
            ("preb95", 7), ("safe95", 7), ("gap95", 7), ("stall", 6), ("WALL/AUD", 8),
            ("quote", 6), ("verdict", 0)]
    units = ["", "", "ms", "ms", "ms", "ms", "STREAM", "STREAM", "STREAM", "ms", "ms",
             "ms", "ms", "@%dms" % args.stall_gate_ms, "s/s", "coal%", ""]

    def row(cells):
        out = []
        for (name, w), c in zip(cols, cells):
            out.append(str(c) if w == 0 else "%*s" % (w, c))
        return " ".join(out)

    print(row([c[0] for c in cols]))
    print(row(units))
    print("-" * W)
    for r in reports:
        s = r["summary"]
        stall = s.get("stall_rate@%d" % args.stall_gate_ms, float("nan"))
        coal = r.get("coalesced_chunk_share", float("nan"))
        print(row([r["level"], "%d/%d" % (r["completed"], r["launched"]),
                   fnum(r["ttfb_ms"]["p50"], 0, 1), fnum(r["ttfb_ms"]["p95"], 0, 1),
                   fnum(r["ttfa_ms"]["p50"], 0, 0), fnum(r["ttfa_ms"]["p95"], 0, 0),
                   fnum(r["stream_rtf"]["p50"], 0, 3), fnum(r["stream_rtf"]["p95"], 0, 3),
                   fnum(r["stream_rtf"]["sd"], 0, 3),
                   fnum(r["prebuffer_ms"]["p50"], 0, 0), fnum(r["prebuffer_ms"]["p95"], 0, 0),
                   fnum(r["safe_play_start_ms"]["p95"], 0, 0),
                   fnum(r["max_gap_ms"]["p95"], 0, 0),
                   ("n/a" if stall != stall else "%.0f%%" % (stall * 100.0)),
                   fnum(r["aggregate_rtf"], 0, 2),
                   ("n/a" if coal != coal else "%.0f%%" % (coal * 100.0)),
                   r["verdict"]]))
    print("-" * W)
    print()

    for r in reports:
        print("C%-2d  %s   (%d/%d completed, %d requests with a measurable cadence, "
              "wall %.1f s)"
              % (r["level"], r["verdict"], r["completed"], r["launched"],
                 r["gates"]["n_cadence"], r["wall_s"]))
        if r.get("verdict_note"):
            print("       note: %s" % r["verdict_note"])
        if not r.get("may_promote"):
            print("       SCREEN ONLY: a wave result may disqualify a configuration, "
                  "never promote one.")
        if r.get("warmup_discarded"):
            print("       warm-up: %d request(s) started in the first %.0f s were "
                  "discarded" % (r["warmup_discarded"], r["warmup_s"]))
        if r.get("windows"):
            print("       SOAK WINDOWS (%d x %.0f s)  -- the trend, not the mean"
                  % (len(r["windows"]), args.window_seconds))
            print("         %-4s %5s %11s %13s %11s %11s"
                  % ("win", "n", "STREAM_RTF", "prebuffer p95", "TTFA p95",
                     "stall@%d" % args.stall_gate_ms))
            for w in r["windows"]:
                st = w.get("stall_rate@%d" % args.stall_gate_ms, float("nan"))
                print("         %-4d %5d %11s %13s %11s %11s"
                      % (w["window"], w["n"], fnum(w["stream_rtf_p95"], 11, 3),
                         fnum(w["prebuffer_p95_ms"], 13, 0),
                         fnum(w["ttfa_p95_ms"], 11, 0),
                         "n/a" if st != st else "%.0f%%" % (st * 100.0)))
            for key, d in sorted(r["drift"].items()):
                mark = {True: "PASS", False: "FAIL", None: "n/a "}[d["pass"]]
                print("         %s drift %-22s %s" % (mark, d["key"], d["why"]))
        for kind in ("mandatory", "preferred"):
            for g in r["gates"][kind]:
                mark = {True: "PASS", False: "FAIL", None: "n/a "}[g["pass"]]
                print("       %s %-9s %-22s %s %s %s%s"
                      % (mark, kind, g["name"],
                         ("n/a" if g["value"] != g["value"] else "%.3f%s"
                          % (g["value"], g["unit"])),
                         g["op"], "%.3f%s" % (g["limit"], g["unit"]),
                         ("   (%s)" % g["note"]) if g.get("note") else ""))
        print("       audio delivered %.1f s total, %.2f s/request (sd %.2f), "
              "%.0f chunks/request"
              % (r["audio_total_s"], r["audio_s"]["mean"], r["audio_s"]["sd"],
                 r["chunks"]["mean"]))
        print("       throughput %.2f audio-s per wall-s  |  aggregate RTF %.3f  |  "
              "per-request wall p50/p95 %.2f/%.2f s"
              % (r["throughput_audio_s_per_s"], r["aggregate_rtf"],
                 r["request_wall_s"]["p50"], r["request_wall_s"]["p95"]))
        print("       TTFB->TTFA (header lead) p50/p95 %s/%s ms  |  "
              "header is sent before synthesis, so TTFB alone is not a latency"
              % (fnum(r["header_to_audio_ms"]["p50"], 1, 0),
                 fnum(r["header_to_audio_ms"]["p95"], 1, 0)))
        print("       mix: %s"
              % (", ".join("%s x%d" % (k, v) for k, v in sorted(r["mix"].items()))
                 or "n/a"))
        print("       bank coverage: %d of %d texts spoken, each %.0fx on average"
              "  (%d request slots)"
              % (r["distinct_texts"], r["bank_size"] or r["distinct_texts"],
                 r["reuse_per_text"], r["request_slots"]))
        # The distribution, not just the median: a serialized server produces bimodal
        # latencies whose p50 looks healthy and whose sd says otherwise.
        print("       DISTRIBUTION over %d completed requests" % r["completed"])
        print("         %-20s %8s %8s %8s %8s %8s %8s %4s"
              % ("", "p50", "p95", "mean", "sd", "min", "max", "n"))
        for label, key in (("TTFB ms", "ttfb_ms"), ("TTFA ms", "ttfa_ms"),
                           ("STREAM_RTF", "stream_rtf"), ("prebuffer ms", "prebuffer_ms"),
                           ("safe_play_start ms", "safe_play_start_ms"),
                           ("max_gap ms", "max_gap_ms"),
                           ("audio s", "audio_s"), ("chunks", "chunks"),
                           ("request wall s", "request_wall_s")):
            d = r[key]
            digits = 3 if key in ("stream_rtf", "audio_s", "request_wall_s") else 1
            print("         %-20s %s %s %s %s %s %s %4d"
                  % (label, fnum(d["p50"], 8, digits), fnum(d["p95"], 8, digits),
                     fnum(d["mean"], 8, digits), fnum(d["sd"], 8, digits),
                     fnum(d["min"], 8, digits), fnum(d["max"], 8, digits), d["n"]))
        if r["summary"].get("n_records"):
            print(PB.format_summary(r["summary"], args.buffers, indent="       "))
        print("       QUOTABLE: %s%s" % (r["quotable"],
                                         "" if not r["quotable_reasons"] else
                                         " -- " + "; ".join(r["quotable_reasons"])))
        if r["failed"]:
            print("       !! %d of %d requests did NOT complete; the statistics above "
                  "cover only the %d that did"
                  % (r["failed"], r["launched"], r["completed"]))
            for e in r["errors"]:
                print("          %s" % e)
        print()

    print("THRESHOLDS IN FORCE (all are flags; change them and the verdicts change)")
    print("  mandatory : completed == launched, STREAM_RTF p95 < %.2f, "
          "stall_rate@%dms == 0" % (args.rtf_hard, args.stall_mandatory_ms))
    print("  preferred : TTFB p95 <= %.0f ms, TTFA p95 <= %.0f ms, STREAM_RTF p95 <= %.2f,"
          % (args.ttfb_pref_ms, args.ttfa_pref_ms, args.rtf_pref))
    print("              required_prebuffer p95 <= %.0f ms, safe_play_start p95 <= %.0f "
          "ms, stall_rate@%dms == 0"
          % (args.prebuffer_pref_ms, args.safe_start_pref_ms, args.stall_gate_ms))
    print("  refusal   : cadence percentiles are NOT QUOTABLE above %.0f%% coalesced "
          "reads (warn above %.0f%%)"
          % (args.coalesced_refuse * 100.0, args.coalesced_warn * 100.0))
    if args.mode == "soak":
        print("  drift     : STREAM_RTF p95 last-vs-best <= %+.3f, prebuffer p95 "
              "last-vs-best <= %+.0f ms, over >= %d windows"
              % (args.drift_rtf_tol, args.drift_prebuffer_tol_ms,
                 args.drift_min_windows))
    print("  GOOD = every preferred gate met - MARGINAL = mandatory met, a preferred one "
          "missed")
    print("  NOT STREAMABLE = a mandatory gate failed - INCONCLUSIVE = nothing to judge "
          "(no request delivered two chunks, so no player was ever simulated)")
    print("  NOT QUOTABLE = the client's reads coalesced badly enough that the cadence "
          "percentiles describe the client")
    print()
    # Capacity is discovered, not prescribed.
    verdicts = {r["level"]: r["verdict"] for r in reports}
    cap = PB.capacity(verdicts)
    if args.mode == "soak":
        if cap is None:
            print("CAPACITY: none of the measured levels qualified as GOOD, so this "
                  "configuration has no operating point.")
        else:
            print("CAPACITY: the operating point is C%d -- the highest GOOD concurrency "
                  "with no gap below it." % cap)
            higher = sorted(c for c in verdicts if c > cap)
            if higher:
                print("          C%d is %s; a GOOD level above a non-GOOD one is a "
                      "measurement to explain, not a product point."
                      % (higher[0], verdicts[higher[0]]))
    else:
        print("CAPACITY: NOT DETERMINED. This was a wave screen; capacity is a soak "
              "result. Levels that")
        print("          failed here can be dropped, but no level passes on this "
              "evidence -- rerun --mode soak.")
    print()
    mixes = {tuple(sorted(r["mix"].items())) for r in reports}
    if len(mixes) > 1:
        print("CAUTION: the class mix is not identical across levels (%s). Part of any "
              "difference between" % " | ".join(", ".join("%s x%d" % kv for kv in m)
                                                for m in sorted(mixes)))
        print("  levels may be workload drift rather than load. Use --same-text, or a "
              "bank whose length divides every level, to remove it.")
        print()
    print("READING THE TABLE: STREAM_RTF < 1 is capacity, not continuity. The columns that "
          "decide whether a")
    print("player stalls are prebuffer, safe_play_start, max_gap and the stall rate. A "
          "level can have")
    print("STREAM_RTF 0.4 and still be MARGINAL because the audio arrives in bursts.")


# --------------------------------------------------------------------------------------
# server lifecycle
# --------------------------------------------------------------------------------------
def port_is_live(host, port):
    try:
        http_get_json(host, port, "/health", timeout=2.0)
        return True
    except (OSError, HttpError, ValueError):
        return False


def start_server(args):
    if port_is_live(args.host, args.port):
        raise SystemExit("REFUSING TO START: something already answers /health on %s:%d. "
                         "Stop it, choose another --port, or attach with --url."
                         % (args.host, args.port))
    cmd = [args.server_bin, "-m", args.model, "-p", str(args.port)]
    if args.server_args:
        cmd += args.server_args.split()
    log = open(args.server_log, "wb") if args.server_log else subprocess.DEVNULL
    print("starting: %s" % " ".join(cmd), file=sys.stderr)
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True)
    t0 = time.time()
    while time.time() - t0 < args.server_timeout:
        if proc.poll() is not None:
            raise SystemExit("server exited with %s before becoming ready (see %s)"
                             % (proc.returncode, args.server_log or "its output"))
        if port_is_live(args.host, args.port):
            print("server ready after %.1f s" % (time.time() - t0), file=sys.stderr)
            return proc
        time.sleep(0.25)
    stop_server(proc)
    raise SystemExit("server did not become ready within %.0f s" % args.server_timeout)


def stop_server(proc):
    if proc is None or proc.poll() is not None:
        return
    # A server holds the whole model resident; a leaked one is expensive.  SIGTERM, then
    # make sure.
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (OSError, ProcessLookupError):
        proc.terminate()
    t0 = time.time()
    while time.time() - t0 < 10.0 and proc.poll() is None:
        time.sleep(0.1)
    if proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (OSError, ProcessLookupError):
            proc.kill()
    proc.wait()


def load_bank(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for ln in f:
            ln = ln.rstrip("\n")
            if not ln.strip() or ln.lstrip().startswith("#"):
                continue
            if "\t" in ln:
                cls, txt = [p.strip() for p in ln.split("\t", 1)]
            else:
                cls, txt = "text", ln.strip()
            if txt:
                rows.append((cls, txt))
    if not rows:
        raise SystemExit("REFUSING TO RUN: the bank %s has no usable lines. A profile run "
                         "against one hardcoded fallback sentence is how a whole level "
                         "ends up qualified on 0.4 s of audio." % path)
    return rows


# ======================================================================================
# THE DECODER-QUANTUM SWEEP
# ======================================================================================
# WHERE THE QUANTUM IS DECIDED, as of d1ffd01:
#
#   src/inference.c:128   slot_stream() emits only when
#                           fresh >= caps->audio_emit_frames  (or the window ended)
#   src/engine_pocket.c   cfg_opt_size(manifest, "audio_emit_frames", ..., 1u)
#   src/tts_engine.h:45   unsigned audio_emit_frames;   /* streaming emit threshold */
#
# So the decoder quantum IS a parameter -- but it is a MODEL-PACK parameter, read from
# model.json at load time.  It is not a CLI flag, not an environment variable and not a
# request field, so it cannot be varied without restarting the server.  This sweep
# therefore materialises one pack VARIANT per quantum: every large file is symlinked and
# only model.json is rewritten, so a variant costs a few kilobytes and no copy of the
# 219 MB safetensors.
#
# THE SECOND GRANULARITY, and what it actually does -- verified by reading the code,
# because the obvious reading is wrong:
#
#   server/main.c:86         #define STREAM_CHUNK 4096u  /* samples per streamed chunk */
#   src/inference.c:29-40    emit_stream_samples() splits the decoded PCM into pieces of
#                            at most STREAM_CHUNK samples, one stream_out_enqueue each
#   server/stream_out.c:169  the writer thread then takes `span = out->queued` -- the
#                            WHOLE readable span -- and writes it as ONE HTTP chunk
#
# So STREAM_CHUNK bounds each enqueue into the ring, and then the writer coalesces
# whatever is queued back into a single HTTP chunk.  It is NOT a delivery ceiling.  The
# measurement confirms it: at 40 frames, q1/q4/q16 deliver 40/10/2 client-visible chunks
# -- exactly ceil(frames/quantum) -- not the ~15 that a 4096-sample ceiling would force.
# Delivery granularity is therefore set by `audio_emit_frames`, which IS what this sweep
# varies, plus whatever extra coalescing happens when the writer is behind.
#
# What that leaves unreachable: there is no way to make delivery granularity SMALLER
# than the decode quantum, and no pacing. To decouple compute granularity from delivery
# granularity -- the principle in .work/streaming-cadence.md section 4, where the server
# aggregates work internally but still delivers small regular PCM increments -- the
# writer would have to cap its span (a max bytes per HTTP chunk, or a paced drain) in
# server/stream_out.c, and STREAM_CHUNK would have to become a runtime parameter rather
# than a #define.  Both are in server/, owned by another lane.  This sweep reports what
# is reachable and refuses to pretend the rest was measured.
QUANTUM_KEY = "audio_emit_frames"


def make_quantum_pack(src_dir, dst_dir, quantum):
    """A pack variant that differs from ``src_dir`` ONLY in the emit quantum.

    Everything except model.json is symlinked, so the variant is kilobytes and the
    weights are byte-identical to the original by construction -- there is no chance of
    a sweep accidentally comparing two different sets of weights.
    """
    src_dir = os.path.abspath(src_dir)
    manifest_path = os.path.join(src_dir, "model.json")
    if not os.path.isfile(manifest_path):
        raise SystemExit("REFUSING TO SWEEP: %s has no model.json" % src_dir)
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    if os.path.isdir(dst_dir):
        for name in os.listdir(dst_dir):
            os.unlink(os.path.join(dst_dir, name))
    else:
        os.makedirs(dst_dir)
    for name in os.listdir(src_dir):
        if name == "model.json":
            continue
        os.symlink(os.path.join(src_dir, name), os.path.join(dst_dir, name))
    manifest[QUANTUM_KEY] = int(quantum)
    with open(os.path.join(dst_dir, "model.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    return dst_dir


def interleave(labels, repeats):
    """A/B/B/A rather than all-of-A then all-of-B.

    This machine drifts, and the reference measured drift between two IDENTICAL runs
    larger than the effect under test.  Running every A first and every B second turns
    that drift into a fake difference between the arms; a palindromic order spreads it
    across both.  Two arms and two repeats give exactly A B B A.
    """
    order = []
    for r in range(repeats):
        order.extend(labels if r % 2 == 0 else list(reversed(labels)))
    return order


def platform_caveat():
    """Say plainly where the number came from.  Production is Linux x86-64 and ARM64."""
    if sys.platform == "darwin":
        return ("macOS / Apple Silicon -- a DEVELOPMENT SIGNAL, NOT A PRODUCTION CLAIM. "
                "Accelerate and the P-core thread-pool default do not exist on Linux; "
                "production is Linux x86-64 and ARM64.")
    if sys.platform.startswith("linux"):
        return "Linux -- the production platform. Record the exact CPU, mask and build."
    return "%s -- not a production platform." % sys.platform


def host_description():
    bits = [sys.platform, os.uname().machine]
    try:
        bits.append("%d cpus" % (os.cpu_count() or 0))
    except Exception:
        pass
    if sys.platform == "darwin":
        try:
            brand = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                   capture_output=True, text=True, timeout=5).stdout.strip()
            if brand:
                bits.append(brand)
        except Exception:
            pass
    return ", ".join(bits)


# --------------------------------------------------------------------------------------
# Serving profiles -- a configuration that cannot be got wrong by hand
# --------------------------------------------------------------------------------------
def apply_profile(args, ap):
    """Fill args from configs/perf/<name>.json, and refuse rather than silently disagree.

    Three things happen here, and the third is the point:

      1. every knob the profile pins is written into args, so the run is the profile;
      2. a flag the caller ALSO passed is a conflict, not an override -- a profile whose
         values can be quietly replaced from the command line qualifies nothing;
      3. a variable the profile declares absent must be absent from os.environ.

    (3) is what this file exists for. Between 2026-09-13 and 2026-09-19 every capacity
    number in docs/ was measured with MYNAH_QUANT_GROUPS exported by hand while the
    shipped binary chose something slower, and nothing in the harness could see the
    difference: the run passed, the product could not reproduce it.
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import perf_profile as PP

    try:
        prof, path = PP.load(args.profile)
    except PP.Bad as exc:
        raise PB.Refusal([str(exc)])
    errs = PP.semantic(prof, path)
    if errs:
        raise PB.Refusal(["%s is not a valid profile:" % path] + ["  " + e for e in errs])

    given = set()
    for tok in sys.argv[1:]:
        if tok.startswith("--"):
            given.add(tok.split("=", 1)[0])

    reasons = []
    for var in PP.forbidden_env(prof):
        if var in os.environ:
            reasons.append("%s is in the environment (=%r) and the profile declares it "
                           "must be ABSENT: %s"
                           % (var, os.environ[var],
                              prof["runtime"]["environment"][var]["why"]))
    wanted = PP.environ(prof)
    for var, value in sorted(wanted.items()):
        have = os.environ.get(var)
        if have is not None and have != value:
            reasons.append("%s is %r in the environment but the profile pins %r"
                           % (var, have, value))
    if reasons:
        raise PB.Refusal(reasons)
    os.environ.update(wanted)

    pinned = {"--server-args": " ".join(PP.server_args(prof))}
    for flag, value in zip(PP.gate_args(prof)[0::2], PP.gate_args(prof)[1::2]):
        pinned[flag] = value
    obj = prof["profile"].get("objective", {})
    if obj.get("preferred_concurrency"):
        pinned["--levels"] = str(obj["preferred_concurrency"])
    pinned["--mode"] = "soak"

    conflict = sorted(f for f in pinned if f in given)
    if conflict:
        raise PB.Refusal(
            ["the profile %s pins these, and they were also given on the command line:"
             % prof["profile"]["id"]]
            + ["  %s (profile says %s)" % (f, pinned[f]) for f in conflict]
            + ["a profile whose values can be overridden from the shell qualifies "
               "nothing. Drop the flag, or run without --profile and own the numbers."])

    for flag, value in pinned.items():
        dest = flag[2:].replace("-", "_")
        cur = getattr(args, dest, None)
        setattr(args, dest, type(cur)(value) if isinstance(cur, (int, float)) else value)

    args.profile_doc = prof
    args.profile_path = path
    return args


def profile_banner(prof, path):
    q = prof["qualification"]
    op = q.get("operating_point") or {}
    out = ["  PROFILE    %s  (%s)" % (prof["profile"]["id"], os.path.relpath(path, "."))]
    out.append("             %s" % prof["profile"]["description"])
    out.append("             status %s%s" % (
        q["status"],
        ("  --  the qualified point is C%d, %s, %ds, %d requests"
         % (op["concurrency"], op["verdict"], op.get("soak_seconds", 0),
            op.get("requests", 0))) if op else ""))
    env = PBENV(prof)
    out.append("             env    %s" % (env or "(none -- the shipped default)"))
    return "\n".join(out)


def PBENV(prof):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import perf_profile as PP
    return " ".join("%s=%s" % kv for kv in sorted(PP.environ(prof).items()))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    srv = ap.add_argument_group("server")
    srv.add_argument("--url", default="", help="attach to a running server, e.g. "
                                               "http://127.0.0.1:8973")
    srv.add_argument("--server-bin", default="build/cpu/mynah-tts-server")
    srv.add_argument("--model", default="", help="model pack; starts a server on --port")
    srv.add_argument("--port", type=int, default=8974)
    srv.add_argument("--host", default="127.0.0.1")
    srv.add_argument("--server-args", default="", help="extra argv for the server")
    srv.add_argument("--server-log", default="", help="file for the server's output")
    srv.add_argument("--server-timeout", type=float, default=300.0)
    srv.add_argument("--profile", default="",
                     help="a serving profile from configs/perf: it supplies the server "
                          "topology, the environment, the bank and every gate, and "
                          "REFUSES the run if the environment contradicts it. This is the "
                          "way to reproduce a qualified operating point without retyping "
                          "it -- see tools/perf_profile.py")

    load = ap.add_argument_group("load")
    load.add_argument("--mode", choices=("wave", "soak"), default="wave",
                      help="WAVE = a screen: C requests fired at t=0, repeated --waves "
                           "times. It may disqualify a level, never promote one. "
                           "SOAK = a qualification: --soak-seconds at fixed C with "
                           "warm-up and a drift gate across windows. Only a soak may "
                           "promote a configuration.")
    load.add_argument("--levels", default="1,2,4",
                      help="concurrency levels, e.g. 1,2,4,8")
    load.add_argument("--waves", type=int, default=3,
                      help="WAVE only: synchronized waves per level "
                           "(requests = level x waves)")
    load.add_argument("--soak-seconds", type=float, default=300.0,
                      help="SOAK only: measured seconds per level, after warm-up")
    load.add_argument("--warmup-seconds", type=float, default=30.0,
                      help="SOAK only: seconds run and DISCARDED before measuring")
    load.add_argument("--window-seconds", type=float, default=60.0,
                      help="SOAK only: window length for the drift gate")
    load.add_argument("--wave-gap", type=float, default=0.0,
                      help="idle seconds between waves")
    load.add_argument("--route", default="/v1/audio/speech")
    load.add_argument("--batch", dest="stream", action="store_false", default=True,
                      help="profile the non-streaming route instead")
    load.add_argument("--bank", default="", help="text bank (one per line, optional "
                                                 "TAB-separated class)")
    load.add_argument("--text", default="", help="a single text, overrides the bank")
    load.add_argument("--voice", default="")
    load.add_argument("--language", default="")
    load.add_argument("--seed", type=int, default=7)
    load.add_argument("--vary-seed", action="store_true",
                      help="use a different seed per request")
    load.add_argument("--same-text", dest="vary_text", action="store_false", default=True,
                      help="every wave reuses the same texts (removes mix drift between "
                           "waves at the cost of exercising fewer lengths)")
    load.add_argument("--max-steps", type=int, default=0,
                      help="per-request decoder step cap sent to the server; the synthetic "
                           "pack needs it (its model.json caps at 8 steps ~ 0.4 s)")
    load.add_argument("--extra-json", default="", help="JSON merged into every request")
    load.add_argument("--timeout", type=float, default=300.0, help="per-request timeout")
    load.add_argument("--warmup", type=int, default=1,
                      help="unmeasured single requests before the first level")

    gates = ap.add_argument_group("verdict thresholds")
    gates.add_argument("--rtf-hard", type=float, default=DEFAULTS["rtf_hard"])
    gates.add_argument("--rtf-pref", type=float, default=DEFAULTS["rtf_pref"])
    gates.add_argument("--ttfb-pref-ms", type=float, default=DEFAULTS["ttfb_pref_ms"])
    gates.add_argument("--ttfa-pref-ms", type=float, default=DEFAULTS["ttfa_pref_ms"])
    gates.add_argument("--prebuffer-pref-ms", type=float,
                       default=DEFAULTS["prebuffer_pref_ms"])
    gates.add_argument("--safe-start-pref-ms", type=float,
                       default=DEFAULTS["safe_start_pref_ms"])
    gates.add_argument("--stall-gate-ms", type=int, default=DEFAULTS["stall_gate_ms"],
                       help="the jitter buffer whose stall rate is the QUALIFYING gate "
                            "(preferred); default 250")
    gates.add_argument("--stall-mandatory-ms", type=int,
                       default=DEFAULTS["stall_mandatory_ms"],
                       help="the jitter buffer whose stall rate is the HARD gate "
                            "(mandatory); default 500")
    gates.add_argument("--buffers", default="100,250,500,1000",
                       help="jitter buffers to simulate, ms")
    gates.add_argument("--coalesced-refuse", type=float,
                       default=DEFAULTS["coalesced_refuse"],
                       help="refuse to quote cadence percentiles above this share of "
                            "already-queued reads")
    gates.add_argument("--coalesced-warn", type=float, default=DEFAULTS["coalesced_warn"])
    gates.add_argument("--drift-rtf-tol", type=float, default=DEFAULTS["drift_rtf_tol"],
                       help="SOAK: allowed STREAM_RTF p95 drift, last window vs best")
    gates.add_argument("--drift-prebuffer-tol-ms", type=float,
                       default=DEFAULTS["drift_prebuffer_tol_ms"])
    gates.add_argument("--drift-min-windows", type=int,
                       default=DEFAULTS["drift_min_windows"])

    sweep = ap.add_argument_group("decoder-quantum sweep")
    sweep.add_argument("--quantum-sweep", default="",
                       help="comma list of audio_emit_frames values to sweep, e.g. "
                            "1,2,4,8. Derives one pack variant per value (model.json "
                            "rewritten, every other file symlinked) and runs them "
                            "INTERLEAVED. Needs exactly one --levels value.")
    sweep.add_argument("--quantum-pack-dir", default="",
                       help="where to materialise the pack variants")
    sweep.add_argument("--repeats", type=int, default=2,
                       help="interleaved repeats; arms run palindromically (A B B A)")

    cmp_ = ap.add_argument_group("compare")
    cmp_.add_argument("--compare", nargs="+", default=None,
                      help="difference saved --json runs; REFUSES (exit 4) when their "
                           "dispatch resolution differs")
    cmp_.add_argument("--allow-unverified-dispatch", action="store_true",
                      help="proceed when a dispatch fact is unrecorded in EVERY run. "
                           "Only correct when sameness is established outside the "
                           "report (same binary, same environment).")

    out = ap.add_argument_group("output")
    out.add_argument("--json", default="", help="write the full report as JSON "
                                                "('-' for stdout)")
    out.add_argument("--marks", action="store_true",
                     help="keep the per-chunk arrival marks in the JSON")
    out.add_argument("--sample-rate", type=int, default=0,
                     help="last-resort override when the server announces no format")
    out.add_argument("--channels", type=int, default=1)
    out.add_argument("--bits", type=int, default=16)
    out.add_argument("--quiet", action="store_true")

    args = ap.parse_args()
    if args.profile:
        try:
            apply_profile(args, ap)
        except PB.Refusal as exc:
            print("REFUSING TO RUN THE PROFILE:", file=sys.stderr)
            for why in exc.reasons:
                print("  %s" % why, file=sys.stderr)
            return 4
    args.buffers = tuple(int(x) for x in args.buffers.split(",") if x.strip())
    if args.compare:
        try:
            return compare_reports(args.compare, args)
        except PB.Refusal as exc:
            print("REFUSING TO COMPARE:", file=sys.stderr)
            for why in exc.reasons:
                print("  %s" % why, file=sys.stderr)
            return 4
    for b in (args.stall_gate_ms, args.stall_mandatory_ms):
        if b not in args.buffers:
            args.buffers = tuple(sorted(args.buffers + (b,)))
    levels = [int(x) for x in args.levels.split(",") if x.strip()]
    if not levels or min(levels) < 1:
        raise SystemExit("--levels must be positive integers, e.g. 1,2,4")
    if args.mode == "soak":
        if args.soak_seconds <= 0:
            raise SystemExit("--soak-seconds must be positive in --mode soak")
        if args.window_seconds <= 0:
            raise SystemExit("--window-seconds must be positive in --mode soak")
        if args.soak_seconds < args.window_seconds * args.drift_min_windows:
            raise SystemExit(
                "REFUSING TO RUN: --soak-seconds %.0f cannot contain the %d windows of "
                "%.0f s the drift gate needs. A soak that cannot show a trend is a wave "
                "with extra steps; either lengthen the soak or lower "
                "--drift-min-windows deliberately."
                % (args.soak_seconds, args.drift_min_windows, args.window_seconds))

    if args.url:
        m = re.match(r"http://([^:/]+)(?::(\d+))?", args.url)
        if not m:
            raise SystemExit("--url must look like http://host:port")
        args.host, args.port = m.group(1), int(m.group(2) or 80)
    elif not args.model:
        raise SystemExit("give either --url (attach) or --model (start a server)")

    if args.text:
        bank = [("cli", args.text)]
    elif args.bank:
        bank = load_bank(args.bank)
    elif os.path.exists(DEFAULT_BANK_FILE):
        bank = load_bank(DEFAULT_BANK_FILE)
    else:
        print("note: %s is missing, falling back to the five built-in texts -- "
              "they are Italian and a level qualified on them says little"
              % DEFAULT_BANK_FILE, file=sys.stderr)
        bank = list(DEFAULT_BANK)

    if args.quantum_sweep:
        try:
            run_quantum_sweep(args, bank, levels)
        except PB.Refusal as exc:
            print("REFUSING TO REPORT THE SWEEP:", file=sys.stderr)
            for why in exc.reasons:
                print("  %s" % why, file=sys.stderr)
            return 4
        return 0

    proc = None
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    t_profile0 = time.perf_counter()
    try:
        reports, prof, health0, health1, proc = run_arm(args, bank, levels)
    finally:
        stop_server(proc)

    duration = time.perf_counter() - t_profile0
    meta = build_meta(args, bank, prof, health0, health1, started, duration)

    if not args.quiet:
        print()
        print_table(reports, args, meta)
        print()
        print("profile took %.1f s of wall clock (%d levels, %d requests total)"
              % (duration, len(reports), sum(r["launched"] for r in reports)))
        print("server counters after the run: %s" % json.dumps(health1.get("jobs", {})))

    if args.json:
        write_json(args, meta, reports)

    # THE REFUSAL.  A run whose marks were mostly already-queued data has measured the
    # client's reader, not the server's emission.  Exit non-zero rather than let a
    # number nobody should trust be scraped out of stdout by the next script.
    refused = [r for r in reports if r["quotable"] == "NOT QUOTABLE"]
    if refused:
        print()
        print("REFUSING TO REPORT CADENCE PERCENTILES for level(s) %s."
              % ", ".join("C%d" % r["level"] for r in refused), file=sys.stderr)
        for r in refused:
            for why in r["quotable_reasons"]:
                print("  C%d: %s" % (r["level"], why), file=sys.stderr)
        print("  Fix the client or the transport (TCP_NODELAY, E5-18) and rerun. "
              "Nothing here is quotable.", file=sys.stderr)
        return 3

    worst = {"GOOD": 0, "INCONCLUSIVE": 1, "MARGINAL": 1, "NOT STREAMABLE": 2,
             "NOT QUOTABLE": 3}
    return max(worst.get(r["verdict"], 2) for r in reports) if reports else 2


def run_arm(args, bank, levels, model_dir=None):
    """One server lifetime: start, warm up, run every level, read /health, return.

    ``model_dir`` overrides ``args.model`` so the quantum sweep can point successive
    arms at successive pack variants.  The caller owns stopping the process.
    """
    proc = None
    saved = args.model
    if model_dir is not None:
        args.model = model_dir
    try:
        if not args.url:
            proc = start_server(args)
        health0 = http_get_json(args.host, args.port, "/health", timeout=10.0)
        prof = Profile(args, bank, health0)

        for i in range(args.warmup):
            w = prof.run_one(0, -1, i)
            if not w.get("ok"):
                raise SystemExit("warm-up request failed, refusing to profile a server "
                                 "that cannot serve one request: %s" % w.get("error"))
        reports = []
        for level in levels:
            if args.mode == "soak":
                if not args.quiet:
                    print("soaking C%d (%.0f s warm-up + %.0f s measured)..."
                          % (level, args.warmup_seconds, args.soak_seconds),
                          file=sys.stderr)
                records, measured, wall, warm_s = prof.run_soak(level)
                windows = prof.soak_windows(measured)
                reports.append(level_report(
                    level, measured, len(measured), wall, args, mode="soak",
                    windows=windows, warmup_s=warm_s,
                    warmup_n=len(records) - len(measured), bank_len=len(bank)))
            else:
                if not args.quiet:
                    print("screening C%d (%d waves)..." % (level, args.waves),
                          file=sys.stderr)
                records, launched, wall = prof.run_level(level)
                reports.append(level_report(level, records, launched, wall, args,
                                            mode="wave", bank_len=len(bank)))
        health1 = http_get_json(args.host, args.port, "/health", timeout=10.0)
    except BaseException:
        stop_server(proc)
        raise
    finally:
        args.model = saved
    return reports, prof, health0, health1, proc


def build_meta(args, bank, prof, health0, health1, started, duration):
    # The dispatch facts.  Two arms may only be differenced when these match; the set is
    # PB.COMPARABLE_KEYS and the check is PB.require_comparable.  Anything the server
    # does not report stays literally "<unrecorded>" -- an unrecorded fact is not an
    # equal fact, and pretending otherwise is how a sweep attributes a difference to the
    # wrong variable.
    #
    # As of d1ffd01 /health reports model, engine, sample_rate, voices and job counters
    # -- and NOT isa, simd, backend, quant, threads or build flags.  Those therefore
    # come from what the harness itself controls (the environment it launched the server
    # with), and anything neither side knows stays "<unrecorded>" so that
    # PB.require_comparable can refuse to certify it.
    arm = {k: health0.get(k, PB.UNRECORDED) for k in
           ("engine", "isa", "simd", "backend", "sample_rate")}
    arm["quant"] = health0.get("quant", os.environ.get("MYNAH_QUANT")
                               or "<pack default>")
    arm["threads"] = health0.get("threads", os.environ.get("MYNAH_THREADS")
                                 or "<server default>")
    arm["build_flags"] = health0.get("build_flags", PB.UNRECORDED)
    arm["server_bin"] = args.server_bin if not args.url else "<attached>"
    arm["server_args"] = args.server_args
    arm["route"] = args.route
    arm["stream"] = bool(args.stream)
    meta = {"started": started, "duration_s": duration,
            "target": "http://%s:%d" % (args.host, args.port),
            "server_mode": "attached" if args.url else "started by this tool",
            "health": health0, "health_after": health1,
            "bank": [t for _c, t in bank],
            "format": repr(prof.fmt) if prof.fmt else None,
            "format_source": prof.fmt_source,
            "host_desc": host_description(),
            "arm": arm, "mode": args.mode,
            "dispatch_desc": "  ".join("%s=%s" % (k, arm[k]) for k in sorted(arm)),
            "platform_caveat": platform_caveat(),
            "argv": sys.argv[1:]}
    return meta


def arm_row(rep, args):
    """The three numbers a serving point is actually chosen on, plus the one it is not."""
    s = rep["summary"]
    return {
        "verdict": rep["verdict"], "quotable": rep["quotable"],
        "prebuffer_p95_ms": rep["prebuffer_ms"]["p95"],
        "stall_gate": s.get("stall_rate@%d" % args.stall_gate_ms, float("nan")),
        "safe_start_p95_ms": rep["safe_play_start_ms"]["p95"],
        "ttfa_p95_ms": rep["ttfa_ms"]["p95"],
        "stream_rtf_p95": rep["stream_rtf"]["p95"],
        "chunks_mean": rep["chunks"]["mean"],
        "max_gap_p95_ms": rep["max_gap_ms"]["p95"],
    }


def run_quantum_sweep(args, bank, levels):
    """Sweep the decoder emit quantum, interleaved, and choose on prebuffer and stall.

    The quantum must be chosen on required_prebuffer and stall_rate, NEVER on
    STREAM_RTF.  In the reference the smallest quantum FAILED on STREAM_RTF (1.005 p95)
    while the largest PASSED on it (0.817) and stalled half the time; aggregate rate
    selected exactly the wrong serving point.  We have an advantage they did not: our
    codec carries state, so chunked-vs-one-shot error stays at 1e-7 down to a one-frame
    chunk (.work E2-3).  A small quantum costs us no correctness, so the cadence knob is
    free for us in a way it was not for them.
    """
    if not args.model:
        raise SystemExit("--quantum-sweep needs --model (a pack to derive variants "
                         "from); it cannot sweep an attached server, because the "
                         "quantum is read from model.json at load time")
    if len(levels) != 1:
        raise SystemExit("--quantum-sweep takes exactly one concurrency level "
                         "(--levels C): a sweep that also varies load has measured two "
                         "things at once")
    level = levels[0]
    quanta = [int(x) for x in args.quantum_sweep.split(",") if x.strip()]
    if not quanta or min(quanta) < 1:
        raise SystemExit("--quantum-sweep must be positive integers, e.g. 1,2,4,8")

    base = os.path.abspath(args.model)
    root = args.quantum_pack_dir or os.path.join(
        os.path.dirname(base), ".quantum-sweep-" + os.path.basename(base))
    os.makedirs(root, exist_ok=True)
    packs = {}
    for q in quanta:
        packs["q%d" % q] = make_quantum_pack(base, os.path.join(root, "q%d" % q), q)

    labels = ["q%d" % q for q in quanta]
    order = interleave(labels, args.repeats)
    print("SWEEP ORDER (interleaved against drift): %s" % " ".join(order),
          file=sys.stderr)

    runs = []
    arms_facts = {}
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    for n, label in enumerate(order):
        print("  [%d/%d] %s ..." % (n + 1, len(order), label), file=sys.stderr)
        proc = None
        try:
            reports, prof, h0, h1, proc = run_arm(args, bank, [level], packs[label])
        finally:
            stop_server(proc)
        meta = build_meta(args, bank, prof, h0, h1, started, 0.0)
        arms_facts.setdefault(label, meta["arm"])
        runs.append({"label": label, "rep": n, "report": reports[0],
                     "row": arm_row(reports[0], args)})

    # THE REFUSAL: every arm must have resolved the same dispatch.  The quantum is the
    # only thing allowed to differ; if the ISA, quantization or thread count moved too,
    # the difference between the arms is not the variable under test.
    #
    # Sameness here IS established by construction -- one binary, one environment, one
    # process launch per arm, and pack variants that symlink every file except
    # model.json -- so facts the server does not report are allowed through.  They are
    # named in the output rather than quietly assumed.
    unverified = PB.require_comparable(arms_facts, allow_unverified=True)

    print()
    W = 108
    print("=" * W)
    print("DECODER-QUANTUM SWEEP at C%d  --  %s" % (level, args.mode.upper()))
    print("  quantum = model.json %r (frames accumulated before decode+emit); "
          "1 frame = %s" % (QUANTUM_KEY, "%.0f ms" % (1000.0 / 12.5)))
    print("  chosen on required_prebuffer p95 and stall_rate@%dms. NOT on STREAM_RTF."
          % args.stall_gate_ms)
    print("  order: %s   (%d repeats, palindromic)" % (" ".join(order), args.repeats))
    print("  dispatch: %s" % list(arms_facts.values())[0].get("engine"))
    if unverified:
        print("  DISPATCH NOT VERIFIED for %s -- /health does not report these. The arms"
              % ", ".join(unverified))
        print("    share one binary, one environment and one launch, so they are the "
              "same by construction,")
        print("    but that is an argument, not a measurement. Make /health report them "
              "to close this.")
    print("  PLATFORM %s" % platform_caveat())
    print("=" * W)
    hdr = ("%-6s %4s %9s %11s %10s %11s %11s %10s %9s  %s"
           % ("arm", "n", "chunks", "preb95 ms", "stall@%d" % args.stall_gate_ms,
              "safe95 ms", "TTFA95 ms", "gap95 ms", "SRTF95", "verdicts"))
    print(hdr)
    print("-" * W)
    per_arm = {}
    for label in labels:
        rows = [r["row"] for r in runs if r["label"] == label]
        per_arm[label] = rows

        def med(key):
            return PB.pct([r[key] for r in rows], 50)
        verdicts = ",".join(r["verdict"] for r in rows)
        print("%-6s %4d %9.1f %11.0f %10s %11.0f %11.0f %10.0f %9.3f  %s"
              % (label, len(rows), med("chunks_mean"), med("prebuffer_p95_ms"),
                 "%.0f%%" % (med("stall_gate") * 100.0)
                 if med("stall_gate") == med("stall_gate") else "n/a",
                 med("safe_start_p95_ms"), med("ttfa_p95_ms"), med("max_gap_p95_ms"),
                 med("stream_rtf_p95"), verdicts))
    print("-" * W)

    # The choice, made explicitly on the gating metrics.
    def score(label):
        rows = per_arm[label]
        return (PB.pct([r["stall_gate"] for r in rows], 50),
                PB.pct([r["prebuffer_p95_ms"] for r in rows], 50))
    ok_arms = [l for l in labels if all(r["verdict"] not in ("NOT QUOTABLE",)
                                        for r in per_arm[l])]
    best = min(ok_arms, key=score) if ok_arms else None
    print()
    if best is None:
        print("NO ARM IS QUOTABLE: the sweep decided nothing.")
    else:
        srtf_best = min(labels, key=lambda l: PB.pct(
            [r["stream_rtf_p95"] for r in per_arm[l]], 50))
        print("CHOSEN ON CADENCE: %s (lowest stall_rate@%dms, then lowest prebuffer p95)"
              % (best, args.stall_gate_ms))
        print("BEST STREAM_RTF  : %s" % srtf_best)
        if srtf_best != best:
            print("  ^ THESE DISAGREE, which is the entire point of the sweep: choosing "
                  "the quantum on aggregate")
            print("    rate would have selected %s, and %s is the serving point."
                  % (srtf_best, best))
        else:
            print("  (they agree here; that is a fact about this run, not a licence to "
                  "choose on STREAM_RTF next time)")
    print()
    print("WHAT WAS SWEPT: model.json %r, read at load time (src/engine_pocket.c, "
          "default 1)." % QUANTUM_KEY)
    print("  It is not a CLI flag, env var or request field, so each arm is a pack "
          "variant and a")
    print("  server restart. Chunks/request = ceil(frames / quantum) confirms the "
          "quantum took effect.")
    print()
    print("NOT SWEPT, and what would have to change:")
    print("  server/main.c STREAM_CHUNK (4096 samples) bounds each enqueue, but")
    print("  server/stream_out.c:169 drains the WHOLE queued span into one HTTP chunk, "
          "so it is")
    print("  not a delivery ceiling -- delivery granularity follows the decode quantum. "
          "There is")
    print("  no way to deliver SMALLER increments than the decode quantum, and no "
          "pacing. Doing")
    print("  that needs a capped or paced writer span in server/stream_out.c and "
          "STREAM_CHUNK as a")
    print("  runtime parameter. Both live in server/, owned by another lane.")
    return runs, per_arm


def compare_reports(paths, args):
    """Difference two or more saved runs, refusing when the dispatch differs."""
    docs = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            docs.append((p, json.load(f)))
    arms = {p: d["meta"].get("arm", {}) for p, d in docs}
    unver = PB.require_comparable(arms, allow_unverified=args.allow_unverified_dispatch)
    print("COMPARABLE: every arm resolved the same dispatch (%s)"
          % docs[0][1]["meta"].get("dispatch_desc", "<unrecorded>"))
    if unver:
        print("  CAVEAT: %s was unrecorded in every run; sameness is assumed, not "
              "measured." % ", ".join(unver))
    modes = {d["meta"].get("mode") for _p, d in docs}
    if len(modes) > 1:
        raise PB.Refusal(["these runs are not the same KIND of measurement (%s): a wave "
                          "screen and a soak are not comparable"
                          % ", ".join(sorted(str(m) for m in modes))])
    print("%-40s %4s %11s %10s %9s  %s"
          % ("run", "C", "preb95 ms", "stall", "SRTF95", "verdict"))
    for p, d in docs:
        for lv in d["levels"]:
            s = lv["summary"]
            st = s.get("stall_rate@%d" % args.stall_gate_ms, float("nan"))
            print("%-40s %4d %11.0f %10s %9.3f  %s"
                  % (os.path.basename(p)[:40], lv["level"],
                     lv["prebuffer_ms"]["p95"],
                     "n/a" if st != st else "%.0f%%" % (st * 100.0),
                     lv["stream_rtf"]["p95"], lv["verdict"]))
    return 0


def write_json(args, meta, reports):
    doc = {"meta": meta, "levels": reports,
           "thresholds": {k: getattr(args, k) for k in
                          ("rtf_hard", "rtf_pref", "ttfb_pref_ms", "ttfa_pref_ms",
                           "prebuffer_pref_ms", "safe_start_pref_ms",
                           "stall_gate_ms", "stall_mandatory_ms",
                           "coalesced_refuse", "drift_rtf_tol",
                           "drift_prebuffer_tol_ms", "drift_min_windows")}}
    text = json.dumps(doc, indent=2, default=str)
    if args.json == "-":
        print(text)
    else:
        with open(args.json, "w") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    sys.exit(main())
