#!/usr/bin/env python3
"""serving_profile.py -- concurrency profile of the mynah-tts HTTP server, with verdicts.

    # start a server on a scratch port, profile C1, C2, C4, tear it down
    python3 tools/serving_profile.py --model models/fake-magpie --levels 1,2,4 \
        --waves 2 --max-steps 64

    # attach to a server someone else is running
    python3 tools/serving_profile.py --url http://127.0.0.1:8973 --levels 1,2,4

What it answers
---------------
Not "how fast is the server" but "at which concurrency can a real player still play".
For every concurrency level it launches C requests at once (a wave), repeats that
``--waves`` times, and reports:

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
# A neutral bank: short, medium and long, so a level is not qualified on one length.
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

# The provisional streaming envelope, ported from qwen-tts
# .work/professional-streaming-architecture.md E8.  Every one of these is a flag.
DEFAULTS = {
    "rtf_hard": 1.00,        # mandatory: STREAM_RTF p95 must be below this
    "rtf_pref": 0.90,        # preferred
    "ttfb_pref_ms": 100.0,
    "ttfa_pref_ms": 500.0,
    "prebuffer_pref_ms": 500.0,
    "safe_start_pref_ms": 1000.0,
    "stall_buffer_ms": 500,  # the buffer size whose stall rate must be zero
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

        k = PB.timeline_kpis(marks, r["t_done_s"], fmt, self.args.buffers)
        rec.update(k)
        rec["ttfa_s"] = k.get("ttfa_s", float("nan"))
        rec["header_to_audio_s"] = (rec["ttfa_s"] - r["ttfb_s"]
                                    if r["ttfb_s"] is not None else float("nan"))
        rec["ok"] = True
        if self.args.marks:
            rec["marks"] = marks
        return rec

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
def level_report(level, records, launched, wall, args):
    ok = [r for r in records if r.get("ok")]
    bad = [r for r in records if not r.get("ok")]
    rep = {"level": level, "launched": launched, "completed": len(ok),
           "failed": len(bad), "wall_s": wall,
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
    rep["distinct_texts"] = len({r.get("index") for r in records})
    rep["verdict"], rep["gates"] = verdict(rep, args)
    return rep


def _g(name, value, op, limit, unit=""):
    if value != value:                      # NaN: not measured
        passed = None
    elif op == "<":
        passed = value < limit
    elif op == "<=":
        passed = value <= limit
    elif op == "==":
        passed = value == limit
    else:
        raise ValueError(op)
    return {"name": name, "value": value, "op": op, "limit": limit,
            "unit": unit, "pass": passed}


def verdict(rep, args):
    """GOOD / MARGINAL / NOT STREAMABLE / INCONCLUSIVE, with every gate carried along.

    * mandatory failed        -> NOT STREAMABLE
    * mandatory ok, preferred failed -> MARGINAL
    * everything ok           -> GOOD
    * nothing to judge (no cadence measured anywhere) -> INCONCLUSIVE, never GOOD by
      default: a level whose requests each delivered one chunk has no continuity evidence.
    """
    s = rep["summary"]
    mandatory = [
        _g("completed == launched", float(rep["completed"]), "==", float(rep["launched"])),
        _g("STREAM_RTF p95", rep["stream_rtf"]["p95"], "<", args.rtf_hard),
    ]
    preferred = [
        _g("TTFB p95", rep["ttfb_ms"]["p95"], "<=", args.ttfb_pref_ms, "ms"),
        _g("TTFA p95", rep["ttfa_ms"]["p95"], "<=", args.ttfa_pref_ms, "ms"),
        _g("STREAM_RTF p95", rep["stream_rtf"]["p95"], "<=", args.rtf_pref),
        _g("prebuffer p95", rep["prebuffer_ms"]["p95"], "<=", args.prebuffer_pref_ms, "ms"),
        _g("safe_play_start p95", rep["safe_play_start_ms"]["p95"], "<=",
           args.safe_start_pref_ms, "ms"),
        _g("stall_rate@%dms" % args.stall_buffer_ms,
           s.get("stall_rate@%d" % args.stall_buffer_ms, float("nan")), "==", 0.0),
    ]
    gates = {"mandatory": mandatory, "preferred": preferred,
             "n_cadence": s.get("n_cadence", 0), "n_records": s.get("n_records", 0)}
    if any(g["pass"] is False for g in mandatory):
        return "NOT STREAMABLE", gates
    if s.get("n_cadence", 0) == 0:
        # Mandatory gates that could be evaluated held, but no request delivered two
        # chunks, so continuity was never observed.  Saying GOOD here would be a claim
        # about a player that was never simulated.
        return "INCONCLUSIVE", gates
    if any(g["pass"] is False for g in preferred):
        return "MARGINAL", gates
    if any(g["pass"] is None for g in mandatory + preferred):
        return "INCONCLUSIVE", gates
    return "GOOD", gates


# --------------------------------------------------------------------------------------
# printing
# --------------------------------------------------------------------------------------
def fnum(v, w=7, p=2):
    return ("%*s" % (w, "n/a")) if v != v else ("%*.*f" % (w, p, v))


def print_table(reports, args, meta):
    W = 116
    print("=" * W)
    print("SERVING PROFILE  %s" % meta["started"])
    print("  server     %s  (%s)" % (meta["target"], meta["server_mode"]))
    print("  model      %s   engine %s   sample_rate %s Hz announced by /health"
          % (meta["health"].get("model"), meta["health"].get("engine"),
             meta["health"].get("sample_rate")))
    print("  audio      %s  (source: %s)" % (meta.get("format"), meta.get("format_source")))
    print("  route      %s   mode %s   waves %d   bank %d texts   max_steps %s"
          % (args.route, "stream" if args.stream else "batch", args.waves,
             len(meta["bank"]), args.max_steps or "-"))
    print("  host       %s" % meta["host_desc"])
    print("=" * W)
    print()
    cols = [("C", 3), ("done/lnc", 9), ("TTFB50", 7), ("TTFB95", 7), ("TTFA50", 7),
            ("TTFA95", 7), ("RTF50", 6), ("RTF95", 6), ("RTFsd", 6), ("preb50", 7),
            ("preb95", 7), ("safe95", 7), ("gap95", 7), ("stall", 6), ("aggRTF", 7),
            ("verdict", 0)]
    units = ["", "", "ms", "ms", "ms", "ms", "", "", "", "ms", "ms", "ms", "ms",
             "@%dms" % args.stall_buffer_ms, "wall/aud", ""]

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
        stall = s.get("stall_rate@%d" % args.stall_buffer_ms, float("nan"))
        print(row([r["level"], "%d/%d" % (r["completed"], r["launched"]),
                   fnum(r["ttfb_ms"]["p50"], 0, 1), fnum(r["ttfb_ms"]["p95"], 0, 1),
                   fnum(r["ttfa_ms"]["p50"], 0, 0), fnum(r["ttfa_ms"]["p95"], 0, 0),
                   fnum(r["stream_rtf"]["p50"], 0, 3), fnum(r["stream_rtf"]["p95"], 0, 3),
                   fnum(r["stream_rtf"]["sd"], 0, 3),
                   fnum(r["prebuffer_ms"]["p50"], 0, 0), fnum(r["prebuffer_ms"]["p95"], 0, 0),
                   fnum(r["safe_play_start_ms"]["p95"], 0, 0),
                   fnum(r["max_gap_ms"]["p95"], 0, 0),
                   ("n/a" if stall != stall else "%.0f%%" % (stall * 100.0)),
                   fnum(r["aggregate_rtf"], 0, 2), r["verdict"]]))
    print("-" * W)
    print()

    for r in reports:
        print("C%-2d  %s   (%d/%d completed, %d requests with a measurable cadence, "
              "wall %.1f s)"
              % (r["level"], r["verdict"], r["completed"], r["launched"],
                 r["gates"]["n_cadence"], r["wall_s"]))
        for kind in ("mandatory", "preferred"):
            for g in r["gates"][kind]:
                mark = {True: "PASS", False: "FAIL", None: "n/a "}[g["pass"]]
                print("       %s %-9s %-22s %s %s %s"
                      % (mark, kind, g["name"],
                         ("n/a" if g["value"] != g["value"] else "%.3f%s"
                          % (g["value"], g["unit"])),
                         g["op"], "%.3f%s" % (g["limit"], g["unit"])))
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
        print("       mix: %s (%d distinct request slots)"
              % (", ".join("%s x%d" % (k, v) for k, v in sorted(r["mix"].items()))
                 or "n/a", r["distinct_texts"]))
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
        if r["failed"]:
            print("       !! %d of %d requests did NOT complete; the statistics above "
                  "cover only the %d that did"
                  % (r["failed"], r["launched"], r["completed"]))
            for e in r["errors"]:
                print("          %s" % e)
        print()

    print("THRESHOLDS IN FORCE (all are flags; change them and the verdicts change)")
    print("  mandatory : completed == launched, STREAM_RTF p95 < %.2f" % args.rtf_hard)
    print("  preferred : TTFB p95 <= %.0f ms, TTFA p95 <= %.0f ms, STREAM_RTF p95 <= %.2f,"
          % (args.ttfb_pref_ms, args.ttfa_pref_ms, args.rtf_pref))
    print("              prebuffer p95 <= %.0f ms, safe_play_start p95 <= %.0f ms, "
          "stall_rate@%dms == 0"
          % (args.prebuffer_pref_ms, args.safe_start_pref_ms, args.stall_buffer_ms))
    print("  GOOD = every preferred gate met - MARGINAL = mandatory met, a preferred one "
          "missed")
    print("  NOT STREAMABLE = a mandatory gate failed - INCONCLUSIVE = nothing to judge "
          "(no request delivered two chunks, so no player was ever simulated)")
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

    load = ap.add_argument_group("load")
    load.add_argument("--levels", default="1,2,4",
                      help="concurrency levels, e.g. 1,2,4,8")
    load.add_argument("--waves", type=int, default=3,
                      help="synchronized waves per level (requests = level x waves)")
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
    gates.add_argument("--stall-buffer-ms", type=int, default=DEFAULTS["stall_buffer_ms"],
                       help="the jitter buffer whose stall rate must be zero")
    gates.add_argument("--buffers", default="100,250,500,1000",
                       help="jitter buffers to simulate, ms")

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
    args.buffers = tuple(int(x) for x in args.buffers.split(",") if x.strip())
    if args.stall_buffer_ms not in args.buffers:
        args.buffers = tuple(sorted(args.buffers + (args.stall_buffer_ms,)))
    levels = [int(x) for x in args.levels.split(",") if x.strip()]
    if not levels or min(levels) < 1:
        raise SystemExit("--levels must be positive integers, e.g. 1,2,4")

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
    else:
        bank = list(DEFAULT_BANK)

    proc = None
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    t_profile0 = time.perf_counter()
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
            if not args.quiet:
                print("running C%d (%d waves)..." % (level, args.waves), file=sys.stderr)
            records, launched, wall = prof.run_level(level)
            reports.append(level_report(level, records, launched, wall, args))
        health1 = http_get_json(args.host, args.port, "/health", timeout=10.0)
    finally:
        stop_server(proc)

    duration = time.perf_counter() - t_profile0
    meta = {"started": started, "duration_s": duration,
            "target": "http://%s:%d" % (args.host, args.port),
            "server_mode": "attached" if args.url else "started by this tool",
            "health": health0, "health_after": health1,
            "bank": [t for _c, t in bank],
            "format": repr(prof.fmt) if prof.fmt else None,
            "format_source": prof.fmt_source,
            "host_desc": host_description(),
            "argv": sys.argv[1:]}

    if not args.quiet:
        print()
        print_table(reports, args, meta)
        print()
        print("profile took %.1f s of wall clock (%d levels, %d requests total)"
              % (duration, len(reports), sum(r["launched"] for r in reports)))
        print("server counters after the run: %s" % json.dumps(health1.get("jobs", {})))

    if args.json:
        doc = {"meta": meta, "levels": reports,
               "thresholds": {k: getattr(args, k) for k in
                              ("rtf_hard", "rtf_pref", "ttfb_pref_ms", "ttfa_pref_ms",
                               "prebuffer_pref_ms", "safe_start_pref_ms",
                               "stall_buffer_ms")}}
        text = json.dumps(doc, indent=2, default=str)
        if args.json == "-":
            print(text)
        else:
            with open(args.json, "w") as f:
                f.write(text + "\n")

    worst = {"GOOD": 0, "INCONCLUSIVE": 1, "MARGINAL": 1, "NOT STREAMABLE": 2}
    return max(worst.get(r["verdict"], 2) for r in reports) if reports else 2


if __name__ == "__main__":
    sys.exit(main())
