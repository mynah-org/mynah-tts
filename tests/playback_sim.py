#!/usr/bin/env python3
"""playback_sim.py -- what a real 1x player would do with one chunk-arrival timeline.

This module is the single definition of every per-request playback metric used by
``tools/serving_profile.py``.  It takes CLIENT RECEIVE MARKS -- ``(t_rel_s, nbytes
[, blocked_s])`` in arrival order, ``t_rel_s`` measured from the moment the request was
sent -- and answers the only question that matters for streaming TTS: *when could a
player start, and would it stall?*

Why this exists at all
----------------------
``STREAM_RTF < 1`` (generation faster than realtime, on average) does NOT prove that a
player never stalls.  STREAM_RTF is a mean rate over the whole stream; stalling is a
property of the WORST moment in the arrival distribution.  A server that emits 2.5 s of
audio every 2.3 s has STREAM_RTF 0.92 and still forces a player to hold ~2 s of prebuffer
before it can start safely: the audio arrives in quanta, and between two quanta the
player drains at exactly 1x.  Capacity and continuity are different measurements, and
only the second one is a streamability verdict.  The derivation is in qwen-tts
``.work/professional-streaming-architecture.md`` (E1, E2: "the cadence law").

So the metrics here are max-lateness statistics, not averages.

No engine constants
-------------------
Nothing in this file knows the sample rate, the frame rate, the codec or the engine.
Bytes become seconds only through an explicit :class:`AudioFormat`, which the caller
builds from what the server actually announced (``X-Sample-Rate`` / ``X-Channels`` /
``X-Bits-Per-Sample``, or a WAV header).  Magpie at 22050 Hz and PocketTTS at whatever
rate it ships both work, and neither is baked in.

Definitions (per request, seconds unless the key says ``_ms``)
-------------------------------------------------------------
* ``ttfa_s``              -- ``t_0``, arrival of the first non-empty audio chunk.
* ``stream_rtf``          -- ``(t_done - t_0) / (audio delivered after the first chunk)``.
                             A mean rate: capacity, not continuity.
* ``required_prebuffer_s``-- ``max(0, max_{i>=1} [(t_i - t_0) - A_i])`` with ``A_i`` the
                             audio delivered before chunk ``i``.  The smallest fixed delay
                             after ``t_0`` at which a 1x player that then never pauses
                             finishes without a single underrun.
* ``safe_play_start_s``   -- ``max_{i>=0} (t_i - A_i)``.  The earliest ABSOLUTE instant
                             after the request was sent at which a player could press
                             play and never starve.  Computed by a direct scan; per
                             request it equals ``ttfa + required_prebuffer``.
* zero-buffer player      -- starts at ``t_0``, pauses whenever its buffer empties:
                             ``underrun_total_s``, ``stall_max_s``, ``stall_count``.
* fixed-buffer player @B  -- starts once B seconds of audio are held (or the stream ended),
                             re-buffers B after an underrun: ``stalls@B``, ``stall_ms@B``,
                             ``stall_max_ms@B``, ``start_delay_ms@B``.  B=0 reproduces the
                             zero-buffer player.
* ``max_gap_s``           -- largest inter-arrival gap after ``t_0`` (raw cadence).
* ``gap_ratio_max``       -- max over chunks of (gap / that chunk's audio duration).

Receive-marker honesty
----------------------
A mark is stamped when the client's reader returns one HTTP chunk, so socket, kernel and
interpreter buffering all sit between the server's write and the mark.  A reader that was
late finds several chunks already queued and returns them microseconds apart: the earlier
ones are then stamped LATE.  When the caller records how long each read blocked, a read
that returned in under ``COALESCE_S`` is counted in ``coalesced_chunks`` and the share is
reported -- every cadence value here is an UPPER bound on the server's real lateness.

Self-test: ``python3 tests/playback_sim.py`` (known-answer timelines, no network).
"""

import math

DEFAULT_BUFFERS_MS = (100, 250, 500, 1000)
COALESCE_S = 0.001          # a read faster than this returned already-queued data


class AudioFormat:
    """Bytes -> seconds, from what the server announced.  No engine defaults."""

    __slots__ = ("sample_rate", "channels", "bits_per_sample")

    def __init__(self, sample_rate, channels=1, bits_per_sample=16):
        if sample_rate <= 0 or channels <= 0 or bits_per_sample <= 0:
            raise ValueError("audio format must be positive: sr=%r ch=%r bits=%r"
                             % (sample_rate, channels, bits_per_sample))
        if bits_per_sample % 8:
            raise ValueError("bits_per_sample must be a whole number of bytes: %r"
                             % bits_per_sample)
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self.bits_per_sample = int(bits_per_sample)

    @property
    def bytes_per_second(self):
        return float(self.sample_rate * self.channels * (self.bits_per_sample // 8))

    def seconds(self, nbytes):
        return nbytes / self.bytes_per_second

    def __repr__(self):
        return ("AudioFormat(sample_rate=%d, channels=%d, bits_per_sample=%d)"
                % (self.sample_rate, self.channels, self.bits_per_sample))

    def __eq__(self, other):
        return (isinstance(other, AudioFormat) and
                (self.sample_rate, self.channels, self.bits_per_sample) ==
                (other.sample_rate, other.channels, other.bits_per_sample))


def _bps(fmt):
    """Accept an AudioFormat or a bare bytes-per-second number."""
    if isinstance(fmt, AudioFormat):
        return fmt.bytes_per_second
    v = float(fmt)
    if v <= 0:
        raise ValueError("bytes per second must be positive, got %r" % fmt)
    return v


# --------------------------------------------------------------------------------------
# small statistics (stdlib only; every aggregate says how many samples it used)
# --------------------------------------------------------------------------------------
def _finite(values):
    return [float(v) for v in values if v is not None and isinstance(v, (int, float))
            and not math.isnan(float(v))]


def pct(values, q):
    """Percentile by nearest rank.  NaN when there is nothing to rank."""
    v = sorted(_finite(values))
    if not v:
        return float("nan")
    return v[min(len(v) - 1, int(round(q / 100.0 * (len(v) - 1))))]


def mean(values):
    v = _finite(values)
    return sum(v) / len(v) if v else float("nan")


def stdev(values):
    """Sample standard deviation (n-1).  NaN below two samples -- one sample has no
    spread, and reporting 0 there would look like agreement that was never measured."""
    v = _finite(values)
    if len(v) < 2:
        return float("nan")
    m = sum(v) / len(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))


def spread(values):
    """The honest five: n, p50, p95, mean, sd, min, max."""
    v = _finite(values)
    return {"n": len(v), "p50": pct(v, 50), "p95": pct(v, 95),
            "mean": mean(v), "sd": stdev(v),
            "min": min(v) if v else float("nan"),
            "max": max(v) if v else float("nan")}


# --------------------------------------------------------------------------------------
# players
# --------------------------------------------------------------------------------------
class Playhead:
    """Incremental zero-buffer playback state (optionally delayed by a fixed prebuffer),
    so a live client can ask "how much has been HEARD?" while the stream still arrives."""

    def __init__(self, fmt, prebuffer_s=0.0):
        self.bps = _bps(fmt)
        self.B = float(prebuffer_s)
        self.t_first = None
        self.avail = 0.0
        self.played = 0.0
        self.t_prev = None
        self.stall_total = 0.0
        self.stall_count = 0
        self.worst_stall = 0.0
        self.deepest_deficit = float("-inf")
        self._in_stall = False

    def on_chunk(self, t, nbytes):
        self.advance(t)
        if self.t_first is None:
            self.t_first = t
            self.t_prev = t + self.B
        self.avail += nbytes / self.bps
        # Data landed: whatever starvation episode was open has just ended.  Without this
        # a stream that is late at EVERY arrival would be reported as one endless stall
        # instead of one stall per gap, and stall_count would understate the damage.
        if self.avail > self.played:
            self._in_stall = False

    def advance(self, t):
        """Move the playhead to wall time t without new data."""
        if self.t_first is None:
            return
        start = self.t_first + self.B
        if t <= start:
            return
        prev = self.t_prev if self.t_prev is not None else start
        gap = t - prev
        if gap <= 0:
            return
        want = self.played + gap
        deficit = want - self.avail
        self.deepest_deficit = max(self.deepest_deficit, deficit)
        if deficit > 0:
            self.stall_total += deficit
            self.worst_stall = max(self.worst_stall, deficit)
            if not self._in_stall:
                self.stall_count += 1
                self._in_stall = True
            self.played = self.avail
        else:
            self._in_stall = False
            self.played = want
        self.t_prev = t

    @property
    def heard(self):
        return self.played

    @property
    def buffered_unheard(self):
        return max(0.0, self.avail - self.played)


def _norm(marks):
    """[(t, nbytes, blocked_or_None)] in arrival order."""
    out = []
    for m in marks:
        t, nb = float(m[0]), int(m[1])
        blocked = float(m[2]) if len(m) > 2 and m[2] is not None else None
        out.append((t, nb, blocked))
    out.sort(key=lambda m: m[0])
    return out


def fixed_buffer_sim(marks, fmt, buffer_s):
    """A 1x player with an audio-based jitter buffer of ``buffer_s``.

    It starts once ``buffer_s`` of audio has arrived (or the stream ended, whichever comes
    first), pauses when its buffer runs dry, and resumes only when ``buffer_s`` of unplayed
    audio is buffered again (or the stream ended).  Nothing after the last arrival can
    stall, so the scan stops there.

    Returns ``(start_delay_s_from_t0, stall_total_s, stall_max_s, stall_count)``.
    """
    bps = _bps(fmt)
    ms = _norm(marks)
    if not ms:
        return float("nan"), float("nan"), float("nan"), 0
    t0 = ms[0][0]
    last = len(ms) - 1
    avail = played = stall_total = stall_max = 0.0
    playing = False
    start_at = None
    stall_from = None
    stall_count = 0
    for i, (t, nb, _b) in enumerate(ms):
        if playing and i > 0:
            gap = t - ms[i - 1][0]
            lead = avail - played
            if gap <= lead:
                played += gap
            else:
                played = avail
                stall_from = ms[i - 1][0] + lead      # the instant the buffer ran dry
                stall_count += 1
                playing = False
        avail += nb / bps
        if not playing:
            if (avail - played) >= buffer_s - 1e-12 or i == last:
                playing = True
                if start_at is None:
                    start_at = t
                elif stall_from is not None:
                    d = t - stall_from
                    stall_total += d
                    stall_max = max(stall_max, d)
                    stall_from = None
    return (start_at - t0 if start_at is not None else float("nan"),
            stall_total, stall_max, stall_count)


def timeline_kpis(marks, t_done, fmt, buffers_ms=DEFAULT_BUFFERS_MS):
    """Every per-request playback metric derived from one arrival timeline.

    ``t_done`` is when the response finished (the last byte / the terminating chunk).
    A request with fewer than two chunks HAS NO CADENCE: those keys come back NaN rather
    than 0, so an aggregate can exclude them instead of averaging a fiction.
    """
    bps = _bps(fmt)
    ms = _norm(marks)
    n = len(ms)
    out = {
        "chunks": n,
        "coalesced_chunks": sum(1 for m in ms if m[2] is not None and m[2] < COALESCE_S),
        "blocked_reads_known": any(m[2] is not None for m in ms),
        "bytes": sum(nb for _t, nb, _b in ms),
    }
    if n == 0:
        out.update({"ttfa_s": float("nan"), "delivered_s": 0.0})
    else:
        out["ttfa_s"] = ms[0][0]
        out["delivered_s"] = sum(nb for _t, nb, _b in ms) / bps

    if n < 2:
        for k in ("stream_rtf", "required_prebuffer_s", "safe_play_start_s",
                  "underrun_total_s", "stall_max_s", "max_gap_s", "gap_ratio_max"):
            out[k] = float("nan")
        out["stall_count"] = 0
        out["gap_ratios"] = []
        for b in buffers_ms:
            out["start_delay_ms@%d" % b] = float("nan")
            out["stall_ms@%d" % b] = float("nan")
            out["stall_max_ms@%d" % b] = float("nan")
            out["stalls@%d" % b] = 0
        return out

    t0 = ms[0][0]
    rest = sum(nb for _t, nb, _b in ms[1:]) / bps
    out["stream_rtf"] = (t_done - t0) / rest if rest > 0 else float("nan")

    # required_prebuffer and safe_play_start, by a direct scan of the timeline.
    A = 0.0
    need = 0.0
    safe = float("-inf")
    for i, (t, nb, _b) in enumerate(ms):
        safe = max(safe, t - A)                      # i = 0 contributes t_0 (A_0 = 0)
        if i > 0:
            need = max(need, (t - t0) - A)
        A += nb / bps
    out["required_prebuffer_s"] = max(0.0, need)
    out["safe_play_start_s"] = safe

    # Zero-buffer player.
    ph = Playhead(bps, 0.0)
    for t, nb, _b in ms:
        ph.on_chunk(t, nb)
    ph.advance(ms[-1][0])                            # nothing after the last arrival stalls
    out["underrun_total_s"] = ph.stall_total
    out["stall_max_s"] = ph.worst_stall
    out["stall_count"] = ph.stall_count

    # Raw cadence.
    gaps = [ms[i][0] - ms[i - 1][0] for i in range(1, n)]
    out["max_gap_s"] = max(gaps)
    ratios = [g * bps / ms[i][1] for i, g in zip(range(1, n), gaps) if ms[i][1] > 0]
    out["gap_ratio_max"] = max(ratios) if ratios else float("nan")
    out["gap_ratios"] = ratios

    # Fixed audio-based jitter buffers.
    for b in buffers_ms:
        start, total, worst, count = fixed_buffer_sim(ms, bps, b / 1000.0)
        out["start_delay_ms@%d" % b] = start * 1000.0
        out["stall_ms@%d" % b] = total * 1000.0
        out["stall_max_ms@%d" % b] = worst * 1000.0
        out["stalls@%d" % b] = count
    return out


def summarize(records, buffers_ms=DEFAULT_BUFFERS_MS):
    """Aggregate per-request dicts.  Every scalar family reports n/p50/p95/mean/sd, and
    ``n_cadence`` says how many requests actually had a cadence to measure."""
    def col(key):
        return [r.get(key) for r in records]

    out = {"n_records": len(records)}
    for key in ("ttfa_s", "ttfb_s", "stream_rtf", "required_prebuffer_s",
                "safe_play_start_s", "underrun_total_s", "stall_max_s",
                "max_gap_s", "gap_ratio_max", "delivered_s"):
        vals = col(key)
        if any(v is not None for v in vals):
            out[key] = spread(vals)

    cadence = [r for r in records
               if not math.isnan(float(r.get("required_prebuffer_s", float("nan"))))]
    out["n_cadence"] = len(cadence)
    for b in buffers_ms:
        key = "stalls@%d" % b
        stalled = [r for r in cadence if (r.get(key) or 0) > 0]
        out["stall_rate@%d" % b] = (len(stalled) / len(cadence)) if cadence else float("nan")
        out["stall_ms@%d" % b] = spread(col("stall_ms@%d" % b))
        out["stall_max_ms@%d" % b] = spread(col("stall_max_ms@%d" % b))
        out["start_delay_ms@%d" % b] = spread(col("start_delay_ms@%d" % b))
        le = [r for r in cadence
              if r.get("required_prebuffer_s", float("inf")) <= b / 1000.0 + 1e-9]
        out["prebuffer_le_rate@%d" % b] = (len(le) / len(cadence)) if cadence else float("nan")

    chunks = sum(r.get("chunks", 0) for r in records)
    coal = sum(r.get("coalesced_chunks", 0) for r in records)
    known = any(r.get("blocked_reads_known") for r in records)
    out["chunks_total"] = chunks
    out["coalesced_chunk_share"] = (coal / chunks) if (chunks and known) else float("nan")
    return out


def format_summary(s, buffers_ms=DEFAULT_BUFFERS_MS, indent="  "):
    """Three standard lines for a harness report."""
    def ms(d, k="p50"):
        v = d.get(k, float("nan")) if isinstance(d, dict) else d
        return "n/a" if v != v else "%.0f" % (v * 1000.0)

    pre = s.get("required_prebuffer_s", {})
    safe = s.get("safe_play_start_s", {})
    l1 = (indent + "PLAYBACK (client-observed, n=%d of %d): required_prebuffer p50/p95 "
          "%s/%s ms (sd %s ms) - safe_play_start p50/p95 %s/%s ms - stall_max p95 %s ms "
          "- underrun_total p95 %s ms - max_gap p95 %s ms"
          % (s.get("n_cadence", 0), s.get("n_records", 0),
             ms(pre), ms(pre, "p95"), ms(pre, "sd"), ms(safe), ms(safe, "p95"),
             ms(s.get("stall_max_s", {}), "p95"),
             ms(s.get("underrun_total_s", {}), "p95"),
             ms(s.get("max_gap_s", {}), "p95")))
    parts = []
    for b in buffers_ms:
        rate = s.get("stall_rate@%d" % b, float("nan"))
        parts.append("@%dms stall_rate %s (total p95 %.0f ms, start p95 %.0f ms)"
                     % (b, "n/a" if rate != rate else "%.0f%%" % (rate * 100.0),
                        s.get("stall_ms@%d" % b, {}).get("p95", float("nan")),
                        s.get("start_delay_ms@%d" % b, {}).get("p95", float("nan"))))
    l2 = indent + "FIXED BUFFER: " + " - ".join(parts)
    coal = s.get("coalesced_chunk_share", float("nan"))
    l3 = (indent + "RECEIVE FIDELITY: %.1f%% of reads returned already-queued data "
          "(cadence values are upper bounds on server lateness)" % (coal * 100.0)
          if coal == coal else
          indent + "RECEIVE FIDELITY: blocked-read times not recorded")
    return "\n".join((l1, l2, l3))


# --------------------------------------------------------------------------------------
# deterministic synthetic timelines -- the self-test below, and anything that needs a
# known answer without a server
# --------------------------------------------------------------------------------------
FMT_22K = AudioFormat(22050)       # Magpie/NanoCodec, used by the self-test
FMT_24K = AudioFormat(24000)       # a different rate, to prove nothing is hardcoded


def chunk_bytes(seconds, fmt):
    return int(round(seconds * _bps(fmt)))


def synth(kind, fmt=FMT_22K, chunk_s=0.5, n=8, t0=0.3, rtf=0.8):
    """Known-answer timelines.  Every chunk carries ``chunk_s`` of audio."""
    nb = chunk_bytes(chunk_s, fmt)
    if kind == "smooth":               # each chunk arrives rtf * chunk_s after the last
        return [(t0 + i * chunk_s * rtf, nb) for i in range(n)]
    if kind == "exact":                # arrives exactly as fast as it is consumed
        return [(t0 + i * chunk_s, nb) for i in range(n)]
    if kind == "bursty":               # one chunk, then quanta of four at a time
        marks = [(t0, nb)]
        for k in range(1, 1 + (n - 1) // 4):
            marks += [(t0 + k * 4 * chunk_s * rtf, nb)] * 4
        return marks
    if kind == "late_first":           # tiny first chunk, a long wait, then smooth
        small = chunk_bytes(0.08, fmt)
        marks = [(t0, small)]
        marks += [(t0 + 2.0 + i * chunk_s * rtf, nb) for i in range(n - 1)]
        return marks
    if kind == "repeated_gap":         # smooth, but every third chunk is 0.4 s late
        marks = []
        t = t0
        for i in range(n):
            if i > 0:
                t += chunk_s * rtf + (0.4 if i % 3 == 0 else 0.0)
            marks.append((t, nb))
        return marks
    if kind == "slower_than_realtime":  # generation slower than playback: stalls forever
        return [(t0 + i * chunk_s * 1.5, nb) for i in range(n)]
    raise ValueError(kind)


# --------------------------------------------------------------------------------------
# self-test: known answers, no network, no server, no model
# --------------------------------------------------------------------------------------
def _selftest():
    ok = True
    cases = []

    def check(name, got, want, tol=1e-6):
        nonlocal ok
        good = (got != got and want != want) or abs(got - want) <= tol
        ok = ok and good
        cases.append((good, name, got, want))
        print("  %s %-52s got %12.6f  want %12.6f"
              % ("ok  " if good else "FAIL", name, got, want))

    def check_true(name, got):
        nonlocal ok
        ok = ok and bool(got)
        cases.append((bool(got), name, got, True))
        print("  %s %-52s %s" % ("ok  " if got else "FAIL", name, bool(got)))

    F = FMT_22K
    print("playback_sim self-test (%r, %.0f bytes/s)" % (F, F.bytes_per_second))

    # --- CASE 1: a stream that CANNOT stall -------------------------------------------
    # 0.5 s chunks arriving every 0.4 s: the player gains 0.1 s of lead per chunk.
    k = timeline_kpis(synth("smooth", F), t_done=4.0, fmt=F)
    check("smooth/required_prebuffer", k["required_prebuffer_s"], 0.0)
    check("smooth/safe_play_start == ttfa", k["safe_play_start_s"], k["ttfa_s"])
    check("smooth/stall_count", k["stall_count"], 0)
    check("smooth/underrun_total", k["underrun_total_s"], 0.0)
    for b in DEFAULT_BUFFERS_MS:
        check("smooth/stalls@%d" % b, k["stalls@%d" % b], 0)
    check("smooth/max_gap", k["max_gap_s"], 0.4)
    check("smooth/gap_ratio_max", k["gap_ratio_max"], 0.8)

    # --- CASE 2: a stream that MUST stall ---------------------------------------------
    # 0.5 s chunks arriving every 0.75 s: the player loses 0.25 s per chunk, forever.
    # Seven gaps of 0.75 s consume 7 x 0.5 s of audio -> 7 x 0.25 s of underrun, and the
    # zero-buffer player stalls at every single gap.
    m = synth("slower_than_realtime", F)
    k = timeline_kpis(m, t_done=m[-1][0], fmt=F)
    check("slow/stall_count", k["stall_count"], 7)
    check("slow/underrun_total", k["underrun_total_s"], 7 * 0.25, 1e-6)
    check("slow/stall_max", k["stall_max_s"], 0.25, 1e-6)
    check("slow/required_prebuffer", k["required_prebuffer_s"], 7 * 0.25, 1e-6)
    check("slow/safe_play_start", k["safe_play_start_s"], 0.3 + 7 * 0.25, 1e-6)
    check("slow/stream_rtf", k["stream_rtf"], (7 * 0.75) / (7 * 0.5), 1e-9)
    # A jitter buffer cannot fix a stream that is slower than realtime: it only defers
    # the first underrun.  The exact count depends on the re-buffer policy, so the
    # falsifiable claim is "still stalls", not a number.
    check_true("slow/a 1 s buffer still stalls", k["stalls@1000"] >= 1)

    # --- CASE 3: exactly at the limit --------------------------------------------------
    # 0.5 s chunks arriving every 0.5 s.  The buffer touches empty at every arrival and
    # never goes below: zero stalls, zero prebuffer, STREAM_RTF exactly 1.
    m = synth("exact", F)
    k = timeline_kpis(m, t_done=m[-1][0], fmt=F)
    check("exact/required_prebuffer", k["required_prebuffer_s"], 0.0, 1e-9)
    check("exact/stall_count", k["stall_count"], 0)
    check("exact/underrun_total", k["underrun_total_s"], 0.0, 1e-9)
    check("exact/stream_rtf", k["stream_rtf"], 1.0, 1e-9)
    check("exact/safe_play_start == ttfa", k["safe_play_start_s"], k["ttfa_s"], 1e-9)
    # One millisecond late on a single chunk, and the same timeline stalls: the limit is
    # a knife edge, which is exactly why a verdict needs a margin and not RTF < 1.
    m2 = list(m)
    m2[4] = (m2[4][0] + 0.001, m2[4][1])
    k2 = timeline_kpis(m2, t_done=m2[-1][0], fmt=F)
    check("exact+1ms/stall_count", k2["stall_count"], 1)
    check("exact+1ms/required_prebuffer", k2["required_prebuffer_s"], 0.001, 1e-9)
    # A 100 ms audio-based buffer does NOT absorb it: the first chunk already carries
    # 0.5 s, so the player starts immediately and has exactly the lead it had before.
    # A jitter buffer only helps when it makes the player WAIT.
    check("exact+1ms/stalls@100", k2["stalls@100"], 1)
    check("exact+1ms/stalls@1000", k2["stalls@1000"], 0)      # this one does wait
    check("exact+1ms/start_delay_ms@1000", k2["start_delay_ms@1000"], 500.0, 1e-6)

    # --- CASE 4: bursty delivery, the cadence law ---------------------------------------
    # 0.5 s at t=0.3, then four 0.5 s chunks together at 1.9 and at 3.5.  STREAM_RTF is
    # 0.8 -- "faster than realtime" -- yet a player starting at first audio runs dry for
    # 1.1 s.  This is the whole point of the module.
    m = synth("bursty", F, n=9)
    k = timeline_kpis(m, t_done=4.0, fmt=F)
    check("bursty/stream_rtf", k["stream_rtf"], (4.0 - 0.3) / 4.0, 1e-9)
    check("bursty/required_prebuffer", k["required_prebuffer_s"], 1.1, 1e-6)
    check("bursty/max_gap", k["max_gap_s"], 1.6, 1e-6)
    check("bursty/stalls@250", k["stalls@250"], 1)
    check("bursty/stall_ms@250", k["stall_ms@250"], 1100.0, 1e-3)
    check("bursty/stalls@1000", k["stalls@1000"], 0)         # waits for the quantum
    check("bursty/start_delay_ms@1000", k["start_delay_ms@1000"], 1600.0, 1e-3)

    # --- CASE 5: a late first chunk ------------------------------------------------------
    m = synth("late_first", F)
    k = timeline_kpis(m, t_done=6.0, fmt=F)
    check("late_first/required_prebuffer", k["required_prebuffer_s"], 2.0 - 0.08, 1e-6)
    check("late_first/safe_play_start", k["safe_play_start_s"], 0.3 + 2.0 - 0.08, 1e-6)
    check("late_first/max_gap", k["max_gap_s"], 2.0, 1e-6)
    check("late_first/stall_count", k["stall_count"], 1)
    check("late_first/stalls@1000", k["stalls@1000"], 0)     # starts once 1 s is held
    check("late_first/start_delay_ms@1000", k["start_delay_ms@1000"], 2400.0, 1e-3)

    # --- CASE 6: repeated gaps ------------------------------------------------------------
    m = synth("repeated_gap", F)
    k = timeline_kpis(m, t_done=6.0, fmt=F)
    check("repeated_gap/stall_count", k["stall_count"], 2)
    check("repeated_gap/required_prebuffer", k["required_prebuffer_s"], 0.2, 1e-6)
    check("repeated_gap/stalls@500", k["stalls@500"], 2)
    check("repeated_gap/stalls@1000", k["stalls@1000"], 0)

    # --- CASE 7: invariants that must hold on every timeline -------------------------------
    for kind in ("smooth", "exact", "bursty", "late_first", "repeated_gap",
                 "slower_than_realtime"):
        m = synth(kind, F, n=9)
        k = timeline_kpis(m, m[-1][0] + 0.1, F)
        check("%s/identity safe = ttfa + prebuffer" % kind, k["safe_play_start_s"],
              k["ttfa_s"] + k["required_prebuffer_s"], 1e-9)
        check("%s/zero-buffer == fixed@0" % kind, fixed_buffer_sim(m, F, 0.0)[1],
              k["underrun_total_s"], 1e-9)

    # --- CASE 8: the sample rate is not baked in --------------------------------------------
    # The same timeline shape at 24 kHz must give the same seconds-domain answers.
    a = timeline_kpis(synth("bursty", FMT_22K, n=9), 4.0, FMT_22K)
    b = timeline_kpis(synth("bursty", FMT_24K, n=9), 4.0, FMT_24K)
    check("rate-independence/prebuffer", b["required_prebuffer_s"],
          a["required_prebuffer_s"], 1e-6)
    check("rate-independence/stream_rtf", b["stream_rtf"], a["stream_rtf"], 1e-6)
    # Stereo/8-bit arithmetic goes through the same door.
    st = AudioFormat(22050, channels=2, bits_per_sample=16)
    check("stereo/bytes_per_second", st.bytes_per_second, 22050 * 4.0)
    check("stereo/seconds", st.seconds(22050 * 4), 1.0)

    # --- CASE 9: degenerate streams -----------------------------------------------------------
    k = timeline_kpis([(0.5, chunk_bytes(1.0, F))], 0.6, F)
    check("single-chunk/chunks", k["chunks"], 1)
    check("single-chunk/prebuffer is NaN", k["required_prebuffer_s"], float("nan"))
    check("single-chunk/stalls@250", k["stalls@250"], 0)
    s = summarize([k])
    check("single-chunk/n_cadence", s["n_cadence"], 0)
    check("single-chunk/n_records", s["n_records"], 1)
    k = timeline_kpis([], 0.0, F)
    check("empty/chunks", k["chunks"], 0)
    check("empty/delivered", k["delivered_s"], 0.0)
    # A stream shorter than the jitter buffer must still start (at the last chunk),
    # not never.
    marks = [(0.2, chunk_bytes(0.1, F)), (0.3, chunk_bytes(0.1, F)),
             (0.4, chunk_bytes(0.1, F))]
    start, total, worst, count = fixed_buffer_sim(marks, F, 1.0)
    check("short-utterance/start_delay", start, 0.2, 1e-9)
    check("short-utterance/stalls", count, 0)

    # --- CASE 10: aggregation honesty -----------------------------------------------------
    recs = [timeline_kpis(synth(kind, F, n=9), 6.0, F)
            for kind in ("smooth", "bursty", "late_first", "repeated_gap")]
    s = summarize(recs)
    check("summary/n_cadence", s["n_cadence"], 4)
    check("summary/stall_rate@250", s["stall_rate@250"], 2 / 4.0)
    check("summary/stall_rate@1000", s["stall_rate@1000"], 0.0)
    check("summary/prebuffer_le_rate@250", s["prebuffer_le_rate@250"], 2 / 4.0)
    check("summary/sd is reported", s["required_prebuffer_s"]["sd"] > 0.0, True)
    check("summary/sd of one sample is NaN", stdev([1.0]), float("nan"))
    check("summary/pct nearest rank p50", pct([1, 2, 3, 4, 5], 50), 3)
    check("summary/pct nearest rank p95", pct([1, 2, 3, 4, 5], 95), 5)
    # Coalesced reads: the third mark returned in 20 us, so it was already queued.
    marks = [(0.3, chunk_bytes(0.5, F), 0.3), (0.7, chunk_bytes(0.5, F), 0.39),
             (0.70002, chunk_bytes(0.5, F), 0.00002)]
    k = timeline_kpis(marks, 1.0, F)
    check("coalesced/counted", k["coalesced_chunks"], 1)
    check("coalesced/share", summarize([k])["coalesced_chunk_share"], 1 / 3.0, 1e-9)

    print()
    print(format_summary(summarize(recs)))
    print()
    bad = [c for c in cases if not c[0]]
    print("  %d checks, %d failed -- %s"
          % (len(cases), len(bad), "PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(_selftest())
