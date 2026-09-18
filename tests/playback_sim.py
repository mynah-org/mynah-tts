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


def timeline_kpis(marks, t_done, fmt, buffers_ms=DEFAULT_BUFFERS_MS, ttfb_s=None):
    """Every per-request playback metric derived from one arrival timeline.

    ``t_done`` is when the response finished (the last byte / the terminating chunk).
    A request with fewer than two chunks HAS NO CADENCE: those keys come back NaN rather
    than 0, so an aggregate can exclude them instead of averaging a fiction.

    ``ttfb_s`` is the only input a timeline cannot contain: the response header arrives
    before any audio, so the caller stamps it.  It is carried here, rather than kept in
    the harness, so that TTFB and TTFA have ONE definition and the difference between
    them (``header_to_audio_s``) is computed once.  This server sends the streaming
    header from the writer thread before synthesis starts, so TTFB can be milliseconds
    while TTFA is seconds; quoting TTFB as "latency" is the classic way to publish a
    number no listener experiences.
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
    out["ttfb_s"] = float("nan") if ttfb_s is None else float(ttfb_s)
    if n == 0:
        out.update({"ttfa_s": float("nan"), "delivered_s": 0.0})
    else:
        out["ttfa_s"] = ms[0][0]
        out["delivered_s"] = sum(nb for _t, nb, _b in ms) / bps
    # TTFB and TTFA are different measurements; the gap between them is the header lead.
    out["header_to_audio_s"] = out["ttfa_s"] - out["ttfb_s"]

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
    for key in ("ttfa_s", "ttfb_s", "header_to_audio_s", "stream_rtf",
                "required_prebuffer_s", "safe_play_start_s", "underrun_total_s",
                "stall_max_s", "max_gap_s", "gap_ratio_max", "delivered_s"):
        vals = col(key)
        if any(v is not None for v in vals):
            out[key] = spread(vals)

    cadence = [r for r in records
               if not math.isnan(float(r.get("required_prebuffer_s", float("nan"))))]
    out["n_cadence"] = len(cadence)
    for b in buffers_ms:
        key = "stalls@%d" % b
        stalled = [r for r in cadence if (r.get(key) or 0) > 0]
        # The COUNT as well as the rate.  A mandatory gate that reads "0.000 == 0.000
        # FAIL" is unreadable: at 54360 requests a rate rounds to zero from 1 stall and
        # from 27, and those are different verdicts about the same server.
        out["stall_n@%d" % b] = len(stalled)
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


# ======================================================================================
# THE QUALIFICATION ENVELOPE  (.work/streaming-cadence.md section 5)
# ======================================================================================
# GOOD = every preferred gate met.  MARGINAL = every mandatory met, a preferred missed.
# NOT STREAMABLE = a mandatory gate failed.  INCONCLUSIVE = nothing to judge.
#
# Which metric GATES is the whole content of the note, so it is stated once, here:
#
#   STREAM_RTF is a CAPACITY metric.  It is mandatory (< 1) because a stream slower than
#   realtime can never be fixed by buffering, and it is preferred (<= 0.90) because
#   running at the edge leaves no margin.  It is NOT a continuity verdict and it must
#   never be the metric that promotes a configuration.
#
#   required_prebuffer p95 and stall_rate@250 are what QUALIFY.  They are max-lateness
#   statistics, they are what a listener experiences, and they are the pair that moved
#   while STREAM_RTF sat still in every quantum sweep on record.
#
# Every threshold is data, printed next to the value it judged, so a verdict can be
# falsified by reading the table instead of trusting it.
ENVELOPE = {
    # mandatory -- failing one is NOT STREAMABLE
    "rtf_hard": 1.00,             # STREAM_RTF p95 <  1.00
    "stall_mandatory_ms": 500,    # stall_rate@500ms == 0
    # preferred -- failing one is MARGINAL
    "ttfb_pref_ms": 100.0,
    "ttfa_pref_ms": 500.0,
    "rtf_pref": 0.90,
    "prebuffer_pref_ms": 500.0,
    "safe_start_pref_ms": 1000.0,
    "stall_pref_ms": 250,         # stall_rate@250ms == 0  -- the qualifying gate
    # strong -- reported, never a verdict on its own
    "rtf_strong": 0.85,
    "prebuffer_strong_ms": 300.0,
    "safe_start_strong_ms": 800.0,
}

# A run whose marks are mostly already-queued data cannot support a cadence percentile.
# The reference declared a run with 33-37% coalesced reads NOT QUOTABLE; we refuse at a
# deliberately stricter 15%, because below that the numbers are still upper bounds and
# above it they are fiction.
COALESCED_REFUSE_SHARE = 0.15
COALESCED_WARN_SHARE = 0.05


def gate(name, value, op, limit, unit="", kind="preferred", note=""):
    """One comparison, carrying everything needed to re-check it by hand.

    ``pass`` is None when the value is NaN -- not measured is not the same as failed,
    and collapsing the two is how an unmeasured level becomes GOOD.
    """
    v = float("nan") if value is None else float(value)
    if v != v:
        passed = None
    elif op == "<":
        passed = v < limit
    elif op == "<=":
        passed = v <= limit
    elif op == "==":
        passed = v == limit
    elif op == ">=":
        passed = v >= limit
    else:
        raise ValueError("unknown comparison %r" % op)
    return {"name": name, "value": v, "op": op, "limit": float(limit),
            "unit": unit, "kind": kind, "pass": passed, "note": note}


def qualify(summary, completed, launched, env=None):
    """The single definition of the streaming verdict.

    Takes an aggregate from :func:`summarize` plus the completed/launched counts, and
    returns ``(verdict, gates)``.  No harness reimplements this: a second copy of an
    envelope is a second product.
    """
    e = dict(ENVELOPE)
    if env:
        e.update(env)
    s = summary or {}

    def sp(key, field="p95"):
        d = s.get(key)
        return d.get(field, float("nan")) if isinstance(d, dict) else float("nan")

    def ms(key, field="p95"):
        v = sp(key, field)
        return v * 1000.0 if v == v else v

    def stall_note(b):
        n = s.get("stall_n@%d" % b)
        total = s.get("n_cadence")
        if n is None or not total:
            return ""
        return "%d of %d requests" % (n, total)

    mandatory = [
        gate("completed == launched", float(completed), "==", float(launched),
             kind="mandatory"),
        gate("STREAM_RTF p95", sp("stream_rtf"), "<", e["rtf_hard"], kind="mandatory"),
        gate("stall_rate@%dms" % e["stall_mandatory_ms"],
             s.get("stall_rate@%d" % e["stall_mandatory_ms"], float("nan")),
             "==", 0.0, kind="mandatory",
             note=stall_note(e["stall_mandatory_ms"])),
    ]
    preferred = [
        gate("TTFB p95", ms("ttfb_s"), "<=", e["ttfb_pref_ms"], "ms"),
        gate("TTFA p95", ms("ttfa_s"), "<=", e["ttfa_pref_ms"], "ms"),
        gate("STREAM_RTF p95", sp("stream_rtf"), "<=", e["rtf_pref"]),
        gate("required_prebuffer p95", ms("required_prebuffer_s"), "<=",
             e["prebuffer_pref_ms"], "ms"),
        gate("safe_play_start p95", ms("safe_play_start_s"), "<=",
             e["safe_start_pref_ms"], "ms"),
        gate("stall_rate@%dms" % e["stall_pref_ms"],
             s.get("stall_rate@%d" % e["stall_pref_ms"], float("nan")), "==", 0.0,
             note=stall_note(e["stall_pref_ms"])),
    ]
    strong = [
        gate("STREAM_RTF p95", sp("stream_rtf"), "<=", e["rtf_strong"], kind="strong"),
        gate("required_prebuffer p95", ms("required_prebuffer_s"), "<=",
             e["prebuffer_strong_ms"], "ms", kind="strong"),
        gate("safe_play_start p95", ms("safe_play_start_s"), "<=",
             e["safe_start_strong_ms"], "ms", kind="strong"),
    ]
    gates = {"mandatory": mandatory, "preferred": preferred, "strong": strong,
             "envelope": e, "n_cadence": s.get("n_cadence", 0),
             "n_records": s.get("n_records", 0)}

    if any(g["pass"] is False for g in mandatory):
        return "NOT STREAMABLE", gates
    if s.get("n_cadence", 0) == 0:
        # Mandatory gates that could be evaluated held, but no request delivered two
        # chunks, so continuity was never observed.  GOOD here would be a claim about a
        # player that was never simulated.
        return "INCONCLUSIVE", gates
    if any(g["pass"] is False for g in preferred):
        return "MARGINAL", gates
    if any(g["pass"] is None for g in mandatory + preferred):
        return "INCONCLUSIVE", gates
    return "GOOD", gates


def capacity(level_verdicts):
    """The operating point: the highest concurrency that is GOOD with NO gap below it.

    Discovered, not prescribed.  ``level_verdicts`` is ``{concurrency: verdict}``.
    A GOOD level above a non-GOOD one does not extend capacity -- an island of GOOD at
    C8 above a MARGINAL C4 is a measurement to explain, not a product point to ship.
    Returns ``None`` when even the lowest level is not GOOD.
    """
    best = None
    for c in sorted(level_verdicts):
        if level_verdicts[c] == "GOOD":
            best = c
        else:
            break
    return best


# ======================================================================================
# REFUSALS  (.work/engineering-method.md section 4: "every tool declares a refusal")
# ======================================================================================
class Refusal(Exception):
    """Raised instead of printing a number nobody should trust."""

    def __init__(self, reasons):
        self.reasons = list(reasons)
        Exception.__init__(self, "; ".join(self.reasons))


def quotable(summary, refuse_share=COALESCED_REFUSE_SHARE,
             warn_share=COALESCED_WARN_SHARE):
    """May this run's CADENCE PERCENTILES be quoted?  ``(status, share, reasons)``.

    ``status`` is ``"QUOTABLE"``, ``"QUOTABLE (warn)"``, ``"NOT QUOTABLE"`` or
    ``"UNKNOWN"``.

    The client mark is stamped when ``read1()`` returns.  A reader that was late finds
    several chunks already queued and returns them microseconds apart, so N server
    emissions appear as N marks in one instant: the earlier marks are stamped LATE and
    every cadence statistic derived from them is fiction in the tail and an upper bound
    everywhere else.  When the share of such reads is high the percentiles measure the
    CLIENT, not the server.

    UNKNOWN when blocked-read times were never recorded -- that is not a pass.  A
    harness that does not record how long each read blocked cannot know whether its own
    numbers are real, and must say so rather than assume zero.
    """
    share = summary.get("coalesced_chunk_share", float("nan")) if summary else float("nan")
    if share != share:
        return ("UNKNOWN", share,
                ["blocked-read times were not recorded, so the coalesced-read share is "
                 "unknown and the cadence percentiles cannot be certified"])
    if share > refuse_share:
        return ("NOT QUOTABLE", share,
                ["%.1f%% of reads returned already-queued data (refuse above %.0f%%): "
                 "the cadence percentiles describe the client's reader, not the "
                 "server's emission" % (share * 100.0, refuse_share * 100.0)])
    if share > warn_share:
        return ("QUOTABLE (warn)", share,
                ["%.1f%% of reads returned already-queued data (warn above %.0f%%): "
                 "cadence values are upper bounds on server lateness"
                 % (share * 100.0, warn_share * 100.0)])
    return ("QUOTABLE", share, [])


# The facts that must be identical before two arms may be differenced.  "Dispatch
# resolution" is the qwen-tts term: which kernels actually ran.  A sweep that changes
# the ISA, the quantization, the thread count or the backend along with the variable
# under test has measured their sum and can attribute it to nothing.
COMPARABLE_KEYS = ("engine", "isa", "simd", "backend", "quant", "threads",
                   "sample_rate", "build_flags", "server_bin", "route", "stream")


def comparable(arm_a, arm_b, keys=COMPARABLE_KEYS):
    """``(ok, differences)`` -- may these two arms be differenced?

    Each arm is a dict of resolved facts.  A key missing from BOTH is not a difference
    (nobody measured it); a key present in one and missing in the other IS, because an
    unrecorded fact is not an equal fact.
    """
    diffs = []
    for k in keys:
        if k not in arm_a and k not in arm_b:
            continue
        va, vb = arm_a.get(k, "<unrecorded>"), arm_b.get(k, "<unrecorded>")
        if va != vb:
            diffs.append("%s: %r vs %r" % (k, va, vb))
    return (not diffs), diffs


UNRECORDED = "<unrecorded>"


def unverifiable_keys(arms, keys=COMPARABLE_KEYS):
    """Keys that are ``<unrecorded>`` in EVERY arm.

    These are the dangerous ones.  Two arms that both say ``<unrecorded>`` compare
    equal, but nobody measured anything: equality of two unknowns is not sameness.  A
    sweep whose quantization silently differed would pass a naive check.
    """
    return [k for k in keys
            if all(str(a.get(k, UNRECORDED)) == UNRECORDED for a in arms.values())]


def require_comparable(arms, keys=COMPARABLE_KEYS, allow_unverified=False):
    """Raise :class:`Refusal` unless every arm PROVABLY resolved the same dispatch.

    Two failure modes, and they are different:

    * a key that differs      -> the arms measured different things.  Always refuses.
    * a key nobody recorded   -> the arms MIGHT have measured different things and
      there is no evidence either way.  Refuses unless ``allow_unverified``, which the
      caller may set only when sameness is established by construction (same binary,
      same environment, same process launch) rather than by the report.
    """
    labels = sorted(arms)
    reasons = []
    for i in range(1, len(labels)):
        ok, diffs = comparable(arms[labels[0]], arms[labels[i]], keys)
        if not ok:
            reasons.append("arm %r and arm %r did not resolve the same dispatch (%s)"
                           % (labels[0], labels[i], "; ".join(diffs)))
    if reasons:
        raise Refusal(reasons + ["refusing to compare arms whose dispatch resolution "
                                 "differs: the difference between them is not the "
                                 "variable under test"])
    unver = unverifiable_keys(arms, keys)
    if unver and not allow_unverified:
        raise Refusal(["no arm recorded %s, so their sameness is an assumption, not a "
                       "measurement" % ", ".join(unver),
                       "refusing to certify comparability from unrecorded facts: make "
                       "the server report them, or pass the flag that says sameness is "
                       "established by construction"])
    return unver


# ======================================================================================
# SOAK DRIFT  -- the gate that separates a screen from a qualification
# ======================================================================================
def drift_gate(windows, key="stream_rtf", field="p95", tol=0.05, min_windows=3):
    """Did the metric hold STILL across a soak's windows, or did it walk?

    ``windows`` is a list of :func:`summarize` aggregates, in time order.  A soak that
    ends worse than it started has not qualified anything, however good its average: a
    configuration whose STREAM_RTF climbs from 0.88 to 1.02 over thirty minutes has a
    perfectly respectable mean and is not shippable.

    The gate compares the LAST window against the BEST window rather than first-vs-last,
    so a single bad warm-up window cannot hide a downward trend behind it.  Returns a
    dict with ``pass`` True/False/None (None = too few windows to judge).
    """
    vals = []
    for w in windows:
        d = w.get(key)
        v = d.get(field, float("nan")) if isinstance(d, dict) else (
            w.get(key, float("nan")))
        vals.append(float(v) if v is not None else float("nan"))
    finite = [v for v in vals if v == v]
    out = {"key": "%s %s" % (key, field), "values": vals, "n_windows": len(windows),
           "tol": tol, "min_windows": min_windows}
    if len(finite) < min_windows:
        out.update({"pass": None, "drift": float("nan"), "best": float("nan"),
                    "last": float("nan"),
                    "why": "only %d window(s) with a value; %d needed to call a trend"
                           % (len(finite), min_windows)})
        return out
    best, last = min(finite), vals[-1]
    if last != last:
        last = finite[-1]
    drift = last - best
    out.update({"pass": drift <= tol + 1e-12, "drift": drift, "best": best,
                "last": last,
                "why": "last window %.4f vs best %.4f = %+.4f (tolerance %+.4f)"
                       % (last, best, drift, tol)})
    return out


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
    check("summary/stall_n@250", s["stall_n@250"], 2)
    check("summary/stall_n@1000", s["stall_n@1000"], 0)
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

    # --- CASE 11: TTFB is not TTFA ----------------------------------------------------
    # The header goes out before synthesis starts, so TTFB is milliseconds while TTFA is
    # seconds.  Both come out of this module so that nothing downstream can quote one
    # for the other.
    k = timeline_kpis(synth("smooth", F), 4.0, F, ttfb_s=0.012)
    check("ttfb/carried", k["ttfb_s"], 0.012)
    check("ttfb/ttfa is not ttfb", k["ttfa_s"], 0.3)
    check("ttfb/header lead", k["header_to_audio_s"], 0.288, 1e-9)
    check("ttfb/absent is NaN not zero",
          timeline_kpis(synth("smooth", F), 4.0, F)["ttfb_s"], float("nan"))

    # === CASE 12: THE CASE THIS FILE EXISTS FOR =======================================
    # A stream that is comfortably FASTER than realtime on average and still stalls.
    # STREAM_RTF passes every RTF gate there is -- mandatory (<1), preferred (<=0.90)
    # and even strong (<=0.85) -- while a player with a 250 ms jitter buffer runs dry.
    # If STREAM_RTF were allowed to promote a configuration, this one would ship.
    m = synth("bursty", F, n=9)              # 0.5 s at 0.3, then 4 chunks at 1.9 and 3.5
    k = timeline_kpis(m, t_done=m[-1][0], fmt=F, ttfb_s=0.01)
    check("HEADLINE/stream_rtf", k["stream_rtf"], (3.5 - 0.3) / 4.0, 1e-9)   # 0.800
    check_true("HEADLINE/stream_rtf passes the MANDATORY gate (<1.00)",
               k["stream_rtf"] < ENVELOPE["rtf_hard"])
    check_true("HEADLINE/stream_rtf passes the PREFERRED gate (<=0.90)",
               k["stream_rtf"] <= ENVELOPE["rtf_pref"])
    check_true("HEADLINE/stream_rtf passes the STRONG gate (<=0.85)",
               k["stream_rtf"] <= ENVELOPE["rtf_strong"])
    check("HEADLINE/required_prebuffer is 1.1 s anyway", k["required_prebuffer_s"],
          1.1, 1e-6)
    check("HEADLINE/a 250 ms buffer stalls", k["stalls@250"], 1)
    v, g = qualify(summarize([k]), completed=1, launched=1)
    # The verdict is the strongest one available: a 500 ms jitter buffer -- half a second
    # of latency the listener pays before a word is heard -- still runs dry.
    check("HEADLINE/verdict is NOT STREAMABLE",
          1.0 if v == "NOT STREAMABLE" else 0.0, 1.0)
    check_true("HEADLINE/the MANDATORY gate that failed is stall@500, not RTF",
               sorted(x["name"] for x in g["mandatory"]
                      if x["pass"] is False) == ["stall_rate@500ms"])
    failed = sorted(x["name"] for x in g["preferred"] if x["pass"] is False)
    check_true("HEADLINE/the PREFERRED gates that failed are prebuffer and stall@250",
               failed == ["required_prebuffer p95", "safe_play_start p95",
                          "stall_rate@250ms"])
    check_true("HEADLINE/NO RTF gate failed anywhere in the envelope",
               all(x["pass"] is not False
                   for x in g["mandatory"] + g["preferred"] + g["strong"]
                   if "STREAM_RTF" in x["name"]))

    # --- CASE 13: the envelope ---------------------------------------------------------
    smooth = [timeline_kpis(synth("smooth", F, n=9), 3.5, F, ttfb_s=0.01)
              for _ in range(4)]
    v, g = qualify(summarize(smooth), 4, 4)
    check("envelope/clean stream is GOOD", 1.0 if v == "GOOD" else 0.0, 1.0)
    v, _ = qualify(summarize(smooth), completed=3, launched=4)
    check("envelope/a lost request is NOT STREAMABLE",
          1.0 if v == "NOT STREAMABLE" else 0.0, 1.0)
    slow = [timeline_kpis(synth("slower_than_realtime", F, n=9),
                          synth("slower_than_realtime", F, n=9)[-1][0], F, ttfb_s=0.01)]
    v, _ = qualify(summarize(slow), 1, 1)
    check("envelope/slower than realtime is NOT STREAMABLE",
          1.0 if v == "NOT STREAMABLE" else 0.0, 1.0)
    one = [timeline_kpis([(0.5, chunk_bytes(1.0, F))], 0.6, F, ttfb_s=0.01)]
    v, _ = qualify(summarize(one), 1, 1)
    check("envelope/no cadence is INCONCLUSIVE, never GOOD",
          1.0 if v == "INCONCLUSIVE" else 0.0, 1.0)
    # Every gate carries the threshold it compared against, so the verdict is falsifiable.
    _, g = qualify(summarize(smooth), 4, 4)
    check_true("envelope/every gate carries its limit",
               all("limit" in x and "op" in x
                   for x in g["mandatory"] + g["preferred"] + g["strong"]))
    check("envelope/stall@250 is a PREFERRED gate",
          1.0 if any(x["name"] == "stall_rate@250ms" for x in g["preferred"]) else 0.0, 1.0)
    check("envelope/stall@500 is a MANDATORY gate",
          1.0 if any(x["name"] == "stall_rate@500ms" for x in g["mandatory"]) else 0.0, 1.0)

    # --- CASE 14: capacity is discovered, and an island does not count -------------------
    check("capacity/contiguous", float(capacity({1: "GOOD", 2: "GOOD", 4: "MARGINAL"})), 2.0)
    check("capacity/island above a gap is not capacity",
          float(capacity({1: "GOOD", 2: "GOOD", 4: "MARGINAL", 8: "GOOD"})), 2.0)
    check_true("capacity/none when C1 already fails",
               capacity({1: "MARGINAL", 2: "GOOD"}) is None)

    # --- CASE 15: the refusals ----------------------------------------------------------
    st, sh, why = quotable({"coalesced_chunk_share": 0.0})
    check("refuse/clean run is quotable", 1.0 if st == "QUOTABLE" else 0.0, 1.0)
    st, _, _ = quotable({"coalesced_chunk_share": 0.08})
    check("refuse/8% warns", 1.0 if st == "QUOTABLE (warn)" else 0.0, 1.0)
    st, _, why = quotable({"coalesced_chunk_share": 0.35})
    check("refuse/35% (the reference's run) is NOT QUOTABLE",
          1.0 if st == "NOT QUOTABLE" else 0.0, 1.0)
    check_true("refuse/and says why", bool(why))
    st, _, _ = quotable({})
    check("refuse/unrecorded blocked reads is UNKNOWN, not a pass",
          1.0 if st == "UNKNOWN" else 0.0, 1.0)

    a = {"engine": "pocket", "isa": "neon", "threads": 4, "quant": "int8"}
    check_true("refuse/identical arms compare", comparable(a, dict(a))[0])
    same, diffs = comparable(a, dict(a, threads=8))
    check_true("refuse/different thread count does not compare",
               (not same) and len(diffs) == 1)
    same, _ = comparable(a, {"engine": "pocket", "isa": "neon", "threads": 4})
    check_true("refuse/an unrecorded fact is not an equal fact", not same)
    check_true("refuse/a key absent from both is not a difference",
               comparable({"engine": "pocket"}, {"engine": "pocket"})[0])
    full = {k: "x" for k in COMPARABLE_KEYS}
    try:
        require_comparable({"q1": dict(full), "q4": dict(full, quant="f32")})
        check_true("refuse/require_comparable raises on a difference", False)
    except Refusal as exc:
        check_true("refuse/require_comparable raises on a difference",
                   "quant" in str(exc))
    check_true("refuse/fully recorded identical arms compare",
               require_comparable({"q1": dict(full), "q4": dict(full)}) == [])
    # Equality of two unknowns is NOT sameness: both arms saying "<unrecorded>" must not
    # certify a comparison.  This is the hole that a naive dict-equality check leaves,
    # and this server's /health reports none of isa/simd/backend/quant/threads today.
    blind = {k: "x" for k in COMPARABLE_KEYS}
    blind["quant"] = UNRECORDED
    try:
        require_comparable({"q1": dict(blind), "q4": dict(blind)})
        check_true("refuse/unrecorded-in-every-arm is not proof of sameness", False)
    except Refusal as exc:
        check_true("refuse/unrecorded-in-every-arm is not proof of sameness",
                   "quant" in str(exc))
    check_true("refuse/but may be allowed explicitly, and says which keys",
               require_comparable({"q1": dict(blind), "q4": dict(blind)},
                                  allow_unverified=True) == ["quant"])

    # --- CASE 16: the soak drift gate ----------------------------------------------------
    def win(v):
        return {"stream_rtf": {"p95": v}}
    d = drift_gate([win(0.88), win(0.89), win(0.90)], tol=0.05)
    check_true("drift/steady soak passes", d["pass"] is True)
    check("drift/measured drift", d["drift"], 0.02, 1e-9)
    d = drift_gate([win(0.88), win(0.95), win(1.02)], tol=0.05)
    check_true("drift/a soak that walks upward fails", d["pass"] is False)
    check("drift/against the BEST window, not the first", d["drift"], 0.14, 1e-9)
    # A bad first window must not hide a downward trend behind it.
    d = drift_gate([win(1.20), win(0.88), win(0.99)], tol=0.05)
    check_true("drift/a bad warm-up window cannot mask the trend", d["pass"] is False)
    d = drift_gate([win(0.88), win(0.90)], tol=0.05)
    check_true("drift/two windows cannot call a trend", d["pass"] is None)

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
