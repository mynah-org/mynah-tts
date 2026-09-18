# Cadence — why `STREAM_RTF < 1` does not mean the player never stops

Source: qwen-tts `.work/professional-streaming-architecture.md` (335 lines) plus
their quantum-floor and lead-in campaigns. The reasoning is reproduced here
because the conclusion alone is useless — it is the derivation that tells us
which knob to turn.

This note also carries **our own lead-in measurement**, which closes one of their
open questions negatively for us (§6).

---

## 1. The two metrics measure different things

```
STREAM_RTF        = (t_done - t_first_chunk) / (audio delivered after the first chunk)
required_prebuffer = max_i [ (t_i - t_first) - (audio owned before chunk i) ]
```

The first is an **average rate** over the stream. The second is a **maximum
latency** statistic over the same timeline. They diverge exactly when delivery is
quantized — which it always is, because audio comes out in decoder chunks.

> A stream can have STREAM_RTF 0.90 and still deliver 2.56 s of audio every
> ~2.3 s. **`STREAM_RTF < 1` does not prove that a player starting at first audio
> never stalls.** It stays the capacity metric; it is not a streamability
> qualification.

Their own earlier documentation said the opposite and called prebuffer and
underrun "diagnostics, not KPIs". That reading is now explicitly retired in their
tree. We should not repeat the mistake: **our `tools/serving_profile.py` already
reports both** — the point is which one gates.

---

## 2. The cadence law

A chunk of `C` frames can only be delivered after `C` model steps plus its decode,
so consecutive deliveries are about `C × w` apart, where `w` is the wall time per
frame. Between two deliveries the player consumes `C × w` of audio holding only
the lead accumulated so far, and the lead grows by `(1 − ρ)` per second of audio
produced.

> A stream absorbs a quantum of `C` frames without stalling only if
> **`lead ≥ C × w ≈ ρ_frame × (audio in the chunk)`**.

At ρ ≈ 0.9 the lead grows by 10% of the audio produced. So a large quantum needs a
lead that a 0.9-RTF stream takes a very long time to accumulate — and that is the
entire mechanism.

Their prediction vs their measurement:

| chunk | audio | predicted `ρ·quantum − lead` | measured prebuffer p95 |
|---|---|---|---|
| 8 frames | 0.64 s | ~0.4-0.5 s | **0.450 s** |
| 32 frames | 2.56 s | ~2.1-2.4 s | 1.22 s (short wave) / **2.37-2.55 s** (mixed bench) |

Independent confirmation at C4: q8 prebuffer p95 **337 ms** vs q32 **2173 ms** in
the wave, **726 ms vs 2544 ms** in the soak — while STREAM_RTF barely moves.
`safe_play_start ≈ TTFA + prebuffer ≈ 3.0 s`, which fails any interactive budget
*regardless of RTF*.

**The loop to break:** fixed per-call decoder cost → pressure toward large chunks
→ large audio quanta → bursty delivery → multi-second prebuffer.

---

## 3. The quantum floor — and why the smallest quantum is not the answer

Full sweep at C4:

| quantum | STREAM p50/p95 | prebuffer p95 | safe start p95 | max gap p95 | stall@250 | chunks |
|---|---|---|---|---|---|---|
| q1 | .962/**1.005** | 224 ms | 535 ms | 273 ms | 0% | 81 |
| **q2** | .863/.892 | **149 ms** | 568 ms | 300 ms | 0% | 43 |
| q4 | .836/.862 | 277 ms | 683 ms | 366 ms | 0% | 24 |
| q8 | **.793/.817** | 333 ms | 606 ms | 602 ms | **50%** | 15 |

**q1 fails**, at 1.005 STREAM p95, despite far more calls — and it does not even
improve TTFA. **q8 has the best RTF and stalls half the time.** After a later
optimisation the choice moved to q4 (0.868, prebuffer 201 ms, stall@250 **0%**)
against q8 at 0.822 with prebuffer 306 ms and stall@250 25%.

> Aggregate RTF alone would select the wrong serving point.

Their quantum ramps per slot, which is the part to copy:

```
decpos == 0      -> 1 frame      (the first chunk IS the TTFA)
decpos < 4       -> 2
decpos < 12      -> 4
otherwise        -> steady-state quantum (default 8, clamped to 32)
```

→ emitted quanta **1, 2, 2, 4, 4, then regime**. First chunk always one frame;
then grow to amortise the fixed per-call cost once the lead exists to pay for it.

The policy they designed but never shipped, and the rule inside it:

- steady state `n = clamp(floor((lead − panic_lead) / cost_per_frame), n_min, n_max)`;
- **load and admission pressure enter only through the utilization estimate and
  the prefill reserve, never through chunk size** — *"shrinking chunks under load
  is the vLLM-Omni death spiral in reverse, and is forbidden by the `n_min` floor."*

---

## 4. The underlying principle

**Decouple compute granularity from delivery granularity.** The server may
aggregate work internally for efficient matrix execution while still delivering
small, regular PCM increments. Never reason "bigger decoder chunk → better
amortised RTF" without the cadence consequence attached.

And **playback lead is the cross-stage currency**: very low lead → deadlines
dominate; moderate → some batching admissible; high → spend the margin on future
work (prefill of the next window). Prefill becomes a deadline-aware job rather
than an unconditional inline stall.

Their recommended scheduling primitive, after comparing pure EDF / least-laxity /
DRR / weighted urgency: **credit-gated EDF with a prefill reserve and
laxity-based admission**. Not something we need now — but it is the shape the
problem converges to, and worth knowing before inventing a different one.

---

## 5. The qualification envelope — adopt as-is

| dimension | mandatory | preferred | strong |
|---|---|---|---|
| correctness | parity PASS, errors = rejects = timeouts = 0 | | |
| TTFB | measured **independently of TTFA** | < 100 ms | |
| TTFA p95 | reported | < 500 ms | ≤700 ms only for materially better continuity |
| STREAM_RTF p95 | < 1 | ≤ 0.90 | ≤ 0.85 |
| required_prebuffer p95 | reported | ≤ 500 ms | ≤ 250-300 ms |
| safe_play_start p95 | reported | ≤ ~1 s | ≤ ~750-800 ms |
| stall_rate@500ms | → 0 at the operating point | | stall@250ms → 0 |
| established streams under admission | no stall induced by a new arrival at a 250-500 ms buffer | | |
| slow/stopped client | unrelated streams untouched | | |

GOOD = every preferred met · MARGINAL = mandatory yes, preferred no ·
NOT STREAMABLE otherwise.

> Capacity is the highest GOOD concurrency with useful margin — **discovered, not
> prescribed.** If C3 is GOOD and C4 is MARGINAL, the operating point is C3.

**Transport caveat, and it invalidates runs.** The client mark is when
`read1()` returns; a late reader finds several chunks already queued and returns
them microseconds apart, so N server emissions can appear as N marks in one
instant. Every mark must carry the time blocked in read; under 1 ms it is a
*coalesced read* and the share is reported per run. A run with 33-37% coalesced
reads is declared **not quotable** for cadence percentages. Our `TCP_NODELAY` item
(E5-18) is what keeps that share near zero.

---

## 6. Leading silence — their half-second, and our measurement

Their finding: an energy-envelope pass over every saved WAV (20 ms frames,
threshold −30 dB of file peak) found a systematic difference in **lead-in
silence** between model classes — median **0.50 s** on the 1.7B against **0.06 s**
on the 0.6B.

| | TTFA p50 | + lead-in | first word heard |
|---|---|---|---|
| 1.7B | 241 ms | +500 ms | **~740 ms** |
| 0.6B | 186 ms | +60 ms | **~250 ms** |

> The engineering numbers say the small checkpoint is ~24% faster to first audio.
> To the ear it is about **three times** faster to the first word. Half a second of
> dead air at the start of every reply is exactly what makes a voice assistant feel
> slow, and no amount of RTF work addresses it.

### Our result: **PocketTTS does not have this problem**

Same method, run on 2026-09-12 over everything in `build/demo/` and the oracle
references (`/private/tmp/.../scratchpad/leadin.py`, 20 ms frames, −30 dB of peak):

| file | dur s | **lead-in s** | tail silence s |
|---|---|---|---|
| EN-1 oracle (PyTorch) | 4.160 | **0.040** | 0.180 |
| EN-2 mynah C | 4.320 | **0.020** | 0.240 |
| IT-1 oracle (PyTorch) | 6.320 | **0.080** | 0.600 |
| IT-2 mynah C | 6.320 | **0.060** | 0.860 |
| IT-3 mynah C (lola) | 3.520 | **0.000** | 0.780 |

Two conclusions, both clean:

1. **PocketTTS is in the 0.06 s class, not the 0.50 s class.** There is no
   half-second of dead air to recover. The item is closed negative — which is
   worth exactly as much as closing it positive, and cost ten minutes.
2. **Our C engine adds no lead-in over the reference** (0.020 vs 0.040 EN,
   0.060 vs 0.080 IT). So this is not a place where our implementation differs
   from upstream.

**One observation worth keeping, not yet a claim:** trailing silence is not small
— 0.24 s EN and 0.78-0.86 s IT. On IT the same text at the same total duration
gives 0.86 s tail in our output against 0.60 s in the oracle. Different sampling
trajectories produce different endings, so this is not necessarily a defect; but
trailing silence is decoder frames spent on nothing, it inflates the RTF
denominator, and at the end of a turn it delays the point where a duplex system
can hand the floor back. Worth a look when we touch EOS handling.

---

## 7. What this changes for us

- **The quantum is a first-class serving parameter**, and it must be chosen on
  prebuffer and stall, never on RTF. We have not swept it.
- **The first chunk is one frame**, always. Cheap to implement, directly buys TTFA.
- **`required_prebuffer` and `stall_rate@250` gate; `STREAM_RTF` reports.**
  `tools/serving_profile.py` already computes all of them — the change is which
  one is allowed to say GOOD.
- Our chunked-vs-one-shot codec error is 2.98e-07 and carried state costs 1e-7
  down to a **1-frame chunk** (E2-3), so unlike Magpie we pay no correctness
  penalty for a small quantum. **The cadence knob is free for us in a way it was
  not for them.** That is probably our single biggest structural advantage on
  streaming, and it came out of a measurement we already did.
