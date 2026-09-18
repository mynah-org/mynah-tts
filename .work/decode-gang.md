# E1-2 — the decode gang, the blast radius, and the quantum ramp

Status: **LANDED (correctness only)** · depends on [engine-seam-refactor.md](engine-seam-refactor.md) E1-1 ·
theory in [serving-design.md](serving-design.md) §5 and [streaming-cadence.md](streaming-cadence.md) §3

Everything below was verified on macOS/arm64, which this repo treats as a
development signal and not a product claim. **Nothing here was benchmarked.**
The performance argument is entirely borrowed — ours and the reference's
earlier measurements — and the only thing this work established on its own is
that the seam admits the batching and that the driver's policies are what they
claim to be.

## Problem

`decode_audio` is declared PER CONTEXT, so the driver could never show an
engine two requests' codec work at the same time. The codec transformer plus
the SEANet conv stack are ~55% of wall time, and per-context that 55% is
multiplied by the number of concurrent streams with nothing shared. The
reference's decoder was 72-80% of the marginal cost of each additional stream,
and batching it was the single change that moved their capacity (§4, §5).

Two other defects showed up in the same code:

1. **`step_live()` retired every live slot when one slot's step failed.**
   Harmless at width 1, the whole server at width 16. Reproduced: with the old
   code, request 1 died carrying *request 0's* error text
   (`fake: request 1000 refuses step 3`).
2. **`emit_batch` failing while attributing the failure to nobody retired
   nobody**, leaving every slot active — the service loop would have taken the
   same step forever.

## What landed

- `decode_audio_batch` **appended to the end** of the engine vtable, with the
  contract written out in `tts_engine.h`: contiguous monotonic ranges per
  context (the seam's existing rule, unchanged), **bit-identity per context**
  regardless of who shared the call, per-context failure in `failed[i]`, and
  driver-owned arrays with one malloc per context.
  The position is load-bearing: both engine vtables are **positional**
  initializers, so a member inserted anywhere else silently shifts every
  pointer after it. Append; never insert.
- `mynah_engine_decode_gang()` in `inference.c` is the **default
  implementation** and the driver's only door: it dispatches to the engine hook
  when present and otherwise loops `decode_audio`. Both engines were left
  **untouched** and take the NULL/loop path. Landing order works: an engine
  gaining a batched codec is one line in its vtable.
- `stream_gang()` forms the gang once per step for the whole batch. Ready slots
  first, then the leader pull-in. **No slot is ever delayed to build a bigger
  gang** — the gang can only grow past what no-wait already requires.
- `step_isolate()` re-steps the batch one context at a time when the batched
  step fails, so one bad request retires one request. This is legal only
  because `step_batch` is atomic over the batch, which is now stated in the
  header as a requirement rather than assumed. `pocket_step_batch` currently
  advances contexts 0..i-1 before refusing i — it is safe at its declared
  `max_batch = 1` and **must be made atomic before that widens**.
- Quantum ramp per slot: 1, 2, 2, 4, 4, then `caps.audio_emit_frames`, clamped
  so it is never above what the engine declared.

## The gate, and the negative controls that make it mean something

`make driver-test` runs the driver against a synthetic engine whose audio is a
pure function of (seed, absolute frame, sample index), so "same audio whoever
you shared a gang with" is an identity rather than a tolerance. 8 requests at
width 4, deliberately out of lockstep.

Everything passed on the first run, which is not evidence, so each assertion
was checked against a driver edited to break exactly the thing it guards:

| control | edit | result |
|---|---|---|
| A | gang takes ready slots only | `no slot was ever pulled into a leader's decode` |
| B | quantum fixed at the steady state | `the ramp produced too few chunks` |
| C | failed step retires the whole batch | `one request's failure retired a sibling` (and request 1 carried request 0's message) |
| D | park ready work in the steady state | rescued by the pull-in — **did not fire**, see below |
| E | park ready work AND remove the pull-in | `ready work withheld 10 times` |

D is worth keeping: parking ready work did **not** show up as a stall, because
a neighbour's leadership dragged the parked slot into the next gang anyway. The
no-wait check is real (E proves it fires) but it is not sensitive to parking on
its own while any slot is leading. A future policy change near the gang must
not be validated by that counter alone.

A first attempt at the "rode along" counter (a member smaller than the widest
member of the same call) did **not** discriminate: control A still passed,
because slots flushing their last frames look the same. Redefined as *handed
less than its own quantum while still having audio to generate*, which by
construction no ready slot can be.

## Acceptance

Clean build, 3 pre-existing warnings and no new ones · `--self-test` ·
`make goldens` 8 checks + batch parity (Magpie byte-identical) ·
`make stream-test` (byte-identical offline↔streaming with the ramp active, so
Magpie's chunked decode really is one-shot-equivalent down to a 2-frame chunk) ·
`make driver-test` · UBSan, ASan and `leaks` clean, each also over the goldens,
the stream test and a real PocketTTS synthesis.

`-Wmissing-field-initializers` had to be turned off for the TUs that include
`tts_engine.h`: an appendable vtable and that warning cannot both hold while
the engines use positional initializers, and the engines were required to stay
untouched. The reason is written next to the pragma.

## What is NOT known

- **Whether any of this is faster.** No benchmark was run and none should be
  quoted from this machine. The gang's value is a Linux question: RTF, TTFA,
  required_prebuffer p95 and stall@250 at fixed concurrency, on the rented box,
  per the protocol in serving-design.md §10.
- **Whether the ramp's break points (4, 12) are right for us.** They are the
  reference's, taken as-is. Our quantum has never been swept, and the cadence
  note says it must be chosen on prebuffer and stall, never on RTF.
- **Whether `MYNAH_GANG_MIN_PENDING = 1` is the right follower floor.** It is
  the value with no downside under the cadence law, not a measured optimum.
- **Whether the leader test should be `quantum >= steady` or
  `pending >= steady`.** As written, a slot flushing its last three frames in
  the steady-state regime makes everyone a follower. It errs toward more
  batching and never toward waiting, which is the safe direction, but it is a
  guess.
