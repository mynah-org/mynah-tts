# Where the TTFA under load actually goes (E9-10)

Measured 2026-09-13 on the GCP Axion (32× Neoverse-V2, gcc 15.2), on the merged
tree carrying all three profile lanes, server `--prefork 16 --prefork-threads 2
--max-batch 16`, `BLAS=none`, `SIMD=auto`.

## The question

The FAST screen reported TTFB p95 489 ms and safe-to-play p95 890 ms at C48,
while the warm prefill measures ~122-166 ms and there are 256 slots for 48
arrivals. Something was waiting. The board said: measure it, do not guess it —
the obvious reading ("serialised admission") had already been falsified once.

## What it is not

**Not the listen backlog.** Under 48 live streams, `connect()` completes in
**0.1 ms p50, 0.3 ms p95**. The reference engine's failure mode — "p95 TTFB
4470 ms with over 97% of the tail invisible before accept()" — is quoted in
`server/prefork.c` and is *not* ours.

**Not connection-thread starvation.** With the same 48 streams running:

| probe | header p50 | p95 |
|---|---|---|
| `GET /health` | **0.7 ms** | 2.3 ms |
| `GET /v1/voices` | 0.9 ms | 2.3 ms |
| `POST` speech, streaming | 33.0 ms | 154.4 ms |

A request that touches no model is answered in under a millisecond while the
box is full. The HTTP side is not the problem.

**Not steady-state latency.** A request arriving into an already-loaded server
gets its first audio at **149 ms p50 / 274 ms p95**, not 890.

## What it is

The screen's number is a **burst** number: a wave launches every request of a
level at one instant. Firing 48 at once, from pre-established connections
released by a barrier, the shape is *flat*:

    first audio, 48 requests, sorted:  580 580 580 ... 583 584 585

Five milliseconds of spread across forty-eight requests. Not a ramp (a queue
draining), not a step (slot groups) — everyone waits for the same thing and is
released together. Two sweeps say what that thing is.

**Linear in slots per worker:**

| slots/worker | first audio | Δ |
|---|---|---|
| 1 | 197.6 ms | — |
| 2 | 389.2 | +191.6 |
| 3 | 581.0 | +191.8 |
| 4 | 774.9 | +193.9 |

**Linear in text length** (48 requests, 3 slots/worker):

| words | first audio |
|---|---|
| 2 | 124.1 ms |
| 12 | 297.9 |
| 24 | 534.1 |
| 40 | 813.9 |

18.2 ms per word over 3 slots = **6.05 ms/word/slot ≈ 4.65 ms/token**, which is
exactly the per-token prefill cost E9-1 measured at two threads. It is the
prefill, run one request at a time inside each worker: `src/inference.c`'s
admission `while` admits every available arrival at once (it is not one per
iteration — that reading was correctly falsified), but calls `slot_start()` —
and therefore the prefill — for each of them in turn, and the batch emits
nothing until they have all finished.

## And yet it is not a scheduling defect

One request's prefill at two threads costs **165.8 ms**, of which **149.9 ms is
the four projections** and 15.9 ms is the flat attention/RoPE/KV half. That is
~331 core-ms of work. Forty-eight of them is 15.9 core-seconds; on 32 cores,
**497 ms**, plus the ~88 ms intercept the text sweep measures for the first
frames = **585 ms predicted against 581 measured.**

Sixteen workers × two threads *is* the machine. Every core is busy doing prefill
that was genuinely asked for. **There is no idle resource to schedule better.**
Serialising the prefills within a worker costs nothing that parallelising them
would recover — the parallelism is already spent across workers.

So the TTFA floor in a burst is *total arriving prefill work ÷ machine*, and the
only way down is to make the prefill cheaper.

## What the prefill costs, and what it is made of

| weights | prefill @ 2 threads | projections |
|---|---|---|
| f32 (`MYNAH_QUANT_GROUPS=none`) | 278.4 ms | 262.4 |
| **f16 (default)** | **165.8 ms** | **149.9** |
| int8 (`MYNAH_QUANT=int8`) | 165.7 ms | 149.9 |
| `MYNAH_QUANT_GROUPS=all` | 165.3 ms | 149.7 |

f16 is already worth **1.68×** over f32. **int8 does not touch the prefill at
all** — identical to the millisecond — while it takes `request.total` from 1848
to 1496 ms and `request.finalize` from 1096 to 743. That is by design and the
design is written down (`src/engine_pocket.c`, `POCKET_QG_DEFAULT_SPEC`): the
backbone is inside the autoregressive loop, where int8's 1-6% per-step error
compounds over ~50 steps of feedback into a different sampled trajectory — the
frame count moves, the EOS step moves, log-mel correlation against f32 falls to
0.73-0.96. f16 measures 1.2e-06 per step and moves nothing.

### The tile sweep was invalid, and the retraction matters

Recorded first as "falsified", which was wrong. The box-local knob changed only
`mynah_transformer_ar_prefill_tile()`. The per-request scratch is reserved with
the **macro** `TAR_PREFILL_TILE` (`src/transformer_ar.c:297`) and the prefill
loop clamps to it:

    if (rows > state->rows.rows_cap) rows = state->rows.rows_cap;   /* 16 */

So above 16 the knob could not take effect at all — the flat 165.3-166.9 ms was
an experiment that never exercised the path it claimed to test. Below 16 it did
work, because `engine_pocket` tiles with the function value before calling in,
which is why tile 2 measured slower. The byte-identity across tiles is still a
real result.

Raising the tile needs the macro raised too, which is a per-request memory
decision (~57 KB of scratch per row). But it is the *second* lever, not the
first: while the kernel walks the batch one activation at a time, a larger tile
buys nothing, because the weight block is re-read per row regardless.

## Where the headroom actually is

**The f16 batched path does not batch.** `qmat_batch_rows()` has a
two-activations-per-SMMLA fast path for INT8 with i8mm — one weight load serving
two rows — and every other encoding, f16 included, falls through to:

    for (size_t b = 0; b < j->batch; ++b)
        qmat_rows_dispatch(&rj, row0, count);    /* one activation at a time */

With a 16-row tile the weight block is read **sixteen times**. `matvec_f16_neon`
is a good GEMV — four weight rows against one activation, 8 FMAs per 64 bytes of
weights, 0.5 FLOP/byte — and a GEMV is memory-bound by construction. Running a
GEMM as B independent GEMVs is exactly the shape that lands at a third of the
roof.

The prefill moves ~151 MFLOP per token through six layers (`3+1+4+4` × 1024²
MACs per layer). For this text at two threads that is roughly **13.6 GFLOP/s per
core**, against the **~41 GFLOP/s per core** NEON f32 roof the seanet lane
measured on this box. **The f16 weight-stationary batched linear is running at
about a third of the roof**, which is the same finding E9-1 handed over and
could not act on because `qmat` was another lane's file.

Two levers, in this order:

1. **Make the f16 batched linear faster.** Three-fold headroom, *no numerical
   question of any kind* — the arithmetic does not change, only its scheduling.
   At C48 a 2× here takes burst TTFA from 581 ms to roughly 330.
2. **int8 for the prefill phase only.** The prefill is not inside the sampling
   loop, so the measurement that rules int8 out for the backbone does not
   directly apply — but it does not clear it either: the prefill writes the KV
   the whole generation attends to, so its error is static rather than
   compounding, which is a different risk and not a smaller one. It needs its
   own quality gate (frame count, EOS step, log-mel corr) before anyone ships
   it, and it costs a second quantized copy of the backbone weights because the
   steps must stay f16.

## Reproduce

`probe_ttfb.py` (shape comparison under load) and `probe_burst.py` (burst shape,
sweeps slots/worker and text length) are in the session scratchpad, not the
repo: they are one-question tools, not a harness.
