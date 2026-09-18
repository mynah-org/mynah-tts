# E9-1 — `prep.decoder_prefill`: what is serial inside it

Status: **IN PROGRESS** (measurement DONE, one fix landed, two asks open)

## The claim under test

E9 profiled one stream on the production box and read `prep.decoder_prefill` at
361 ms on one core, 241 ms on eight, 241 ms on sixteen — 28.7% of a request's
wall, all of it before the first sample. The item concluded "about 240 ms is
irreducibly serial on this machine" and offered it as the reason safe-to-play is
908 ms at C48.

The region was one marker around a whole phase, so the conclusion was an
inference from a shape, not a measurement of a part.

## What the measurement actually shows

GCP Axion, 32x Neoverse-V2, gcc 15.2, `SIMD=auto`, pocket-en, one stream,
26-token text, `--max-steps 48`. `--runs 1` against `--runs 8` separates the
FIRST call of the region from the steady-state ones, because the cost map prints
a call count:

| cores | first call | steady state | difference |
|---|---|---|---|
| 1 | 442.3 | 230.0 | **212.3** |
| 4 | 282.2 | 68.2 | **214.0** |
| 8 | 255.1 | 39.4 | **215.7** |
| 16 | 242.9 | 27.5 | **215.4** |

The steady-state prefill divides **8.4x across sixteen cores**. It was never the
problem. What does not move is a constant ~215 ms that the first call pays and no
later call pays, and it is *exactly* constant — 212 / 214 / 216 / 215 — which is
what a fixed amount of single-threaded work looks like and what neither a
dependency chain nor an under-threaded kernel looks like.

**It is the quantized weight cache being built.** `MYNAH_QUANT_GROUPS=none`
removes it entirely (first call 38.2 ms, steady 37.4 ms, difference **0.8 ms**),
so it is not a page-in of the mapped f32 weights. The line is
`qmat.c:cache_insert()` — `qmat_f16_pack()` / `quantize_weight_int8()` — called
from `mynah_qmat_linear_batched_qt()` under `cache->mutex`, converting the
backbone's 24 tensors, 302 MB of f32 into 151 MB of f16, on one thread, while
every other thread waits on a mutex it cannot help with. The first thing in the
process that ever touches those tensors is the text prefill of the first request.

## The split, as a permanent region

`prep.prefill_proj` (new, level 1) brackets the prefill's four linear
projections and nothing else; its parent's self time is then the half no hook can
hand to the pool — attention scores, softmax, RoPE, the KV writes, the norms,
GELU, the residuals. With the pack moved out (below):

| cores | `prep.decoder_prefill` | `prep.prefill_proj` | self |
|---|---|---|---|
| 1 | 230.2 | 218.9 | **11.3** |
| 4 | 66.9 | 55.5 | **11.4** |
| 8 | 39.8 | 28.4 | **11.3** |
| 16 | 27.0 | 15.6 | **11.4** |

The genuinely serial, genuinely single-threaded part of the prefill — everything
that is not a projection — is **11.3 ms and does not move**. It is not worth a
thread pool. The projections divide 14x.

## What was done

`pocket_prepack()` in `src/engine_pocket.c` builds every cache entry the group
spec selects at engine init, through `mynah_qmat_linear_resolved_qt` with the
same name, weight pointer, shape and qtype the graph will use, so the entry that
exists afterwards is the entry that would have existed anyway. Reported as
`runtime.weight_prepack` under `runtime.model_load`. `MYNAH_PREPACK=0` disables
it; it is skipped under `MYNAH_QMAT_ACT_STATS`, where the throwaway row it
computes would enter the activation statistic.

Guarded once per cache: `serve()` calls `engine->model_init` per driver call, so
without the guard the second and later calls paid one full matvec per tensor to
discover the entries already existed (+2.1 ms per serve at 16 threads, +14.2 ms
at one).

## What it is NOT

It does not make a single one-shot synthesis faster. The work moved; it did not
shrink. Process wall for one CLI synthesis is unchanged within noise. What
changed is *when* it is paid and what the report calls it:

- for a server it is paid once, at `serve()` entry, before the scheduler admits
  anything — off the first request instead of on it;
- for a one-shot caller it is `runtime.model_load` instead of 28.7% of a
  decoder region;
- E9-1's causal claim is **wrong**: `server/main.c` already warms up before the
  accept loop, so no client request has ever paid this. It cannot be why
  safe-to-play is 908 ms at C48.

## What is actually left in the prefill, per request

At the topology E9-3 measured as the winner (`16x2`, two threads per worker) the
warm prefill is **~122 ms** for a 26-token text, and it is linear in text tokens
(~4.65 ms/token at two threads, ~1.1 at sixteen). That is real TTFA and it is
already parallel — it is starved of threads, not of parallelism. It is 60%
projection, and the projection runs at ~20 GFLOP/s/core against a ~41
GFLOP/s/core NEON f32 roof.

## Open asks (other lanes)

1. **qmat**: hoist the pack out of `cache->mutex` — build the entry into a local,
   take the lock only to re-check and publish — and expose
   `mynah_qmat_cache_prepack(cache, name, w, n, k, qtype)` so an entry can be
   built without a throwaway matvec, and without a `cache_lookup` API this file
   has to work around with a memo table. Then `pocket_prepack` can
   `mynah_parallel_for` over tensors and ~270 ms of serial model load becomes
   tens of ms.
2. **qmat / sgemm**: the weight-stationary f16 batched linear is now 60% of the
   warm prefill and runs at about half the f32 NEON roof per core. A real f16
   microkernel is worth more here than anywhere else in the prefill.
3. **server**: the warm-up is deliberately after the fork, so each of sixteen
   workers builds its own 151 MB f16 copy of the same weights. Model open already
   happens before the fork "so the mapped weights are one physical copy behind
   the whole tree"; the packed copies are not. An engine-level prepack hook
   called from `mynah_tts_model_open_device` would make them one copy-on-write
   copy — that needs `tts_engine.h` and `src/mynah_tts.c`, which this lane did
   not touch.
4. **driver**: `serve()` runs `engine->model_init` per call, ~57 ms of self time,
   flat in core count (JSON parse, 214-tensor resolve, 26 voice files opened and
   validated, SentencePiece parse). Once per server, but once per call for a
   one-shot `mynah_tts_synthesize()`.

## A defect found on the way

`MYNAH_RGN_MODEL_LOAD` and `MYNAH_RGN_DECODE_GANG` both evaluated to **44**:
MODEL_LOAD was written with no value under `RT_PARALLEL_WAIT = 43`, and
DECODE_GANG declares `= 44` two lines later. Every merge summed their times and
printed the sum under whichever `rgn_find()` reached first, so
`driver.decode_gang` has never appeared in a report and `runtime.model_load` has
never been only a model load. MODEL_LOAD is renumbered to 52 and
`rgn_unique_check()` now refuses the table at startup.

## Gate

PocketTTS byte-identical to the pinned head under default, `MYNAH_QUANT=int8`,
`MYNAH_QUANT=int4`, `MYNAH_QUANT_GROUPS=none` and `=all`, en and it, several
speakers, seeds and text lengths, single and `--batch 4`.
