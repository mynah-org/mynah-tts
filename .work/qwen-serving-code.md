# qwen-tts v2 serving path — read from the CODE

**Target repo** `/Users/gabrielemastrapasqua/source/personal/qwen-tts`
**Branch** `feature/x86-amx-vnni-oss`
**HEAD** `dc8bc486d4e1683dcc0c91c309b43f7a979d4f01`
**Tree state at read time** no tracked modifications; one untracked file,
`configs/perf/aws-c9g-8xlarge-32c-arm-v2-all-on.json` (flagged where I use it — it is
uncommitted work, not a committed claim).
**Method** whole-file reads of `qwen_tts_server.c` (3194 l), `qwen_tts.c` §serve\_continuous
(2870–4330), `qwen_tts_thread.c` (910 l), `main.c` serve wiring, plus the batched Talker /
CP / vocoder kernels the loop calls. Notes were read separately and are treated as a
hypothesis, never as a source. Every claim below carries `file:line`; anything I could not
cite is marked **(inference)**.

Read-only pass. Nothing in either repo was modified except this file.

---

## 1. ADMISSION — what they do, what we do, what would move our TTFB

### 1.1 The correction that matters most: we do not admit one per iteration either

The brief's premise is that our scheduler admits at most one request per iteration, so 30
simultaneous arrivals wait behind 7–8 iterations. **That is not what our code does.**

`src/inference.c:754`:

```c
while (!drained && used < max_batch &&
       (sink->running == NULL || sink->running(sink->ud) != 0)) {
```

It is a `while`, not an `if`. It finds the next free slot (`:757–759`), calls
`sink->next_job(..., block)` with `block = (used == 0u)` (`:765`), and on success calls
`slot_start` (`:786`) — which runs `engine->ctx_new` **and `engine->prepare`** inline
(`src/inference.c:202–208`). There is no `break` on success. So on a burst, one worker
fills **every** free slot in a single iteration, paying one full `prepare` per admission,
back to back, on the scheduler thread.

So the 500 ms TTFB p95 is not seven iterations of waiting. It is **seven serialised
`prepare()` calls inside one iteration**, with the 8th admitted request's header written
only after all seven in front of it have prefilled. Same wall clock, different mechanism —
and a different fix. `queued_total=0 queue_peak=0` is exactly consistent with this: the
queue never builds because the scheduler drains it instantly into slots; the wait is *after*
the pop, inside the admission loop.

This matters because two of the three obvious fixes (a bigger queue, a faster queue handoff)
address a thing that is not happening.

It does **not** follow that the third obvious fix is right. qwen made it and measured a severe
regression — §1.5, which is the section to read if you only read one.

### 1.2 How many admissions per iteration in qwen — three paths, and the default is "all of them"

`qwen_tts.c:3495` opens the same shape:

```c
for (int b = 0; b < B; b++) {
    if (active[b]) continue;
    if (dec_on && atomic_load(&dec_busy[b]) != 0) { ... continue; }
```

Three mutually exclusive bodies follow.

**(a) Default — inline, monolithic, every free slot** (`qwen_tts.c:3692–3715`). Reached when
`adm_slice == 0` and `use_helper == 0`, which is the shipped configuration (see §1.4).

```c
int block = (n_active == 0);
int pf_got = sink->next_job(sink->ud, &req, &t, block);
if (!pf_got) { ... if (block) break; continue; }
int prc = 0, pl = 0;
ADMIT_PREFILL(b, req, prc, pl);      /* 3705 */
...
ADMIT_INSTALL(b, req, t, pl);        /* 3714 */
```

No `break` after `ADMIT_INSTALL`. `ADMIT_PREFILL` (`qwen_tts.c:3273–3296`) is a full
`qwen_tts_generate(ctx, req.text, NULL, NULL)` with `ctx->prefill_only = 1` — the whole
prompt, all layers, one call. So **qwen's default admits up to `B - n_active` requests per
iteration, each costing a complete inline prefill**, exactly as ours does. Their scheduler is
not structurally better here; it is identical.

**(b) Sliced — exactly one slice per iteration, opt-in** (`qwen_tts.c:3507–3632`). Gated on
`adm_slice > 0 && !use_helper`, where `adm_slice` comes from `qwen_prefill_slice_tokens()`
(`qwen_tts.c:3388`), which reads `QWEN_PREFILL_SLICE` and **defaults to 0**
(`qwen_tts.c:1990–2010`). When on:

- at most **one** pending admission exists at a time (`adm_pending_t adm;` — a single
  struct, `qwen_tts.c:3401`);
- one `qwen_talker_prefill_range(ctx, ..., adm.done, adm.done + a_S)` per iteration
  (`qwen_tts.c:3603`), then an unconditional `break` at `qwen_tts.c:3631`;
- when the worker is idle it takes the whole remainder in one visit:
  `int a_S = (n_active == 0 && !adm_slice_idle) ? a_rem : qwen_prefill_slice_next(...)`
  (`qwen_tts.c:3599–3601`) — cold-start TTFA is deliberately not slowed;
- a prompt that would *populate* the prefix cache cannot be sliced and falls back to one
  monolithic prefill for that admission (`qwen_talker_prefill_plan` returns 1,
  `qwen_tts_talker.c:1904–1908`; handled at `qwen_tts.c:3551–3584`).

The slicing boundary is legal because a token walks all layers inside its own slice and reads
earlier tokens only through the bf16 KV cache (`qwen_tts_talker.c:1871–1877`). It has a state
oracle, `--prefill-slice-check`, that compares KV rows and `dec_x` against the monolithic arm
rather than comparing audio (`main.c:1803–1890`) — the reasoning at `main.c:1795–1802` is
worth reading on its own: a WAV cannot answer this question, because a correct slicing and a
slicing that never ran produce the same file, while one flipped argmax produces a completely
different but equally valid utterance.

**(c) Helper — off-thread prefill, still one at a time** (`qwen_tts.c:3634–3691`). Gated on
`QWEN_PREFILL_HELPER=1` **and** `qwen_pool_concurrent_submit_ok()` (`qwen_tts.c:3421–3422`).
One helper thread with a full context clone (`qwen_tts.c:3423`) pulls jobs itself
(`prefill_helper_main`, `qwen_tts.c:2515–2583`), prefills into its own KV cache, and pushes a
heap-copied `prefilled_t` onto a bounded queue of capacity 2 (`qwen_tts.c:3426`). The frame
loop then does only `memcpy` per layer (`qwen_tts.c:3652–3656`) — microseconds, not a
prefill. The loop `continue`s, so it can install several ready prefills per iteration.

**(d) M1 late admission** (`qwen_tts.c:3963–4040`), `QWEN_ADMIT_M1`, default off. After the
main step, it admits **one** request into a slot that freed *during* this iteration and
immediately runs head + sample + CP + embed for that slot alone
(`qwen_tts.c:3994–4010`), so the new request gets its first frame in the same iteration
instead of the next. Disabled automatically when the helper is on
(`qwen_tts.c:3438–3442`) — M1 is defined against inline admission.

### 1.3 Where in the iteration, what state it owns, why that bounds it

Iteration order in `qwen_tts_serve_continuous` (all `qwen_tts.c`):

| # | phase | line |
|---|-------|------|
| 1 | **admission scan (all free slots)** | 3495–3716 |
| 2 | codec head GEMM over active slots | 3792 |
| 3 | sample per slot (+ EOS / cap / cancel) | 3797–3802 |
| 4 | code predictor, batched | 3807 |
| 5 | record frame, embed, decode / enqueue | 3823–3958 |
| 6 | M1 late admission (opt-in) | 3963–4040 |
| 7 | batched vocoder pass (`dec_batch`) | 4042–4106 |
| 8 | batched Talker step, ragged | 4135 |
| 9 | `pos[b]++`, `sframe[b]++` | 4145 |

Two consequences worth copying:

- **Admission is at the top, and the head/sample run before the Talker step.** A slot
  installed at step 1 already has `last_hidden` written by `ADMIT_INSTALL`
  (`qwen_tts.c:3306`, an rms\_norm of `ctx->dec_x` left by the prefill), so it produces its
  **first frame in the same iteration it is admitted**. There is no dead iteration.
- **The first decode unit is one frame.** `qwen_tts.c:3833–3837`:
  `if (decpos[b]==0) target = 1; else if (decpos[b] < 4) target = 2; else if (decpos[b] < 12)
  target = 4; else target = g_stream_dec_chunk;` The ramp 1 → 2 → 4 → 8 means first audio
  costs exactly one Talker/CP frame plus one 1-frame vocoder call.

**What the admission owns, and why only one prefill can be in flight.** `ADMIT_PREFILL` runs
on the *shared engine context*: `ctx->kv_cache_k/v`, `ctx->dec_x`, `ctx->bg_text_content_len`,
`ctx->stream_trailing_text`, `ctx->prefill_embeds`. `ADMIT_INSTALL` then memcpys
`ctx->kv_cache_*` into `bb->kv_k/v` at the slot's offset (`qwen_tts.c:3299–3305`). There is
exactly one such scratch context, so a second prompt build would destroy the first — stated
in so many words in the sliced-admission comment, `qwen_tts.c:3503–3506`:

> *"While it is pending no other request is prefilled, because the partial KV of this one
> lives in ctx->kv_cache and the prompt state (dec_x, bg_text_content_len, trailing text)
> lives in ctx -- a second prompt build would destroy both."*

**Can a second admission overlap a first?** Only via the helper, and only because the helper
owns a *separate cloned context* (`qwen_tts_clone_for_worker`, `qwen_tts.c:912–990`: fresh
`kv_cache_k/v` at `kv_max = 2048`, fresh `dec_*`, fresh `cp_*`, shared read-only weights).
Even then it is one prefill at a time — one helper thread. The clone is the enabling fact,
not the thread.

**What happens to an established stream's cadence during an admission.** Nothing protects it
in the default path: the frame loop is one thread and the prefill is inline, so every active
stream loses the full prefill duration from its frame budget. `pf_admit` is accumulated
separately and reported with the honest label *"what a new arrival costs the requests already
in flight"* (`qwen_tts.c:4229`). Three mechanisms exist to bound it, **all default-off**:

- slicing (§1.2b) — costs the streams one slice instead of a whole prompt;
- `QWEN_PREFILL_LOW_MS` with the helper — the helper's pool dispatches wait while a HIGH
  submitter has dispatched within the last 200 µs window (`qwen_tts_thread.c:546–559`,
  `qwen_tts.c:2526–2539`), so prefill work only fills the loop's idle windows;
- `QWEN_TTFA_PRIORITY=N` (`qwen_tts.c:3752–3765`) — the inverse trade: while a newly admitted
  slot has `sframe < N`, the whole batch narrows to that one slot, with an anti-starvation
  `freeze_cap` (`qwen_tts.c:3043`, `3779–3783`). It buys the new arrival's TTFA by stalling
  everyone else's cadence outright.

### 1.4 What actually ships — and this is the real answer to our TTFB

`QWEN_PREFILL_SLICE` appears in **zero** shipped configs, **zero** files under `docs/`,
nothing in `PLAN.md` or `README.md` (verified by grep over `configs/ docs/ PLAN.md README.md`).
Same for `QWEN_ADMIT_M1` and `QWEN_PREFILL_HELPER` — mentioned in `docs/feature-flags.md:239`
and `:267` as opt-ins, set in no profile. The sliced-admission experiment was built, given a
state oracle, registered in the flag table (`qwen_flag_scope.h`), and **not adopted**.

What they ship instead is admission *control*. `configs/perf/axion-c4a-highcpu32-0p6b-all-on.json`
is our hardware almost exactly — GCP c4a-highcpu-32, **Neoverse-V2, 32 physical cores, no SMT,
one NUMA node** — and its `server` block is:

```json
"prefork_workers": 4, "threads_per_worker": 8, "batch_size": 8,
"max_queue": 0, "queue_timeout_ms": 0, "max_request_seconds": 60
```

`max_queue: 0` is the load-bearing number. In the code it produces
`reject_full_at_parent = (g_cfg_max_queue == 0 && !getenv("QWEN_QUEUE_UNBOUNDED"))`
(`qwen_tts_server.c:2756`): the prefork parent keeps polling the listening socket even when
every worker slot is occupied, accepts, and answers **503 immediately**
(`qwen_tts_server.c:3101–3112`) instead of letting the client sit in the backlog. Inside a
worker the same bound is `jq_push` (`qwen_tts_server.c:1678`):

```c
if (q->cap >= 0 && atomic_load(&g_srv.running) + q->count >= g_srv.slots + q->cap) {
    pthread_mutex_unlock(&q->mtx); return 0;     /* -> 503 "server at capacity: queue full" */
}
```

And their own qualification note for that profile
(`configs/perf/axion-c4a-highcpu32-0p6b-all-on.json`, `qualification.notes`) says:

> *"C16 preferred; C20 is the soft realtime edge; C32 is admission-only. … C16 had zero
> errors/rejects/timeouts and STREAM_RTF p95 0.881 … C20 is an edge with STREAM_RTF p50/p95
> 1.02/1.09 and playback degradation … Status stays unqualified"*

So: 32 slots exist, 16 are the recommended operating point, 20 is the edge, and the profile is
still marked `"status": "unqualified"`. **They do not run their slots full, and they refuse
rather than queue.** We run 30 concurrent on 32 slots and complete 90/90 at STREAM\_RTF p95
0.922 — which is a *better* throughput result than the number in their own config, on the same
silicon family. Our problem is not that our scheduler is worse than theirs; it is that we are
operating past the point where they chose to stop, and the tail we are paying is the admission
convoy.

### 1.5 They ran the obvious fix. It was a 5× regression.

This is the single most important thing in this document, and I nearly recommended the
opposite before finding it. `PLAN.md:131–149`, verbatim:

> *"C12-WIN-10 Admission slicing … **CLOSED 2026-09-10 on Turin: state parity CORRECT,
> serving behaviour a severe REGRESSION**, flag stays default-off and unpromoted. … Closed-loop
> C12, 10 minutes per arm, frozen profile, one variable: **completed 1294 -> 721 (-44 %), TTFA
> p95 183 -> 1574 ms, STREAM_RTF p95 0.922 -> 4.609**, and 7 server request timeouts against 0
> on the control. It made the established-stream interference it was built to remove about five
> times worse."*

And then the part that makes it directly about our proposed change
(`PLAN.md:141–145`, verbatim):

> *"Note what it did NOT test: all four workers report `mean_slices=1.00`, so with a warm
> prefix cache the admission prefill is ~1 new token (seq\_len=10, prefix=9) and **nothing was
> ever actually split**. The damage therefore comes from the sliced-admission **PATH**, not from
> slicing -- **most likely the one-admission-per-frame-iteration break stealing iterations from
> running streams.**"*

Read that against the code: with `mean_slices = 1.00` the slicing arithmetic was a no-op, so
the *only* behavioural difference left between the arms is the `break` at `qwen_tts.c:3631` —
i.e. **exactly the "admit at most one per iteration" change**, in isolation, measured on four
workers for ten minutes. It cost 44 % of completions and took STREAM\_RTF p95 from 0.922 to
4.609.

Their post-mortem does not fully explain *why* one `break` does that, and neither can I from a
read. Two candidates, both marked **(inference)**:

- **Refill latency compounds.** With cap 4 and four workers, every completion leaves a hole
  that now takes an extra whole iteration to fill; at C12 the batch spends a growing fraction
  of its iterations under-occupied, and the per-slot cost rises as occupancy falls (their own
  lane law, `.work/32c-serving-utilization-architecture-review-20260909.md`: `T(B) ≈ 40 +
  13.5·B` ms, so four slots cost 94 ms and one costs 53 — under-occupancy is expensive per
  request, not cheap).
- **Per-slice weight preparation.** `qwen_talker_prefill_range` re-runs
  `tk_prefill_weight_f32` for all six weight matrices **per layer, per call**
  (`qwen_tts_talker.c:2025–2042`), where the monolithic `qwen_talker_prefill` pays it once.
  It is skipped when `use_matmat` is on (`:2025`, `mm_env` resolved once at `:1978–1980`), and
  the frozen Turin profile does set `QWEN_PREFILL_MATMAT=1` — so this is probably *not* what
  bit them, but it is a trap anyone re-implementing the path would fall into.

Their own conclusion (`PLAN.md:148–149`): *"Any retry needs a redesign of the admission path
first, plus a cold-prefix workload so real multi-slice prefills occur."*

**Also note the caveat they put on their own A/B** (`PLAN.md:145–148`): a first attempt with a
true-wave arrival model was declared void, because with `n_active == 0` the code takes the whole
prefill in one slice by design (`qwen_tts.c:3599–3601`) — *"a wave that releases every request
into an idle engine cannot reach the mechanism at all."* **Our 90-request C30 wave has exactly
that shape.** If we A/B an admission cap with a wave harness, we will measure nothing, twice,
and conclude it is free.

### 1.6 The TTFB numbers that do exist

I said in an earlier draft that there were none. There are, in the notes — not in any committed
config. `.work/dl1-decoder-lane-split-20260909.md:284–297`, a **4×8 cap-4** screen on
c8a.8xlarge (32-core Zen 5, one CCX per worker), elastic lane on, one wave per level,
self-labelled *"SCREEN grade"* and *"What this is not: a qualification"* (`:307`):

| bank | C | TTFB p95 | TTFA p95 | STREAM p50/p95 | prebuffer p95 | stall @100/@250/@500 |
|---|---|---|---|---|---|---|
| short | 8 | **49 ms** | 183 | .730/.751 | 143 | 0/0/0 % |
| short | 12 | **103 ms** | 265 | .793/.814 | 247 | 67/0/0 % |
| short | 16 | **133 ms** | 341 | .864/.919 | 360 | 100/0/0 % |
| long | 16 | **143 ms** | 341 | .863/.881 | 358 | 100/0/0 % |

Two things follow, and they point in opposite directions.

**It puts our 500 ms in family, not out of it.** Their TTFB p95 roughly **doubles from C8 to
C12 and rises again to C16** — 49 → 103 → 133 — on *half* our per-worker slot count. That is
the admission-convoy signature: TTFB grows with arrivals-per-worker-per-iteration. Extrapolating
that curve to C30 on cap 8 does not obviously land below 500 ms. We may not have a defect.

**It also shows the operating point they chose.** At C16 their STREAM p95 is 0.919 short /
0.881 long — and we do 0.922 at **C30**. We are running roughly twice their concurrency at the
same realtime factor. Our 500 ms TTFB is the price of that, and it may be the correct trade
rather than a bug. Worth settling deliberately rather than by default.

### 1.7 What specifically would change our TTFB

Ranked by expected effect per unit of risk. Note that in **all** of these, TTFB is still "time
to admission" — qwen made the same choice we did, deliberately and with a comment saying why
(`qwen_tts_server.c:1993–1997`: *"The batched synchronous path used to wait for the first
generated audio before sending the response headers. That made TTFB equal TTFA and hid
admission/prefill latency from the client metric."*). Nobody gets to send the header earlier
without making the metric lie.

Three candidates. **None of them is a recommendation on its own** — read §1.5 before acting on
any of them, because the first is the change qwen measured as a severe regression and the
second is the change qwen closed. The ordered plan is in WHAT TO PORT at the end, and it puts
measurement ahead of all three.

1. **Cap admissions per iteration.** Our `while` at `src/inference.c:754` becomes
   admit-at-most-`k`. With `k = 1` the 8th arrival's header goes out after 7 *frames* rather
   than 7 *prefills*. Mechanically it is three lines; empirically it is
   `STREAM_RTF p95 0.922 → 4.609` on their box (§1.5). The two facts are not in conflict — the
   change is cheap and its *consequence* is what is expensive — and that is exactly why it must
   be measured before it is believed.
2. **Slice the prefill** (their `QWEN_PREFILL_SLICE` shape). Reduces what each admission costs
   the streams already running, which is the *other* half of the tail. Needs a resumable range
   prefill and a state oracle. qwen built exactly this, proved its state parity exact, and
   closed it unpromoted (`PLAN.md:133–134`).
3. **Off-thread prefill on a cloned context** (their helper). Removes the prefill from the
   frame loop entirely. Highest effort, and it buys less than it looks like it should because
   the pool serialises submitters anyway (§3.2) — their own measured helper arm cost
   `TTFA p95 172 → 683 ms` for a STREAM gain of ~0.05 (asserted in
   `.work/c12-win-admission-slicing-20260910.md`, provenance not stated in that file).

---

## 2. BATCHING AND MICRO-BATCHING, AS THE CODE DOES IT

### 2.1 Batched across requests vs looped per request — Talker step

`batch_talker_step_impl`, `qwen_tts_talker.c:2708–2791`. Per layer:

**Batched (one GEMM over the active columns):**

| tensor | call | line |
|---|---|---|
| Q, K, V (fused) | `qwen_batch_proj_qkv` | 2736 |
| attention out proj `wo` | `batch_proj_q` | 2762 |
| gate+up fused | `batch_proj_q` | 2766 |
| down | `batch_proj_q` | 2770 |
| codec head (in the loop, not here) | `qwen_batch_proj` | `qwen_tts.c:3792` |

**Looped per slot:**

| work | line |
|---|---|
| input rms\_norm | 2734–2735 |
| q\_norm / k\_norm per head, RoPE, KV bf16 store | 2743–2753 |
| **causal attention** `qwen_causal_attention_bf16kv` | 2754–2761 |
| swiglu | 2768–2769 |
| residual rms\_norms, final talker\_norm | 2763–2765, 2772–2784 |

The attention is the ragged part and it is **not batched at all** — one call per slot, over
that slot's own `pos`. Everything with a shared weight matrix is batched; everything that is
per-sequence stays scalar. That is the whole design and it is the right one.

### 2.2 Where the batch width is decided, and how ragged lengths are handled

**Decided per iteration, at `qwen_tts.c:3790`:** `qwen_batch_pack_active(bb, step_active)`.
`qwen_tts_talker.c:2406–2416`:

```c
for (int b = 0; b < bb->B; b++) if (active[b]) bb->act_idx[n++] = b;
if (n < 1) { bb->act_idx[0] = 0; n = 1; }
...
bb->B_eff = n;
```

`B_eff` is the number of active slots and `act_idx` is the compact list. The GEMM then
**gathers** those columns into a contiguous `Xt`, runs one `qwen_matmat_bf16`, and **scatters**
back (`qwen_batch_proj`, `qwen_tts_talker.c:2245–2255`). So the pack is a gather/scatter over
an index list — **no padding anywhere**, and the inactive slots' memory is never touched by the
GEMM. `QWEN_BATCH_NO_BEFF=1` disables the packing and runs the full `B` (`:2418–2427`); the
solo fast path `QWEN_BATCH_NO_SOLO` disables the `n_act == 1` shortcut (`:2429–2437`).

**Eligibility for a slot to be in `step_active`** is decided in three successive narrowings
(`qwen_tts.c:3731–3783`), applied in this order:

1. `step_active = active` (default: everything admitted steps);
2. the **lead gate** (`QWEN_STREAM_LEAD_GATE`, default off, `qwen_tts_server.c:1952–1954`):
   a stream whose *delivered* audio lead exceeds `QWEN_STREAM_LEAD_TARGET_MS` (default 250)
   is parked for this frame (`sink_step_allowed`, `qwen_tts_server.c:2090–2111`); the first
   frame is always eligible and a cancellation always passes through
   (`qwen_tts_server.c:2093`, `2098–2101`). If *every* slot parks, the loop sleeps 1 ms
   rather than spinning (`qwen_tts.c:3743–3749`);
3. `QWEN_TTFA_PRIORITY` (default 0) can collapse the batch to a single newest slot (§1.3);
4. `QWEN_BATCH_TALKER=N` caps the width round-robin (`qwen_tts.c:3768–3778`).

A slot is also skipped for *admission* — not for stepping — while the decoder lane still owns
it: `if (dec_on && atomic_load(&dec_busy[b]) != 0) continue;` (`qwen_tts.c:3497`).

### 2.3 The batched vocoder — ragged by concatenation, and only under BLAS

`qwen_speech_decoder_decode_streaming_batch`. **Two definitions**:

- `qwen_tts_speech_decoder.c:4499` — the real one, inside `#ifdef USE_BLAS` (`:4497`);
- `qwen_tts_speech_decoder.c:4952` — the `#else`, which is `sd_batch_fallback`
  (`:4485–4495`): a plain `for` loop of single-item calls.

So **a build without BLAS has no batched vocoder at all**, silently, and the server still
prints `"[serve] batched speech decoder ON (one pass over the decoder weights for all active
slots)"` (`qwen_tts_server.c:2352`). That is a gate worth checking on any box before believing
a decoder-batching number.

Ragged handling in the BLAS path is **concatenation, not padding**. `sd_rag_t`
(`qwen_tts_speech_decoder.c:3674–3679`) is `{ n, len[], off[], total }` with
`rag_recompute` laying items end to end (`:3687–3691`). The `kernel == 1` convolutions become
one GEMM over the concatenated time axis of width `total` (`:4023–4027`); the `kernel > 1`
convolutions do an im2col that clips each item to its own `[off, off+len)` window
(`:4050–4055`) and carries a per-item causal tail (`rag_save_tail`, `:3694–3700`). A
same-length group of 2–3 additionally takes a shared-weight multi-context kernel
(`:3985–4019`).

### 2.4 Where a late joiner gets folded in

There is no separate join path. `ADMIT_INSTALL` (`qwen_tts.c:3298–3325`) writes the slot's KV,
`last_hidden`, sampler parameters, RNG, trailing-text tail, and `active[b] = 1`. From the next
statement on, the slot is indistinguishable from one that has been running for 200 frames:
`qwen_batch_pack_active` picks it up by index in the same iteration. **Ragged positions are
what make this free** — `pos_arr` is per slot (`qwen_batch_talker_step_ragged(..., pos, ...)`,
`qwen_tts.c:4135`; consumed as `POS_B(b)` at `qwen_tts_talker.c:2726`), so a slot at position
512 and a slot at position 3 share the same GEMM and differ only in their own attention call.

### 2.5 The mechanism we are missing: one dispatch per step, not 112

This is the part that is not in the notes' advertising and is the largest structural
difference. `tk_region_run` / `tk_region_task` (`qwen_tts_talker.c:2580–2706`) runs **the whole
Talker layer stack as a single persistent parallel region**: one `qwen_parallel((size_t)team,
tk_region_task, &r)` (`:2703`) with **8 spin barriers per layer** (`:2594, 2600, 2619, 2621,
2629, 2631, 2638, 2640, 2653`) instead of one pool entry and exit per projection. The header
comment states the arithmetic: *"Eight barriers per layer, one dispatch per step instead of
112"* (`:2503–2504`), and *"Kernels and partition are unchanged, so the hidden states are
bit-identical"* (`:2504–2505`).

Its gates matter and are easy to miss:

- decided **once**, in a function-local `static int region_mode` (`:2663`), from
  `QWEN_TK_REGION != "0"`, `qwen_parallel_team() >= 2`, `bb->B >= 2`, and **all six weights
  int8 with no q4** (`:2667–2684`);
- re-checked **per call**: `if (region_mode == 0 || bb->force_matvec || BW < 2 || BW > 16)
  return 0;` (`:2689`). So **a solo request never uses the region** — with one active slot the
  loop falls back to the 112-dispatch path;
- `team` is re-read per step (`qwen_parallel_team()`, `:2701`), so the region follows the
  elastic lane's width from frame to frame;
- two runners: x86 VNNI in-region row blocks, or KleidiAI prepared-state on Arm (`region_mode
  == 2`, `:2679–2684`) — the Arm one packs the activation once on the region leader and shares
  it across the team behind a barrier (`tk_region_kai_prep`, `:2541–2553`).

The code predictor has the same treatment (`cp_frame_region_run`, one dispatch per decode
frame, `qwen_tts_code_predictor.c:1494`) plus a solo bypass that re-points `ctx->cp_kv_*` at
the slot's slice and calls the single-request predictor (`:1512–1532`).

---

## 3. THE THREAD POOL AND THE LANES, AS THEY EXIST IN CODE

All in `qwen_tts_thread.c`. Three backends; the pthread one (`#else`, `:268`) is Linux
production. macOS uses GCD `dispatch_apply` (`:64–117`) and has **no lane at all** (`:103–114`
are all stubs) — a number taken on macOS says nothing about the lane.

### 3.1 How a region is dispatched

`qwen_parallel`, `:528–612`:

1. lane redirect first — if `g_lane_tls` is set, the call goes to `lane_parallel` and never
   touches the engine pool (`:532`; the comment at `:530–531` calls this "the single choke
   point");
2. serial if the pool is uninitialised or `nt == 1` (`:533–537`);
3. optional LOW-priority wait (`:546–556`);
4. `pthread_mutex_lock(&P.submit_mtx)` (`:557`) — **held until `:610`, i.e. across the entire
   region including the completion wait**;
5. `need` is computed (`:562–570`);
6. `P.job = &job`, generation bumped with `need` packed into its low 16 bits
   (`QWEN_GW_MAKE`, `:286–291`, stored at `:576–580`);
7. conditional broadcast only if a *needed* worker is actually parked
   (`:581–586` — it masks `P.sleep_mask` against `need_mask`, so it does not wake sleepers it
   does not need);
8. the caller runs its own chunks (`:589`);
9. completion wait (`:594–608`).

### 3.2 What the barrier actually costs

Two different barriers, and the distinction matters.

**Pool completion barrier** (`:595–607`): the submitter spins `qwen_pool_spin()` iterations of
`qwen_cpu_relax()`, then takes `P.mtx` and parks on `P.complete`. Workers do the symmetric
thing on `P.wake` (`:410–430`). Cost is bounded by the spin budget and then becomes a futex
sleep.

**The killer default is `qwen_pool_narrow()`**, `:326–333`, read from `QWEN_POOL_NARROW` and
**defaulting to off**. At `:565`:

```c
if (!qwen_pool_narrow()) need = P.nworkers;
```

So by default **every dispatch wakes and waits on the whole team, regardless of `nt`**. A
two-chunk region on a 8-thread worker still requires all 7 workers to enter `run_chunks`,
find `job->next >= nt`, increment `P.completed`, and be waited for. On a serving loop that
dispatches often, that is the dominant fixed cost — and it is exactly what the persistent
region (§2.5) exists to amortise.

**Region barrier** (`qwen_barrier_wait`, `:20–34`): a sense-reversing spin barrier with **no
park and no budget** — it spins on `b->phase` forever. The comment justifies it
(`:16–18`: *"every participant is a distinct thread already running … and the phases are
microseconds long"*) and that holds **only** if the region was dispatched with `nt == team`
and no participant gets descheduled. On an oversubscribed box or a mask that overlaps another
worker's, one preempted thread spins the other seven. (inference: they never say this; it is a
property of the loop as written.)

### 3.3 Is the spin budget read per dispatch or cached?

**Called per dispatch, computed once per process.** `qwen_pool_spin()` (`:317–325`) caches in a
function-local `static int v` on first call. It is *called* at `:410` (worker wake loop),
`:595` (submitter wait), `:752` (lane worker) and `:863` (lane submitter) — so the call is per
dispatch, the `getenv` is not. Default is **65536 on Linux/aarch64, 4096 elsewhere**
(`:312–316`) — a 16× difference, and our target is the aarch64 side of it. The shipped Arm
profile pins `QWEN_POOL_SPIN=65536` with the honest comment *"Arm-specific starting policy;
must be requalified on the target host"*
(`configs/perf/axion-c4a-highcpu32-0p6b-all-on.json`).

There is one place the budget **is** dynamic: the lane worker halves it to ≤1024 when no
decoder unit is in flight and elastic mode is on (`:752–753`), so lane cpus park quickly and
give themselves back to the engine between units.

### 3.4 The decoder lane — creation, teardown, and unsatisfiable requests

**Prepare** (`qwen_lane_split_prepare`, `:675–733`) — Linux only, returns 0 elsewhere
(`:729–732`). Reads the *current affinity mask* (`:680–683`), splits the last `N` cpus off as
the decoder set and the rest as the step set (`:690–694`), and in the static (non-elastic) mode
calls `sched_setaffinity` on the **calling thread** *before* the pool is spawned
(`:717`) — because "pthreads inherit the creating thread's mask" (`:715–716`). Ordering is a
hard requirement and the prefork child honours it: `qwen_lane_split_prepare` at
`qwen_tts_server.c:2912–2913`, `qwen_set_threads` at `:2914`.

**Team start** (`qwen_lane_team_start`, `:785–814`) creates `L.dec_cpus - 1` workers pinned to
`L.dec_set` (`:807–811`, pinning at `:741`) and returns `nworkers + 1`. In elastic mode it
*also* re-pins the engine pool one worker per cpu and the caller to cpu 0 (`:789–799`), so that
capping the width to `step_width` cleanly excludes the decoder cpus.

**Teardown** (`qwen_lane_team_stop`, `:816–826`): sets `L.stop`, bumps the generation with
`need = 0`, broadcasts, joins. Called from `qwen_tts.c:4267` together with
`qwen_pool_set_width(0)`.

**When a lane is requested but cannot be satisfied** — three distinct behaviours, all
explicit, none silent:

| condition | behaviour | line |
|---|---|---|
| `QWEN_SD_LANE_SPLIT >= ncpus in mask` | prints *"split DISABLED (the engine needs at least one cpu)"*, returns 0 | `qwen_tts_thread.c:684–688` |
| not Linux | returns 0, nothing changes | `:729–732` |
| ctx clone or `pthread_create` fails | prints *"decoder thread requested but clone/thread failed — staying inline"*, frees the clone | `qwen_tts.c:3031–3034` |
| requested engine threads exceed the step side | prints *"requested %d engine threads, step side has %d cpus: keeping %d"* and treats the request as a **ceiling** | `:718–725` |

**Elastic width** is the mechanism worth studying. `dec_enqueue` sets
`qwen_pool_set_width(dp->step_width)` when the first unit is queued (`qwen_tts.c:2821`), and
`dec_worker_main` restores `qwen_pool_set_width(0)` when the queue drains
(`qwen_tts.c:2781–2784`). `qwen_parallel` reads it per dispatch (`:566–570`) and
`qwen_parallel_team()` reports the capped value (`:614–619`), which is why the persistent
regions follow it without being told.

**The mailbox contract** is bounded and audited: one unit in flight per slot, enforced by
`dec_wait_idle` blocking *only that slot's* producer (`qwen_tts.c:2657–2665`), with an explicit
overrun counter that is asserted to stay zero and printed at shutdown
(`g_lane_overrun`, `qwen_tts.c:2792–2799`, reported at `:4191–4192`: *"mailbox overruns %lld
(must be 0)"*). Preallocated per-slot job and codes buffers avoid a heap allocation per unit
(`qwen_tts.c:2801–2806`, cap 32 frames).

---

## 4. SURPRISES — things with no note, guards describing bugs that are gone, inert knobs

**4.1 `QWEN_DECODER_BATCH` is inert in their own shipped Arm profile, and the server still
prints that it is ON.** The chain:

- the server sets it on by default and announces it:
  `setenv("QWEN_DECODER_BATCH", "1", 0)` then prints *"batched speech decoder ON (one pass over
  the decoder weights for all active slots)"* (`qwen_tts_server.c:2349–2353`);
- `dec_on` becomes 1 whenever the lane started (`qwen_tts.c:2990`);
- `dpool.batch` is set **only when there is no lane**: `(!dpool.lane && getenv(...))`
  (`qwen_tts.c:3006–3007`);
- and the inline path is killed outright: `if (dec_batch && dec_on) dec_batch = 0;`
  (`qwen_tts.c:3099`).

The Arm 32-core profile sets `QWEN_SD_LANE_SPLIT=4`, `QWEN_SD_LANE_ELASTIC=1` **and**
`QWEN_DECODER_BATCH=1` with the rationale *"keep server batching semantics"*
(`configs/perf/axion-c4a-highcpu32-0p6b-all-on.json`). With the lane on, that flag reaches no
code. What actually batches there is the 2-slot multislot cohort (`QWEN_SD_MULTISLOT=2`), which
is a different, narrower mechanism: at most 3 slots, **same pending length only**
(`dec_enqueue_cohort` returns 0 on any length mismatch, `qwen_tts.c:2843–2844`).

This is the exact failure mode the brief says it got burned by this week, present in their
repo, in a committed config, with a justification that the code contradicts.

**4.2 Their own `--effective-config` would not catch 4.1, and that is the more interesting
half.** `qwen_effective_config_report` (`qwen_tts_dispatch.c:201–257`) prints
requested/effective/why for every set flag and marks a flag IGNORED when the owning subsystem
declares it inert (`qwen_pool_flag_inert`, `qwen_tts_thread.c:894–910`) or when its kernel is
not compiled (`:237–247`). The scope table is **generated from the sources** by
`tools/flag_parity.py` so it cannot drift (`:158–159`), and it covers every serving flag I
checked (16/16 present in `qwen_flag_scope.h`). But inertness is *owner-declared*, and nobody
declared `QWEN_DECODER_BATCH` inert-under-lane — so it prints `honoured`. The mechanism is
excellent and the coverage is a policy gap, not a bug. Worth copying **with** a rule that the
owner of a mutual exclusion must declare it.

**4.3 The startup banner in the batched server is stale and says the opposite of what the code
does.** `qwen_tts_server.c:2471`:

```
"  POST /v1/tts/stream   — generate speech (chunked PCM, single clone)\n"
```

But `reader_main` routes a stream request to the **batch** queue with
`j->req.want_stream = is_stream` (`:1915–1923`), and only diverts to the single-clone worker
when `needs_single` — which is set exclusively by a non-empty `instruct` (`:1809`) or
`voice_design` (`:1817`). Streaming has been fully batched for some time; the banner still
describes the old topology.

**4.4 In the default (synchronous) output mode the frame loop writes PCM to client sockets
itself, with no send timeout.** `QWEN_SERVER_ASYNC_OUTPUT` is unset by default
(`stream_output_enabled`, `qwen_tts_server.c:315–318`), so `j->out == NULL` and `sink_on_chunk`
calls `send_pcm_chunk` directly on the scheduler thread (`qwen_tts_server.c:2052`).
`set_client_timeout` sets only `SO_RCVTIMEO` and `TCP_NODELAY` (`:1434–1441`); `SO_SNDTIMEO` is
set **only** in `stream_output_start` (`:458–462`), i.e. only on the async path. So one slow
consumer can block the frame loop — and every other slot's cadence — inside a `write()`, with
no timeout at all. Their own Arm profile pins `QWEN_SERVER_ASYNC_OUTPUT=0` ("same synchronous
output contract"). Our `server/stream_out.c` exists precisely to avoid this; that is one place
we are ahead and should stay ahead.

**4.5 `stream_output_enabled()` calls `getenv` on every admission** — no static cache
(`qwen_tts_server.c:315–318`), called from `sink_next_job` at `:1989`. Every other predicate in
the file caches (`qwen_cancel_on_disconnect` `:206–210`, `qwen_f2_trace` `:1649–1656`,
`qwen_server_strict` `:627–634`). Harmless, but it is the sort of inconsistency that says the
async-output path was added later and not swept.

**4.6 The non-region Talker fallback normalises inactive slots.** `qwen_tts_talker.c:2734–2735`:

```c
for (int b = 0; b < B; b++)
    qwen_rms_norm(bb->x_norm + (size_t)b * h, bb->x + (size_t)b * h, l->input_norm, 1, h, eps);
```

No `ACTIVE_B(b)` guard, though the same function guards every other per-slot loop (`:2744`,
`:2755`). The region path does it correctly (iterates `j < BW` over `act_idx`,
`qwen_tts_talker.c:2589`). So on a q4 build, or with `B_eff == 1`, or `B_eff > 16` — all the
cases that fall out of the region — the fallback pays `num_layers × (B − B_eff)` wasted
rms\_norms per step. Small, but it is dead work on exactly the configurations that already lost
the region's dispatch saving.

**4.7 `QWEN_ADMIT_UTIL` only exists at `--batch-size 2`.** `admit_util = qwen_admit_util_requested() && cap == 2`
(`qwen_tts_server.c:2789`), with an explicit refusal message for any other size (`:2793–2796`).
It is the LS-4 utilisation-aware admission: the parent reads each child's last iteration time
from a `MAP_SHARED` page (`:2798–2813`) and grants **one** transient extra slot to the
healthiest full worker (`:3049–3100`), with staleness and warm-up rejected by name
(`qwen_admit_util_sample_ok`, `:1624–1645`). The sample is published sequence-last so a partial
refresh reads as stale with no lock (`qwen_tts.c:3466–3473`). The design is good; the `cap == 2`
restriction means it cannot apply to the 4×8 topology at all.

**4.8 `js_value` / `json_validate_object` validate the body properly, but every scalar is then
read with `strstr`.** `json_extract_number` (`qwen_tts_server.c:94–103`) does
`strstr(json, "\"seed\"")` over the raw body, so a key name occurring inside a *string value*
matches first. `reject_unknown_fields` (`:757–796`) does track string state and depth
correctly. Off the brief's path, but it means `"text": "the \"seed\" of it"` can shift a
parameter.

---

## WHAT I READ

Whole files: `qwen_tts_server.c` (3194), `qwen_tts_thread.c` (910), `qwen_tts_server.h`,
`qwen_tts_batch.h`, `qwen_tts_thread.h`. Whole functions/regions: `qwen_tts.c` 2440–2600
(prefill queue + helper), 2640–2870 (decoder pool, mailbox, cohort), **2870–4330
(`qwen_tts_serve_continuous`, complete)**, 912–1000 (`clone_for_worker`), 1984–2046 (slice
knobs, caps); `qwen_tts_talker.c` 1861–2060 (`prefill_plan` / `prefill_range`), 2230–2300 +
2400–2440 (`batch_proj`, `pack_active`), 2495–2706 (`tk_region_*`), 2708–2830
(`batch_talker_step_impl`); `qwen_tts_code_predictor.c` 1490–1620; `qwen_tts_speech_decoder.c`
3665–3700, 3981–4055, 4470–4520, 4930–4957; `qwen_tts_dispatch.c` 120–265; `main.c` serve
wiring (764–780, 855–910, 1029–1047, 1231–1250, 3062–3093) and `--prefill-slice-check`
(1795–1900). On our side, read-only: `mynah-tts/server/main.c` 460–540, `src/inference.c`
187–215 and 505–535 and 740–835, `src/engine_pocket.c` 3315–3335. Configs:
`configs/perf/*.json` (all `server`/`streaming` blocks; `axion-c4a-highcpu32-0p6b-all-on.json`,
`arm-product.json`, `turin-c8a-32c-vnni-product.json` in full), `configs/perf/schema.json`,
`docs/feature-flags.md` (grep-scoped).

A separate pass inventoried the twelve `.work/` notes plus `docs/batching.md`,
`docs/pipeline.md` and `PLAN.md` as *claims to check*, never as a source. Three of those claims
were load-bearing enough that I re-read them myself before using them, and all three are quoted
here from the file rather than from the summary: `PLAN.md:131–149` (the C12-WIN-10 closure,
§1.5), `.work/dl1-decoder-lane-split-20260909.md:284–297` (the 4×8 TTFB table, §1.6), and
`PLAN.md:245–253` (the Arm lane-split scope, which turned out not to be the contradiction it
looked like).

## WHAT I FOUND

1. **Their default admission is ours**: fill every free slot in one iteration, one full inline
   prefill each (`qwen_tts.c:3692–3715`). The sliced, helper and M1 alternatives all exist and
   are all default-off.
2. **Our premise about our own code is wrong**: `src/inference.c:754` is a `while`, not an
   `if`. Our TTFB tail is a convoy of serialised `prepare()` calls inside one iteration, not a
   queue of iterations.
3. **They ran "admit one per iteration" and it was a −44 % / 5× regression** (`PLAN.md:131–149`),
   with `mean_slices=1.00` proving the slicing itself was a no-op and the `break` at
   `qwen_tts.c:3631` was the only live variable. Their own A/B design note also warns that a
   *wave* workload — the shape of our C30 harness — cannot reach the mechanism at all
   (`PLAN.md:145–148`, `qwen_tts.c:3599–3601`). §1.5.
4. **Their TTFB p95 on a 4×8 box is 49 / 103 / 133 ms at C8 / C12 / C16**, rising with
   arrivals-per-worker — the convoy signature — at *half* our per-worker slot count and *half*
   our concurrency, for the same STREAM\_RTF we hit at C30. §1.6.
5. **Their answer to the burst is admission control, not faster admission**: `max_queue: 0`,
   immediate 503 at the prefork parent (`qwen_tts_server.c:2756`, `3101–3112`), and a qualified
   operating point of **C16 preferred / C20 edge on 32 slots** on Neoverse-V2 — with the
   profile still marked `"unqualified"`.
6. **The batch is a gather/scatter pack over an index list, never padded**
   (`qwen_tts_talker.c:2406–2416`, `2245–2255`); ragged positions are a per-slot `pos_arr`, and
   the only thing that stays scalar is the per-sequence attention (`:2754–2761`).
7. **The batched vocoder packs ragged by concatenation** (`sd_rag_t`,
   `qwen_tts_speech_decoder.c:3674–3691`) and **only exists under `USE_BLAS`** (`:4497` vs
   `:4952`).
8. **The mechanism with no note: one pool dispatch per Talker step instead of 112**
   (`tk_region_run`, `qwen_tts_talker.c:2499–2706`), bit-identical, gated on int8 + `2 ≤ B_eff
   ≤ 16`, and re-reading the team per step so it follows the elastic lane.
9. **`QWEN_POOL_NARROW` defaults off**, so every dispatch wakes and barriers the whole team
   regardless of `nt` (`qwen_tts_thread.c:565`). The region path is what makes that affordable.
10. **`QWEN_DECODER_BATCH` is dead whenever the lane is on** (`qwen_tts.c:3006, 3099`) while the
   server prints it ON (`qwen_tts_server.c:2352`) and their shipped Arm profile sets it with a
   rationale the code contradicts.

## WHERE CODE AND NOTES DISAGREE

Checked against the code, not against each other. (The notes survey ran as a separate pass; the
items below are the ones I verified myself from code + committed configs, which is the only
form I would act on.)

- **"Sliced admission" as a shipped property — it is not.** `QWEN_PREFILL_SLICE` defaults to 0
  (`qwen_tts.c:1990–2010`), appears in **no** config under `configs/`, and appears **nowhere**
  in `docs/`, `PLAN.md` or `README.md`. It is real, tested and unadopted. Anything describing
  qwen's live admission as sliced is describing an experiment.
- **`QWEN_DECODER_BATCH=1` "keeps server batching semantics" on the Arm 4×8 profile — it does
  nothing there.** Config rationale vs `qwen_tts.c:3006` and `:3099`. §4.1.
- **The server banner says `/v1/tts/stream` is served by a "single clone"** —
  `qwen_tts_server.c:2471` vs the batch routing at `:1915–1923`. Streaming is batched. §4.3.
- **`streaming.decode_chunk: 4` in the profile vs a compiled default of 8**
  (`qwen_tts.c:3131`). Not a contradiction *if* the profile is launched through
  `tools/perf_profile.py`, which maps `decode_chunk → QWEN_STREAM_DECODE_CHUNK`
  (`tools/perf_profile.py:208–209`). Launched as a raw `./qwen_tts --serve`, the profile's
  steady-state quantum silently doubles. Their own `docs/feature-flags.md:48` and `:270`
  document the default as 8.
- **`QWEN_ADMIT_UTIL` as a general admission policy** — it is refused for any
  `--batch-size != 2` (`qwen_tts_server.c:2789, 2793`), so it cannot exist on the 4×8 topology
  anybody actually ships.
- **The `.work/` line numbers have drifted about 150–250 lines.**
  `.work/c12-win-admission-slicing-implementation.md` cites `ADMIT_PREFILL` at
  `qwen_tts.c:3154–3176` (actual: `3273–3296`), the admission loop at `3339` (actual: `3495`),
  `ADMIT_INSTALL` at `3178` (actual: `3298`), and `qwen_batch_pack_active` at
  `talker.c:2045–2056` (actual: `qwen_tts_talker.c:2406–2416`). The 32-core review cites
  `sd_batch_fallback` at `sd.c:3852–3856` (actual: `qwen_tts_speech_decoder.c:4485–4495`).
  **The claims at those sites are still correct** — the spec's *"`sink_next_job` pops a request
  (qwen\_tts.c:3339 loop, **up to B per iteration**)"* is exactly what `:3495–3715` does today.
  Only the coordinates are stale. Worth saying because a line-number citation that no longer
  resolves reads as a wrong claim when it is a right one.
- **`docs/pipeline.md` says the speech decoder runs on its own thread and costs ≈ 0 on the
  critical path** — the code makes `dec_on` conditional on `QWEN_DECODER_THREAD` or a started
  lane (`qwen_tts.c:2990`), both default-off, so the default is inline on the loop thread. The
  32-core review agrees with the code against the doc and puts the decoder at ~72–80 % of the
  marginal slot cost. `docs/pipeline.md` is an M1-era document (mtime 2026-06-04) and is the
  most stale file in their tree.
- **`docs/batching.md` lists server request-batching as "to build"** — it has been shipped and
  measured for months. Same vintage problem (2026-06, M1 microbenches). Both docs are still
  reachable from the repo root with no staleness banner.
- **The 32-core review and I reach the same conclusion about `QWEN_DECODER_BATCH` by different
  routes, and both are true.** It says the flag reaches `sd_batch_fallback` on VNNI — *"the
  server's 'batched speech decoder ON' banner is misleading here"*. I found it is also killed
  outright whenever the lane is on (`qwen_tts.c:3006, 3099`). Two independent reasons, one
  misleading banner (`qwen_tts_server.c:2352`), and a committed config that sets the flag with
  a rationale neither route supports. §4.1.
- **Not a contradiction, though it reads like one:** `PLAN.md:253` says *"split=2 was slower,
  so no lane split is promoted in the Arm profile"*, while
  `configs/perf/axion-c4a-highcpu32-0p6b-all-on.json` sets `QWEN_SD_LANE_SPLIT=4`. The scopes
  differ — `arm-product.json` (the generic Arm profile) has `QWEN_SD_LANE_SPLIT: null`, and the
  axion file is host-scoped and marked `"status": "unqualified"`. I checked this one expecting
  a contradiction and did not find one; recording it so nobody else spends the same hour.

**Measurements I could verify from code vs measurements that are assertions.** Verifiable from
code: the *shape* of every claim above — dispatch counts, batched-vs-looped tensors, defaults,
gates — plus one number I could **recompute exactly**: the 32-core review's *"225 spin barriers
per Talker step"* is 8 × 28 + 1, and the code has exactly 8 `qwen_barrier_wait` calls per layer
plus one before the loop (`qwen_tts_talker.c:2594, 2600, 2619, 2621, 2629, 2631, 2638, 2640,
2653`). Also verifiable as structured data: the operating point in
`configs/perf/axion-c4a-highcpu32-0p6b-all-on.json` (`STREAM_RTF p95 0.881` at C16, `p50/p95
1.02/1.09` at C20, mel correlation `0.94558–0.96306`, `"status": "unqualified"`) and the
regression in `PLAN.md:137–140`.

**Only ever assertions in prose**, with provenance quality that varies by an order of
magnitude. The best of them state commit, binary SHA-256, host, harness and pre-declared
thresholds (`.work/ls4-utilization-aware-admission-20260908.md`,
`.work/p4-prefork-admission-bound-20260907.md`, `.work/mt4-transport-boundary-20260907.md`);
the lane and TTFB tables are self-labelled *"SCREEN grade"*, *"one wave"*, *"What this is not:
a qualification"*. And one quantity — the cost of an inline prefill, which is the whole subject
of §1 — appears as **four different numbers across four documents**: 58 ms (32-core review),
108–240 ms (`PLAN.md:42`), 60–240 ms *"on the old engine"* (the C12 spec), and ~600 ms when a
profile accidentally pinned `QWEN_PREFILL_MATMAT=0` (32-core review §19.6, which calls that a
defect). If we take one number from their corpus, it should not be this one.

**Correction to an earlier draft of this document:** I wrote that no qwen TTFB number existed.
That was wrong about the notes and right about the configs — see §1.6 for the numbers and their
grade.

## WHAT TO PORT, IN ORDER

**Ordering changed after reading `PLAN.md:131–149` (§1.5).** My first draft put "cap admissions
per iteration" first. That is the change qwen measured as **−44 % completions and STREAM\_RTF
p95 0.922 → 4.609**, in isolation, on four workers. It is now item 3, behind the instrumentation
that would tell us whether it is even the right target.

**1. Port the per-iteration `[STAGE]` line. Nothing else until this exists.**
Their shape (`qwen_tts.c:4159–4173`): per iteration emit `active`, `step`, `admit_ms`,
`prefill_ms`, `head_ms`, `sample_ms`, `cp_ms`, `decode_ms`, `talker_ms`, `output_ms`,
`queue_wait_ms`, `wall_ms`, and — the part that makes it honest —
`serial_ms = wall − Σ(buckets)`, clamped at 0 (`:4157–4158`), so the buckets are forced to
account for the wall clock instead of quietly summing to less than it. Add the bucket label
from `qwen_tts.c:4229` in spirit: *"admission + prefill: what a new arrival costs the requests
already in flight."*
*Smallest experiment:* emit it under an env flag, run one C30 wave, read `admit_ms` p95 and
`serial_ms`. **This answers the whole question:** if `admit_ms` p95 is ~400 ms we have the
convoy and item 3 is worth its risk; if it is ~40 ms the 500 ms is somewhere else entirely and
items 2–3 are void. We are currently reasoning about our admission cost without having measured
it once — the 70 ms/prefill figure behind the convoy model is my arithmetic, not a number.

**2. Count our dispatches per batched step** (their `qwen_parallel_meter` /
`qwen_parallel_meter_read`, `qwen_tts_thread.c:48–57`). Ten lines, no behaviour change.
It tells us whether item 4 is worth anything and it is the denominator every latency number
is measured against.
*Smallest experiment:* add the counter to `src/threads.c`, run one C30 wave, print
dispatches/step and mean chunks/dispatch. **Decision rule:** ~1 dispatch/step → skip item 4
entirely; ~100 → item 4 is the largest structural win available.

**3. Only then, an admission cap — as a flagged experiment with a kill switch, not a default.**
`src/inference.c:754`'s `while` becomes admit-at-most-`k`. Given §1.5 this is not a fix, it is
a hypothesis with a known failure mode, and it must be run the way their void A/B was not:
- **the workload must be closed-loop, not a wave.** Their first attempt was declared void
  because a wave releases everything into an idle engine and `n_active == 0` bypasses the
  mechanism by design (`qwen_tts.c:3599–3601`, `PLAN.md:145–148`). Our 90-request C30 wave has
  exactly that shape and would measure nothing.
- **watch STREAM\_RTF p95 and completions, not TTFB.** Their regression was invisible in TTFA
  for the first arm and showed up as a 5× STREAM\_RTF blow-up and 44 % fewer completions.
- **stop rule, declared in advance:** abandon if STREAM\_RTF p95 rises at all or completions
  fall by more than 2 %.

**4. One dispatch per step for the batched backbone** (their `tk_region_run`,
`qwen_tts_talker.c:2499–2706`). This is the mechanism our `threads.c` most plausibly lacks, and
it is orthogonal to admission: it cuts the per-step pool cost for *every* frame, which is the
denominator TTFB is measured against.
*Smallest experiment:* gated on item 2's number. Their own corroboration that this is where the
time goes: `.work/32c-serving-utilization-architecture-review-20260909.md` counts **943 spin
barriers per frame** (Talker region 225, CP region 718) against **one** pool entry each — and
the Talker figure checks out exactly against the code, 8 barriers × 28 layers + 1
(`qwen_tts_talker.c:2594, 2600, 2619, 2621, 2629, 2631, 2638, 2640, 2653`). That is the note I
would trust most in their whole corpus, because it is the one I could recompute.

Three things I would **not** port. **The sliced prefill** — built, given a state oracle, and
closed as a severe regression (§1.5); their own verdict is that a retry *"needs a redesign of
the admission path first"* (`PLAN.md:148`). **`QWEN_ADMIT_UTIL`** — it refuses any
`--batch-size != 2` (`qwen_tts_server.c:2789`), so it cannot exist on a 4×8 topology.
**`QWEN_TTFA_PRIORITY`** — it buys a new arrival's TTFA by stalling every established stream
(`qwen_tts.c:3752–3765`), which is the trade we are explicitly not making at STREAM\_RTF p95
0.922 with zero stalls.

One thing to port as a *rule*, not code: their `--effective-config` idea that every declared
flag prints requested / effective / why, with inertness **declared by the owning subsystem**
(`qwen_tts_dispatch.c:201–257`, `qwen_tts_thread.c:894–910`). §4.1 and §4.2 together are the
cautionary tale: the mechanism is excellent and it still let a config ship a flag that reaches
no code, because nobody declared the mutual exclusion. If we copy it, copy it with the rule
that whoever writes `if (a && b) b = 0;` owes an inertness declaration.

## WHAT REMAINS UNKNOWN

- **The cost of our `prepare()` at C30.** Everything in item 1 turns on it and I did not
  measure it — I read `pocket_prepare → pocket_seed_context`
  (`src/engine_pocket.c:3315–3326`) and stopped, per the brief. The 500 ms / 7 ≈ 70 ms per
  prefill implied by the convoy model is **(inference)**, not a measurement.
- **How many dispatches per step our `threads.c` issues.** Item 3's value is entirely
  conditional on this and I did not open `src/threads.c`.
- **Whether their production Arm binary is built with BLAS.** The batched vocoder does not
  exist without it (`qwen_tts_speech_decoder.c:4497` / `:4952`) and the Arm profile's
  `gemm_backend` is "KleidiAI INT8/Q4/BF16", not OpenBLAS. If `USE_BLAS` is off there, the
  multislot cohort path is the *only* decoder batching on Arm and the `QWEN_DECODER_BATCH`
  finding in §4.1 is even stronger. Decidable from their Makefile; I did not read it.
- **Whether `qwen_causal_attention_bf16kv` is internally parallel.** I established it is
  called once per slot (`qwen_tts_talker.c:2758`, `2614`) but did not read the kernel, so I
  cannot say whether the per-slot attention loop is itself a pool dispatch — which would change
  the dispatch arithmetic in §2.5.
- **Whether their C8/C12/C16 TTFB curve (§1.6) survives at C30 and cap 8.** It is SCREEN grade,
  one wave per level, cap 4, and stops at C16 — the regime we care about is off the end of it.
  The comparison is suggestive, not decisive, and I would not put it in front of anyone as
  "qwen does 133 ms and we do 500 ms" without saying cap 4 / C16 / one wave in the same breath.
- **What actually broke in the C12-WIN-10 A/B.** Their post-mortem says *"most likely"* and
  leaves it open (`PLAN.md:143–145`); my two candidates in §1.5 are inference. Until someone
  explains why a single `break` costs 44 % of completions, item 3 of the port list is a
  hypothesis and not a plan.
- **`aws-c9g-8xlarge-32c-arm-v2-all-on.json` is untracked.** I used it only to confirm the
  4×8 / batch 8 / `max_queue 0` topology, which the committed
  `axion-c4a-highcpu32-0p6b-all-on.json` states identically. Nothing above rests on the
  untracked file alone.
