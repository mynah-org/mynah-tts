# E5 — Streaming server v2, ported from qwen-tts

Status: **OPEN** · CPU only · audit 2026-09-12

Supersedes the July server roadmap (`PLAN.md` §24, P0-P3), which is closed.

## The finding that reorders this epic

**`synthesize_slots` (`graph.c:4046-4200`) is a closed batch.** It receives
`count` slots, prepares them all (`:4055-4057`), sizes scratch on exactly those
(`:4061-4077`), and loops (`:4085-4182`) until every one is finished. **There is
no admission point — no `next_job` anywhere.** The "continuous" in the comment
at `:4041-4045` means only that a slot reaching EOS drops out and the rest
continue at reduced width. `tests/test_server.sh:190-215` confirms this: it
checks that a mid-flight arrival is *served*, in the **next** batch.

So removing the global mutex is **not** the first step, and on its own would not
buy concurrency. The first step is giving the driver a sink with `next_job`.

## The defect that existed, and was worse than stated — fixed 2026-09-12

`stream_callback` did three **blocking** socket writes from the synthesis
thread, inside the `g.synth_lock` critical section, on a fd with `SO_SNDTIMEO`
of 30 s. This note used to say that blocked the process for 30 seconds. **It was
unbounded**: the old `write_all` restarted a fresh `SO_SNDTIMEO` on every
partial send, so a client that reads slowly rather than not at all extends the
stall indefinitely. Measured at **66.5 s** before anyone gave up.

Fixed by `server/stream_out.{c,h}` (E5 step 3). Same scenario, measured:

| | second client served after |
|---|---|
| before (synchronous writes under the lock) | **66.50 s** |
| after (async writer, 1 MiB queue) | **4.81 s** |
| after (64 KiB queue, hits the cancel path) | **2.64 s** |

The residual 4.81 s is the synthesis itself: `g.synth_lock` is still there, by
design, until step 4.

Implementation note worth keeping: the queue is a **single-byte ring**, not a
list of malloc'd chunks, which is the only way to get "no malloc per chunk"
honestly — one allocation at start, PCM16 conversion into a reused scratch
*outside* the lock, then a memcpy into the ring under it. The writer writes
straight from the ring **without the lock**, which is safe because `queued` only
shrinks once a write completes, so the producer can never touch the region in
flight. ThreadSanitizer was run specifically to validate that. A side benefit is
that the writer coalesces a backlog into one HTTP chunk.

An aborted stream is never silent: the server logs the byte count, the queue
high-water mark and the refusal count.

## What to port, and what it depends on

| Item | qwen-tts source | Note |
|---|---|---|
| Three-role topology | `qwen_tts_serve_batched:2402-2520` | readers parse HTTP and never synthesize (`reader_main:1845`), one scheduler owns the ctx (`scheduler_main:2269`), a single worker handles what the batch cannot |
| Sink contract | `qwen_tts.h:647-659` | `next_job / on_done / on_chunk / running / cancelled / on_reject / step_allowed`; admission happens **inside the frame loop** (`qwen_tts.c:3515, 3697`) with `block = (n_active == 0)` |
| Async output writer | `qwen_tts_server.c:280-539` | detached thread, bounded queue (1 MiB), `SO_SNDTIMEO`, atomic refcount 2, PCM conversion outside the lock. **Backpressure is cancel, not block.** |
| Cancel on disconnect | `peer_hung_up:212-217`, `write_all_or_gone:219-233` | treats EAGAIN as a disconnect rather than spinning |
| Header at admission | `sink_next_job:1989-2007` | published when admitted, not at first chunk |
| Warm-up ramp | `qwen_tts.c:3830-3837` | `1,2,2,4,4 → chunk` |
| Prefork (Linux) | `:2746-3186` | needs `after_fork` first — see risk 2 |

Copy literally: **`qwen_sd_stream_state_t`** (`qwen_tts.h:310-337`) — codec
streaming state **outside** the model object, one per slot. Do **not** replicate
`ctx->sd_stream` (`qwen_tts.h:455`), the leftover single-path default.

## Port plan, ordered by risk: 0, 1, 3, 2, 4, 5, 6, 7

**0. Synthetic pack + goldens.** Done (E1-0). Without it no step below has a
gate.

**1. Split reader from synthesis; the job owns the fd.** `synth_ticket` moves
off the worker's stack (`main.c:481-482`) onto the heap. `handle_speech` parses,
tokenizes, enqueues and returns; the scheduler owns the fd, responds and closes.
`handle_connection` stops closing unconditionally (`:666`) and returns an
"fd handed off" flag. *Breaks:* double close, fd leak, the 503 path.
*Gate:* full `make server-test`, `make leaks`.

**3. Async writer** (before 2, because it is independent and de-risks 4). New
`server/stream_out.{c,h}`, added to `SERVER_SOURCES` (`Makefile:125`). Port
`qwen_tts_server.c:280-539` stripped. `stream_callback` becomes an enqueue: no
socket write from the synthesis thread, and no per-chunk `malloc`/`free`
(`main.c:327,344`). *Gate:* `mixed` (`test_server.sh:216-243`), `make leaks` —
detached thread plus refcount is where memory goes missing.

**2. `next_job` inside `synthesize_slots`** — the real work. Add
`mynah_graph_sink { next_job, on_done, cancelled, running }` beside
`mynah_graph_job` (`graph.h:21-31`); move `slot_prepare` into an admission block
at the top of the `for(;;)`; finalize and release **per slot** as `active` drops.
Three things break:

- **Scratch sizing.** `k_max` is computed once over the slots present
  (`:4062-4069`) and `batch_scratch_init` sizes `qx = batch * k_max` (`:135`).
  A wider late arrival writes out of bounds. Worse: `batch <= 1u` returns
  without allocating (`:130`), so a service that starts at one slot and grows
  passes `qx_scratch == NULL` into `mynah_qmat_linear_batched`, which has a
  guard (`qmat.c:833`) and **silently falls back to the per-row path, losing all
  batching**. Fix: derive `k_max` from `model->info`, allocate once for
  `MYNAH_MAX_BATCH`.
- **`MYNAH_MAX_BATCH` (`graph.c:12`) vs `MYNAH_GRAPH_MAX_JOBS` (`graph.h:15`)** —
  two unlinked constants sizing the same stack arrays. Unify first.
- `dump_all = count == 1u` (`:4050`) loses meaning; parity dumps depend on it.

Bit-identity holds: `mynah_qmat_linear_batched` (`qmat.c:820`) quantizes each
activation row independently, so B never enters the arithmetic. Admission
changes **when** a row joins, not how it is computed.

**4. Streaming enters the same driver; the mutex disappears.** The lock protects
two things: the model's mutable caches, and the fact that
`mynah_tts_stream_flush` (`mynah_tts.c:413-434`) synthesizes **on the HTTP
thread**. The second vanishes once the stream branch builds a job with
`callback = stream_out_enqueue` and enqueues it like any other — the driver
already supports a callback sink (`graph.h:25-27`, `graph.c:3907-3968`), so
there is no second path (rule 7). Then `g.synth_lock` has no contender: delete
lines **110, 112, 451, 460, 745, 824** and the header comment. Do **not** replace
it with finer-grained locks — that would mean a second owner. Make the invariant
checkable instead: assert the synthesis entry point runs on the scheduler
thread.

**5. Cancel on disconnect.** `poll()` with `POLLRDHUP` behind `#ifdef`. Query
`cancelled()` in `slot_advance` and at the top of the loop. Add a wall-clock
deadline: `max_steps` bounds steps, not time.

**6. Warm-up ramp — only after `decode_audio` carries state.** For Magpie each
chunk re-decodes `STREAM_CONTEXT_FRAMES 32` frames of context (`:3929-3931`), so
a one-frame chunk costs 33 frames of codec; for PocketTTS it costs 1, because
E2-3 chose carried state. Note `STREAM_EMIT_FRAMES 16` is 744 ms at Magpie's
21.5 fps but **1.28 s at PocketTTS's 12.5 Hz** — above the p95 prebuffer this
epic wants to publish. The constant does not survive the second engine.

**7. Prefork** — blocked on risk 2.

## Multi-language changes the scheduler

The six PocketTTS models share nothing (`pocket-tts-model-facts.md` §10), and
batching is weight-stationary. So:

- `g.model` becomes a registry `{lang → model, tokenizer, voices}`;
  `resolve_voice` (`main.c:222-235`) becomes per-language, since a voice KV is
  only valid for its own model.
- The scheduler keeps **one slot group per resident language** and iterates over
  groups, not slots: one turn = one pass over one language's weights.
- The job queue is partitioned per language; rejection on full is per language,
  because capacity is per language.
- **Not round-robin per frame.** Every language switch re-reads the full weights
  from DRAM. At int8, ~110 MB at ~60 GB/s is ≈2 ms, and a batch-1 frame is also
  ≈2 ms, so a quantum of 16-32 frames is the right order — **to be measured**.
- The latency cost is real: at K=32 a stream waits up to ~2.6 s while another
  language runs. The alternatives are a shorter quantum (pays bandwidth) or
  **one process per language**, which given §10 is probably the right default,
  with in-process multi-language as the low-concurrency mode.
- Residency: 6 × 110 MB int8, 6 × 219 MB BF16. Needs a cap and LRU eviction.
- `language` (`main.c:381-382`) stops being a tokenizer detail and becomes a
  **routing key**, validated before admission — unknown language is a 400, not a
  500 halfway through a stream.

## Long-form / incremental input

`stream_push` (`mynah_tts.c:392-411`) only accumulates ids; `flush` (`:413-434`)
calls the graph once and sets `flushed = 1`. Dropping that flag is not enough:
all decode state lives in `synth_slot`, which is **local to
`synthesize_slots`'s stack** (`:4207`) and destroyed by `slot_release`. A second
flush re-encodes the text, rebuilds the KV, re-prefills and re-seeds the RNG.

What is actually needed:

1. slots outliving the call — step 2 already gives this
2. a per-slot input channel drained before `embed_audio_frame` (`:4090`). **Hard
   for Magpie**: `encode_text` output is the cross-attention memory and
   `decoder_cache_init` fixes its length, so extending mid-decode means
   recomputing cross-attention KV. **Easy for PocketTTS**: text is a LUT prefill
   into the backbone KV, so appending tokens is an incremental prefill.
3. the 50-token chunking **reproduced, not fixed** — it is a model property. It
   belongs in `engine_pocket.c`, and the seam behaviour must be compared against
   the oracle. **E2-5 has not been run yet.**
4. codec state across chunks: option A, already decided (`pocket-tts-oracle.md`
   E2-3). Both pieces mandatory — conv ring buffers *and* the position counter
   advancing by `encoder_stride = 16` per latent frame, the one that is
   forgotten and fails silently.
5. length is no longer known up front: `max_steps`/`max_raw_length` (`:3781-3785`)
   and the one-shot `codes` allocation (`:3787`) assume it is. PocketTTS needs no
   historical code buffer at all — one more reason not to generalize Magpie's
   into the seam.
6. `stream_close` (`:436-440`) frees directly, safe only because `flush` is
   synchronous; with a decode in flight it must signal cancel and wait.
   `flush` idempotence (`test_stream.c:106-113`) and callback abort (`:117-129`)
   are existing contracts.

## Do not port

`qwen_tts_serve_ex` + `g_serialize_synth` (it *is* the mutex we are removing);
`handle_tts`/`handle_tts_stream`, which write `ctx->stream`/`ctx->audio_cb` —
per-request state on the global context, a rule-3 violation; the `jq_single`
lane (exists for `instruct`/`voice_design`, would be a second synthesis path);
`qwen_compose_*`; the `[LIFE]/[PATH]/[TTFA2]` telemetry and ~25 timestamps per
job; `qwen_admit_util_*`; `elastic_plan/apply`; admission slicing (Magpie's
prefill is the baked context, PocketTTS's is ≤50 tokens over 6 layers — measure
first); the CUDA/Metal blocks; `stream_output_enabled()` as an **env opt-in** —
port the async writer as the only behaviour, since an unexercised alternative
path is a broken path.

## Risks found by reading the code

1. **`src/threads.c` degrades to inline, silently.** `mynah_parallel_for`
   `trylock`s (`:141`) and runs the region **serially** if the pool is busy
   (`:142`). Invisible today with one synthesis thread. The moment two callers
   overlap, one runs single-threaded with no error — it will look like "the
   decoder got slow". Corollary: **do not parallelise slots above
   `parallel_for`.** Also `blas_set_threads` (`:140,143,157`) is process-global
   and is set and restored around the region; two nearby callers clobber each
   other's thread count.
2. **No `after_fork`.** `g_pool_once` is a static `PTHREAD_ONCE_INIT` (`:73`), so
   a forked child believes the pool is initialised and inherits `g_workers > 0`
   with **no live workers** — `cond_broadcast` to nobody (`:150`) then
   `while (g_pending > 0) cond_wait` (`:154`), an immediate deadlock. Prefork
   needs the equivalent of `qwen_tts_thread.c:472-484` first. `qmat.c:586` and
   `graph.c:1733/1741` can also be inherited locked: fork only from the main
   thread, before any synthesis.
3. **Scratch sized on the initial batch** — see step 2. The `batch <= 1u` early
   return is the silent one.
4. **BNNS cache keyed by `pthread_t`, never pruned.** `graph.c:1687` (owner
   field), `:1863` (lookup), `:1922-1927` (insert), freed only at `:1698`. Every
   thread that decodes creates its own copy of every filter and a dead thread
   leaks it until model close. Design constraint: **threads entering the codec
   must be persistent and fixed in number.**
5. The blocking write under the lock — see "A defect that exists today".
6. **`synth_ticket` on the worker's stack** (`:481-482`) written by the scheduler
   (`:113,150`). Correct today; the moment a deadline or client cancel exists,
   the worker can leave `batch_submit` while the scheduler still holds the
   pointer. Heap + refcount, like qwen's `batch_job_t`.
7. **Routing by `strncmp` on the request-line prefix** (`:652-663`):
   `POST /v1/audio/speechXYZ` matches. Once the fd changes owner, an
   accidentally-matched route is an fd nobody closes. Port `http_precheck`
   (qwen `:1230-1276`).
8. One broadcast condvar for all tickets (`:151`, wait `:171`) — fine at 8
   streams, wasteful at 64. One `done` per job.
9. **No per-request wall-clock limit.** `max_steps` bounds steps, not time; with
   continuous admission a pathological slot holds its batch share forever.
10. **`peer_hung_up` is OS-asymmetric.** `POLLRDHUP` is Linux-only and `POLLHUP`
    on a half-closed TCP is not portably raised. On macOS detection arrives at
    the first failed write. Say so; do not promise more.
11. **Two uncoordinated minimum-length thresholds** (`:3971` vs `:3873,3786`). If
    a sink cancels before 4 stacks, `slot_finalize` takes
    `generated_raw == 0 → slot_fail` (`:4014-4016`) and **a cancellation is
    reported as a synthesis error**. Needs a distinct "cancelled" outcome.
12. **The warm-up ramp moves chunk boundaries**, and for Magpie changes which
    frames get re-decoded with context (`:3929-3931`). That re-decode's
    bit-identity at ≥32 frames is **a claim in the comment (`:3917-3928`), not a
    measurement**, and a one-frame chunk exercises it for the first time.
    Introduce the ramp only after the goldens are re-established.

## Acceptance gate

- **N concurrent streams, N > 1**, each byte-identical to the same request run
  alone. The headline gate.
- `make server-test` fully green; `make goldens` unchanged.
- A soak with zero stalls in `playback_sim` and a published p95 prebuffer.
- Client disconnect frees the slot within one frame, reported as cancelled, not
  failed.
- `grep -n pthread_mutex_lock server/*.c` shows only the connection queue, the
  job queue and the output writer.
- `make leaks` and `make ubsan` clean.

Out of scope: TLS, auth, HTTP/2, WebSocket, Opus, GPU.

## Pre-existing test flakiness, to not mistake for a regression

`tests/test_server.sh`'s `batching` check compares wall-clock times taken with
`date +%s` — one-second granularity — on a synthetic workload that runs in about
200 ms. Over 20 runs per binary it failed 1/20 before the async writer and 3/20
after, with the identical message ("four concurrent requests (1 s) were slower
than four serial (0 s)"). That check makes no streaming request at all, so the
writer cannot affect it; the check needs sub-second timing, not the writer needs
fixing. Filed here so the next person does not bisect it.
