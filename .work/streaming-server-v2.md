# E5 — Streaming server v2, ported from qwen-tts

Status: **OPEN** · CPU only · partly independent of E1-E3

Supersedes the July server roadmap (`PLAN.md` §24, P0-P3), which is closed:
robustness, tests, usability and continuous batching all landed. This epic is
about the next limit, which is **concurrent streaming**.

## The problem, precisely

`server/main.c:451-460` wraps the entire `stream_open` / `stream_push` /
`stream_flush` sequence in `pthread_mutex_lock(&g.synth_lock)`. **One stream at a
time per process.** Offline requests go through the batch scheduler
(`server/main.c:110-172`, 2 ms window) and are fine; streaming does not.

This was a reasonable choice for a 357M model. For a 109.5M model it is the
binding constraint: at int8 one PocketTTS stream needs ≈1.5 GB/s of weight
traffic against ~59 GB/s on an M1, so the hardware can carry many streams and the
mutex is what stops it.

Second gap: **long-form does not exist.** `mynah_tts_stream_push`
(`src/mynah_tts.c:392-411`) only accumulates token IDs; `flush` is one-shot
(`mynah_tts.c:419-422`). There is no incremental input into a running decode and
no cross-flush state. PocketTTS needs exactly that — upstream chunks at
`MAX_TOKEN_PER_CHUNK = 50` and claims unbounded input length.

## What we already have and keep

- continuous batching, up to 16 slots, **bit-exact per request** (audio does not
  depend on who you batched with), measured 1.63× at 8 in flight
- one state machine for offline and streaming, differing only by sink
  (`synthesize_slots`, `graph.c:4046`) — `CLAUDE.md` rule 7 holds, do not fork it
- bounded suffix decoding with constant per-chunk cost (`graph.c:3907-3968`)
- OpenAI-shaped routes, worker pool, bounded queue with `503 + Retry-After`

## What to port from qwen-tts

| Item | Where it lives upstream | Why |
|---|---|---|
| **Three-role topology** | `qwen_tts_server.c` | readers parse HTTP read-only, one scheduler owns `ctx` and is the only synthesizer, workers handle special requests. Removes the need for a global lock instead of holding it less. |
| **Prefork pinned (Linux)** | `qwen_tts_serve_prefork` | W workers forked *after* weight load (shared CoW), pinned to contiguous core slices, fds passed via `SCM_RIGHTS`. Needs `after_fork` on the pool — see E4. |
| **Async output writer** | `stream_output_*`, `qwen_tts_server.c:286-540` | dedicated writer thread, **bounded** chunk queue, send timeout, refcount, `failed_enqueues` / `peak_bytes` counters. Ours writes blocking from the engine thread. |
| **Cancel on disconnect** | `peer_hung_up`, `POLLRDHUP` | a client that hangs up must stop costing CPU |
| **Warm-up chunk ramp** | `1,2,2,4,4 → CHUNK` | first chunk out early, then amortize |
| **Lane split** | `qwen_lane_*` (E4) | separate the AR step team from the decoder team so the conv stack does not starve the step loop |
| **Admission header at admit time** | — | publish the stream header when admitted, not at first chunk |

Design to port as a **pattern**, not code: qwen-tts's decoder keeps its streaming
state **outside** the model object (`qwen_sd_stream_state_t` +
`_decode_streaming_st()` + `_decode_streaming_batch()`). That is exactly the
contract N concurrent causal streams need, and it is what `stream.c` should look
like after E1.

## The cadence rule — read this before choosing a chunk size

qwen-tts's `.work/professional-streaming-architecture.md` shows that
**`STREAM_RTF < 1` does not prove a player will not stall.** What matters is
prebuffer and gap distribution, not mean throughput. Their measured trade-off:
chunk 8 → p95 prebuffer 0.45 s; chunk 32 → 1.22-2.55 s, with STREAM_RTF barely
moving. Port their `playback_sim` idea (stall rate, max gap, safe start) before
claiming a streaming target.

## Multi-language changes the scheduler

Measured in [pocket-tts-model-facts.md](pocket-tts-model-facts.md) §10: the six
PocketTTS language models share **nothing** — not the codec, not the latent
space. Continuous batching is weight-stationary, so **slots batched together must
share a language**. Two consequences for the scheduler:

- slot groups are per language, not one global pool of 16
- a process serving N languages holds N weight sets resident (≈110 MB each at
  int8), so "how many languages per process" becomes a deployment knob worth
  measuring rather than assuming

This does not change the single-language design; it changes admission. Decide it
before E5-1, because it affects who owns the slots.

## Work items

1. Remove the global stream mutex by making the scheduler the sole owner of
   `ctx` — do not replace it with a finer-grained lock.
2. Streaming requests enter the **same** slot driver as offline ones. No second
   path (`CLAUDE.md` rule 7).
3. Async output writer with a bounded queue and send timeout.
4. Cancel on disconnect.
5. Long-form: incremental `push` into a running decode, persistent conv and
   upsampler state across flushes, text chunking at the engine's chunk limit.
6. Prefork pinned on Linux, after E4's `after_fork`.
7. `playback_sim` + a soak test; publish p50/p95 TTFA, prebuffer, stall rate.

## Acceptance gate

- **N concurrent streams, N > 1**, with per-request output byte-identical to the
  same request run alone. This is the headline gate.
- `tests/test_server.sh` still fully green, including batch↔stream parity.
- A soak run at the advertised concurrency with zero stalls in `playback_sim` and
  published p95 prebuffer.
- Client disconnect mid-stream frees the slot within one frame.
- No `pthread_mutex_lock` around whole-request synthesis anywhere in `server/`.
- `make leaks` and `make ubsan` clean.

## Explicitly out of scope

TLS, auth, HTTP/2, WebSocket, Opus. Chunked raw PCM over HTTP is the transport,
as it is upstream in both projects. GPU residency is deferred with E4.
