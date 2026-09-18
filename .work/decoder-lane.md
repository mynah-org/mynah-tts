# The decoder lane, the warm-up, and the two measurement-shaped knobs

Status: **LANDED, OFF BY DEFAULT** · E5-20, E5-21, E5-22, E4-16a ·
E5-5 **BLOCKED**, see §6.
Design reference: [`serving-design.md`](serving-design.md) §5, §7 ·
traps: [`linux-production.md`](linux-production.md) §2, §4, §9.

Everything here was executed on **GCP Neoverse-V2, 32 cores, SMT off, gcc
15.2**, and on an M1 for the paths macOS can reach. **No performance number in
this note is ours.** Every figure quoted is the reference implementation's,
carried over so that the defaults and the refusals have a reason attached; the
capacity campaign that would produce our own numbers is a separate job and was
deliberately not run.

---

## 1. E5-20 — the warm-up goes through the request path's own reset

The bug being avoided is the reference's, and it is the kind that never
reproduces under load: its warm-up ran on whatever state the CLI had left in
the context, primed per-request state for a configuration no request uses, and
**the first real request came out different from every one after it**
(serving-design §7). It shows up only on a fresh process, so a load test never
sees it, and every A/B measured afterwards silently contains it.

What landed: `--warmup N` (default 1, `0` disables) builds an ordinary
`synth_job`, puts it on the ordinary queue, and lets the scheduler admit it
through `sink_next_job`. It therefore walks the same reset a request walks --
`slot_start` → `ctx_new` + `prepare`, steps, `slot_retire` → `ctx_free` -- and
leaves the process in exactly the state a finished request leaves it in. There
is nothing for it to fork because there is no second path for it to take.

The `is_warmup` flag gates exactly two things, both about the *connection*
rather than about synthesis: there is no prefork slot to hand back, and the
completion is not counted in `jobs.completed` (a load test differences that
against the requests it issued). If it ever starts gating something inside the
driver, the warm-up has stopped being the request path and the item is void.

Placement is the item: after the scheduler thread exists, after the fork (the
caches are per process; see `prefork.h` on the codec's `pthread_t`-keyed filter
cache), before the accept loop. The listening socket is already bound, so a
client that connects during the warm-up waits in the backlog — `/health`
therefore answers exactly when the server is warm.

**Gate** (`tests/test_server.sh`, check `warmup`), three-cornered because two
of the corners would each pass for the wrong reason alone:

| | |
|---|---|
| `warm1 == cold1` | the warm-up changed no audio at all |
| `cold1 == cold2` | request one equals request two *without* a warm-up, so the property holds on its own merits and the warm-up is an optimisation, not a patch over a divergence |
| `cold1 != other-input` | anti-vacuity: the server is not returning the same bytes for everything |

Passes single-process and `--prefork 4`, on macOS and Linux.

---

## 2. E5-21 — the decoder lane

A private **pinned** team on the last N cpus of a worker's slice, its own submit
lock, a bounded one-unit-per-slot mailbox. Off by default.

### The mechanism, which is the part worth copying exactly

The redirection is **inside the dispatch primitive**, keyed on a thread-local
tag, not by discipline at call sites — one branch at the top of
`mynah_parallel_for()`:

```c
pf_pool *const pool = g_on_lane ? &g_lane : &g_engine;
```

Everything a lane thread dispatches — conv panels, SGEMM slices, im2col, and
every call site nobody has written yet — lands on the lane team because of that
line. A convention at the call sites would have to hold for code that does not
exist yet; this holds by construction.

There are two `pf_pool` instances sharing no state at all: separate mutex,
generation, job slots and workers. That is the answer to the reference's first
failed attempt.

### The two failures the design is avoiding, and the refusals they bought

| attempt | measured there | what it forces here |
|---|---|---|
| second submitter on the **engine** pool | TTFA p95 167 → **1200 ms** | the lane has its own pool and its own lock |
| private team, **unpinned** | 21 threads on 8 cores | `mynah_lane_split_prepare()` REFUSES when it cannot pin |

And the split sweep, which is why this is a gate and not a knob: at B4,
4+4 / 5+3 / 6+2 measured STREAM p95 **0.997 / 1.113 / 1.364** against 1.203
inline, mailbox wait 3.6 → 8.0 → **16.6 ms/frame**. 2 step + 6 decoder was
killed earlier still (+17 ms of per-slot region work at two threads). **On a
narrow lane the decoder is slower than inline.** So both sides need
`MYNAH_LANE_MIN_CPUS` (4) and a request that cannot have them is refused with
the reason printed, never rounded into a split that "works".

Refusals verified on the Linux box:

```
--prefork 4 --decoder-lane 6   → "8 cpus (pool width 8) split 2+6, and both sides need at least 4 …"
--decoder-lane 4 (no prefork)  → "ignored: the lane is a split of a worker's PINNED cpu slice …"
macOS, any split               → "cannot pin … an UNPINNED private team is the reference's second failed attempt"
```

`/health` reports `decoder_lane` as the width **actually running**, so a lane
that was asked for and refused reads 0 there and its reason is in the log.
"I asked for a lane" and "I have a lane" are different claims.

### The ordering that is not optional

```
sched_setaffinity(child) → mynah_threadpool_after_fork() → mynah_lane_split_prepare() → (first dispatch builds the engine pool)
```

because pthreads inherit the creating thread's mask. Inside `prepare` the same
rule applies again: pin self to the **lane** cpus, create the lane threads so
they inherit it, then move self to the engine cpus so the lazily-built engine
pool inherits *that*. Lane threads re-assert their own mask as well, so the
invariant does not depend on the creator's ordering surviving a future edit.

Verified on Linux, `--prefork 4 --decoder-lane 4`:

```
worker 0  lane pinned to 4..7     main thread affinity 0-3
          mynah-lane 4-7 · mynah-lanew 4-7 · mynah-pool 0-3
```

### The bounded contract

At most one unit in flight per slot, at most one quantum accumulated behind it.
The driver reaps non-blocking at the top of every iteration and blocks in
exactly two places, both about the slot it is blocking on:

1. that slot has a full quantum waiting behind a unit still in flight;
2. that slot is about to be retired and its context freed — **never free state
   the lane is decoding**.

No slot ever waits for another slot's decode. A violation prints
`MAILBOX OVERRUN … the bounded contract was violated` and increments a counter
that must stay 0. **Measured 0** over 24 concurrent streaming requests × 3
rounds on `models/pocket-en` with the lane engaged.

### The trade nobody should discover by accident

`a477b19` landed a batched vocoder for PocketTTS while this was in flight, and
**the lane and the decode gang are alternatives, not layers.** Both attack the
same cost from opposite directions: the gang makes one wide call where there
would have been N narrow ones; the lane keeps the N calls but takes them off
the loop thread. A lane unit is per slot by construction, so running the lane
means calling `decode_audio` per context and not `decode_audio_batch`.

That is a scheduling choice and never a numerical one — the seam requires
`decode_audio_batch` to be bit-identical per context to `decode_audio` on the
same range (`tts_engine.h`) — so whichever runs, each request gets the same
bytes. **Which is faster is unmeasured.** The default is therefore the gang,
which is what ships, and turning the lane on prints what it is giving up.

### Evidence

Byte-identity held everywhere with the lane engaged on Linux at 4+4:

- `tests/test_server.sh --prefork 4 --decoder-lane 4` — every check, including
  `stream==batch`, `repro x3`, `batching`, `admission`, `mixed`;
- `make server-concurrency-test MODEL_DIR=models/pocket-en` with the lane on —
  113 byte comparisons of which 28 must-differ controls, C=2/4/8, forward,
  reverse and ragged arrival, across four worker processes;
- the same suite under **ASan** and under **UBSan** (the stock `make asan` /
  `make ubsan` targets do not build the server, so the sanitized server was
  built explicitly — this is where a use-after-free of the in-flight unit or of
  a retired context would surface, and nothing did).

---

## 3. E5-22 — the spin budget

`MYNAH_POOL_SPIN`, default **65536 on Linux/aarch64 and 4096 elsewhere**,
reported in the dispatch table as `pool.spin` with its source and with the
park/spin-win counters beside it.

The reference's sweep, which is where the default comes from:

| spin | STREAM p95 | ctx switches/s |
|---|---|---|
| 256 | 0.999 | 197,000 |
| 4096 | 0.893 | 38,000 |
| **65536** | **0.808** | **7,600** |

**Not tuned here, and the number does not port**: on x86 8c the aarch64 value
is worse and on Graviton5 the curve is non-monotonic (linux-production §9). The
row says `TRANSFERRED not measured here` for exactly that reason.

Why it is the right shape for us: our own thread-pool work measured ~20 µs of
wake-up per region, so a region must be worth ≥200 µs to pay for itself. The
spin sweep is the same wall from the other side, and it says the fix is not
fewer regions but not parking between them.

**Correctness is independent of the value.** The publisher takes the mutex and
broadcasts unconditionally, and a worker re-checks the generation under that
same mutex before waiting, so the spin is an optimisation layered outside a
lock protocol that was already correct. The store-buffer litmus that deadlocks
a "publish, then peek at `sleeping`" pool on x86 and silently does not on ARM
(linux-production §8) cannot arise. **We develop on ARM**; that bug would not
have shown up here, which is why it is designed out rather than tested for.

Observed on the Linux box at idle-ish load: 1831 parks against 702 spin wins.
That is a datum, not a result — it is whatever the run happened to do.

---

## 4. E4-16a — `OPENBLAS_THREAD_TIMEOUT=1`

**Interim compensation for a dependency E4-16 is removing, not design**, and
the code says so at both sites. When the BLAS call goes, the block goes with it.

Without it an idle OpenBLAS team spins. The reference measured TTFA C=1 at
**108 ms bimodal** against 66 ms stable, and 42,500 against 12,000 context
switches per second. Bimodal is the word that matters: a single run looks
definitive whichever mode it draws.

Set from a constructor in `src/threads.c` (every binary links it; there is no
one `main()` to put it in), only when unset — an explicit value always wins.

**Claim versus fact, and the fact is weaker than the claim.** A shared
`libopenblas` is initialised before the executable's own constructors, so if
that build reads the variable in its constructor our write is too late. The
only channel guaranteed to be read is the process environment before `exec`.
The `blas.thread_timeout` row reports which of the two happened rather than
reporting success.

---

## 5. One thing found while verifying, not fixed

**OpenBLAS builds a team sized to the HOST, per worker process, whatever the
worker's slice is.** On the 32-core box, measured by thread-name inheritance
(a new thread inherits the creator's `comm`, so the team shows up wearing the
scheduler's name):

| | threads named `mynah-sched` in one worker |
|---|---|
| default | **32** |
| `OPENBLAS_NUM_THREADS=1` | 1 |

A worker pinned to 8 cpus therefore carries a 32-thread BLAS team, and the
parent reaches `fork()` with 32 threads alive — which is why the prefork
precondition warning fires on every Linux start. This is linux-production trap
1, now measured here rather than quoted.

It is **not** tuned in this change, deliberately: their own follow-up is that
lowering BLAS threads improves RTF and **worsens TTFA by 30%**, because prefill
is all BLAS with no concurrent decoder, and that a rigid `engine + BLAS = cores`
partition is wrong. `mynah_blas_set_threads()` exists and has no caller; giving
it one is a measurement, not a patch.

While confirming this, the prefork fork-precondition warning was found to be
**stale**: it still named `g_blas_mu` and `g_stats_mutex` as having no atfork
handler, which `c9e31e4` had already fixed. A log line describing a bug
somebody already fixed is worse than no line — the reader budgets for a hang
that cannot happen and stops reading the rest of the warning, which is still
true. Corrected in place, with the audit table updated rather than deleted.

---

## 6. E5-5 — blocked, and where

**Long-form as specified cannot be built from the driver.** The gate is "a text
synthesized in N pushes is byte-identical to the same text synthesized in one",
which requires appending text tokens to a context that is already generating.
Nothing in the seam or in the engine supports that, and both files that would
have to change are owned elsewhere.

What is already done, and changes the shape of the remaining work:
`pocket_decode_audio` **already carries codec state across calls** — conv ring
buffers plus the position counter, cross-checked every frame against the
decoder transformer's offset, with non-contiguous ranges refused outright. The
codec half of E5-5 is finished. The driver half is finished too: `serve()` is
already a continuous service loop whose slots outlive any single call, and
`stream_gang` already decodes in contiguous monotone ranges.

What is missing is the text half, and it is one hook:

```c
/* Append text tokens to a context that has already been prepared, as an
 * incremental prefill of the backbone KV. Bit-identical to having prepared
 * with the concatenated text, for tokens generated after the append. */
int (*append_text)(mynah_engine_ctx *ctx, const int *text_ids, size_t count,
                   char *error, size_t error_capacity);
```

`mynah_transformer_ar_prefill` already appends at the current offset and is
documented bit-identical to the same positions run through `_step`, so the
graph cooperates. The blockers are **allocation and length accounting**, all
inside `pocket_ctx_new`: `text_ids`, `text_embed`, `backbone_capacity`
(`voice_positions + text_length + max_steps + 1`, which becomes the KV's
`max_seq_len`), `latents[max_steps]`, `codec_positions`, and the `max_steps`
budget whose exhaustion forces an EOS. Every one of them is sized from a text
length known at admission.

Magpie is a different and harder problem: `magpie_prepare` fixes the
cross-attention memory length at `decoder_cache_init`, so extending the text
mid-decode means recomputing every layer's cross K/V — as
`streaming-server-v2.md` already predicted.

**Do not** implement this as "chunk the text into N sequential requests". The
audio would not be byte-identical to the one-shot and the gate would have to be
weakened to admit it, which is the wrong direction.
