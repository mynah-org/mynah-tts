# E5-9 — multi-language serving: the decision, then the mechanism

Status: **DECIDED and BUILT** · CPU only · 2026-09-12 · supersedes the
"Multi-language changes the scheduler" section of
[`streaming-server-v2.md`](streaming-server-v2.md), which framed the choice and
guessed at the answer.

## The fact this is all downstream of

The six PocketTTS language models are independently trained and share nothing:
relative L2 ≈ √2 on every probed tensor, in the backbone, in the flow head, in
the conditioner embedding **and in the Mimi codec**; 2 of 214 tensors are
bit-identical and both are deterministic sinusoidal constants
([`pocket-tts-model-facts.md`](pocket-tts-model-facts.md) §10). So:

- one pack per language, ≈300 MB each on disk;
- a voice KV, a latent and a decoder state are meaningless outside the model
  that produced them;
- **continuous batching cannot mix languages**, because a batch is weight
  stationary: the whole point of the batch is that one read of the weights
  serves every slot in it.

**Magpie is not in this position and must not be dragged into it.** One Magpie
pack carries `languages: [...]` — twelve of them — served by one set of weights,
with `language` selecting a tokenizer and nothing else. A Magpie batch may mix
languages and always could. The constraint is a PocketTTS constraint, and the
design has to say so rather than generalise it.

## The hole as it stood at 65a352b

`mynah_graph_serve_continuous()` binds one `mynah_tts_model` for the scheduler's
entire life, and in `server/main.c` `language` reaches only
`mynah_tokenizer_encode()`. On the PocketTTS path (`g.sp != NULL`) it is read
from the body and then **never used at all**. A two-language deployment had no
defined behaviour; a single-language PocketTTS deployment had a worse one —
`{"language":"de"}` against `models/pocket-it` was silently synthesized with
Italian weights and answered 200.

## The two candidates, weighed

### A. Model registry in the driver
`g.model` becomes `{lang → model, tokenizer, voices}`; the scheduler keeps one
slot group per resident language and a turn is one pass over one language's
weights.

### B. One worker process per language
`server/prefork.c` already forks W workers, already routes descriptors over
`SCM_RIGHTS`, and already owns a per-worker slot count. Give each worker one
pack and route on language.

| | A: registry in the driver | B: one process per language |
|---|---|---|
| **Resident memory, N languages** | N packs in one address space | N packs in the parent, one per worker; `mmap`ed weights are page-cache shared across processes, so the bytes are counted once either way. **A wash** — and it is worth saying so, because "B duplicates the weights" is the intuition and it is wrong. What is per-process is the *derived* state (quantized weights, codec filter cache), and that is per-worker under A too the moment W > 1. |
| **Language switch inside a turn** | Every switch re-streams the whole weight set from DRAM: ≈110 MB int8 at ≈60 GB/s ≈ 2 ms, against a batch-1 frame of ≈2 ms. A quantum of 16–32 frames is the only way to amortise it, and at K=32 and 12.5 Hz a waiting stream pays up to **≈2.6 s** before its language runs. | **Zero.** A worker holds one pack for its life. Weight-stationary is not a policy, it is the topology. |
| **Starvation** | One scheduler, several groups, so a busy language can hold the turn. The fix is a fairness or credit gate — and every fairness/credit gate the reference implementation built was measured and lost (parked 95.8% of checks, STREAM p95 0.838 → 0.986; [`serving-design.md`](serving-design.md) §9). Building one here means building the thing the doctrine forbids. | Impossible by construction: capacity is statically partitioned, so language A's traffic cannot consume language B's cores. **The honest cost is the mirror image** — idle German workers stay idle while English queues. Static partitioning buys isolation and sells elasticity, and that trade should be made explicitly rather than discovered. |
| **Language nobody holds** | Tempting to load on demand — 300 MB and a multi-second cliff in the middle of a serving loop, plus an LRU and a residency cap nobody has measured. | Nothing to decide: no worker holds it, so it is refused at the router in microseconds with a reason and a counter. "When the machine is full, refuse" generalises cleanly to "when the language is absent, refuse". |
| **batch == solo (E5-8)** | The driver, the scratch sizing (`k_max` is per model) and the admission point all become language-aware. That is precisely the code the byte-identity gate is about, and it would have to be re-proved. | **Nothing in the driver changes.** A worker process has exactly one model for its life, which is what the driver already assumes. Byte-identity is preserved because the synthesis path is not touched. |

### Recommendation: B, one worker process per language

Not because it is less code — it is not — but because of the last two rows. B
turns "a batch must not mix languages" from an invariant somebody has to
maintain into a fact about the address space: **a process holds one model, a
batch is drawn from one process, therefore a batch is one language.** There is
no check to forget, because there is nothing to check. And it leaves the slot
driver, where the E5-8 gate lives, untouched.

A is also the more expensive of the two exactly where this epic is weakest. E5
exists because TTFA collapses under concurrency; A's language quantum adds a
second, larger source of head-of-line delay on the same metric.

### What would change my mind

- **A measured quantum cost far below the estimate.** The ≈2 ms switch is
  arithmetic from bandwidth, not a measurement. If a real sweep showed a switch
  amortising at K=2–4 frames (≈160–320 ms), A's latency objection would mostly
  evaporate and its elasticity would start to matter.
- **Many languages, little traffic each.** At 6 languages and one request per
  minute, static partitioning wastes five sixths of the machine and A's ability
  to lend idle capacity is worth real money. B's answer today is "run fewer
  workers per language"; that answer gets thin below one.
- **A deployment that cannot fork** (a hard single-process constraint, an
  embedder of the library rather than the server). B has nothing to offer there
  and A would have to be built.
- **Measured residency pressure in the parent.** B's parent maps every pack. If
  that were ever shown to cost real RSS rather than address space, the packs
  would have to be opened in the children instead — a change to B, not a reason
  for A.

None of these is true today, and the first one is a measurement anybody can run
later without undoing this work.

## The mechanism, as built

### A pack declares its language, or declares that it has none

`model.json`'s scalar `"language"` (`"english"`, `"italian"`, … — PocketTTS
writes it; Magpie does not) is parsed in `src/mynah_tts.c` into
`mynah_tts_model_info.language`. **Empty means the weights are not bound to a
language**, which is the true statement about Magpie, and every decision below
keys off that emptiness rather than off the engine name. Rule 6: no
`strcmp(engine, "pocket")` in the routing path.

Matching a requested name to a declared one is case-insensitive and exact, plus
an unambiguous prefix of ≥2 characters, so a client may send `it` or `italian`
to `models/pocket-it`. Ambiguity is not resolved, it is refused. The resident
names are printed by `/health` and by the banner, so the client never has to
guess which spelling the pack chose.

### One `-m` per language

`-m` repeats. The first pack is the default, used by a request that names no
language. Two packs claiming the same language, or an extra pack whose weights
are language-agnostic, is a startup error — in both cases a request could not be
routed and the server would have to pick one silently.

More than one pack implies prefork: if `--prefork` was not given, W defaults to
the pack count, and the banner says that it did. `--prefork W` with W below the
pack count is an error rather than a silent drop of a language.

### The parent classifies; it still does not parse HTTP

The router has to know a connection's language before it can pick a worker, and
that is a real weakening of prefork.h's "the parent never parses HTTP". What it
does instead is bounded and non-consuming: `MSG_PEEK` of at most 8 KiB, a scan
for one JSON key, no interpretation of method, headers, framing or body
semantics, and not one byte consumed — the worker still reads the whole request
itself. A connection whose prefix has not arrived yet is parked in a pending set
with its own deadline, the same shape as the lingering-close set, so the
single-threaded router never blocks on a slow client.

**The classifier does not run at all when the fleet holds one language.** That
is not an optimisation, it is the guarantee that the existing path is
untouched: with one pack the router is byte-for-byte the code that shipped at
65a352b.

A prefix with no `"language"` key — `GET /health`, or a speech request that
omits it — is routed to the default group. A name no group holds is refused at
the router.

### Rungs, per language

Capacity is per language, so rungs 1–3 are evaluated per language group: the
least-loaded worker **of that group**, a queue bound of `queue_per_worker × live
workers of that group`, and the deadline checked at that group's head. The queue
is one array with a group tag rather than N arrays; scanning it in arrival order
per group gives each group its own FIFO. Without this a head-of-line entry for a
busy language would block a ready entry for an idle one — a new starvation
channel introduced by the very design that was chosen to prevent starvation.

### `language_not_served` is a rung, in the ladder's own vocabulary

`MYNAH_PREFORK_REFUSE_LANGUAGE_NOT_SERVED`, appended to `mynah_prefork_refusal`
with its own counter and its own `error.code`, exactly as the header demands
("never collapse two of these into one"). It is the first rung whose status is
**400, not 503**, and the first whose `error.type` is
`invalid_request_error`: the machine is not full and retrying unchanged will
fail identically. That forced a `status`/`type` column into the `REFUSAL` table
rather than the hardcoded `server_error` that was there.

It is raised in two places and they are not redundant:

- the **router**, when no group holds the language — it has the fleet's map;
- the **worker**, when the body names a language its own pack is not bound to —
  it is the only process that has parsed the body, and it is the rung that makes
  the *single-process* server correct. That is the case that was silently
  mis-served before this change.

### What a request with no language means

It means "this server's language", not `"en"`. The old default of `"en"` in
`handle_speech` was harmless while `language` only chose a Magpie tokenizer and
would have been a disaster here: every unspecified request to `models/pocket-it`
would have been refused for asking for English. Magpie still falls back to
`"en"` exactly as before, because its weights are not bound and its tokenizer
needs some language.

## Acceptance

- two-language server (`models/pocket-en` + `models/pocket-it`) serving both,
  each byte-identical to the same request run alone on a single-pack server;
- a request for an absent language refused **400 `language_not_served`** with
  the counter incremented, in both topologies;
- `/health` lists the resident languages and the fleet's capacity split; the
  banner prints the same table;
- `make goldens`, `make driver-test`, `make stream-test`, `tests/test_server.sh`
  in both modes unchanged.

## What is deliberately not here

- **No on-demand loading and no LRU eviction.** Residency is decided at startup
  and printed. A language is held or it is refused; there is no third state that
  costs 300 MB and a multi-second stall in the middle of a serving loop.
- **No cross-language borrowing of idle capacity.** That is utilization-aware
  admission wearing a different hat, and it was measured and lost
  ([`serving-design.md`](serving-design.md) §6).
- **No in-process registry as a "low-concurrency mode".** Two serving shapes for
  one job is the second implementation AGENTS.md rule 7 forbids, and the one
  that gets less traffic is the one that rots.
