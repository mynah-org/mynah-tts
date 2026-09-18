# The `transformer_ar` sliding window — contract, and how it is guarded

**Board item:** `PLAN.md` E2-4.
**Owns:** `src/transformer_ar.c`, `src/transformer_ar.h`,
`tests/test_transformer_ar_window.c`, `make window-test`.

## Problem

`mynah_transformer_ar_config.context` is a sliding attention window. The
PocketTTS backbone passes `0` (unlimited); the Mimi decoder transformer passes
`250` (`engine_pocket.c`: `codec.context = cfg->codec_tf_context`,
`tools/convert_pocket.py: DECODER_TRANSFORMER_CONTEXT = 250`). **No reference
dump we hold reaches it**, and the first thing to correct is how far away it was
assumed to be.

The decoder transformer does *not* run at the 12.5 Hz latent rate. It sits after
`mimi.upsample`, whose kernel `[512, 1, 32]` gives
`codec_upsample_stride = 16`, and `engine_pocket.c` sizes it accordingly:
`codec_positions = codec_frames * upsample_stride`. So it runs at
**12.5 x 16 = 200 Hz**, and 250 positions is **1.25 s of audio** — not the ~11.6 s
a 21.5 fps reading suggests, and not a long-form edge case. Every PocketTTS
utterance longer than 1.25 s already runs the windowed path in production.
`tools/oracle_pocket.py` records `MAX_CALLS_PER_MODULE = 4`, i.e. the first 20 ms,
so the code path that serves essentially all real audio has been compared
against nothing.

Two paths landed that have never been run past 250 together: the batched
prefill tile (`_prefill`, 16 rows) and the single `_step`. A window makes the
per-row attention span position-dependent, so the tile is the first place a
window off-by-one shows up — and it shows up silently, as plausible audio.

The Python oracle cannot run in this session (needs `uv`, torch and gated
weights). The guarantee below is therefore **model-free**: synthetic weights, an
independent double-precision reference of the *contract*, and two structural
invariants that hold for any weights at all. That is weaker than oracle parity in
one respect (it cannot tell us the contract matches Kyutai's) and stronger in
another (it does not depend on a checkpoint being present, so it runs in
`make test`, UBSan and ASan on every machine).

## The contract

Stated first, then checked against the code — not read off the code.

**W1 — `context == 0` is unlimited.** No window, `lo = 0` always.

**W2 — the window is the `C` most recent positions, self included.** For a query
at absolute position `p` with `context = C > 0`, the attended key set is the
closed range `[lo, p]` with

```
lo   = (p + 1 <= C) ? 0 : p + 1 - C
span = p - lo + 1 = min(p + 1, C)
```

**W3 — the engage boundary.** `lo` is `0` for every `p <= C - 1` and becomes
non-zero first at `p = C`. `span` reaches its maximum `C` at `p = C - 1` and
stays there. So `p = C - 1` is the last position that sees position 0, and
`p = C` is the first that does not.

**W4 — the KV cache is absolute-indexed and does not wrap.** The block is
`[2][max_seq_len][num_heads][head_dim]` and position `p` lives at slot `p`, for
the life of the state. Consequences, all of them load-bearing:

- memory is `O(max_seq_len)`, **not** `O(context)` — a 250-window does not cap
  the footprint of a long utterance, and `engine_pocket.c` sizes the codec state
  as `codec.max_seq_len = codec_positions`, the whole utterance. At 200 Hz,
  `d_model = 512`, 2 layers, K and V, f32 that is **8.2 MB per 5 s request and
  49 MB per 30 s request**, against **2.0 MB flat** for a ring sized to the
  window. That is a per-slot cost in a continuously batched server and it grows
  with utterance length, so it is a real decision and not a detail — but it is a
  *performance* decision, deliberately out of scope here, and correctness comes
  first (`.work/serving-design.md` is where it would land);
- `max_seq_len` is a hard cap, not a recycling point. `_step` returns `-1` when
  `offset == max_seq_len`; `_prefill` returns `-1` when `offset + n_tokens >
  max_seq_len`. Both refuse **before** mutating anything, so the offset is
  unchanged after a refusal;
- there is no ring-buffer modular index anywhere, so the classic silent defect
  for this shape — an off-by-one at a wrap — cannot occur. The failure mode that
  *can* occur is the opposite one: a caller that sizes `max_seq_len` to
  `context` gets a hard `-1` at position `C`, not wrapped audio.

**W5 — RoPE is absolute and is never re-based.** Both `q` and `k` at position
`p` are rotated by `p`, and `k` is stored *post-rotation*. A key written at
position 3 keeps its position-3 rotation forever, including long after the
window has moved past it. The window changes *which* keys are read, never their
rotation. This is what makes the score `q_p · k_j` a function of `p - j` only.

**W6 — the window length is shared, the window is per row.** `context` is a
config field and `tar_config_same()` requires every state in a `_step_batch` to
agree on it; but `lo` and `span` come from each row's own `position`, so in one
batched step some rows can be in the windowed regime and others not.

**W7 — the prefill tile is a scheduling unit, never a numerical one.** Inside a
tile every row's K/V is written before any row attends (two separate passes over
`b` in `tar_forward_rows`), and a row at `p` reads `j <= p` only, so the later
positions of its own tile are present in the cache but unreachable. With no
`linear` hook installed the tile falls back to the same per-row matvec `_step`
uses, so tile and step must agree **bit for bit**.

## Code, checked against the contract

`tar_window_start()` (`src/transformer_ar.c:544`) is W2 verbatim, including the
`p + 1 <= C` guard that makes W3 come out right (`p = C - 1` → `C <= C` → `lo =
0`; `p = C` → `C + 1 > C` → `lo = 1`). W4: the only cache index in the file is
`(lo + j) * attn_dim`, and the two bounds checks are `state->offset >=
config->max_seq_len` in `_step` / `_step_batch` and `end > config->max_seq_len`
in `_prefill`; there is no `%` on a position anywhere. W5: the RoPE table is
built once over `[0, max_seq_len)` in `_state_new` and indexed
`state->rope_cos + position * half`, with `position` the absolute
`refs[b].position` — no `lo`, no offset-relative index. W6/W7 as described.

**No defect found in the window logic itself.** The value of E2-4 is therefore
the guard, not a fix: the contract above was previously implicit, and the only
windowed case under test was `context = 2` at six positions. Nothing in
`src/transformer_ar.c`'s inference path changed — verified, not asserted:
compiling the canonical `transformer_ar.c` and this one with identical flags and
comparing the disassembly function by function gives **27 of 27 non-self-test
functions instruction-identical**. To repeat it: extract the canonical
`src/transformer_ar.c` from `65a352b` to a scratch file, `cc` both with the same
flags to `.o`, disassemble each (`otool -tvV`, or `objdump -d`), split the
listing per symbol and compare every symbol except
`_mynah_transformer_ar_self_test` — normalising hex operands first, because
branch targets move when the translation unit grows and were the only thing that
differed. That is why a PocketTTS or Magpie generation cannot have moved.

## What was untested before this change

`mynah_transformer_ar_self_test` case 5 exercises `context = 2` over 6
positions; case 10 exercises a ragged `_step_batch` at `context = 2` with
offsets 1/3/5. Case 9 — the only multi-tile test, 35 positions — runs at
**`context = 0`**. So:

- the tile path had never been run with a window engaged, at any length;
- nothing crossed the engage boundary more than once;
- nothing ran at the production `context = 250`;
- the capacity cap (W4) was only tested through `_set_offset`, not through a
  `_step` that runs out of cache.

## The guard

Two pieces, because they answer to two different gates.

### Inside the shipped binary: `mynah_transformer_ar_self_test` cases 9b and 9c

`make self-test` is what runs on a Linux box that has the binary and nothing
else, so the minimum has to travel with it:

- **9b** — a 300-position prefill against the same positions stepped one at a
  time, **bit for bit**, at `context = 70` (which does not divide the 16-row
  tile, so the engage boundary lands mid-tile), plus a hard `-1` for the step
  past `max_seq_len`. Case 9 above it runs at `context = 0`, so before this the
  tile path had never been run windowed at any length.
- **9c** — the window's edge asserted directly rather than by two of our own
  paths agreeing: one layer, one varying perturbation, and the two-sided
  bit-exact claim that an output moves **iff** the perturbed position is inside
  its window. 9b alone cannot see a defect that moves the tile and the step
  equally, and control 7 below proves it: 9b was silent on it and 9c is not.

### `tests/test_transformer_ar_window.c` — `make window-test`, wired into `make test`

Synthetic weights, `d_model = 24`, `heads = 2`, `head_dim = 8` (so
`attn_dim = 16 != d_model`, which catches a dimension conflation the module
self-test cannot — there `d_model == attn_dim == 8`), `ffn_dim = 40`.
21 cases, all passing:

1. **`window/reference`** — an independent double-precision implementation of
   W1-W7 against `_step`, at `(C=7, T=64)` and `(C=13, T=59)` — nine and four
   and a half window lengths — at `(C=0, T=40)` as an unwindowed control, and at
   the production `(C=250, T=520)`. Max relative error 4.7e-07 to 8.3e-07
   against a 2e-04 gate.
2. **`window/tile-vs-step`** — `_prefill` against `_step`, **bit for bit**, at
   `(C=17, T=200)` (deliberately not a multiple of the 16-row tile) and at
   `(C=250, T=300)` with one and two layers. 13 and 19 tiles.
3. **`window/receptive-field`** — the razor. With `num_layers = 1` the set of
   inputs that can reach output `p` is exactly `[p - C + 1, p]` by W2. Perturb
   the input at one position `j` and re-run: outputs at `p - j > C - 1` come
   back **bit-identical**, outputs at `p - j <= C - 1` **differ**. Two-sided, so
   it pins the edge from both directions, and it shares no arithmetic with the
   reference in (1). Verified at `C = 250, j = 260`: field exactly [260, 509].
   `num_layers = 2` asserts the widened bound `2 * (C - 1)`.
   **The perturbation has to vary across the dimension.** A constant offset is
   LayerNorm's null space, so `+= 2.0f` perturbs nothing: measured response
   7e-07 at `p == j` and 2e-07 after it, i.e. the first version of this case
   passed entirely on rounding noise. With a varying bump it is 3.7 and 0.1-0.5.
4. **`window/translation`** — RoPE's base. A constant background with one
   distinct position `a`. With one layer every background position has the same
   pre-RoPE q, k, v, so a score is `R(p)q . R(i)k = q . R(i-p)k`: a function of
   the relative offset alone (W5). Put `a` at two far-apart absolute positions
   and the responses at equal relative offset must match — measured 4.8e-07 to
   7.7e-07. Put it at position 0 instead and the response must differ from the
   translated one for every `d <= C - 2` and match from `d = C - 1` on: at
   `C = 13` that is 2.4e-02 against 3.6e-07, a margin of 66000x; at `C = 250`
   the missing key carries 1/250 of the softmax so the margin narrows to 89x,
   which is why the assertion is a ratio and the bit-exact pin on the same
   boundary lives in (3).
   **An earlier version of this case fed the same vector at every position and
   expected the output to settle once the window filled. It is vacuous**: `v` is
   not rotated, so a constant input gives every position the same `v`, and a
   convex combination of identical vectors is that vector whatever the attention
   weights are. The measured "settling curve" was flat at 3e-07 from position 0.
5. **`window/capacity`** — W4 as an executable assertion, at `C = 37` and
   `C = 250`: a state with `max_seq_len = C` refuses the step at `p = C` with
   `-1`, leaves its offset at `C`, survives the refusal, and refuses an
   overflowing `_prefill` whole. This is what "where does the cache wrap?"
   reduces to here — it does not — and it is what stops someone turning it into
   a ring without noticing the contract changed.
6. **`window/ragged-batch`** — six states at 0, `C-2`, `C-1`, `C`, `3C+5` and
   `7C`, prefixes laid down through the tile path, then stepped together for six
   consecutive steps so rows *cross* the engage boundary mid-run, each row
   compared bit for bit against the same state stepped alone, at `C = 13` and
   `C = 250`, with and without a `linear_rows` hook. The case also asserts the
   rows are not all identical, so it cannot be blind to cross-talk, and that row
   1 really did cross the boundary.

   The hook must call `mynah_matvec_bias_f32`, not an equivalent scalar loop
   written out longhand: the contract is bit equality per row against the
   *unhooked* path, and a hand-written loop is not bit-equal to the
   SIMD/Accelerate matvec. Writing it out by hand produced a failure that looked
   like a batching defect and was not.

## Negative controls

A test that has never failed is a hypothesis. Nine mutations were applied to
`src/transformer_ar.c` one at a time, both gates run, and the mutation reverted.
Two consecutive full runs were byte-identical.

| # | Mutation | `make self-test` | `make window-test` |
|---|---|---|---|
| 1 | window one position too wide (`position <= context`) | caught | caught |
| 2 | window one position too narrow (`position + 2u - context`) | caught | caught |
| 3 | RoPE offset taken relative to `lo` instead of absolute | caught | caught |
| 4 | window ignored whenever more than one row is in flight | caught | caught |
| 5 | keys read from 0 while the span stays windowed (ring-style misindex) | caught | caught |
| 6 | KV cache written modulo `context` instead of absolutely | caught | caught |
| 7 | span silently capped at 64 (invisible below `C = 64`) | **caught only after 9c** | caught |
| 8 | RoPE table built only `context` long and indexed modulo it | caught | caught |
| 9 | RoPE table valid only to position 256, index clamped | **MISSED** | caught |

Two honest readings of that table:

- **the pre-existing self test was better than E2-4's framing assumed.** Simple
  window off-by-ones (1, 2, 5) and cache-shape errors (6, 8) were already caught
  at `context = 2` over six positions. What it could not see is defects with a
  *threshold*: 7 and 9 are silent until the sequence is longer than a block size
  or a table length, and nothing in this repo ran a windowed sequence that long.
- **9 still escapes the shipped self test**, and knowingly. Catching it needs a
  value-level contract check at `T > 256` — the f64 reference or translation
  invariance — which is `make window-test`, i.e. `make test`, i.e. UBSan and
  ASan, but not a bare `--self-test` on a Linux box. Closing that would mean
  moving the f64 reference into the shipped binary; it was not judged worth the
  size, and this line is the record of the decision.

## What still needs the oracle

The tests above prove the C implementation is *self-consistent* and matches the
contract W1-W7. They cannot prove W2 is Kyutai's rule rather than ours — a
window of `C` versus `C + 1`, or "excluding self", would pass every test here if
the contract and the code agreed on the wrong thing.

Exactly one dump settles it — and `tools/oracle_pocket.py` **cannot produce it
today**. It hard-codes `MAX_CALLS_PER_MODULE = 4`, i.e. steps 0..3, which is
precisely the prefill-versus-steady-state question and not the window question.
Two things have to change before the run:

1. in `tools/oracle_pocket.py`, replace the flat `MAX_CALLS_PER_MODULE` cap with
   a call-index allowlist so the dump stays bounded — record only calls
   `{0, 1, 248, 249, 250, 251, 252, 500, 501}` of
   `mimi.decoder_transformer.transformer.layers.0`;
2. pass a text long enough that the codec transformer is actually called 502
   times.

```bash
# with the weights present and uv available
make oracle-pocket \
  ORACLE_POCKET_LANG=english ORACLE_POCKET_VOICE=alba \
  ORACLE_POCKET_OUT=build/oracle-pocket-window \
  ORACLE_POCKET_TEXT="<~30 s of text>"      # needs the --text passthrough too
# or directly:
uv run --with pocket-tts --with numpy --with scipy python tools/oracle_pocket.py \
    --language english --voice alba \
    --text "<~30 s of text>" \
    --out build/oracle-pocket-window
```

What to compare, and the acceptance gate:

- the recorded input/output of
  `mimi.decoder_transformer.transformer.layers.0` at calls
  `{248, 249, 250, 251, 252, 500, 501}`, replayed through
  `mynah_transformer_ar_step` on a state seeded with the same history,
  rel_l2 `< 1e-4`;
- the discriminating pair is `p = 249` and `p = 250`. Under W2/W3, `249` is the
  last position that attends to position 0 and `250` is the first that does not.
  If upstream's rule were `[p - C, p]` (span `C + 1`) the divergence appears at
  `250` and not at `249`; if it were `[p - C + 2, p]` it appears at `249`. A
  single position past the boundary cannot tell those apart, which is why the
  dump must contain both sides of it;
- `p = 500, 501` confirm the rule does not drift after two window lengths, which
  is the one thing a two-position dump cannot show.

The utterance has to be **> 1.25 s of audio** (250 positions at 200 Hz) before the
codec transformer reaches the window at all, and **> 2.5 s** before call 500
exists; anything shorter dumps the unwindowed path and proves nothing about
E2-4. A ~30 s text gives ~6000 calls and plenty of headroom.

## Gates run

Development machine, Apple M-series, macOS, `SIMD=auto` (`-march=native`),
Accelerate. Every number here is a correctness result; nothing was timed.

| Gate | Result |
|---|---|
| clean `make` from `make clean` | pass, no new warnings (two pre-existing `BNNSFilter` deprecations in `src/conv1d.c`, untouched) |
| `make self-test` | PASS, now including window cases 9b and 9c |
| `make window-test` | 21/21 cases pass |
| `make test` | self-test + driver + **window** + playback-sim + python tools, all pass |
| `make goldens` (`models/fake-magpie`) | PASS, 8 checks + batch parity — Magpie byte-identical, not moved |
| `make stream-test MODEL_DIR=models/fake-magpie` | stream equivalence: PASS |
| `make ubsan` | full `make test` under UBSan, clean |
| `make asan` | full `make test` under ASan, clean |
| `make leaks` | 0 leaks, 0 bytes (`--self-test`) |
| `leaks --atExit` on the window test binary | 0 leaks, 0 bytes |
| PocketTTS generation, default and int8 | **NOT RUN** — no PocketTTS pack is present and the weights are gated. Covered instead by the object-code comparison above: the inference path is instruction-identical, so generation cannot have changed. To run it when weights exist: `make synthesize MODEL_DIR=models/pocket-en ...` and `MYNAH_QUANT=int8 ...` at this commit and at `65a352b`, then `cmp` the WAVs. |

## Not done, deliberately

- **The memory shape of W4 is left alone.** A 250-window on a 200 Hz transformer
  with an absolute-indexed cache costs 8.2 MB per 5 s request and 49 MB per 30 s
  request, against 2.0 MB flat for a ring sized to the window. That is a real
  per-slot cost in a continuously batched server, and it is a *performance*
  decision that belongs with `.work/serving-design.md`, not in a correctness
  change. What this change does is make the contract explicit and pin it with
  `window/capacity`, so that turning the cache into a ring later is a visible
  decision with a test to update rather than a silent one.
- **No second implementation.** The window rule is stated once in
  `tar_window_start()` and once, independently, in the test's `contract_window_start()`,
  which exists so the test asserts the contract rather than the code.
