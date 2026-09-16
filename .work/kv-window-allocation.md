# The KV cache allocated one slot per position a windowed attention can never read

Found 2026-09-16 while re-measuring E10-12, which claimed "277 MB of KV
`calloc`'d per request". The claim was right about the allocation and wrong
about what it costs, and the real defect underneath was sharper than either.

## What E10-12 said, and what measuring said

`calloc` of a large block comes from `mmap`, so the pages are faulted on use.
Measured, one process, `models/pocket-en`, same binary:

| utterance | samples | max RSS |
|---|---|---|
| 1 word | 9,600 | 754 MB |
| 9 words | 71,040 | 777 MB |
| 96 words | 608,640 | 834 MB |

**+80 MB across 63× the audio**, not 277 MB per request. An RSS figure quoted
from an allocation size is a figure nobody measured.

## The defect the measurement exposed

`mynah_transformer_ar_config` carries `context`, a sliding attention window.
The backbone passes 0 (unlimited); **Mimi's decoder transformer passes 250**.
The KV cache was sized from `max_seq_len` either way — and for the codec that
is `(max_steps + 1) * upsample_stride` = **24,016 positions**:

    2 layers x 2 (K,V) x 24016 positions x 512 floats x 4 = 196.6 MB

of which 250 positions — 2 MB — can ever be read. The attention loop had said
so since it was written: it reads `[tar_window_start(position, context),
position]` and nothing else.

It survived precisely *because* the pages were lazy. The allocation was wrong;
the RSS was not, so nothing ever pointed at it.

## The fix, and why it is a compaction and not a ring

A windowed cache now holds `context + slack` positions with slot 0 at a moving
absolute position `kv_base`. When a write would run past the end, the still
reachable tail is moved down and the base advances.

A **ring buffer** would be the obvious structure and is the wrong one here: it
puts a wrap and a branch inside the hottest loop in the codec to save an
amortised memmove of about one position per step. Compaction keeps the
attention reading **one contiguous span**, so the inner loop is untouched —
the only change there is `lo - kv_base` instead of `lo`.

Slack is `context` (floored at the prefill tile), so a compaction runs at most
once every `context` positions and moves at most `context` of them.

`context == 0` keeps `kv_positions == max_seq_len` and `kv_base == 0`, so the
backbone — and every voice-prefix path, which is the one place the
`[2][max_seq_len][H][D]` layout contract is load-bearing — is untouched.
`_state_kv()` returns NULL and `_state_load_kv()` refuses on a windowed cache
rather than handing back a pointer whose meaning quietly changed.

## And the allocation that was left largest

With the KV bounded, the biggest remaining per-context block was the **RoPE
table**: 24,016 positions x 32 halves x 2 x 4 bytes = **6.1 MB**, rebuilt
identically in every context, for a quantity that is a pure function of
(position, head_dim, max_period). Now shared, immutable, keyed on exactly what
it is a function of, freed at exit.

## Result

| | before | after |
|---|---|---|
| codec KV, per context | 196.6 MB | 4.1 MB |
| RoPE, per *additional* context | 6.1 MB | 0 |

Measured as peak `vsz`, one process, a 96-word utterance: **−184.5 MB**
(436,313,584 → 436,124,672 KB). At `--batch 4`: a further **−18.2 MB**, which
is 3 x 6.1 to the megabyte — the RoPE arithmetic, confirmed rather than
assumed.

Resident, same utterance: 834 → 798 MB. Small, and that is the point: the
saving is address space and the pages were never all touched. It matters for
concurrency headroom, page-table pressure and any host with strict overcommit
— not for the RSS of one request, which is what the original item implied.

**The audio is byte-identical**, checked against the WAV from before the
change.

## What the gate catches

A windowed cache must produce bit-identical output to the full-length one, or
a request's audio would depend on how much memory the process felt like using.
`mynah_transformer_ar_kv_window_force()` exists so both run in ONE process over
the same weights and inputs — the same argument as `mynah_qmat_i8mm_force()` —
and the test asserts the two arms really allocated different caches before
comparing, so it cannot pass by being blind.

| mutation | caught by |
|---|---|
| compact to the LAST row's window instead of the first | the existing long windowed prefill |
| the read ignores `kv_base` | windowed-vs-full, position 21 |
| the write wraps (`% kv_positions`) instead of compacting | windowed-vs-full, position 21 |
| the compaction moves nothing | windowed-vs-full, position 21 |

The first one matters most: a prefill tile's rows write before any of them
attend, so the bound is the **lowest** position in the tile, not the highest.
Compacting to the last row's window drops what the first row still needs, and
only a multi-row prefill crossing a compaction can see it.
