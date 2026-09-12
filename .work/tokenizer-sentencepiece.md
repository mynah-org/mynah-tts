# E3-2 — SentencePiece Unigram in C11

Status: **IMPLEMENTED** 2026-09-12 — `src/tokenizer_sentencepiece.{c,h}`, 1734 LOC,
wired into `CORE_SOURCES` and `--self-test`, parity gate `make tokenizer-parity`

All numbers below were read from the five real `tokenizer.model` files and
cross-checked against `sentencepiece` 0.1.98. Where something is asserted from
upstream source rather than measured, it says so.

## The finding that shrank this task

**The normalizer is `identity` with a zero-length `precompiled_charsmap`.** No
NFKC, no darts-clone trie to decode. That was expected to be most of the work
and it does not exist.

Proof, two ways. By size: a model trained with `nmt_nfkc` carries a 237,561-byte
charsmap; these files are **59–61 KB in total**, and field 2 of
`normalizer_spec` measures literally 0 bytes in all five. By behaviour, on the
real Italian model:

| input | pieces | note |
|---|---|---|
| `ﬁne` (U+FB01) | `▁ <0xEF> <0xAC> <0x81> ne` | NFKC would give `▁fine` |
| `Ｈｅｌｌｏ` fullwidth | pure byte fallback | |
| `é` | `▁e <0xCC> <0x81>` | no NFC composition |

**Policy**: reject a model whose normalizer is not `identity` or whose charsmap
is non-empty, with a message naming both. Do not approximate it — that is the
`CLAUDE.md` rule about not silently substituting a different normalizer. If one
is ever needed, the converter walks the darts trie offline and exports a sorted
`src→dst` table into the pack (~80 LOC in C, inspectable data), rather than
shipping 237 KB of opaque blob and a trie walker.

## File format, verified

```
ModelProto
  field 1 (LEN) x4000  SentencePiece pieces
  field 2 (LEN) 283 B  TrainerSpec
  field 3 (LEN)  18 B  NormalizerSpec
```

Pieces are `{piece, score}` or `{piece, score, type}`; `score` is **float32**
(wire type 5), not double. Type enum: `NORMAL=1 UNKNOWN=2 CONTROL=3
USER_DEFINED=4 UNUSED=5 BYTE=6`. Identical histogram in every language:
**1 UNKNOWN + 3 CONTROL + 256 BYTE + 3740 NORMAL = 4000**, zero USER_DEFINED,
zero UNUSED.

TrainerSpec fields that matter — **the numbers are counter-intuitive and getting
them wrong yields plausible but false values**: `model_type = 3` (not 4),
`vocab_size = 4`, `treat_whitespace_as_suffix = 24`, `byte_fallback = 35`,
`unk_id = 40`, `pretokenization_delimiter = 53`.

NormalizerSpec, identical in all five:
`name="identity"`, `charsmap=<0 bytes>`, `add_dummy_prefix=1`,
**`remove_extra_whitespaces=0`** (explicitly false, against the default true),
`escape_whitespaces` absent → true.

| lang | sha256[:12] | max piece bytes |
|---|---|---|
| english | d461765ae179 | 9 |
| italian | 6583b974a11b | 9 |
| german | 389079b9c67c | **10** |
| spanish | aac2b96478e3 | 9 |
| portuguese | 3aa51309c55f | **10** |

Root `tokenizer.model` is byte-identical to `languages/english/`, which is
identical to both other English revisions.

## Normalization: four rules, ~80 LOC

1. `add_dummy_prefix`: prepend one `' '` to the raw input.
2. Per-character UTF-8 validation: invalid → **U+FFFD, consuming exactly one
   byte**. "Invalid" is strict: overlong, encoded surrogates, > U+10FFFF,
   5-byte, truncated. `ED A0 80` becomes **three** U+FFFD, not one.
3. `remove_extra_whitespaces = 0`, so the collapse branch is **off**:
   `"ciao  mondo"` → `▁ciao ▁ ▁mondo`.
4. `escape_whitespaces`: only `0x20` becomes U+2581. **`\t`, `\n`, `\r`, `\f`
   and NBSP are untouched** and fall through to byte fallback.

Empty raw input → empty output, checked *before* the dummy prefix; `" "` gives
two tokens.

## Encoding

Lattice positions are **characters, not bytes**. Only `NORMAL`, `USER_DEFINED`
and `UNUSED` pieces enter the index; `UNKNOWN`, `CONTROL` and `BYTE` do not, so
`"<s>"` in text must produce `▁ <0x3C> s <0x3E>` and not the reserved id.

```
best[0] = 0
for c in 0..nchar-1:
    has_single = false
    for each piece p prefixing buf[starts[c]..] and ending on a char boundary:
        if type(p) == UNUSED: continue
        cj = char_index(end of p)
        score = (type(p) == USER_DEFINED) ? (cj - c) * 10.0f : score(p)
        if cj - c == 1: has_single = true
        if best[c] + score > best[cj]: relax
    if !has_single: relax best[c+1] with unk_score   /* UNK spans ONE CHARACTER */
```

`unk_score = min_score - 10.0f` where **`min_score` is over NORMAL pieces only**
— including BYTE/CONTROL (score 0.0) would give -10.0 and byte fallback would
essentially never be chosen. Arithmetic is **float32**, matching upstream; keep
the DP loop away from `-ffast-math` reassociation. Tie-break is strict `>` with
predecessors visited in increasing position order (0 actual ties observed across
59,525 relaxations, so documented but unexercised by real data).

Byte fallback expands the bytes of the **normalized** buffer, not the input: an
invalid `0xFF` yields `<0xEF> <0xBF> <0xBD>`. Consecutive UNKs are never merged.

`max_sentencepiece_length = 6` is a **training** parameter in characters. The
probe bound is in **bytes** and must be computed from the pieces: 9 or 10
depending on language. Byte-token ids happen to be `4 + b` everywhere but must
be built by scanning `type == BYTE` pieces (`CLAUDE.md` rule 6). `unk_id` comes
from the piece of type UNKNOWN, not from TrainerSpec field 40 — that is what
upstream's `InitializePieces` does.

## Structure and API

Open-addressed hash on `(ptr, len)`, FNV-1a, capacity 8192, keys pointing into
the mmap. At most 6 probes per position because pieces are ≤ 6 characters. A
trie buys nothing here: encoding happens in the prefill, not the AR loop, and
the whole vocabulary is ~22 KB, which sits in L2.

New component, **not** an extension of `src/tokenizer.c` (that one is Magpie's
BYT5/G2P/IPA, different contract). Style follows `src/weights.h`.

```c
int  mynah_sp_open(const char *path, mynah_sp **out, char *err, size_t cap);
void mynah_sp_close(mynah_sp *sp);
int  mynah_sp_encode(const mynah_sp *, const char *text, size_t text_length,
                     int **out_ids, size_t *out_count, char *err, size_t cap);
int  mynah_sp_encode_into(const mynah_sp *, const char *, size_t,
                          int *ids, size_t capacity, size_t *out_count, ...);
size_t mynah_sp_vocab_size(const mynah_sp *);
int    mynah_sp_piece(const mynah_sp *, int id, const char **piece, size_t *len);
int    mynah_sp_self_test(char *err, size_t cap);   /* model-free */
```

`(ptr, len)` rather than a bare `const char *`: `"\x00abc"` is legal input and
encodes to `▁ <0x00> a b c`.

**~710 LOC** in the `.c`, ~45 in the header: protobuf reader 70, ModelProto
parse and guards 130, index 80, UTF-8 decoder 45, normalizer 85, Viterbi +
fallback 100, API/mmap/lifecycle 90, self-test 110.

## Validation

The algorithm above was **reimplemented and diffed against `sentencepiece`
0.1.98** before being written down:

```
str level   : 0 / 30,170 mismatches   (6,034 strings x 5 languages)
bytes level : 0 / 35,080 mismatches   (7,016 strings x 5 languages)
float32 DP  : 0 / 20,080 mismatches
```

`sp.encode()` accepts `bytes`, which is what makes the invalid-UTF-8 paths
testable at all. Plan: `tools/oracle_pocket_tokenizer.py` emits
`build/oracle-tokenizer/<lang>.jsonl` with `{"hex": ..., "ids": [...]}` — hex so
invalid bytes survive the round trip — and the C side replays and compares,
reporting the first mismatch. Generated, never committed (rule 8).

Corpus: curated sentences per language (numbers, dates, currency, URLs,
apostrophes, guillemets, em dashes); Unicode stress (ligatures, fullwidth,
combining vs precomposed, ZWJ emoji, variation selectors, flags, U+10FFFF);
whitespace (doubles, tabs, CRLF, NBSP, a literal U+2581); adversarial vocabulary
(`<s>`, `<unk>`, `<0x41>` as literal text); and byte-level fuzz — every truncated
prefix of valid sequences, the invalid catalogue (`C0 80`, `ED A0 80`,
`F4 90 80 80`, …), and 9,000 random byte strings. Plus one 100,000-character
string for overflow and prefill cost.

## Traps

The ones that silently produce wrong tokens rather than failing:

1. Assuming NFKC. Read the field; reject what you cannot reproduce.
2. `remove_extra_whitespaces = 0` against the library default.
3. Only `0x20` is escaped; tabs and NBSP are not whitespace here.
4. UNK spans a **character**, not a byte.
5. `min_score` over NORMAL only.
6. CONTROL/UNKNOWN/BYTE must not be segmentable — otherwise `<s>` in user text
   maps to the reserved id, which is a prompt-injection surface.
7. `UNUSED` is skipped at encode time **and does not count toward
   `has_single_node`**; `USER_DEFINED` scores `nchar * 10.0` instead of its own
   score. *Asserted from upstream `unigram_model.cc`, not measured — these five
   files contain none of either, so only the synthetic self-test can cover it.*
8. Byte fallback expands the normalized buffer.
9. `max_sentencepiece_length` is characters and is training-time; the probe
   bound is bytes and per-language.
10. Protobuf field numbers are counter-intuitive; guards
    (`model_type == 1`, `vocab_size == piece_count`, `vocab_size == n_bins` from
    `model.json`) are what catch a wrong mapping.
11. Unknown protobuf fields must be skipped; groups (wire type 3/4) rejected
    rather than looped on.
12. Text may contain NUL.
13. No BOS/EOS is added, and the pack verifier should assert no emitted id is
    ≥ 4000 — row 4000 of the `[4001, 1024]` embedding is padding.

## Implementation result — 2026-09-12

`src/tokenizer_sentencepiece.c` (1669) + `.h` (65). Core is ~800 LOC; the
model-free self-test is ~600, because it builds a synthetic `ModelProto` by hand
with cases designed to *discriminate*: a `USER_DEFINED` piece whose stored score
is -1000 (using it instead of `nchar * 10` would lose to UNK), an `UNUSED`
single-char piece that would win if it were not skipped, a `min_score` that
changes if BYTE/CONTROL are included, and a TrainerSpec `unk_id` of 77 that
differs from the UNKNOWN piece's index.

**Parity: 45,197 cases across five languages, 2,116,052 ids, one known
divergence.** Clean under UBSan and ASan; `leaks` reports zero.

`make tokenizer-parity` replays `build/oracle-tokenizer/<lang>.jsonl` through
`tests/test_tokenizer_sp.c`.

### The note was wrong about float32

This document said to use float32 "matching upstream". Measured, that is the
worse choice.

On the deliberately adversarial 100,000-character random case the accumulated
Viterbi path score reaches ≈ -3e5, where a float32 ULP (~0.03) is **larger than
the gap between competing segmentations**. Ties then resolve arbitrarily. In
float32 this implementation diverged from sentencepiece on **6 ids out of
102,899** for English (three sites, each a two-character span split the other
way: `ah|i` versus `a|hi`) plus one German case. Switching the DP accumulator to
double: English, Italian, Portuguese and Spanish became exact over the whole
corpus; German keeps one divergence on that same 100k case.

Proof it is precision and not a bug: **the divergent substring, tokenized in
isolation, matches exactly.** It only diverges when preceded by 37,699 tokens of
accumulated score.

So: **double accumulator**, 4 bytes more per character in an array that only
exists during prefill.

### The residual, stated rather than hidden

`tests/test_tokenizer_sp.c` defines `PARITY_GUARANTEED_BYTES = 65536`. Below it,
any mismatch fails the gate. Above it, a divergence is **reported loudly** and
does not fail, because finite precision cannot resolve those ties at all and
neither implementation is "right".

Nothing in the product reaches that regime: PocketTTS chunks text at 50 tokens
and the pack bounds text length. The bound is documented so nobody later
"fixes" a non-bug or, worse, widens the tolerance to make a real failure pass.

### Deviations from the spec above, and why

- **`remove_extra_whitespaces = 1` is rejected, not implemented.** Implementing
  an untested normalizer branch is exactly the silent substitution `CLAUDE.md`
  forbids. Same for `treat_whitespace_as_suffix` and a non-empty
  `pretokenization_delimiter` — **two guards this note did not list** but which
  would change tokenization without failing.
- `escape_whitespaces = 0` **is** supported: two lines, and it is upstream's
  behaviour.
- `byte_fallback` is inferred from the presence of all 256 BYTE pieces when the
  TrainerSpec is absent, so a ModelProto without one stays openable.
- The `vocab_size == n_bins` check is **not** here: that is a pack-level check,
  and this component does not know about `model.json`.
- Duplicate pieces: first wins, silently, as upstream's `InsertIfNotPresent`.
- Hash capacity and `max_piece_bytes` are derived at runtime, not hardcoded to
  8192/9/10 (rule 6).
- Two API additions: `mynah_sp_open_memory` (needed by the self-test) and
  `mynah_sp_unk_id` (so a caller can assert no emitted id is reserved).

Everything else in this note was confirmed correct: protobuf field numbers, the
type histogram, the `identity` normalizer with a zero-length charsmap,
`remove_extra_whitespaces = 0`, the absent `escape_whitespaces`, and all three
rows of the behaviour table.
