# ingot

**One C11 library for reading GGUF and safetensors.** No dependencies, no build
system required, nothing written to stderr, and quantized weights handed back
in the format they were stored in — dequantizing is something you ask for, not
something that happens to you on `open()`.

```c
#include <ingot/gguf.h>

char err[256];
ingot_gguf *g;
if (ingot_gguf_open(&g, "model.gguf", err, sizeof err) != 0) {
    fprintf(stderr, "%s\n", err);   /* the library never prints; you do */
    return 1;
}
const ingot_tensor *t = ingot_gguf_find(g, "blk.0.attn_q.weight");  /* O(1) */
const void *weights = ingot_gguf_data(g, t);                        /* mmap  */
ingot_q4_k_matvec(weights, rows, cols, x, y);                       /* no copy */
ingot_gguf_close(g);
```

```c
#include <ingot/safetensors.h>

ingot_st *st;
ingot_st_open_dir(&st, "./Qwen3-8B", err, sizeof err);   /* index.json, shards, or a single file */
const ingot_st_tensor *e = ingot_st_find(st, "model.embed_tokens.weight");
const uint16_t *bf16 = ingot_st_data(st, e);
```

## Why it exists

Every engine that loads weights ends up writing this parser again, and every
copy is subtly different: one handles `Q8_0`, another checks
`offset % alignment`, a third has an O(1) name index, a fourth reads `BF16` —
and no two have the same three. ingot is that parser written once, with the
strictest version of each check and no fixed caps that truncate a model in
silence.

## What you get

| | |
|---|---|
| **GGUF v2/v3** | full metadata KV store (typed scalars, arrays, vocabularies), split files (`-00001-of-00003.gguf`), zero-copy `mmap` **and** a `pread` twin for when you cannot map |
| **safetensors** | single file, a directory (`model.safetensors.index.json` → glob fallback), or an explicit shard list; `__metadata__` exposed; `F32/F16/BF16/F8_E4M3/F8_E5M2/I8..I64/U8..U64/BOOL` |
| **Every ggml type has a geometry** | including `IQ*`/`TQ*`, so those files open and account their bytes exactly, and the error says `IQ4_XS` instead of `unknown type 23` |
| **Every quantization** | all 33 ggml block types decode — the K-quants, the whole `IQ1`/`IQ2`/`IQ3`/`IQ4` codebook family, ternary `TQ`, microscaling `MXFP4`/`NVFP4` — verified bit-for-bit against llama.cpp. See **[docs/QUANTS.md](docs/QUANTS.md)** |
| **Encoding** | `F16 BF16 F32 Q4_0 Q4_1 Q5_0 Q5_1 Q8_0 Q4_K Q6_K`, so a converter needs nothing else |
| **Kernels** | matvec + batched matmat for `Q2_K Q3_K Q4_K Q5_K Q6_K Q8_0`, with NEON / AVX2 / ARM-SDOT / ARM-SMMLA paths, runtime-detected |
| **One call for any type** | `ingot_matvec(type, …)` / `ingot_matmat(…)` take the hand-written kernel when there is one and decode row-by-row when there is not, so a loader handed an arbitrary GGUF does not have to branch on the format |
| **Writing** | GGUF and safetensors writers, including "hand me f32 and a target type" |
| **Page-cache control** | `prefault`, `dontneed`, `drop_cache` — the difference between a checkpoint that frees its 23 GB and one that fights your GPU for unified memory |
| **O(1) lookup** | FNV-1a index over tensor names and metadata keys |

## Documentation

| | |
|---|---|
| **[docs/QUANTS.md](docs/QUANTS.md)** | every quantization format, what is supported, `bits/weight`, and why `Q4_K_M` is a recipe rather than a type |
| [include/ingot/gguf.h](include/ingot/gguf.h) | GGUF reader: tensors, metadata KVs, split files |
| [include/ingot/safetensors.h](include/ingot/safetensors.h) | safetensors reader: shards, `index.json`, page-cache control |
| [include/ingot/quant.h](include/ingot/quant.h) | decode, encode, kernels, the precision contract, threads |
| [include/ingot/wfile.h](include/ingot/wfile.h) | one handle for either container |
| [include/ingot/write.h](include/ingot/write.h) | writing both formats |
| [examples/minimal.c](examples/minimal.c) | the whole library in 80 lines |

The headers are the reference: each one opens with what it is for and what it
refuses to do.

## Deliberately not

No compute graph. No tensor type with broadcasting. No allocator or arena — you
get pointers into the mapping or you pass a buffer. No imposed memory layout for
kernels: the layout a Metal simdgroup wants and the one a CUDA row-split wants
are different, and a loader that picks for you is a loader nobody adopts. No
tokenizer, though the GGUF metadata that carries one is fully readable.

If the container half grows past ~1,500 lines, something got in that shouldn't
have. The quantization half is allowed to be big — that is where the formats
live.

## Build

```sh
make                # libingot.a, ingot-dump, examples
make test           # 274 checks, no model and no Python needed
make help           # every target, with a line each
```

Also useful:

| target | |
|---|---|
| `make test-leaks` | the suite under macOS `leaks` — **the memory gate on a Mac**, where ASan tends to hang |
| `make test-asan` | the suite under AddressSanitizer + UBSan (use this on Linux) |
| `make fuzz` / `fuzz-leaks` | mutation-fuzz both readers (`ROUNDS=n`) |
| `make core-only` | the readers without the quantization half, to prove the split is real |
| `make amalgam` / `amalgam-test` | regenerate the two-file build and run the whole suite against it |
| `make gen-tables` / `gen-fixtures` | re-derive the IQ codebooks and oracle fixtures from llama.cpp's `gguf` package |
| `make check-real MODEL=…` | cross-check a real checkpoint against an independent parse |

Or skip all that entirely — copy **two files** in and build the `.c` like any
other source:

```sh
cp ingot/amalgam/ingot.h ingot/amalgam/ingot.c yourproject/vendor/
cc -std=c11 -O2 yourproject/vendor/ingot.c ... -lpthread -lm
```

`amalgam/` is generated from `src/` by `make amalgam`, and `make amalgam-test`
runs the **whole** suite against the generated pair — a source file added to
`src/` but forgotten in the generator fails there rather than shipping broken.
The header parses as C++ too. `-DINGOT_NO_KERNELS` halves it (141 KB → 70 KB of
object code) by dropping the quantization half.

Otherwise: `cc -Iinclude -c src/*.c`. C11, POSIX, `-lpthread -lm`.

Only the container half is mandatory (`dtype.c`, `gguf.c`, `safetensors.c`,
`wfile.c`, `write.c`). `cpu.c`, `dequant.c`, `kernels.c`, `generic.c` and
`quantize.c` are optional — a tool that only wants to inspect a file should not
link the SIMD. `make core-only` builds and proves that split on every commit.

## Inspecting a file

```sh
$ ingot-dump model.gguf
GGUF v3  model.gguf
  architecture: llama
  alignment:    32
  shards:       1
  metadata:     26 keys
  tensors:      291

type census:
  F32         81 tensors     52.14 MiB
  Q4_K       189 tensors   3821.02 MiB
  Q6_K        21 tensors    412.66 MiB
  TOTAL      291 tensors   4285.82 MiB
```

`-v` lists every tensor and metadata key. Point it at a directory and it
resolves the safetensors shards.

`examples/minimal.c` is the same idea in 80 lines: open anything, find the
widest 2D tensor, multiply a vector through it without branching on the format.

## The precision contract

From two tokens up, the batched `Q4_K`/`Q5_K` kernels quantize the
**activations** to int8 by default: 1.5–2.5× faster, at roughly 2.4e-3 relative
error instead of the last few ulp of a reordered sum. Everything else stays an
exact reorder.

This matters when the CPU result is the reference a GPU path is measured
against, so the library tells you instead of making you guess:

```c
const double budget = ingot_matmat_is_exact(tokens) ? 1e-5 : 5e-3;
```

## Any type, one call

```c
const ingot_tensor *t = ingot_gguf_find(g, "blk.0.ffn_down.weight");
ingot_gguf_matvec(g, t, x, y);      /* Q4_K? Q6_K? IQ4_XS? F16? it does not matter */
```

`ingot_gguf_matvec` reads `rows`/`cols` off `ne` (so the ggml dimension flip
happens once, here, instead of at every call site) and routes to the specialized
kernel when the type has one. `ingot_has_kernel(type)` says whether it did, for
a caller choosing a quantization on speed grounds.

`ingot_q4_k_matmat_exact()` and `ingot_q5_k_matmat_exact()` are always exact
whatever the default is; `INGOT_SDOT=0` in the environment turns the int8 path
off globally. A parity gate that picks 5e-3 where 1e-5 is owed passes for the
wrong reason.

## Threads

ingot does not own a thread pool — every consumer already has one, and two
pools fighting over the same cores is a measurable loss. Hand yours in:

```c
static void my_pool(size_t n, ingot_range_fn fn, void *user) { /* ... */ }
ingot_set_parallel_for(my_pool);
```

The default runs inline.

## Testing

`make test` runs 235 checks. It builds synthetic containers in a temp directory and checks that
valid ones parse *and* that malformed ones are rejected cleanly, with a message
and without a crash: bad magic, v1, truncated headers, payloads past EOF,
absurd string lengths, alignments that are not powers of two, unknown tensor
types, unpadded safetensors headers, byte counts that disagree with
shape × dtype, tensor names duplicated across shards.

That half is the point. A reader that has only ever seen well-formed files is a
reader nobody has tested — and writing the tests first is what caught four real
bugs on the way in, every one of them the silent kind:

- **`Q8_1` was sized at 36 bytes** where ggml says 40 (`d`/`s` widened from
  f16 to f32 at some point), which silently mis-sizes every tensor of that
  type. Caught by cross-checking all 34 geometries against `GGML_QUANT_SIZES`,
  now a standing test rather than something done once.
- **`Q6_K` decoded from the wrong offsets.** One decoder read the block as
  `{half d; ql[128]; qh[64]; scales[16]}` and walked the quants linearly. ggml
  puts `d` at the **end** and interleaves the quants in two halves of 128. Every
  Q6_K tensor it read was wrong — silently, because the wrong layout still fits
  in exactly 210 bytes. Caught by cross-checking the two independent decoders in
  `src/dequant.c` and `src/kernels.c`: the other five K-quants agree
  bit-for-bit, that one did not.
- **`Q8_0` dequant read one eighth of every row**, leaving the rest of the
  output uninitialised — the 256-element decoder was handed a 34-byte stride.
- **The SIMD dispatchers segfaulted on a null pointer** instead of returning
  `-1`: the scalar kernels validate their arguments, the NEON entry points
  jumped straight in.

Two more gates run outside `make test` because they are loops rather than
assertions:

- `make fuzz` writes a valid file, flips a few bytes, opens it and touches
  everything a consumer would touch. 6000 rounds, ~15% of which still parse —
  and those are the interesting ones, because they walk the accept path with
  values no valid writer would produce.
- `make test-leaks` runs the suite under macOS `leaks`. It is the memory gate
  here: ASan tends to hang on macOS, and a hung run reads as a slow test rather
  than a broken one. It immediately found a handle leak ASan had not reported.

Two tests go outside the library for their answers, because fixtures built
with ingot's own understanding of a format cannot catch a shared misreading:

- `tests/test_oracle.c` decodes fixtures whose expected values came from
  llama.cpp's own `gguf` package. All 23 decodable block types match **bit for
  bit**, on pseudo-random blocks that walk the whole codebook.
- `tools/check_against_python.py` reparses a real safetensors file with nothing
  but `struct` and `json` and compares every tensor. It agrees on 3,235 tensors
  across four real checkpoints.

## Endianness

Both formats are little-endian by definition and ingot reads them byte by byte,
so the parsing is host-independent. Bulk conversion of 4- and 8-byte element
types currently assumes a little-endian host — fine on x86-64 and aarch64,
which is everywhere this runs today.

## Not implemented

`Q1_0`, and only because llama.cpp's own reference package has no decoder for
it either — there would be nothing to verify against. Its geometry is known, so
such a file opens, accounts its bytes exactly, and fails by name. Details in
[docs/QUANTS.md](docs/QUANTS.md).

## License

MIT — see [LICENSE](LICENSE). The ggml block layouts are ggml's (MIT); the
decoders are written from the spec and checked against it.
