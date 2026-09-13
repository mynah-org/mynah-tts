# The Linux build posture: SIMD=auto, the ISA guard, the link matrix, the stamp

Status: **DONE** for E4-12, E4-13, E4-14, E4-15 and the dispatch report's ARM
blind spots · **E4-18 closed as not-applicable, by measurement**.
Worked 2026-09-13 against `d7281b9`, verified on macOS/M-series and on the
project's Linux box (GCP Axion c4a, 32× Neoverse V2, gcc 15.2, OpenBLAS,
glibc 2.43).

Companion: [`linux-production.md`](linux-production.md) is the specification
this implements; [`dtype-and-fallbacks.md`](dtype-and-fallbacks.md) is where the
first two Linux-invisible defects were found.

---

## 1. What changed

| item | file | what it does now |
|---|---|---|
| E4-15 | `tools/simd-auto.sh`, `Makefile` | `SIMD=auto` resolves against `/proc/cpuinfo` **and** a compiler capability probe, and prints the resolution (`make simd-auto`) |
| E4-12 | `src/dispatch.c`, `src/dispatch.h`, `cli/main.c` | `mynah_dispatch_isa_guard()`: a definite ISA mismatch exits with a message instead of SIGILL, before any allocation |
| E4-13 | `.github/workflows/build.yml` | a 16-entry `link-only` job across SIMD and BLAS profiles on x86 and ARM |
| E4-14 | `Makefile` | `build/cpu/.build-flags`: every object depends on the effective flags |
| — | `src/kernels.c/.h`, `src/dispatch.c` | five ARM rows that did not exist, and an `IDLE HARDWARE` footer |
| E4-18 | *(nothing)* | measured instead; see §6 |

New: `SIMD=portable`, `EXTRA_CFLAGS`, `make simd-auto`, `make simd-auto-test`,
`tests/test_simd_auto.sh`, `tests/fixtures/cpuinfo/`.

---

## 2. E4-15 — and a correction to its premise

The item says VNNI "would never be selected in production however well it is
implemented". **That half is not true of this tree, and the distinction
matters.** `src/qmat.c` carries its VPDPBUSD kernels behind
`__attribute__((target(...)))` and picks one with a runtime CPUID probe, so VNNI
is reachable in every x86 build with no build flag at all. What *was* broken is
the other half: `SIMD=auto` appended a fixed `-mavx2 -mfma` to every Linux x86
build **without asking the CPU anything**, so

- a host without AVX2 got a binary that dies on its first `vpaddd`, and
- a host with AVX-512/VNNI/BF16 got no acknowledgement that those units exist,
  which is what would strand a future *flag-gated* kernel.

The resolver applies the double test from `linux-production.md` trap 6 — kernel
flag **and** `cc_ok`, each probing an actual intrinsic, never the flag alone.

**What it deliberately does not pass.** Not `-mavx512f/-mavx512bw/-mavx512vl`,
even when the host has them: no f32 kernel in `src/` dispatches on AVX-512 (the
`isa.x86.avx512f` row says so in words), so the only effect would be wider
autovectorization — an unmeasured change that also pins the artifact to the
build host. `SIMD=avx512` stays the explicit opt-in. Not a VNNI flag either,
for the reason above: a flag that implies a kernel choice it does not make is
how the false AVX-512 claim in `README.md` happened in the first place. Both
are **reported** in `SIMD_AUTO_REJECTED`, which is the part that was missing.

**Consequence that had to be fixed in the same change**: `auto` now resolves for
the *build* host, so the x86 release job in CI was moved off `auto` to
`SIMD=avx2`. Left alone it would have stamped release binaries with whatever ISA
the GitHub runner happened to have.

**Testable without an x86 host.** `tools/simd-auto.sh --cpuinfo PATH --arch ARCH`
reads any cpuinfo file; `tests/fixtures/cpuinfo/` holds Zen 5 EPYC, Sapphire
Rapids (AMX), Haswell, Skylake-SP (AVX-512 without VNNI), Westmere (no AVX2) and
the project's own Neoverse V2, captured from the box. `make simd-auto-test`
runs 23 checks on any machine and is wired into `make test`. The compiler half
of the double test is explicitly **skipped** there (`--no-cc-check` prints
`[cc check SKIPPED]`) because only a compiler targeting the fixture's
architecture could exercise it, and pretending otherwise is the exact species of
unverifiable claim this lane exists to stop.

AMX is detected and refused on the reference's own numbers: their AMX 8-core
host qualified C2 while their VNNI 32-core did C12. It stays E4-7, last.

---

## 3. E4-12 — the guard, and the one rule that keeps it safe

~30 lines in `src/dispatch.c`, called as the first statement of `main()`.
Two design points worth keeping:

**It reads the raw predefined macros, not `MYNAH_DISPATCH_HAS_*`.** Those fold
in `MYNAH_DISABLE_SIMD`, which answers "which kernel dispatches" — the wrong
question. `SIMD=scalar` compiles out our intrinsics but does **not** stop the
compiler autovectorizing with whatever `-march` allowed, so a `SIMD=scalar`
build on a host whose `-march=native` implies SVE is still full of SVE. This is
the one place in that file entitled to read `__AVX2__` and `__ARM_FEATURE_SVE`.

**Only a definite absence is fatal.** Every probe is a tri-state and the guard
fires on `0` alone. `-1` means we could not ask — no `getauxval`, a CPUID leaf
the hypervisor hid, an emulator — and exiting over our own ignorance would be a
worse bug than the one being prevented. This is the emulator case from
`dtype-and-fallbacks.md`, where the layer did not expose F16C through CPUID.

The message names the instruction set the binary needs, the profile it was built
with, and the feature list the CPU reports. `--dispatch-map` prints the same
verdict as an `ISA GUARD:` line so it can be checked without a mismatched host.

---

## 4. The dispatch report's ARM blind spots

On the box, `/proc/cpuinfo` advertises `sve sve2 svei8mm svebf16 i8mm bf16`. The
report printed **one** of those six. `isa.arm.sve` did not exist at all, so a
clean-looking report was silent about four vector units on the production CPU
that this runtime does not touch — precisely the failure the report exists to
catch. `isa.arm.bf16` existed but its reason never mentioned that the hardware
in front of it has the unit.

Five rows now resolve through `mynah_kernels_isa_kernels()` (`src/kernels.h`), a
bitmask exported by the file that would hold the kernel. Every bit is 0 today
and **each 0 is a statement**: a kernel author flips its bit in the same edit
and the row follows. `resolved` is obtained by calling that predicate, never by
restating a `#if` in `dispatch.c` — the drift the header spends a paragraph
warning about. `MYNAH_DISPATCH_HAS_BF16_KERNEL` was deleted for the same reason.
When `src/qmat.c` grows an SVE int8 kernel it registers its own probe over the
same row id; the registry replaces by id, by design.

A new footer closes the general case rather than this instance of it:

```
IDLE HARDWARE: 5 units this CPU has and this binary has no kernel for:
  isa.arm.sve isa.arm.sve2 isa.arm.svei8mm isa.arm.svebf16 isa.arm.bf16
```

It is derived from the rows — `supported=yes` with `compiled=no` is exactly that
pair — so a feature can never again be idle *and* unmentioned. It works on x86
too, where it names `amx_int8` and `avx512_bf16`.

Observed on the box: 53 rows, `IDLE HARDWARE: 5`, `ISA GUARD: ok`, canary ok.

---

## 5. E4-13 found a real defect on its first run — twice

Both are pre-existing, both are in files this lane does not own, and **both were
invisible on macOS**, which is the entire thesis of the item.

**A. `SIMD=scalar` does not compile at all.** `src/qmat.c:1798`
(`qmat_weight_rel`) calls `qmat_f16_to_f32()` and reads `qmat_entry.f16`, which
exist only under `MYNAH_QMAT_F16` — and `-DMYNAH_DISABLE_SIMD` switches that
off. Identical failure on macOS/clang 17 and Linux/gcc 15.2. Introduced by
`d1ffd01`, long before this job existed. It is the `linux-production.md` trap 10
shape exactly: a symbol used outside the guard that defines it, in the one
configuration nobody builds. Three entries in the matrix carry
`known_broken: true` and `continue-on-error`; **that flag comes out in the same
commit that fixes `qmat.c`**, or the entry stops meaning anything.

**B. Every non-`-march=native` ARM profile fails `--self-test` on Linux/gcc.**

```
qmat self-test failed: qmat f16 pack disagrees with the scalar reference
at 4124: 66088.4844 -> 0xb01a, reference 0x7c00
```

66088.48 is past the f16 maximum (65504); the reference saturates to `+inf`
(`0x7c00`) and the pack produces `0xb01a`, which is not a saturation of anything.
Reproduced on `SIMD=portable`, `SIMD=neon`, `SIMD=portable BLAS=none`, and both
`-march=armv8-a` baselines. **`SIMD=auto` (`-march=native`) passes, and
macOS/clang passes every one of the same profiles** — so it is a baseline-ARM
codegen difference in the f16 pack, at the overflow boundary that
`dtype-and-fallbacks.md` §5 says the self-test was written to cover. The
self-test earned its place again; the ISA it was earning it on was the wrong one.

This is why the link-only job runs `--version` and `--dispatch-map` and **not**
`--self-test`: the item asks for link-only, and adding the runtime check would
paint the job red for a defect belonging to another file. Add `--self-test`
there in the same commit that fixes `qmat.c`.

Everything else links and starts, on both machines:

| profile | macOS arm64 | Linux aarch64 |
|---|---|---|
| `SIMD=portable` / `neon` / `auto` | link ok | link ok |
| `BLAS=scalar` / `none` / `openblas` | link ok | link ok |
| `SIMD=portable BLAS=none` | link ok | link ok |
| `SIMD=portable EXTRA_CFLAGS=-march=armv8-a` | link ok | link ok |
| `SIMD=portable BLAS=none -march=armv8-a` | link ok | link ok |
| `SIMD=scalar`, `SIMD=scalar BLAS=scalar` | **fails to compile** (A) | **fails to compile** (A) |

`SIMD=avx2` / `SIMD=avx512` are in the CI matrix and **were not run by this
lane**: there is no x86 host here.

---

## 6. E4-18 — measured, and therefore not written

The item predicts glibc's mmap threshold costs us what it cost the reference:
78 mmap + 154 munmap and 11,899 allocations per request, cured by a per-stream
bump arena. It is a prediction, so it was measured before it was believed.
An `LD_PRELOAD` counter on the box (`malloc/calloc/realloc/free/posix_memalign/
mmap/munmap`), Magpie `fake-magpie`, 5 tokens:

| | reference, before | reference, after | **mynah-tts, measured** |
|---|---|---|---|
| allocations / request | 11,899 | 41 | **2,786** (2,608 malloc + 178 calloc) |
| `mmap` / request | 78 | 1.3 | **3** (35 total, 32 of them process startup) |
| `munmap` / request | 154 | 1.3 | **2** |

And the decisive one — count against `--max-steps` 20 / 40 / 80 / 160:

```
malloc 2608  calloc 178  realloc 18  mmap 35  munmap 2      (all four runs)
```

**The count does not move over an 8× range in frames**, while total bytes do.
The allocations are per-request and per-stage; the autoregressive loop allocates
nothing, which is what `CLAUDE.md` requires and what an arena would be built to
achieve. `BLAS=none` and `BLAS=scalar` are lower still (1,962 + 138, **2 mmap**);
the 646 extra allocations and 33 extra mmaps under `BLAS=openblas` are
OpenBLAS's own, and they leave with it under E4-16.

The structural reason we do not have their problem: `src/codec_nanocodec.c`
allocates one `columns_workspace` per decode and passes it into
`mynah_conv1d_causal`, so the im2col scratch is reused instead of re-allocated
per convolution. That is the data-structure fix they arrived at, already in
place.

**E4-18 is closed as not-applicable**, not deferred. A per-stream bump arena
here would be an unmeasured change against a problem this tree does not have.
Recording the numbers so nobody re-runs the hunt. If the decode gang or a
resident server later moves allocations into the frame loop, the instrument is
eight lines of `LD_PRELOAD` and the test is the constant-count check above.

---

## 7. Gates

Both machines: clean build, **no new warnings** (macOS 3 warnings before and
after; Linux gcc 15.2 22 lines / 13 unique, byte-identical sets),
`--self-test` PASS, `make goldens` 8 checks + batch parity PASS, `make
window-test` all cases, `make driver-test` PASS, `make simd-auto-test` 23/23.

PocketTTS `models/pocket-en`, seed 42, 40 steps, default and `MYNAH_QUANT=int8`:
**byte-identical to `d7281b9` on both machines**, checked by building pristine
`d7281b9` beside the change and hashing. (The macOS and Linux hashes differ from
each other, as the repo's numerical contract allows and requires.)

**Not claimed: any RTF, TTFA or throughput number.** This lane measured counts,
resolutions and link results only.

---

## 8. What this lane did not do, and who has to

1. **`src/qmat.c:1798`** — three lines to put the `QMAT_F16` branch of
   `qmat_weight_rel()` behind `#if MYNAH_QMAT_F16`. Unblocks `SIMD=scalar`
   everywhere; then delete `known_broken` from the three CI entries.
2. **`src/qmat.c` f16 pack at the 65504 boundary on baseline ARM/gcc** — §5B.
   Then add `--self-test` back to the link-only job.
3. **`src/threads.c`** — `blas.thread_timeout`, `blas.threads_owned`,
   `pool.spin` and `pool.decoder_lane` resolve UNKNOWN on Linux (0 UNKNOWN at
   `d7281b9`). The rows and their reasons are already written; they need the
   predicates the reasons name.
4. **Every x86 ISA claim in §2 is untested on hardware.** The fixtures test the
   resolution table, not a CPU. The first x86 evidence will come from the
   `SIMD=auto resolution on this runner` step now in CI, and a real number needs
   the EPYC box.
