# The first Linux ARM run — what the production target said back

Status: **DONE (correctness)** · measured 2026-09-13 on GCP Axion c4a,
Neoverse-V2, 32 cores SMT off, 80 MiB L3, one NUMA node, gcc 15.2, OpenBLAS.
Companion: [linux-production.md](linux-production.md) for what to expect from
the box, [dtype-and-fallbacks.md](dtype-and-fallbacks.md) for the previous
round of the same class of bug.

**No performance number here, by instruction.** The box is shared and the
capacity campaign is a separate job. Everything below is correctness and
dispatch resolution.

---

## 1. The headline: batching was not deterministic on ARM hardware that has i8mm

`--self-test` failed on Linux aarch64 at HEAD:

```
qmat batched qtype=1 differs at row 0 col 10: -1.29924119 vs -1.29924107
```

qtype=1 is **INT8**, not f16. The one-ULP look of it is misleading, and the
first instinct — "another E4-20, the test is wrong" — was wrong. Three
measurements settled it, all on the box, 96x256 INT8, batch 1..6:

**(a) The integers are exact.** With `scales = sx = 1` and `bias = NULL` the
float epilogue is the identity on the int32, and SMMLA agrees with SDOT on
**0 of 96 rows differing**. The 2x2 outer product accumulates the same products
in a different order and integer addition does not care. That half of the
kernel's comment was always true.

**(b) The floats are not.** Same shape, real scales: **71 of 192 rows differ,
up to 2 ULP**. `(float)s * scales[row] * sx + bias[row]` is a three-factor
product, `-ffast-math` implies `-fassociative-math`, and GCC 15 groups the four
copies in `matvec_q8_pair_i8mm`'s epilogue differently from the one in
`matvec_q8`. The existing `self_test_i8mm_identity` asserted float bit-identity
and **passed** — because 13x37 is too small a shape to reach the divergence. A
gate that passes because it never drives down the road is what let this ship.

**(c) And therefore a row's answer depended on where it sat in the batch.**
This is the part that makes it a defect rather than an ULP argument. One
activation row, fixed weights, four different answers:

| where the row was | differs from the row alone |
|---|---|
| alone (batch 1, the early return) | — (it *is* the reference) |
| lead of an SMMLA pair (even position) | 3 of 96 cols, 1 ULP |
| follower of an SMMLA pair (odd position) | 5 of 96 cols, 1 ULP |
| SDOT tail of an **odd** batch | **39 of 96 cols, 2 ULP** |

Position in the batch is decided by arrival order under concurrent serving. So
a request's audio depended on who else was in flight — the one thing the
weight-stationary batched linear promises not to do. Not a tolerance question.

**(d) The generic path is exactly right.** With the SMMLA wiring off, the same
sweep gives **0 differing columns at every position and every batch size**,
including the odd tail and the 32-row blocking. So the guarantee is achievable
and is already achieved; SMMLA was the sole defect.

### Why the fix is a gate and not a repair

E4-20 already closed the repair. Hoisting the epilogue into a shared helper
does not work — the compiler inlines and then reassociates per site. Forcing
one order with `noinline` would change `matvec_q8`'s codegen, and `matvec_q8`
is the kernel the int8 goldens are frozen against. There is no source-level
spelling of "these two kernels must round identically" under `-ffast-math`.

So `matvec_q8_pair_i8mm` stays compiled and self-tested, and stays out of the
path that must be deterministic. `MYNAH_QMAT_I8MM=1` wires it in for a future
measurement, and `--self-test` then fails **on purpose**. The
`isa.arm.i8mm` dispatch row says all of this in words rather than reporting ON.

**The lesson, which is the same one twice now:** two textually identical
epilogues are not one epilogue. Determinism comes from *one compiled kernel*,
not from careful transcription. Any future batching kernel — bf16/BFMMLA is the
obvious next one — inherits this and must be checked with
`self_test_batch_membership`, which varies the batch a row sits in and asserts
bit-equality with the row alone. Verified to catch the defect: with the wiring
forced on and `self_test_batched` disabled, it fires by name.

## 2. The server did not compile on Linux

`server/http_util.c:20: implicit declaration of pthread_setname_np`. glibc
hides it behind `_GNU_SOURCE`; macOS declares it unconditionally. One guarded
`#define` at the top, the same one `server/prefork.c` already carried. Swept
the tree for the rest of the class — `sched_setaffinity`/`CPU_SET` are in
`prefork.c` which is already guarded, `clock_gettime`/`strcasecmp` are covered
by the Makefile's `-D_DEFAULT_SOURCE`, and `memmem` is our own. Nothing else.

## 3. One dispatch row lied under BLAS=none

`codec.seanet_gemm` read the compiled column as `Accelerate || OpenBLAS`, which
predates `src/sgemm.c` serving `BLAS=none`. A `BLAS=none` build reported
**compiled=no on a row that then resolved ON**. Added
`MYNAH_DISPATCH_HAS_OWN_SGEMM`.

## 4. What the box answered — first time these predicates have run on production silicon

Default build (`BLAS=auto` → OpenBLAS), 46 rows, **0 UNKNOWN**, gate canary ok.

- `quant.int8_kernel` → **`neon-sdot`**, not i8mm. That is correct and not a
  disappointment: the row describes `matvec_q8`, the single-vector decode
  matvec, where SMMLA would waste half its lanes by design. i8mm has its own
  row.
- `isa.arm.i8mm` → compiled yes, supported **yes** (first time), resolved
  **OFF**, with §1 as the reason.
- `quant.f16` → **`neon`** (vcvt_f32_f16, four rows per activation load).
- `isa.arm.bf16` → compiled **no**. The hardware has bf16/svebf16 and we have no
  kernel. That is the honest state and the next ISA item.
- `sve2` / `svei8mm` / `svebf16` are in `/proc/cpuinfo` and have **no row at
  all** — we emit nothing for SVE. Worth a row that says NOT IMPLEMENTED
  rather than silence.
- `pool.threads` → 32; `blas.threads_owned` → ON.

`BLAS=none`: `sgemm.provider` → `mynah`, `codec.seanet_blas` → `mynah-sgemm`,
self-test PASS, 0 UNKNOWN. `BLAS=openblas` explicit matches `auto`.

## 5. Gates

Linux ARM: `make` clean, `make server` (now), `--self-test` PASS, `make
goldens` **8/8 + batch parity PASS**, `make window-test` PASS, `make
driver-test` PASS, PocketTTS EN synthesis 59520 frames @ 24 kHz, peak 21139,
under both OpenBLAS and BLAS=none. macOS: same gates, plus PocketTTS output
**byte-identical to CANONICAL_HEAD across f32, int8 and f16**, and UBSan clean
(122 checks).
