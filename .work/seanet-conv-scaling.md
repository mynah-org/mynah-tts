# codec.conv_stack: the quarter of the wall that did not divide

Pinned at `1a93cbafa41da8bbafae7d1c47a191744cd94b06`, branch
`docs/plan-board-pocket-tts`.  Machine: 32x Neoverse-V2, gcc 15.2, `SIMD=auto`
and `SIMD=portable`, `BLAS=none/mynah-sgemm`.  Files: `src/seanet.c`,
`src/sgemm.c`, `src/sgemm.h`.

## Problem

One stream, `MYNAH_COST_MAP=2`, 59 decode calls:

| region | 1 core | 16 cores | speedup |
|---|---|---|---|
| `codec.transformer` | 678.6 ms | 150.7 ms | 4.50x |
| `codec.conv_stack` | **609.3 ms** | **238.3 ms** | **2.56x** |
| `step.backbone` | 568.0 ms | 119.7 ms | 4.74x |

Reproduced exactly.  Amdahl on the region total says ~35% of it is serial and
names no line, which is why the first thing built here was instrumentation.

## Evidence, not a reading of the loop

Two profilers, both env-gated and off by default, both shipped:

* `MYNAH_SEANET_PROFILE=1` -> `[SEANET-PHASES]` (src/seanet.c).  Splits
  `codec.conv_stack` by WHO RUNS THE LOOP: a phase either dispatches to the
  pool or runs on the calling thread.  The table is the ceiling, read off.
* `MYNAH_SGEMM_PROFILE=1` -> `[SGEMM-SHAPES]` (src/sgemm.c).  The shape
  histogram AS EXECUTED, with per-shape wall and the task count each call asked
  the pool for.

### The three candidate mechanisms, decided

**1. The dependency chain — NOT the limit.**  The stack is sequential across
layers, but the individual layers are not too small to fill sixteen cores: the
three transposed convolutions (1.77-1.83 ms each on one thread) get 11.5-14.3x
from sixteen.  Shape, by itself, is not the ceiling.

**2. The panel/narrow family — NOT wrong.**  The histogram as executed confirms
sgemm.h's documented 28%: `m=512 n=16 k=512` is 413 of 1475 calls (28.0%), and
it takes `MYNAH_SGEMM_FAMILY_NARROW` as designed.  The kernel choice is right.

**3. The dispatch count — REAL, and measurable to a number.**  25 GEMM
dispatches per frame, 1475 per utterance, mean 51 us per call.  Every one is
below the 200 us the pool's own header says a region must be worth.  Fitting
`t16 = overhead + t1/16` over the eleven measured shapes gives **26-44 us of
fixed pool cost per dispatch on this box** (not the ~20 us the header quotes),
so ~52 ms of the 235 ms region is barrier.  The clean two-sided proof is in the
histogram: the three calls worth ~1.8 ms on one thread get 11.5-14.3x; every
call worth 110-170 us gets 2.4-4.7x; and `m=1 n=1920 k=64` was actually SLOWER
on the pool (11.07 us -> 21.15 us).

**4. And the mechanism nobody listed, which was bigger than all three.**  At
sixteen threads, **66% of the region was element-wise glue on the calling
thread**:

| phase | par | 1 core | 16 cores | speedup |
|---|---|---|---|---|
| `elu` | no | 76.8 ms | **77.0 ms** | **1.00x** |
| `conv.gather` | no | 71.6 ms | **57.8 ms** | 1.24x |
| `conv.gemm` | yes | 150.3 ms | 54.4 ms | 2.76x |
| `convtr.gemm` | yes | 299.7 ms | 25.5 ms | 11.7x |
| `conv.bias` | no | 1.5 ms | 6.6 ms | **0.22x** |
| rest (serial) | no | 14.4 ms | 13.4 ms | ~1x |

`elu` is a scalar `expf` loop over 530k activations per frame that nothing ever
dispatched.  `conv.gather` re-gathers one kernel tap of the weight into a dense
matrix **on every frame** — 1.96M gathered floats per frame, of which the entry
convolution's 7 x 512 x 512 is 93.6%, and the gathered values depend only on
weights that never change.  `conv.bias` got 4.5x SLOWER with sixteen threads
because the caller re-touches output lines sixteen workers just dirtied.

## Fix — fewer, larger regions; no new arithmetic

1. **`mynah_sgemm_f32_conv_taps` (src/sgemm.c).**  A causal conv1d with kernel K
   was K pool dispatches, each needing a dense op(A) gathered first.  Now it is
   ONE region: a task owns a block of C's ROWS across every column group and
   every tap, gathers its own rows, and multiplies them while they are still in
   its L1.  K dispatches become one and the gather becomes free.
   Byte-identical, not "agrees to 1e-6": the gather is a copy; the row plan is
   made from (m, n, k) for ONE tap exactly as the K calls made it, so every
   output element keeps its micro-kernel instantiation; the tap loop stays
   outermost so the beta chain still rounds each tap to f32.
   `mynah_sgemm_self_test()` asserts `memcmp` against the K-call sequence over
   eight shapes x two betas and refuses if the fused path never ran.
   **Guarded by `MYNAH_SEANET_OWN_SGEMM`**: on an Accelerate or OpenBLAS build
   `sea_sgemm` is that library's `cblas_sgemm`, and routing taps through our
   kernels instead would be a numerical change on macOS wearing a scheduling
   change's clothes.
2. **Parallel ELU (src/seanet.c).**  Element-wise, disjoint writes, so
   bit-identical by threads.h's own contract.  Thresholds derived from the two
   measured numbers (35 us per region, ~2.5 ns per element): parallel above 16k
   elements, tasks no smaller than 8k, chunks rounded to 64 floats so no
   element moves between a vector body and a scalar remainder.
3. **`want == 1` now means what it says (src/sgemm.c).**  A GEMM below
   `SG_PARALLEL_MIN_WORK` still dispatched, because the COLUMN grid is planned
   from the shape and produced 8 tasks for the m=1 matvec.  Those tasks now run
   inline, in order — same task code, disjoint outputs, bit-identical.

## Result

`codec.conv_stack`, paired interleaved runs, median of within-round ratios:

| threads | HEAD (median) | lane (median) | median of within-round ratios |
|---|---|---|---|
| 1 | 625.9 ms | 624.0 ms | 1.00 |
| 8 | 243.3 ms | 124.5 ms | 1.95 |
| 16 | 241.6 ms | **103.7 ms** | **2.30** |
| 32 | 268.9 ms | 113.2 ms | 2.37 |

Scaling 1 -> 16 cores: **2.59x -> 6.02x** (the target was 4.3x, its neighbour's).
Re-taken on the exact shipped binary while another lane had ~9 cores of the
box: base 253.9 ms, lane 115.4 ms, median ratio **2.25** -- the paired protocol
holds the ratio across load 0 to load 9 (2.27 / 2.28 / 2.30 / 2.25 in four
sessions), which is the whole reason it is paired.

Whole request, `request.total`, 16 threads, quiet box: 834.6 ms -> 684.6 ms,
**-18% of total wall**; RTF 0.186 -> 0.155.  Under contention the same pair
compresses to 1.09, because a contended parallel region inflates on both sides;
the 1.22 is the number that belongs to a dedicated box.
Serial fraction of the region: 66% -> 23%.

Note for whoever owns the pool: **HEAD gets nothing from 8 -> 16 cores on this
region (243 -> 240 ms) and both builds REGRESS from 16 -> 32 cores.**  That is
the barrier growing with width and it is not in these files.

## What is left, and why it stops here

23% of the region is still serial: `convtr.scatter` 7.3 ms, `conv.bias` 6.9,
`residual_add` 4.1, `conv.window` 2.4, `convtr.copy` 2.0, `convtr.fill` 0.8.
Each individual call is 14-42 us, i.e. **below the measured 35 us dispatch
cost** — the pool cannot help them.  Folding them into a neighbouring GEMM
region would work but changes where the f32 rounding happens (the bias would go
from two roundings to one), so it is a numerical change and needs its own
qualification, not a performance argument.  `convtr.scatter` additionally
cannot be split on the GEMM's row blocks at all: rows of the tap matrix that
share an output channel overlap-add into the same positions.
