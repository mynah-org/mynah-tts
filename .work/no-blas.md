# Removing BLAS — decision, and the surface it actually covers

**Decision (author, 2026-09-12):** we do not want OpenBLAS, and ideally no
external BLAS at all. The reason is not performance, it is **ownership**:
OpenBLAS ships its own thread pool with its own policies, and every one of those
policies is a trap you have to remember to check, on every host, forever. The
same removal is a planned future task on qwen-tts.

This note records how big the job actually is here. The answer is much smaller
than it is there, and it gets smaller still if we do it in the right order.

Related: [`linux-production.md`](linux-production.md) §BLAS oversubscription,
[`cpu-kernels-arm-x86.md`](cpu-kernels-arm-x86.md).

---

## 1. The surface, measured by grep, not by memory

**One BLAS function is used in this repo: `cblas_sgemm`. Three call sites.**

| site | what it serves | reached by |
|---|---|---|
| `src/backend.c:377` | the backend vtable's `sgemm` | **everything** that goes through `mynah_graph_sgemm` — `conv1d.c` (2), `engine_magpie.c` (7) |
| `src/backend.c:177` | `matmul_block`, row-blocked multi-row matmul split over our pool | Magpie prefill/encoder |
| `src/seanet.c:53` | `sea_sgemm`, the conv/convtranspose GEMM fast path | **the PocketTTS decoder**, 2 call sites (`:295`, `:538`) |

Plus a fourth thing that is not BLAS but travels with it: the **BNNS filter cache**
in `conv1d.c`, Accelerate-only, therefore macOS-only, therefore not on the
production path at all.

### What this means for PocketTTS specifically

The PocketTTS graph does **not** go through `mynah_graph_sgemm`. Its backbone and
flow-head linears go through `mynah_qmat_linear` / `mynah_qmat_linear_batched` —
our own kernels, our own ISA dispatch, our own pool. Checked:
`engine_pocket.c:622,631,639,682`.

> **The entire BLAS dependency of the PocketTTS production path is
> `sea_sgemm`, two call sites in `seanet.c`.**

**Correction (2026-09-12, from building it):** that holds for the *quantized*
path. `engine_pocket.c:1166,1237` can still reach `mynah_backend_matmul` for an
unquantized projection, so `sea_sgemm` is the whole f32-GEMM surface only when the
groups are doing their job.

Everything else in the table above is Magpie's, and Magpie is not what we are
optimising for Linux servers.

That is the finding that changes the size of this task. "Remove BLAS" sounds like
replacing a library; here it is **writing one f32 GEMM for two known call sites.**

---

## 2. Why it is load-bearing today

Two reasons, and only one of them is real.

**Real:** the SEANet conv stack was 76.4% of wall at RTF 2.03. Rewriting it as one
GEMM per kernel tap took it from 8570 ms to 237 ms — **36×** — and that GEMM is
`cblas_sgemm`. Delete BLAS naively and we give back the single largest win in the
repo.

**Not real, but decisive in practice:** the fallback is a naive `i,j,p` triple
loop (`backend.c:386-399`). It is a *correctness* reference, not an
implementation. So "no BLAS" today does not mean "slightly slower", it means
"unusable", and that is the only reason the dependency looks structural.

Replacing the fallback with a real blocked GEMM is therefore the whole task. There
is no other coupling.

---

## 3. The shapes we have to be good at — measured, and not what I assumed

**Correction.** This section previously said the shapes were "skinny and long",
that `n` large meant the inner loop had plenty of work and the packing would
amortise, and that this was explicitly not the square-GEMM case where a general
BLAS earns its keep. The first and third points survive. **The second is wrong**,
and the histogram says so.

Instrumented over one PocketTTS utterance, 43 frames, **1075 GEMM calls, 11
distinct shapes** (counts are `43 x kernel_size`; 25 GEMMs per frame, 22 conv1d
and 3 convtranspose):

| calls | site | m | n | k | transA |
|---|---|---|---|---|---|
| **301** | conv1d | 512 | **16** | 512 | 0 |
| 129 | conv1d | 64 | 480 | 128 | 0 |
| 129 | conv1d | 32 | 1920 | 64 | 0 |
| 129 | conv1d | 128 | 96 | 256 | 0 |
| 129 | conv1d | 1 | 1920 | 64 | 0 |
| 43 | convtr | 512 | 480 | 128 | 1 |
| 43 | convtr | **3072** | **16** | 512 | 1 |
| 43 | convtr | 1280 | 96 | 256 | 1 |
| 43 | conv1d | 64 | 1920 | 32 | 0 |
| 43 | conv1d | 256 | 96 | 128 | 0 |
| 43 | conv1d | 128 | 480 | 64 | 0 |

The single most frequent shape — **28% of all calls** — is `m=512, n=16, k=512`,
and the widest is `m=3072, n=16, k=512`. That `n=16` is the **frame batch**, not
a long axis. These are narrow-RHS matmuls, much closer to a handful of matvecs
than to a panel GEMM, and no packing cost amortises over sixteen columns.

**Consequence for `mynah_sgemm_f32`: `n=16` and `n=1920` are two different
kernels, not one tuned shape.** The `n=16` family wants the weights streamed once
with the sixteen columns held in registers — which is the same weight-stationary
argument the batching lane makes for the codec transformer, on the same tensors.
The `n=480/1920` family is an ordinary panel GEMM.

Note also `m=1, n=1920, k=64`: 129 calls of a single output row. That is a matvec
wearing a GEMM's clothes and should never reach a packed kernel at all.

## 4. What removal deletes as a bonus

All of this exists only to compensate for a thread pool we do not own:

- `src/threads.c:232-305` — `PF_HAVE_BLAS_KNOB`, the weak `openblas_set_num_threads`,
  the clamp-on-region-entry, the `OPENBLAS_NUM_THREADS`-wins rule, and the
  `mynah_blas_owned()` predicate that answers "did the clamp actually happen?";
- `src/dispatch.c:466-568` — three dispatch rows (`blas.accelerate`,
  `blas.openblas`, `blas.threads_owned`), one of which is an *unknown* because the
  honest answer is "we cannot tell from here";
- `src/backend.c:377` — the `mynah_blas_set_threads(mynah_num_threads())` call on
  every codec GEMM, whose comment says it exists because that path "bypasses the
  pthread pool";
- the Makefile's `BLAS ?= auto` detection and its three-way `Accelerate / OpenBLAS
  / scalar` split, with three build configurations to test instead of one;
- from [`linux-production.md`](linux-production.md): `OPENBLAS_THREAD_TIMEOUT=1`
  (TTFA 108 ms **bimodal** → 66 ms stable, 42.5k → 12k context switches/s) and the
  21%-of-time-in-the-scheduler oversubscription trap. Both stop being our problem.

That last one is the point the author made: it is not that OpenBLAS is slow, it
is that **a second thread pool inside our process is a permanent source of
non-reproducible measurements**, and every profile has to carry env vars that
must be *absent* to be valid.

---

## 5. Sequencing — do it in this order, not the obvious one

1. **int8/int4 first.** Every linear that moves to qmat leaves the f32 GEMM
   surface. On PocketTTS that has already happened for the backbone and the flow
   head; the remaining f32 is the SEANet decoder. Quantising the decoder shrinks
   the problem before we solve it.
2. **Dump the shape histogram** (§3). Two call sites.
3. **Write `mynah_sgemm_f32`** — packed, register-blocked, dispatched on our own
   pool, NEON + AVX2/AVX-512 + scalar reference, `--self-test` against the triple
   loop. This replaces `backend.c`'s fallback, so it also serves Magpie for free.
4. **Flip the default to `BLAS=none`** and keep `BLAS=openblas` / `Accelerate` as
   a *comparison* build only, so the A/B is always available and never the
   default.
5. **Delete** the compensation machinery in §4 once the comparison build is the
   only thing that references it.

Steps 3 and 4 are separable from everything else in E4/E5 and do not block them.

---

## 5b. Landed — the kernel, not the default (`fa3df67`)

`src/sgemm.c` exists: four families chosen by one function the runtime and the
report both call, a scalar reference as the definition of correctness, and a
narrow/panel boundary **derived** from the register file rather than chosen.
`BLAS=none` is a first-class build that links neither Accelerate nor OpenBLAS.
The default is unchanged and byte-identical.

**The numerical result inverted the expected direction.** Isolated against the
in-tree scalar convolution reference, with two controls proving the GEMM is the
only thing that changed:

| | max abs | samples differing |
|---|---|---|
| **our GEMM**, f32 | **1 LSB** | 0.12% |
| **our GEMM**, int8 | **1 LSB** | 0.16% |
| Accelerate, f32 | 8 LSB | 18.5% |
| Accelerate, int8 | 360 LSB | 94.5% |

Twenty to nine hundred times closer to the reference than the incumbent, which has
been shipping since E3-4.

**Two things the default flip must not ride along on.** Dropping Accelerate also
drops vForce's `vvtanhf` in the GELU for libm's `tanhf` — on a continuous-AR model
that compounds, and it is a separate numerical qualification. And the BNNS filter
cache in `conv1d.c` goes with it. Neither is a GEMM decision.

**A false gate worth remembering:** `make ubsan` and `make ubsan BLAS=none` wrote
to the same directory, so the second run re-tested the first one's
Accelerate-linked binary and printed a green PASS on top. The sanitizer directory
now carries the BLAS name and the run prints its link count as proof.

---

## 6. Acceptance gate

- a default `make` on Linux x86-64 and ARM64 links **neither** `-lopenblas` nor
  `-framework Accelerate`, and `make info` says so;
- SEANet oracle parity unchanged (currently 5.7e-07 against a 1e-4 budget) and
  chunked-vs-one-shot unchanged (2.98e-07);
- `mynah_sgemm_f32` self-test against the scalar triple loop passes on every
  compiled ISA;
- **RTF not worse than the BLAS build** on both ARM and x86 Linux, measured with
  the §10 protocol in [`serving-design.md`](serving-design.md) — a WAVE screen is
  not enough to promote this;
- the dispatch report loses three rows and gains one: which `sgemm_f32` kernel
  resolved, and why.

**Open risk, stated honestly:** a hand-written GEMM beating OpenBLAS on the
*general* case is hard. Beating it on two known skinny shapes, with the packing
done once and the threading already ours, is normal work — but it is real work,
and if the measurement says we lost, the answer is to keep the comparison build
and say so, not to ship a regression for architectural tidiness.
