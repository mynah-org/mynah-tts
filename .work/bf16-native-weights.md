# E4-8 — the checkpoint is bf16, the CPU has bf16, and we use neither

Item: `PLAN.md` E4-8. Files it would touch: `src/weights.c`, `src/qmat.{c,h}`,
`src/kernels.c` (the ISA inventory), `src/dispatch.c` (the row's reason).

Status: **ANALYSED, not implemented.** Everything below is either read out of the
code or measured on the production box; the projected speedups are arithmetic and
are labelled as such.

## What the binary says about itself

`mynah-tts --dispatch-map` on the Axion:

    IDLE HARDWARE: 5 units this CPU has and this binary has no kernel for:
      isa.arm.sve  isa.arm.sve2  isa.arm.svei8mm  isa.arm.svebf16  isa.arm.bf16

    isa.arm.bf16   compiled=no  supported=yes
      "no bfdot/bfmmla and no bf16 weight type; src/weights.c widens bf16 to f32
       at load and f32 paths stay f32"

That row has been printing the answer for days. It took being asked whether
anything from the qwen-tts flag set transfers here to go and read it.

## Four of the five are not worth writing, and one measurement says so

    $ cat vl.c   # svcntb() on the box
    SVE compiled in, vector length = 128 bits
    HWCAP2 bf16=1 i8mm=1 svebf16=1 svei8mm=1

**Neoverse-V2's SVE is 128 bits wide — the same width as NEON.** So `sve`,
`sve2` and `svei8mm` are not a widening on this host: they would buy predication
and vector-length-agnostic code, which is portability and tail handling, not
throughput. `svebf16` is the same instruction as `bf16` through a different
register file. Writing three SVE kernels to chase "five idle units" would have
produced approximately nothing, and the one-second program above is what says so
rather than a guess about a microarchitecture.

**What is left is `bf16`, and it is the one that changes the arithmetic.**

## What happens today to a bf16 checkpoint

`models/pocket-en/model.json` declares `"dtype": "bfloat16"`. The path from the
file to a multiply, read out of the code:

    file: bf16, mmap'd, 2 bytes/weight
      -> ingot_st_to_f32()        malloc'd f32 copy, 4 bytes   src/weights.c:122
      -> qmat_f16_pack()          f16 copy, 2 bytes            src/qmat.c:354
      -> kernel                   vld1q_f16 -> vcvt_f32_f16
                                  -> vfmaq_f32                 src/qmat.c:1716

Three representations to arrive back at the width we started from, and the
multiply is an f32 FMA: **4 MACs per 128-bit instruction**.

`BFMMLA` (`vbfmmlaq_f32`) takes bf16 inputs and produces a 2x2 f32 accumulator
tile: **16 MACs per 128-bit instruction, accumulating in f32**. `BFDOT` is 8.

That f32 accumulation matters for a reason already on the board: `FMLAL` was
rejected because it needs the activation narrowed to f16, which is what the
weight spec exists to prevent. BFMMLA does not narrow the accumulator. It is a
different instruction with a different property, and the earlier rejection does
not cover it.

## Where it would and would not help

The batched linear has two regimes and they want opposite things.

| | rows | arithmetic intensity | bound by |
|---|---|---|---|
| AR step | 1-6 | one activation per weight block | **memory** |
| text prefill | 16 (`TAR_PREFILL_TILE`) | sixteen | **compute** |

E9-11 measured the batched f16 kernel memory-bound at batch 3 and *used* that
fact: it spends a quarter more arithmetic to halve weight traffic and wins. So
BFMMLA buys the AR step nothing — bf16 and f16 are both two bytes, the traffic is
identical, and the step is not waiting on the multiplier.

The prefill is the opposite. One `[4096][1024]` projection at 16 rows is 67M MACs
against 8 MB of weights, about 8.4 MACs per byte. Two threads at roughly 8
MACs/cycle give order 48 GMAC/s while 35 GB/s of bandwidth can feed order 280
GMAC/s at that intensity. **The prefill is compute-bound by roughly six times**,
which is exactly the condition under which a 4x arithmetic instruction is worth
having.

## The honest size of the prize

E9-12 refused int8-in-the-prefill on this arithmetic: `prep.prefill_proj` is
**35.7 ms, 4.9% of `request.total`**, so 1.5x on it is 1.6% of a request. The
same discipline has to be applied here, and the answer is different only because
the question changed.

That 4.9% was measured on a short text and against *request total*. What binds
the product now is **TTFA under load**, and the number that sets it is the
prefill of a LONG text: **190 ms at C1** (E10-16). Projections scale with tokens,
so on a ~80-token text the projection half is order 110 ms of that 190. A 2-4x
there is **55-80 ms off the longest prefill**, and TTFA p95 currently sits at
**492 ms against a 500 ms gate** — the binding gate of the whole campaign.

It is also the one lever that pays twice: less prefill work per request means
less prefill work per AR step, which is what `MYNAH_PREFILL_STEP_MS` is rationing
and what stops C98 today.

Two more effects, smaller but free:

* **the f32 intermediate disappears.** E9-1 measured the first-call pack
  converting "302 MB of f32 into 151 MB of f16" for the backbone. A bf16-native
  path reads the mapped file directly: no malloc'd f32 copy, no pack pass.
* **no re-encoding.** Today is `bf16 -> f32 -> f16`. It is lossless (f16's 10
  mantissa bits hold bf16's 7, and no weight in this pack leaves f16's exponent
  range) so this is *not* a quality argument — it is two conversions that a
  bf16-native path simply does not perform.

## What would have to be written

1. **A bf16 weight type.** `src/weights.c` currently has one non-f32 answer:
   convert to f32 and cache. It needs to be able to hand back the mapped bf16
   bytes untouched, which means a dtype on `mynah_tensor` rather than an implied
   f32.
2. **A scalar reference first**, per the coding rules: bf16 x bf16 -> f32 with
   the same blocking as the NEON kernel, so the vector path has something to be
   equal to.
3. **The NEON BFMMLA kernel** in `src/qmat.c`, behind the same runtime probe
   `i8mm` already uses — compiled unconditionally, selected by a CPU check, never
   a compile-time gate, so one binary still runs everywhere.
4. **The inventory row.** `mynah_kernels_isa_kernels()` is what makes
   `--dispatch-map` print `compiled=yes`; the reason string in `src/dispatch.c`
   has to stop saying NOT IMPLEMENTED.
5. **x86.** The rule is that no kernel is done until both ISAs exist. AVX512-BF16
   (`VDPBF16PS`) is the counterpart and `cpu_has_bf16()` already probes for it on
   x86. It may be written blind, as the VNNI kernels were, but it must exist and
   self-test.

## Acceptance gate

* `--self-test` compares the BFMMLA kernel against the scalar bf16 reference at
  every width 1-17, by `memcmp` on the f32 output, and mutation-tests both.
* Audio is **bit-identical** to the f16 path or the difference is explained and
  bounded: the same weights at the same precision through a different
  instruction order may differ in the last ULP, which the project's numerical
  policy allows, but a *systematic* difference is a bug and not a rounding story.
* `prep.prefill_proj` on the box, paired, same build otherwise. **Re-take on the
  Axion before any number here is quoted** — E9-11's 1.62-1.71x is a macOS
  number that has never been re-taken on the production target, and this note
  must not repeat that mistake.
* The serving gate that actually matters: a 30-minute C98 soak. C98 fails today
  on 12 stalls in 18008 with TTFA p95 497.6; if the prefill gets cheaper, that is
  the level that should move.

## What this is not

It is not KleidiAI. That was rejected here on specific grounds — 40 files of
hand-written assembly, a second packed copy of every weight, and no committed
head-to-head against their own SMMLA — and none of those grounds have changed.
The transfer from qwen-tts is the **idea** that a bf16 checkpoint on a bf16 CPU
should not be doing f32 arithmetic, not the dependency. qwen-tts states the cost
in its own profile notes: "vnni-product pins f32 prefill, which on a BF16 CPU
costs ~600 ms per admission".
