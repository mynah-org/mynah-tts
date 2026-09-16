# `step.backbone` is 49% of the wall and eight cores buy 15% of it

Measured 2026-09-16, this Mac, `BLAS=none`, `models/pocket-en`, seed 1, the
same 11-word utterance throughout. A development signal, not a product claim —
but the *shape* is the finding, and the shape is not host-specific.

## The measurement

`MYNAH_COST_MAP=2`, `step.backbone`, five runs per thread count, median:

| threads | median | min | max | spread |
|---|---|---|---|---|
| 1 | 220.6 ms | 216.5 | 232.4 | 7% |
| 2 | 227.1 ms | 216.6 | 231.4 | 7% |
| 8 | 191.8 ms | 186.0 | 219.8 | 18% |

**1.15× from eight cores**, on the region that is **48.7%** of `request.total`.

## Why, in one number

One AR step reads every backbone weight exactly once:

    6 layers x (qkv 1024x3072 + oproj 1024x1024 + ffn1 1024x4096 + ffn2 4096x1024)
      = 75.5M weights = 151 MB at f16, per step

At the measured 4.2-4.8 ms per step that is **32-38 GB/s**, and it does not
move with the thread count. The step is one activation against 151 MB of
weights: arithmetic intensity is one MAC per two bytes, so this is a memory
wall and not a parallelism problem. Adding cores to it is adding cores to a
`memcpy`.

## It also explains a result already on the board

`PLAN.md` E10's rejected list carries "storing f32 weights instead of f16 —
**2.22× slower on the backbone**, measured". That is exactly what a
bandwidth-bound region does when its bytes double, and it is a second,
independent confirmation of the same wall from a measurement taken for another
reason.

## What follows, and what does not

There are only two levers on a bandwidth wall, and neither is a kernel:

1. **Fewer weight bytes.** int8 halves the traffic against f16, which is why
   `.work` keeps circling it (E3-5c, E9-12). It is blocked on quality and the
   block is real: the backbone is *inside* the AR loop, where a per-step error
   compounds over ~50 steps into a different trajectory, which is the whole
   reason `POCKET_QG_DEFAULT_SPEC` pins it to `:f16`. int4 is further in the
   wrong direction. This is a quality problem wearing a performance problem's
   clothes, and no kernel work moves it.
2. **More activations per weight read.** One weight pass serving B requests
   divides the traffic by B, which is what the weight-stationary batched
   kernels already do and what the serving topology decides. This is why
   `16x2` versus `1x32` (E10-10) is a *bandwidth* question and not a
   scheduling one, and why the simultaneous-worker shape screen is the right
   next experiment rather than another kernel.

**Do not** spend effort on: threading this region harder, a better f16 matvec,
or a different f16 packing (all four candidate layouts already lost — E10's
rejected list). The kernel is not what is limiting it.

## A measurement hazard, recorded because it nearly became a number

The first reading of `step.backbone` taken for this note was **356.0 ms** at
two threads — 1.6× the median of the five-run sweep that followed at the same
thread count on the same binary. It was a single run taken immediately after a
build, so the machine was neither quiet nor warm.

Nothing was published from it, but it would have made "eight cores buy 15%"
read as "eight cores buy 85%", which is the opposite conclusion and would have
sent the next person to thread the region harder. `CLAUDE.md` already says to
measure differences rather than absolutes on a noisy machine; the specific
lesson here is narrower — **a single cost-map reading right after a build is
not a measurement**, and this region's spread is 7-18%, so anything under a
five-run median is noise.
