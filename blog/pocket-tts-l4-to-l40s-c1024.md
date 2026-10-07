---
title: "A 2x GPU that did not scale: taking Pocket TTS from an L4 to 1024 streams on one L40S"
published: false
description: "From a ~740 audio-s/s plateau with the GPU idle 40 % of the time to 1024 concurrent TTS streams at RTF p95 0.746 on one NVIDIA L40S. The GPU was never the problem: one host thread was."
tags: tts, cuda, performance, opensource
cover_image:
---

I rented a GPU roughly twice as fast as the one I had, ran the same build on it, and got almost
nothing. Throughput went flat at ~740 audio-seconds per second, and `nvidia-smi` said the card was
idle 40 % of the time.

This is the story of how that same NVIDIA L40S ended up streaming **1024 concurrent text-to-speech
streams** with the GPU 95 % busy, and why almost none of the fixes touched a GPU kernel.

## TL;DR

On an NVIDIA L4 the Pocket TTS server is GPU-bound (knee at C288-C320). On an NVIDIA L40S the same
build stopped at **~740 audio-s/s** whatever the concurrency. Four days of measuring later, the L40S
holds **C1024 at stream RTF p95 0.746, 1283 audio-s/s,
first audio p95 106 ms, GPU 95 % busy at 342 W** of its 350 W cap. Along the way:

- **`nvidia-smi dmon`'s SM % is a time share, not occupancy.** 60 % means "nothing at all runs 40 %
  of the time". The GPU was not underfilled; it was waiting.
- The serving loop is **one scheduler thread doing host work in series with the GPU work**. On an L4
  that host time is 10-15 % of an iteration; on an L40S it was more than half.
- **Eleven small host-side changes**, each bit-identical behind its own flag, cut stream syncs per
  iteration from 17-20 to ~9.5-11.8. Now on by default.
- The server **stopped silently at ~1000 streams**: not the engine, the 1024 file-descriptor soft
  limit. C1024 had never actually been measured before that was found.
- **Admission was the real ceiling.** Building a fresh request context under load cost ~13 ms;
  renewing a pooled one costs 0.1-0.3 ms. That single change took C1024 from failing to passing.
- On another host, **`cudaFree` at admission synchronised the whole device** (11-30 ms per new
  request). A fixed-size device slot pool with no driver call on take removed it.
- **Ping-pong half-batches** hid 92 % of the host time and were still 7-8 % slower once the GPU
  was busy. Kept as an experimental opt-in.
- **A 4-vCPU affinity experiment** (server pinned to 4 vCPUs with `taskset`) still held C768 on the L40S, about 3 % below the whole host.

All numbers below are 2-minute closed-loop screens unless noted, not 30-minute qualifications. RTF
is synthesis time divided by audio duration, per stream; the gate is **stream RTF p95 ≤ 0.88, zero
stalls at a 250 ms buffer, zero failures**.

---

## The project, in two paragraphs

**[Mynah TTS](https://github.com/mynah-org/mynah-tts)** is a fast native C11 text-to-speech
inference engine, in the spirit of llama.cpp: no Python at runtime, CPU, Metal and CUDA backends, one
binary plus a model pack. It runs Kyutai's Pocket TTS (6-layer and 24-layer English models) among
others, from the command line or through a streaming HTTP server. This post is
about that server on CUDA, with the 24-layer model.

It lives in **[mynah-org](https://github.com/mynah-org)**, a small open-source GitHub organisation
created to build lightweight native engines for the voice stack:
[mynah-tts](https://github.com/mynah-org/mynah-tts) (speech synthesis),
[mynah-asr](https://github.com/mynah-org/mynah-asr) (streaming and offline speech recognition:
Parakeet, Canary, Nemotron), [mynah-slm](https://github.com/mynah-org/mynah-slm) (small language
models: Qwen3, Gemma next) and [ingot](https://github.com/mynah-org/ingot) (a zero-dependency C11
GGUF/safetensors reader). Everything is MIT licensed, and contributions — code, bench results from
your own hardware, bug reports — are welcome.

---

## 1. The puzzle: a bigger GPU that does not get faster

The CUDA path keeps everything resident: weights, KV caches, codec state. Every live stream advances
by one frame per iteration through one batched autoregressive step, then one batched "gang" decode
turns each stream's new latent into 80 ms of PCM, which is streamed to its client. Continuous
batching: a stream that ends frees its row, and a queued request takes it on the next iteration.

On the L4 that design had already been pushed a long way. The history, from `PLAN.md` and
`docs/performance.md`:

| date | L4, 24L model | highest C | audio-s/s | RTF p95 | TTFA p95 |
|---|---|---:|---:|---:|---:|
| 2026-09-28 | first qualified point, 2 x 30-min soaks, WER checked | 160 | 184.5 | 0.855 | 155 ms |
| 2026-10-02 | seven new defaults (shared voice-prefix KV, split decode attention, bf16 Linears through cuBLASLt, fused SEANet decoder, one host sync per frame, per-request memory diet), 10-min soak | 288 | 341 | 0.821 | 141 ms |
| 2026-10-04 | + int8 backbone KV, 2-min screen | 320 | 338 | 0.855 | 146 ms |

On the L4 the GPU is the wall: 86-91 % busy at its 72 W cap, host time 13-15 ms per iteration. More
streams just make every step longer.

Then the first L40S screen, on a small 4-vCPU host with the load generator on the same box:

| build | C | audio-s/s | stream RTF p95 | TTFA p95 | GPU W / `dmon sm` |
|---|---:|---:|---:|---:|---|
| main (384-row compile-time cap) | 384 | 740 | 0.488 | 87 ms | ~241 / 62 % |
| rows raised to 768 | 512 / 640 / 768 | 715 / 718 / 726 | 0.667 / 0.826 / 0.970 | 118 / 147 / 175 ms | 238 / 60 % |

Throughput flat at ~715-740 audio-s/s from C384 to C768, at 238-241 W of 350 W. More rows made each
step longer and added nothing. My first written explanation (still in `docs/performance.md`, under a
"Superseded" banner) was that one batched step does not fill 142 SMs. It was a reasonable story, and
it was wrong.

## 2. Measuring before guessing

### `dmon sm` is a clock, not a ruler

`nvidia-smi dmon`'s `sm` column is the share of time in which **at least one kernel** is running. It
says nothing about how many SMs that kernel uses. So "62 %" does not mean "the kernels use 62 % of the
GPU"; it means "for 38 % of the time, nothing is queued at all". Underfilled kernels would call for
fatter kernels. Idle gaps call for asking who is not feeding the GPU.

### Two engines on one GPU

The cheapest experiment that separates the two readings needs no code: run two servers with 320 rows
each on two ports, with two clients, against one server at 640.

| setup | audio-s/s | RTF p95 | GPU busy | power |
|---|---:|---:|---:|---:|
| one engine, 640 rows | 799-877 | ~0.68-0.79 | ~62 % | ~261 W |
| two engines x 320 rows | 465 + 448 = **913** | 0.68 / 0.70 | **93 %** | 300 W |
| two engines x 320, with MPS | 908 | | | |

Two engines push the GPU to 93 % busy. MPS adds nothing, so there is no hidden concurrency headroom
inside the kernels. **The idle time is between steps**: one engine's GPU work fills the other's host
gaps. That killed two items on the list on the spot (a decode on a second CUDA stream, and two row
groups on two streams), and it set the target: one engine, with the gaps gone, rather than two
engines with smaller, less efficient batches.

### The load generator is not the cap

`tools/pocket_ladder.py` gained `--client-procs N`, and a stand-in server paced at RTF 0.5
(`tools/gpu/stub_stream_server.py`) measures a client's ceiling anywhere: on an M1 one client process
carries C512 (12,800 chunks/s) using 22 % of a core. On the L40S, one and four client processes gave
827 and 808 audio-s/s. Ruled out.

### Where the loop actually spends its time

`MYNAH_SERVE_PROFILE=1` now splits each iteration into **device wait** (time blocked inside a
backend sync) and **host** (everything else), and prints a table of syncs per call site. At C640 on a
~30-CPU host:

- device wait **44.7 %** of the loop, host **55.1 %**, **31.7 ms of host per iteration** while the
  GPU had nothing queued;
- two real GPU waits per iteration: the AR step (~10.8 ms) and the gang decode (~11.7 ms);
- the decoder's CUDA graph was **re-recorded on almost every step**, because gang membership changes
  every iteration (retire, admit, swap-remove) and the exact-shape match failed;
- a second flow-head pass ran whenever any row ended, which with ~5.5 completions per iteration was
  nearly every step.

The model that came out of it: one iteration is about 22-26 ms of GPU work plus about 32 ms of host
work, **in series**. On an L4 the GPU half is 2.5-3x longer, so the same host time is only 10-15 % of
the loop, which is exactly why the L4 runs 85-95 % busy and the L40S does not. A faster GPU does not
shorten the host half. It just makes it visible.

## 3. The host is part of the benchmark

If one host thread sets the pace, the host is part of the benchmark. The same GPU model on
different rented boxes gave results up to 2x apart. All of these were Vast.ai instances:

| GPU | host CPU | what we found | verdict |
|---|---|---|---|
| L40S | AMD EPYC 9534, ~3.7 GHz | host 25-29 ms/iteration at C640-C896 (base) | fastest host seen; first package A/B |
| L40S | Intel Xeon Platinum 8558, capped at 2.1 GHz (`powersave`, no turbo), 4 NUMA nodes, load average ~31 | host **61 ms/iteration** unpinned, 46 ms pinned; C832 at 635 audio-s/s, RTF p95 1.55 | rejected, only relative A/B quoted |
| L40S | AMD Threadripper 7960X | GPU throttling to **630 MHz at 87 °C** under load | rejected |
| L40S | Intel Xeon Gold 6430, ≤ 2.6 GHz, 2 NUMA nodes, driver 575 | GPU 70 °C at its full 2520 MHz; host 33.5 ms/iteration (base, pinned) | **the reference box** for every L40S number below |
| RTX 6000 Ada | AMD EPYC 7C13, ~3.1 GHz, 1 node, driver 565 | every admission paid 11-30 ms of device-side allocation | usable; led to section 8 |
| L4 | AMD EPYC 7702 | 72 W cap, ~1250 MHz under `sw_power_cap`, normal | the L4 regression checks |

Three things came out of this that are now in [`docs/benchmarking.md`](https://github.com/mynah-org/mynah-tts/blob/main/docs/benchmarking.md):

**Pin the server to the GPU's NUMA node.** On the 4-node Xeon the scheduler thread had landed on CPU
170, a node away from the GPU. `nvidia-smi topo -m` gives the GPU's CPU list; `taskset -c` on that
list took host time per iteration from 61 to 46 ms, with no code change. The load generator goes on
another node.

**Check the clock you actually have.** `lscpu` and `cpufreq/scaling_governor` before anything else.
Inside a container you usually cannot change the governor, so a `powersave` host at 2.1 GHz stays
that way, and a loop bound by one thread runs at that speed.

**Run a 3-minute check before an A/B.** One level, three minutes, with temperature, SM clock, power
and the active throttle reasons sampled:

```bash
THERM=1 TAG=therm LEVELS=768 DUR=180 tools/gpu/knee_closed.sh
```

`0x0` or `0x4` (software power cap) is fine. `0x20` / `0x40` (thermal slowdown), `0x8` (hardware
slowdown) or a clock far below boost means the box is not comparable to anything: move on before it
eats an hour.

## 4. Eleven small wins, each one alone

The profile pointed at a long list of small host costs rather than one big one. Each became a work
item with **its own environment flag, default off, bit-identical when on**. The ones that shipped:

| item | what it removes |
|---|---|
| L6 deferred release | a full stream drain every time a finished context is parked; it is fenced with an event instead |
| L7 cancellation every 4 iterations | one mutex and one `poll()` per stream per iteration to ask whether the client is still there |
| L8 `writev` | three `send()` calls per HTTP chunk, now one |
| L10 epoch duplicate check | two O(rows²) pointer scans per step (~205k compares each at 640 rows) |
| L11 decoder table patch | re-recording the whole SEANet decoder graph on every membership change; the graph's pinned tables are patched and relaunched |
| L12 survivor flow reuse | the second flow-head pass (a sync, a 2.6 MB gather and an upload) whenever a row ends |
| L19 delivery threads | the int16 convert, enqueue and one futex wake per stream, moved to 1/2/4 helper threads chosen from the usable CPUs |
| L20 PCM direct | two memcpys plus a `calloc`/`free` per row per frame; rows borrow pointers into the pinned PCM buffer |
| L21 lazy hidden state | a 2.6 MB hidden-state download and a finite scan every step; checked on the device instead |
| L22 KV table cache | rewriting the layers x rows KV metadata every step; only changed rows are rewritten |
| L24 validate once | decoder op compatibility checked per decoder at open, not rows x ops per step |

(L9, a FIFO prefill scan that was O(rows²) per step, is a pure fix with no flag.)

**Identity first.** Concurrent serving is not deterministic even in pedantic mode (base against base
at C8 matched 164 of 240 WAVs: batch composition follows arrival timing, and cuBLAS picks its
algorithm by width). So identity is checked where composition *is* fixed: a 32-request CLI burst with
fixed seeds, and the server at C1 with fixed seeds and request ids. Every flag alone and all eleven
together: 32/32 and 166-169 of 166-169 identical.

**Then speed, as a package, then leave-one-out.** At 1-minute levels the noise was ±5-10 %, bigger
than most single-flag effects, so: the whole package against the base at 2-minute levels, then remove
one flag at a time.

On the L40S with the fast EPYC host, the package moved the 0.88 gate from C640 to C768 (RTF p95
0.911 → 0.857 and 0.855 in two runs), and C896 went from 46,640 stalls to zero. On the reference Xeon
Gold 6430 box:

| C | base | all 11 flags |
|---:|---|---|
| 768 | 773 / 0.919 ✗ / 157 ms | 867 / 0.829 / 141 ms |
| 896 | 785 / 1.034, 80,583 stalls | 891 / 0.915, 0 stalls |

(cells: audio-s/s / RTF p95 / TTFA p95)

Stream syncs per iteration went from 17-20 to ~9.5-11.8, host time per iteration from 33.5 to
28.4 ms, and decoder graph re-records from ~2,000-2,400 per level to 1-160. Even on the rejected
2.1 GHz host the package was +10 % throughput with half the stalls.

**And on the L4**, where these changes matter least, the full leave-one-out ran at C288 and C320,
with the reference arms repeated at the start and end (drift 0-3 audio-s/s). The package gave
+4 % audio-s/s and −0.03 RTF p95, reproduced; removing L19 or L12 cost the most, and nothing
regressed. That satisfied the close-out rule — bit-identical, safe on both GPUs, a gain on one with
no loss on the other — and on 2026-10-07 all eleven became defaults, each keeping `=0` as a rollback.

Then a plateau again, just higher: with the defaults, C1024 on the L40S came in at 880 audio-s/s and
RTF p95 1.076. Still host-bound. But C1024 had a problem of its own first.

## 5. The silent exit at 1024

An early run with a 1024-row build stopped between C896 and C1024. No crash, no error: the server
printed its normal `[SERVE]` summary, answered the queued requests with `503 "server is shutting
down"`, and left.

Reading the code, no box needed, narrowed it to one line. The driver has exactly one exit, and if it
had stopped on its own the scheduler would have printed "the synthesis driver stopped on its own";
that line was not in the log. So the shutdown came from the outside — from the accept loop:

```c
fd = accept(accept_fd, NULL, NULL);
if (fd < 0) {
    if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK ||
        errno == ECONNABORTED) continue;
    break;                       /* silent; EMFILE lands here */
}
```

Every stream holds a socket for its whole life. Add the listener, stdio, the CUDA driver's
`/dev/nvidia*` descriptors and log files, and C896 needs about 940 descriptors. The default soft
`RLIMIT_NOFILE` on Linux is **1024**. At C1024 `accept()` returns `EMFILE`, the loop breaks without a
word, and the server performs a perfectly orderly shutdown. The diagnosis was confirmed with one
experiment: `ulimit -n 65536` before the server, and C1024 ran clean (880 / 1.076, 0 failures).

The fix is two parts:

- at start-up the server **raises its soft limit to the hard limit** (no privilege needed) and logs
  it — on the box, `open-file limit raised from 1024 to 1048576`;
- on `EMFILE` the accept loop **logs at most once a second, waits 20 ms and keeps going** while
  in-flight streams close theirs. The backoff matters: the listener stays readable and the loop
  would otherwise spin.

Only the hard limit needs to be right, and whatever starts the process sets it:
`LimitNOFILE=65536` in a systemd `[Service]`, `--ulimit nofile=65536:65536` for `docker run`. [`docs/server.md`](https://github.com/mynah-org/mynah-tts/blob/main/docs/server.md#open-file-limit) has the table.

The uncomfortable consequence: **C1024 had never actually been measured before that day.**

## 6. Admission was the real ceiling

With the descriptors fixed, the profile at C1024 showed host time per iteration around 44 ms with the
eleven defaults — and the one term growing with load was admission. At ~900 rows about eight
requests finish and eight are admitted per iteration, independent of host speed. The `[CTX]` line
broke the context build down:

- a **fresh** request context built while the server is full cost **~13 ms**;
- the same context renewed from a pool costs **0.1-0.3 ms**.

A Pocket context carries tens of megabytes of host state — the backbone's host KV mirror, the Mimi
window, the SEANet arena, projection scratch — most of which a device-owned row never reads. Building
it fresh means `calloc`, memsets and page faults on admission and an `munmap` on retirement; the cost
grows with the number of live contexts.

**A1a, `MYNAH_CTX_HOST_POOL=1`**, parks the host half of a finished context and renews it for the
next admission. Renewal is byte-equivalent to a fresh build for everything a request can read; the
KV is not zeroed, because attention only reads positions the request itself wrote. `=2` zeroes it
anyway to prove that; both were identical (CLI 32/32, server C1 166/166 and 150/150).

| C | 11 defaults | + A1a | GPU busy | power |
|---:|---|---|---:|---:|
| 768 | 867 / 0.829 / 141 ms | 1110 / 0.638 / 111 ms | 85 % | 316 W |
| 896 | 891 / 0.915 ✗ | 1108 / 0.742 / 128 ms | 83 % | 318 W |
| 1024 | 880 / 1.076 ✗, 162,015 stalls | **1112 / 0.852 ✓ / 147 ms**, 0 stalls | 84 % | 319 W |

Host time per iteration fell from ~44 to 26.5 ms, and device wait went back to about half the loop.
**C1024 passed the gate for the first time.** It is the largest single step in the whole story, and it
was not a GPU change at all.

## 7. Hiding the GPU waits

The remaining host time still ran in series with two GPU waits per iteration. The obvious move is to
queue the next GPU work before doing the host work.

**L13, step overlap (`MYNAH_CUDA_STEP_OVERLAP=1`).** After step k's emit, decode and delivery, queue
AR step k+1 immediately, then do retire, admission and cancellation while it runs. Identity held (CLI
32/32 with 124 of 128 steps queued ahead; server C1 identical). Throughput moved a little: on the L4
+1 %, on the L40S 882-901 audio-s/s at C768 against 867. But **TTFA p95 rose by 45-60 ms**, more than
the one iteration expected. A KO as a default.

The reason was the retire order. The closed-loop client sends its next request when the previous one
is reported done. L13 retired *after* the launch, so the re-request always arrived just after the last
point where a new row could still join the queued step, and waited a whole extra iteration — and then
its own. The design model puts the cost for Poisson arrivals at only ~5 ms; a closed loop always lands
on the worst phase.

**L13b, decode-ahead (`MYNAH_CUDA_DECODE_OVERLAP=1`).** Split the gang decode into submit and
collect. Finish AR k, emit, submit decode k with a fence; retire *now*, under the decode; prefill and
late admission; launch AR k+1; admission and cancellation while the GPU runs both; collect the PCM by
event and deliver. Retiring before the launch puts the re-request back in front of the pass.

**L13d, first frame first (`MYNAH_CUDA_FIRST_FRAME_FIRST=1`).** New streams' first frames are decoded
as their own small gang and delivered before the main gang, so first audio no longer waits for a
1000-row decode.

All three were identical in every arm (CLI 32/32, CLI `--stream` 32/32, server C1 166/166, including
with `MYNAH_CUDA_DECODE_CHECK=1`, which lands every gang inside its submission as a race check).

| C | + L13 | + L13 + L13b | + L13b + L13d |
|---:|---|---|---|
| 768 | 901 / 0.785 / 186 ms | 979 / 0.727 / 167 ms | 970 / 0.761 / **118 ms** |
| 896 | 899 / 0.914 ✗ / 216 ms | **961 / 0.853 ✓** / 195 ms | 947 / 0.899 / 139 ms |
| 1024 | 894 / 1.065 / 246 ms | 943 / 0.998 / 227 ms | 940 / 1.038 / 160 ms |

(without A1a; cells audio-s/s / RTF p95 / TTFA p95)

L13b passed C896 on its own and lowered TTFA compared with L13. L13d trades 1-2 % throughput for
50-70 ms lower TTFA p95. With the GPU waits hidden, device wait fell to ~6 % of the loop and the host
half was ~90 % — which is exactly why A1a, which shrinks the host half, compounds with it.

**Combined** (eleven defaults + A1a + L13 + L13b + L13d): **C1024 at 1157 audio-s/s, RTF p95 0.846,
TTFA p95 130 ms, zero stalls, GPU 87 % busy at 327 W.** Re-run the next day on the defaults tree with
no flags set except the opt-ins, it reproduced within 1 audio-s/s (1156 / 0.843 / 127 ms, 88 %).

## 8. A platform surprise: `cudaFree` at admission

The next box was an RTX 6000 Ada on an EPYC 7C13, driver 565. Same 48 GB class as the L40S. Every
arm was 2-3x slower, including the eleven defaults alone: **C768 at 376 audio-s/s and RTF p95 2.08**.

`[CTX]` showed it immediately: the device half of a new context (`cuda_backbone`) cost **11-30 ms per
admission**, against under 1 ms on the L40S host. The cause: when a parked backbone KV cache in the
slot pool is too small for the new request, it is freed and a new one allocated — and `cudaFree`
**synchronises the device behind whatever is queued**. The mallocs alone cost ~0.4 ms; on this host
and driver, each of those syncs cost a whole step.

**A1b, `MYNAH_CUDA_SLOT_FIXED=1`**: every pooled request set keeps a backbone KV of at least a fixed
size F and is parked whole, so taking one from the pool makes no `cudaFree`, `cudaMalloc`, VMM call or
synchronous memset. At C1, `cuda_backbone` went from 26.9-29.8 ms to 0.05 ms. At C768 on the Ada:
**455 → 1114 audio-s/s, RTF p95 1.714 → 0.708**, host per iteration 62-100 → 23 ms.

And at C896 it ran out of VRAM. The first version planned its cap at model load, before the server's
start-up width-bucket walk; the walk's long requests left fixed caches of up to ~650 positions in the
pool, 833 fixed rows were live, 467 takes went "over cap" back to the free-and-allocate path, and the
warm-up's ~40 GB left no room for 896 x 25.5 MiB on 48 GB.

**Sizing v2** fixed the plan rather than the mechanism:

- the server marks the end of its start-up walk and prefill; the engine then frees the walk's caches,
  measures free memory and caps at `min(rows + rows/32, (free − max(2 GiB, total/16)) / F)`;
- **F = 384 stored positions** (19.1 MiB with int8 KV at 24 layers, ~26 s of audio), moved once to
  the p95 of real request lengths after 1024 served requests;
- a row that outgrows its cache takes a spare, a larger parked cache, or a new allocation within the
  cap — the copy is stream-ordered and the old cache is parked, **never freed**, so growth has no
  device-wide sync either.

Identity held again (32/32, 32/32, 166/166, including at 64 positions to force growth). On the L40S
reference box, on top of the combined configuration:

| C | combined | combined + A1b v2 |
|---:|---|---|
| 896 | 1161 / 0.734 / 110 ms | 1263 / 0.669 / 96 ms, 94 %, 337 W |
| 1024 | 1156 / 0.843 / 127 ms, 88 % | **1283 / 0.746 / 106 ms, 95 %, 342 W** |

The cap planned 1056 fixed caches and all 1056 were live; 60,819 of 61,920 admissions made no driver
call; the mean context build fell to 0.40 ms (device half 0.006 ms); VRAM ended at 34 GB against 39 GB
without A1b.

> **30-minute soak at C1024, every win on:** 328,625 requests, 0 failures, **0 stalls** above 250 ms,
> stream RTF p95 **0.746** (p99 0.766), first audio p95 **106 ms**, max inter-chunk gap 131 ms, a
> client prebuffer of 23 ms at p95. First 5 minutes vs last 5 minutes: 1403 vs 1427 audio-s/s, RTF
> p95 0.747 vs 0.744, GPU 337 W / 72 °C vs 341 W / 74 °C, 96 % busy throughout. No drift.

+11 % at C1024 with real margin under the gate, and the GPU at 95 % busy and 342 W of 350 W: as
close to "the GPU is the limit" as this card gets.

## 9. What did not win

**L13 alone.** Covered above: ~+1-4 % throughput for +45-60 ms TTFA p95. Kept opt-in, superseded by
L13b.

**Async admission.** Building contexts on helper threads (`MYNAH_ASYNC_ADMIT` with its inline threshold
at 0) produced **72 "CUDA: out of memory" step failures** at C768 with only 32 of 46 GB in use. Not a
real OOM: a recoverable failed allocation leaves its error as the thread's last CUDA error, and the
next ordinary launch check read it back as its own. One failure, reported 72 times, and it disabled
the one-sync path for good. The backend's error check now clears a non-sticky
`cudaErrorMemoryAllocation` after reporting it (and never touches a sticky one). That fix shipped; the
async-admission arm itself has not been re-measured with it, and with A1a making admission cheap
there is less left for it to win.

**Ping-pong half-batches (L26, `MYNAH_CUDA_PINGPONG=2`).** The idea was to reproduce the 93 % of
"two engines" inside one engine: two row groups on one CUDA stream, ordered with events; while the GPU
runs group A's step and decode, the scheduler does group B's host work, then they swap. Per-group
scratch and decoder graphs ("lanes"), a group-local order so a split burst is bit-identical to two
half bursts (32/32), fence-based finishes.

The first run hit a start-up out-of-memory at 1024 rows (group B's scratch was sized at full width).
With B at half width and `--max-batch 768` on the RTX 6000 Ada it worked as designed — **92 % of the
host time ran under the other group's GPU work** — and was still **7-8 % slower than L13 + L13b**
once the GPU was ≥ 93 % busy. Two half-width steps carry extra fixed GPU cost per cycle, and when the
GPU is nearly full, hiding the host better cannot pay for it. Opt-in, labelled experimental.

## 10. Small hosts

The first L40S screen ran on a 4-vCPU host and plateaued at 740 audio-s/s, so the obvious worry was
that the large-row configuration needs a large host. It does not. With the server confined by
`taskset` to two physical cores and their SMT siblings on the GPU's node (the delivery helpers then
default to one), the combined configuration measured:

| C | audio-s/s / RTF p95 / TTFA p95 | GPU busy |
|---:|---|---:|
| 512 | 1131 / 0.445 / 70 ms | |
| 640 | 1135 / 0.549 / 87 ms | |
| 768 | 1126 / 0.658 / 102 ms | 87 % |

About **3 % below the same server on the whole host**, at ~31 ms host per iteration. C896 (≈ 0.77) and
C1024 (≈ 0.88, right at the gate) are *projections* from the C512-C768 slope, not measurements, and
that run did not include A1b, which cuts admission host time further. On four vCPUs: plan for
C768-C896 and screen C1024 before relying on it. What matters is not core count but single-core
speed, because one thread sets the pace.


This is a **4-vCPU affinity experiment**: `taskset` on a larger host keeps that host's memory bandwidth, caches, clock and NUMA layout, so it is not a measurement on a real 4-vCPU instance; qualify one before relying on it.


With the fixed slot pool on as well, the same 4-vCPU affinity run held a **10-minute soak at C1024**:
108,837 requests, 0 failures, 0 stalls, RTF p95 0.752, first audio p95 107 ms, against 0.746 /
106 ms on the whole host. Within noise.

## Bonus: the 6-layer model on the same card

With the 24-layer model at ~96 % GPU busy, the obvious next question was the
small English model: a transformer 4× shallower should go 4× further, right?

```text
C1024   RTF p95 0.562   1802 audio-s/s   comfortable
C1536   RTF p95 0.835   1776 audio-s/s   near the safe knee
C1792   RTF p95 0.965   1773 audio-s/s   realtime edge
C2048   RTF p95 1.112   1744 audio-s/s   overloaded, 570,980 stalls
```

It does not. **Reducing transformer depth 4× raised end-to-end throughput only
~1.3-1.4×.** At 1,500-2,000 rows both the GPU's power budget (~340 W of 350 W)
and the host's per-row work (51-56 ms per iteration, up from ~41 ms) matter, so
the transformer stopped being the dominant term. The GPU stays at 89 % busy
even while C2048 falls apart: the limit is composite, not "CUDA cores full".
Which GPU components make up the remainder is the job of a kernel-level
profile, not of these numbers.

> **30-minute soak at C1536:** 535,593 requests, 0 failures, 0 timeouts; stream RTF p95 **0.823**
> (p99 0.846), first audio p95 **113 ms**, client prebuffer 44 ms at p95; first vs last 5 minutes
> 1934 vs 1968 audio-s/s, RTF p95 0.821 vs 0.823, GPU 344 W / 73 °C vs 346 W / 75 °C at 91 % busy,
> 0 stalled requests in either window. **102 requests (0.02 %) stalled above 250 ms, all in the first
> ~50 s**, when the load stepped from C768 to C1536 and 768 requests were admitted at once; after that
> ramp, none. In steady state C1536 holds; a strict zero-stall gate needs a gentler ramp (or a planned
> capacity below C1536).

That is the whole story of this work in one line: optimise A, B shows up;
optimise B, C shows up; change the model and the bottleneck moves again. There
is no single "an L40S does Cxxxx": there is this engine, this model, this
request mix and these bottlenecks.

## 11. The evolution in one table

| stage | GPU / host | C passing | audio-s/s | RTF p95 | TTFA p95 | GPU busy / power |
|---|---|---:|---:|---:|---:|---|
| first qualified point (30-min soaks) | L4 / EPYC 7702 | 160 | 184.5 | 0.855 | 155 ms | — |
| 2026-10-02 defaults (10-min soak) | L4 | 288 | 341 | 0.821 | 141 ms | — |
| + int8 backbone KV | L4 | 320 | 338 | 0.855 | 146 ms | — |
| + 11 host flags (defaults) | L4 | 320 | 354 | 0.820 | 142 ms | 90 % / 71 W |
| first L40S screen | L40S / 4-vCPU host | 384 | 740 | 0.488 | 87 ms | 62 % / ~241 W |
| 768-row build, base | L40S / EPYC 9534 | 640 | 873 | 0.681 | 114 ms | 67 % / 274 W |
| base | L40S / Xeon Gold 6430 | none (C768 0.919) | 773 | — | 157 ms | 61 % / 260 W |
| 11 flags (defaults) | L40S / Xeon Gold 6430 | 768 | 867 | 0.829 | 141 ms | 67 % / 279 W |
| + L13 + L13b | same | 896 | 961 | 0.853 | 195 ms | 74 % / 300 W |
| + A1a host-context pool | same | **1024** | 1112 | 0.852 | 147 ms | 84 % / 319 W |
| combined (A1a + L13 + L13b + L13d) | same | 1024 | 1157 | 0.846 | 130 ms | 87 % / 327 W |
| **combined + A1b v2 fixed slots** | same | **1024** | **1283** | **0.746** | **106 ms** | **95 % / 342 W** |
| combined, server on 4 vCPUs | same box, `taskset` | 768 (top level run) | 1126 | 0.658 | 102 ms | 87 % / ~325 W |
| **30-minute soak, every win on** | L40S / Xeon Gold 6430 | **1024** | ~1400¹ | **0.746** | **106 ms** | 96 % / 340 W |

C1024 is the top of the `ROW_CAP=1024` build, so the last rows are "passes at the cap", not "the
knee". The L40S rows are 2-minute screens except the 30-minute soak.

¹ Audio-seconds of the requests that finished in a 5-minute window, divided by the window. The
2-minute screens report a lower figure (1283 at C1024): `pocket_ladder.py` counts the requests
sent after warm-up and divides by the time until the last of them finishes, so the drain tail,
when concurrency is already falling, dilutes a short screen. Over 30 minutes the tail is
negligible and the aggregate (1393) matches the steady 5-minute windows. Compare screens with
screens, not with the soak.

## 12. Lessons, and what is next

1. **One flag per idea, default off, A/B on and off**, same box, same session, a fresh server per
   arm. Defaults only after that, with `=0` kept as a rollback.
2. **Prove identity where it can be proved** (CLI burst, streamed burst, server at C1), then check
   the path was exercised: an overlap test with zero launched steps passes without testing anything.
3. **Two-minute levels, a written gate.** The first level after start-up is noisy; judge the later
   ones.
4. **Measure differences, not absolutes.** The same L40S gave 635 and 1283 audio-s/s on two hosts.
   A package that is +10 % on a bad host is still a +10 % package.
5. **Check the box first.** Clock and governor, NUMA, driver, open-file limit, three minutes of
   thermals. Every one of those bit us once.
6. **Ask which instrument you are reading.** `dmon sm` at 62 % first read as "underfilled kernels".
   It meant "idle", and the whole fix list followed from reading it correctly.
7. **Let the program say where it dies.** The silent exit was solved by listing the driver's exits
   and noticing which log line was missing, not by re-running the ladder.

**Next.** A1a, L13b + L13d and A1b were bit-identical and recommended for large-row serving, but
stayed opt-in until an L4 regression run passed — the same rule the eleven defaults went through.
That run is the next section; they are defaults now. After that: a 30-minute soak at C1024 on the L40S (everything above is a screen), re-running the RTX 6000
Ada with A1b v2, and a build beyond `ROW_CAP=1024`, since at 95 % busy with RTF p95 0.746 there is
still margin under the gate that the row cap does not let us use.

## 13. Back to the L4: it got faster too

I optimized for the faster GPU. Then I went back to the slower one — and it got faster too.

The L4 is the opposite case from the L40S: it is GPU-bound, 88-90 % busy with the eleven defaults,
so hiding host time should buy little there. The rule was still to check it before promoting the
large-row set, on a Vast.ai L4 (EPYC 7702 host), default `ROW_CAP=384` build, 2-minute levels,
audio-s/s / stream RTF p95 / TTFA p95:

| level | the eleven defaults | + A1a + A1b (pools only) | + A1a + L13/L13b/L13d + A1b |
|---|---:|---:|---:|
| C256 | 354 / 0.715 / 117 ms (19 stalls) | 373 / 0.632 / 106 ms | **382 / 0.634 / 100 ms** (0 stalls) |
| C288 | 361 / 0.735 / 128 ms | 373 / 0.702 / 118 ms | **378 / 0.712 / 112 ms** |
| C320 | 367 / 0.788 / 137 ms | 381 / 0.757 / 126 ms | **382 / 0.775 / 121 ms** |

The GPU went from 88-90 % to 98 % busy: even on the L4 the host loop was leaving 10 % of the card
idle. The pools do most of the work (they remove the per-admission context build and the
`cudaFree` behind it); the overlap package adds a few audio-s/s and another 5-6 ms off first audio,
at a small RTF cost that stays far inside the gate. The identity checks passed on the L4 as well
(32/32, 32/32, 95/95), so A1a, L13 + L13b + L13d and A1b are now on by default for CUDA serving,
each with an `=0` rollback (`MYNAH_CUDA_STEP_OVERLAP=0` turns the whole overlap off). Ping-pong
stays experimental.

With a `ROW_CAP=512` build and everything on, the L4 knee moved up one step:

```text
C320   RTF p95 0.775   382 audio-s/s   comfortable
C352   RTF p95 0.846   383 audio-s/s   passes the 0.88 gate
C384   RTF p95 0.914   385 audio-s/s   over the gate, still 0 stalls
C416   RTF p95 0.999   379 audio-s/s   realtime edge, 0 stalls
C448   RTF p95 1.057   386 audio-s/s   overloaded, 16,954 stalls
```

The old defaults stopped around C320; the new ones pass C352. Throughput sits at ~380-385
audio-s/s with the GPU 98-99 % busy at its 72 W cap: on this card the limit really is the GPU now,
which is what an L4 should look like.

## Try it / links

The full copy-paste version — build, NUMA check, pinned server, 4-vCPU and L4 variants, load test —
is the **[large-row quick start in `docs/cuda-serving.md`](https://github.com/mynah-org/mynah-tts/blob/main/docs/cuda-serving.md#quick-start-large-row-serving)**.
The short form, for one 48 GB GPU (the large-row variables are defaults now, so the server line sets
none of them):

```bash
make cuda cuda-server CUDA_ARCH=sm_89 ROW_CAP=1024
nvidia-smi topo -m            # the GPU's CPU list, e.g. 32-63,96-127
ulimit -Hn                    # must exceed the target concurrency

MYNAH_THREADS=1 MYNAH_QUANT_GROUPS=none \
taskset -c 32-63,96-127 ./build/cuda/mynah-tts-server --device cuda -w 8 \
  --max-batch 1024 --max-inflight 1024 --max-pending 2048 \
  -p 8080 -m models/pocket-english-24l
```

Then run the 3-minute thermal check and one closed-loop level from another NUMA node, as described in
[`docs/benchmarking.md`](https://github.com/mynah-org/mynah-tts/blob/main/docs/benchmarking.md). Pass is `rtf95 ≤ 0.88`, `st250 0`, `fail 0` at the
level you will serve.

**Links**

- Mynah TTS: <https://github.com/mynah-org/mynah-tts> (MIT)
- The mynah-org organisation — mynah-tts, mynah-asr, mynah-slm, ingot: <https://github.com/mynah-org>
- Bench results from your own GPU and host are the most useful contribution: open an issue with the
  `[SERVE]` and `[CTX]` lines and the box details from `docs/benchmarking.md`.
- Per-GPU thresholds and host lessons: [`docs/performance.md`](https://github.com/mynah-org/mynah-tts/blob/main/docs/performance.md#pocket-cuda-serving-thresholds-2026-10)
- Every serving flag, its default, measured effect and rollback: [`docs/cuda-serving.md`](https://github.com/mynah-org/mynah-tts/blob/main/docs/cuda-serving.md#serving-flag-reference)
- Bench method, host list and identity protocol: [`docs/benchmarking.md`](https://github.com/mynah-org/mynah-tts/blob/main/docs/benchmarking.md)
