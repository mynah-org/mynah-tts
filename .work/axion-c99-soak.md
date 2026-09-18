# The Axion session: what a real target machine corrected, and what it refused

Host: GCP Axion, 32x Neoverse-V2, 80 MiB L3, 62 GB, Linux 7.0, gcc 15.2,
`BLAS=none` (the Linux default), branch `docs/plan-board-pocket-tts`.
Pack `models/pocket-en`. 2026-09-17.

Everything below was measured there. Nothing in this note is from a laptop.

## It corrected two conclusions taken on Apple silicon

### 1. The int8 conv path LOST at 16 threads, and the reason was not the kernel

| threads | f32 | conv int8 | conv + convtr int8 |
|---|---|---|---|
| 16 | **68.9 ms** | 70.3 | 69.5 |
| 4 | 129.7 | 114.5 (1.13x) | **83.9 (1.55x)** |
| 2 | 224.8 | 189.3 (1.19x) | **120.8 (1.86x)** |

`codec.conv_stack`, medians of five, same utterance. RTF at two threads went
0.272 -> 0.243.

The Mac had measured 1.20x at two threads and I carried that forward as if it
described the target. At sixteen threads it is a small **loss**.

### 2. The activation quantization pass was the serial fraction

|  | quant | gemm | quant share |
|---|---|---|---|
| 2 threads | 3.766 ms | 16.725 ms | 18.4% |
| 16 threads | **3.805 ms** | 7.709 ms | **33.0%** |

Flat in thread count while the GEMM it feeds scales. It ran on the calling
thread. That is Amdahl and nothing else, and it is the whole of why int8 lost
at sixteen: the laptop never ran wide enough for the serial term to show.

Per shape at sixteen threads, before the fix:

| shape | f32 | int8 | |
|---|---|---|---|
| 512x16x512 7 taps | 5.61 ms | 2.49 | **2.25x** |
| 128x96x256 3 taps | 1.98 | 2.25 | 0.88x |
| 256x96x128 1 tap | 1.68 | 1.94 | 0.87x |
| 64x480x128 3 taps | 2.67 | 4.84 | **0.55x** |

The last one spends 2.38 ms of its 4.84 in the quantization pass alone,
against 2.67 ms for the whole f32 call. Fixed: the pass is on the pool,
bit-identical (disjoint writes, one gather slice per task).

**What this means for serving.** A worker runs 2-4 threads, not 16, so the
regime that decides capacity is the one where int8 wins by 1.55-1.86x. And the
codec is the PER-SLOT term -- qwen-tts's `T_frame(B) = a + b*B` with the
backbone as `a` and the codec as `b` -- so that factor is capacity, not
cosmetics.

## The capacity question, and a textbook refusal

Five runs. The first three are recorded because the mistakes in them are the
point.

**Run 1** -- 6 levels, 16x2, the old five-sentence bank. Every level NOT
QUOTABLE: the client coalesced 37.8% of reads at C16 falling to 15.2% at C100.
The direction is the diagnosis -- worse at LOW load means the load generator
cannot keep up with a server delivering faster than it reads. The instrument
was the limit, not the server.

**Run 2** -- my error, recorded as one: I changed **two variables at once**
(14x2 instead of 16x2, and `--same-text`) and everything got worse with no way
to attribute it. `--same-text` synchronises requests into bursts, which makes
the coalescing it was meant to fix worse.

**Run 3** -- 16x2, 180 s, C100, single variable changed: MARGINAL, coal 14%,
the only failing gate `stall_rate@250ms = 0.1%` against a required zero.

**Run 4 -- C99, 600 s, English `medium` from the v2 bank: GOOD.**

    99  25014/25014  TTFB95 70.9  TTFA95 178  RTF p50 0.665 p95 0.779
        preb 0/0 ms  safe95 179  gap95 147  stall@250ms 0%  coal 14%
    CAPACITY: the operating point is C99

    PASS drift stream_rtf p95          last 0.7777 vs best 0.7777 = +0.0000
    PASS drift required_prebuffer_s p95 last 0.0000 vs best 0.0000 = +0.0000

Drift exactly zero over ten windows, and the prebuffer zero at p50 AND p95: a
player can start immediately.

**Run 5 -- C99, 1800 s, the MIXED v2 bank: NOT STREAMABLE.**

    99  54645/54645  TTFB95 56.8  TTFA95 310  RTF p50 0.682 p95 0.831
        preb95 21  gap95 305  stall@250ms 0.8%  coal 14%
    FAIL mandatory stall_rate@500ms  0.001 == 0.000
    FAIL preferred stall_rate@250ms  0.008 == 0.000
    (both drift gates PASS: rtf +0.0176 over ten 3-minute windows)

54,645 requests, none failed, and one in a thousand still stalls **with half a
second of buffer**, which is a mandatory gate.

The mix is the difference: `long x6600, medium x25245, short x13000,
conversational x9800`, audio **1.04 s to 18.48 s, sd 4.39** against the
medium-only sd of 0.298. Long requests hold a slot and the cadence of everyone
else pays for it.

**So C99 is not an operating point on a realistic workload, and the
medium-only GOOD is a screen result that must not be quoted as capacity.**
That is exactly the separation qwen-tts's doctrine insists on -- "a screen is
never a qualification" -- demonstrated on us, in one session, with the two
verdicts an hour apart.

## E10-3b answered: the pre-fork sharing is real

`/proc/PID/smaps_rollup`, four prefork workers:

    worker   RSS 567 MB   PSS 124 MB   Private_Dirty 13 MB

PSS is what the process actually owns. The weights are shared after the fork
and a worker dirties **13 MB** of its own, so four workers cost about
300 + 3x124 MB rather than 4x567. E10-3's pre-fork warm does what it claimed.

## And the spin budget derived itself

`spin source=calibrated (0.33 ns/relax measured here, target 35 us)` -> **65536**,
which is the Axion's measured knee. E10-8 replaced a transferred constant with
a time budget; on the machine the constant came from, the budget lands on it.

## What is still owed

1. **The operating point on the mixed bank.** The stall at 500 ms is 0.1% --
   one level down almost certainly clears it. A descending sweep C64/C80,
   5-minute soaks, ~15 minutes:

       MYNAH_QUANT_GROUPS=codec_transformer:int8,codec_conv:int8,codec_convtr:int8,backbone:f16,flow_net:f16,conditioner:f16 \
       python3 tools/serving_profile.py --mode soak --model models/pocket-en \
         --port 8806 --levels 64,80 --bank tests/load_texts_en_v2.txt --language en \
         --soak-seconds 300 --warmup-seconds 40 --window-seconds 60 \
         --server-args "--prefork 16 --prefork-threads 2 --max-batch 8"

2. **E10-10, the topology, still measured only at low concurrency.** `16x2`
   means sixteen independent weight streams against one memory roof, and the
   backbone is bandwidth-bound. qwen-tts's finding -- "two workers read the
   same weights twice per frame-time against one memory roof" -- predicts that
   at high concurrency a WIDER worker (`8x4`, `4x8`) amortises more and should
   win. Same command as above with the three shapes.

3. **A second VM as load generator.** 14% coalescing is under the 15% refusal
   but not by much, and it is the client, on the same 32 cores as the server.
   A small instance in the same zone removes the doubt from every number here.

4. **`codec_convtr` is OFF by default and every capacity number above was
   measured with it ON.** The box now justifies turning it on -- 1.86x on the
   per-slot term at serving thread counts -- but the cost is the one already
   recorded (SNR 28.9-32.9 dB against 36.4-37.9, log-mel nearly unchanged),
   and that is a product decision, not a measurement. Until it is taken, the
   numbers above do not describe the shipped configuration.

---

# Second session, 2026-09-18: the tail does not close by lowering concurrency

Same box (new address), `main` at `8aa714a` cloned fresh, same pack, same
`16x2 --max-batch 8`, codec fully int8 including `codec_convtr`. Everything
below is from that machine.

## Two numbers in the instrument could not be read, and were fixed first

**The mandatory gate printed `FAIL stall_rate@500ms 0.000 == 0.000`.** Over
54360 requests a rate rounds to zero from one stalling request and from
twenty-seven, and those are different verdicts about the same server.
`summarize()` now carries `stall_n@<buffer>`, `gate()` carries a note and
`qualify()` fills it, so the line reads `(5 of 53895 requests)`. Every number
in this section depends on that fix: the whole finding below is a count of
five.

**Bank coverage was wrong, not merely unreadable.** `distinct_texts` counted
distinct request *indices*, and the index is a monotonic counter, so it always
returned the request count -- a 30-minute soak reported "54360 distinct request
slots" whatever the bank held. The text is `bank[index % len(bank)]`, so that
is what is counted now. Verified against the run that prompted it: the
round-robin cursor predicts `short 12805 / medium 24699 / conversational 9604 /
long 6468 / italian 784` over 54360 requests on the v2 bank, and the server
measured `12783 / 24696 / 9604 / 6489 / 788`, the difference being requests in
flight at the cut. The bank is fully exercised: **277 of 277 texts, each ~195x**
over thirty minutes.

## A contaminated hour, thrown away

An ASR benchmark started on the same machine at 10:14:20 taking ~10.5 of 32
cores. It overlapped the last minute of a C92 screen and all of a C96 screen.
Both were discarded: a capacity soak on a shared machine measures the other
workload. Recorded because the temptation to keep a number that "looks about
right" is exactly the failure this board exists to prevent.

## The three levels, one variable apart

| | C99 (30 min) | C98 (30 min) | C96 (30 min) |
|---|---|---|---|
| completed | 54645/54645 | 54360/54360 | 53895/53895 |
| stall@500ms | 0.1% (~55) | < 27 | **5** |
| stall@250ms | 0.8% | 0.7% | **233 (0.43%)** |
| RTF p95 | 0.831 | 0.817 | **0.783** |
| prebuffer p95 | 21 ms | 7 ms | **0 ms** |
| TTFA p95 | 310 ms | 308 ms | 308 ms |
| drift RTF | +0.0176 | +0.0025 | +0.0033 |

All three are NOT STREAMABLE on the same mandatory gate, and all three pass
every other gate including both drift gates. Three levels bought a factor of
ten on the failing quantity and did not reach zero. **The tail is not a
capacity problem in this range**: it is the duration spread (audio 1.04-19.04 s,
sd 4.45) and it would take a much lower level to close, if concurrency closes
it at all.

## A hypothesis, stated at 10 minutes and falsified at 30

The C96 screen showed `required_prebuffer` max **495.4 ms** over 17968 requests
while `stall_rate@500ms` failed on 3. Those are not contradictory: the gate
simulates a buffer of 500 ms **of audio**, and the server delivers at RTF 0.68,
so half a second of audio accumulates in ~340 ms and that player starts
*earlier* than the worst request needs. The reading was that a client waiting
500 ms **of wall clock** would have had zero stalls, with 4.6 ms of margin.

The 30-minute run refuses it: **`required_prebuffer` max 535.0 ms** over 53895
requests. The margin was 4.6 ms on a 10-minute sample of a tail; at three times
the sample the tail moved past the threshold, exactly as a 4.6 ms margin
deserves. A ten-minute screen cannot qualify a tail, which is the same lesson
as "a screen is never a qualification", now demonstrated on a *statistic*
rather than on a workload.

**What survives.** Over 53895 requests at C96 on the mixed bank, no request
needed more than 535 ms of lead. A client with a **600 ms** prebuffer plays all
of them without a single underrun. That is a product number and it is measured,
not extrapolated -- but it is a statement about a 30-minute sample of a tail,
and it is the kind of number that moves when the sample grows.

## What this leaves owed

1. **E10-10, the topology, is now the first experiment, not the third.**
   Every number above is `16x2`. `step.backbone` is bandwidth-bound, so
   sixteen workers are sixteen weight streams against one memory roof. If a
   wider worker (`8x4`, `4x8`) shortens the long-request tail, it moves the
   only quantity still failing. Two 10-minute screens at C96, ~22 minutes.
2. **The load generator is still on the server's cores** -- 13.6% coalesced
   reads, under the 15% refusal but not by much, and it is the client competing
   with the thing it measures.
3. **`codec_convtr` is still OFF by default and every number here was measured
   with it ON.** Unchanged from the first session, and still a product call.
