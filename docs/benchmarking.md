# Benchmarking the Pocket CUDA server

How the GPU serving numbers in [performance.md](performance.md) were measured,
so they can be reproduced on another box or used to A/B a change. Everything
here is a **screen**: 2-minute levels that find a threshold and compare
configurations. Promoting a level to a production setting still needs the
30-minute soaks of [cuda-serving.md](cuda-serving.md) section 10.

The suite has four parts, in this order:

1. prepare the host (build, clock, NUMA, open files, thermals);
2. prove audio identity for any change that claims it;
3. run the knee: one server, a few closed-loop levels, one summary line each;
4. read the server's own profile to see where the time went.

## 1. Build

```bash
make cuda cuda-server CUDA_ARCH=sm_89 ROW_CAP=1024
```

`ROW_CAP` is the number of rows one CUDA process may step together (default
384, `src/row_cap.h`). `--max-batch` and `--max-inflight` cannot exceed it.
The build stamps the value and rebuilds when it changes. Use 384 on an L4 and
1024 on a 48 GB GPU (L40S, RTX 6000 Ada).

## 2. Prepare the host

On a GPU this fast the serving loop is bound by one scheduler thread, so the
host changes the result as much as the GPU does. Check these before the first
level and write them down with the results.

**CPU and clock.** A 2.1 GHz host doubled the host time per iteration against
a 3.7 GHz one on the same GPU class.

```bash
lscpu | grep -E 'Model name|MHz|NUMA'
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null
nproc; cat /sys/fs/cgroup/cpu.max 2>/dev/null       # the container's CPU quota
```

**NUMA.** On a multi-node host, pin the server to the GPU's node and the load
generator to another node.

```bash
nvidia-smi topo -m          # "CPU Affinity" / "NUMA Affinity" of the GPU
lscpu | grep 'NUMA node'    # the CPU list of each node
```

For example, on a 2-node Xeon Gold 6430 host with the GPU on node 1:
`PIN=32-63,96-127 CLIPIN=0-31,64-95`.

**Open files.** Every stream holds a socket. The server raises its soft limit
to the hard limit at start-up and logs `open-file limit raised from A to B`;
if the hard limit is below the target concurrency, raise it first
([server.md](server.md#open-file-limit)):

```bash
ulimit -Hn; ulimit -n 65536
```

**Thermals: a 3-minute pre-check.** One box had its GPU throttling to 630 MHz
under load, which no A/B on it could have shown. Run one level for 3 minutes
with the clock sampled:

```bash
THERM=1 TAG=therm LEVELS=768 DUR=180 tools/gpu/knee_closed.sh
```

The `therm` lines are temperature, SM clock, power and the active throttle
reasons. `0x0` (none) and `0x4` (software power cap, normal at the power limit)
are fine. `0x20` / `0x40` (thermal slowdown) or `0x8` (hardware slowdown), or a
clock far below the GPU's boost clock, mean the box is not comparable: move to
another one.

## 3. Audio identity

A change that claims "same audio" is checked where batch composition is
deterministic. Under concurrency it is not: which requests share a step
follows arrival timing, and cuBLAS picks its algorithm by width, so even two
runs of the same binary at C8 differ.

Run the same three checks for the reference arm and the changed arm (flag off
and on, or two trees), then compare the WAVs by SHA-256.

```bash
M=models/pocket-english-24l
T="The train leaves the central station at a quarter past eight and arrives just before noon, so there is plenty of time for lunch before the meeting starts."
export MYNAH_QUANT_GROUPS=none MYNAH_THREADS=1      # plus the arm's own variables

# 1. CLI burst: one batch of 32 requests, seeds 1000-1031 (batched step, retire)
./build/cuda/mynah-tts --synthesize $M --text "$T" --lang en --speaker 0 --seed 1000 \
  --batch 32 --device cuda --output $ARM/cli/out.wav
# 2. the same burst through the streaming callbacks (the WAVs are the streamed PCM)
./build/cuda/mynah-tts --synthesize $M --text "$T" --lang en --speaker 0 --seed 1000 \
  --batch 32 --stream --device cuda --output $ARM/str/out.wav
# 3. the server at C1: streaming path, delivery, decoder graph; fixed seed and request ids
./build/cuda/mynah-tts-server --device cuda -w 8 --max-batch 16 --max-inflight 16 -p 18080 -m $M &
python3 tools/pocket_ladder.py --port 18080 --levels 1 --mode closed --warmup 0 --duration 40 \
  --timeout 120 --corpus tools/corpus/pocket_v2_en.jsonl --voices alba,marius --seed 1234 \
  --save-audio $ARM/srv --out $ARM --tag ident
kill -INT %1
```

Compare each directory of one arm with the same directory of the other. The
server run is time-bounded, so the two arms can save a different number of
WAVs; compare the common ones (same tag, so the same file names):

```bash
python3 - ref/srv arm/srv <<'EOF'
import hashlib, os, sys
h = lambda d: {f: hashlib.sha256(open(os.path.join(d, f), "rb").read()).hexdigest() for f in os.listdir(d)}
a, b = h(sys.argv[1]), h(sys.argv[2]); c = sorted(set(a) & set(b))
print("files %d common %d identical %d" % (len(a), len(c), sum(a[k] == b[k] for k in c)))
EOF
```

Pass: identical equals common in all three (32/32, 32/32, and e.g. 166/166).
Also run the reference arm twice: if it does not match itself, the check is
not measuring anything. And make sure the changed arm really took the new
path: with `MYNAH_SERVE_PROFILE=1` the CLI and the server print a line for each
serving feature (for example `[SERVE] step overlap (MYNAH_CUDA_STEP_OVERLAP): 124 of 128 steps
queued ahead (124 launched)`; 0 launched means the path was not exercised).
Each default or opt-in feature also prints one start-up line when it is on.

## 4. The knee

`tools/gpu/knee_closed.sh` starts one server, runs the levels in order and
prints one line per level. It is configured by environment variables (the
header of the script lists them all):

```bash
TAG=combined LEVELS="768 896 1024" PRE=640 PROCS=4 DUR=120 \
  PIN=32-63,96-127 CLIPIN=0-31,64-95 \
  ENVS="MYNAH_CTX_HOST_POOL=1 MYNAH_CUDA_STEP_OVERLAP=1 MYNAH_CUDA_DECODE_OVERLAP=1 \
        MYNAH_CUDA_FIRST_FRAME_FIRST=1 MYNAH_CUDA_FAST_FIRST_CHUNK_WAIT_US=3000" \
  timeout 1800 tools/gpu/knee_closed.sh
```

(the combined configuration of performance.md, run on an NVIDIA L40S on
2026-10-06; the 11 default flags need no variable):

```
== combined [MYNAH_CTX_HOST_POOL=1 ...] procs=4 pin=32-63,96-127 ready in 170 s, VRAM 27395 MiB
  C896 aps 1158 rtf95 0.738 ttfa95 111ms gap95 127ms st250 0 fail 0 | cli_cpu 121% | gpu 329W sm 89%
  C1024 aps 1157 rtf95 0.846 ttfa95 130ms gap95 144ms st250 0 fail 0 | cli_cpu 119% | gpu 327W sm 87%
  VRAM end 39361 MiB
  [SERVE] loop ...
  [SERVE] device wait 7.2% of loop (97777 syncs, 0.38 ms mean, 11.34 per iteration); host 92.5% (55.44 ms per iteration); blocked 0.3%
```

What it does, and why:

- **One fresh server per configuration**, `--max-batch` and `--max-inflight`
  at the highest level, `MYNAH_SERVE_PROFILE=1`, `MYNAH_THREADS=1`. It waits
  for `/health`, prints the start-up time and VRAM, and at the end stops the
  server with SIGINT so the profile report lands in `server.log`.
- **Closed loop**: at level C, C clients each send their next request as soon
  as the previous one ends (`tools/pocket_ladder.py --mode closed`), v2 corpus
  (`tools/corpus/pocket_v2_en.jsonl`), four voices, seed 1234.
- **2-minute levels after a 15-second warm-up.** At 1-minute levels the noise
  was ±5-10 %, as large as the effects being measured.
- **A warm-up level first** (`PRE`, 45 s, not reported). The first level after
  start-up is the noisiest (the same configuration swung 888-983 audio-s/s
  there).
- **`--client-procs 4`.** At C1024 the load generator itself used up to 1.3
  cores; spread over four processes it stays out of the result (1 and 4
  processes gave the same throughput at C640). The summary's `cli_cpu` is the
  load generator's own CPU. `tools/gpu/stub_stream_server.py` measures a
  client's ceiling without a GPU.
- **`nvidia-smi dmon` during each level**: `gpu` is the mean power and `sm` the
  mean SM time share of the samples with `sm` > 5 %. `sm` is the share of time
  at least one kernel runs, so 60 % means the GPU is idle 40 % of the time.
  It does not measure how full the SMs are.

The summary line: `aps` audio seconds produced per second, `rtf95` the stream
RTF p95 (per-request streaming realtime factor), `ttfa95` time to first audio
p95, `gap95` the p95 of each request's largest gap between chunks, `st250` the
playback stalls with a 250 ms buffer, `fail` the failed requests. The gate
used in [performance.md](performance.md) is `rtf95` ≤ 0.88 with `st250` 0
and `fail` 0. The full per-level summary is `$OUT/$TAG/c<C>-summary.jsonl`,
one record per request in `c<C>-c<C>.jsonl`.

**Comparing configurations.**

- Run the arms in the same session, on the same box, one server each, in an
  ABA order (or A B ... A), so the drift between the two A runs shows the
  noise. Compare differences, not absolute numbers from another day.
- Judge on the levels around the threshold; a level far below it says little.
- Wrap every run in `timeout`, and launch long chains detached (for example
  `tools/gpu/detach.sh`). When stopping a chain by hand, stop the knee too, not
  only the server: a knee left behind will run its cleanup at the wrong time.
  `knee_closed.sh` only ever signals the server it started.

## 5. Reading the serving profile

With `MYNAH_SERVE_PROFILE=1` the server prints, at shutdown:

| line | what to read |
|---|---|
| `[SERVE] step cost per width` / `B<n> n=... mean ... worst ...` | GPU+host time of a step at each executed width |
| `[SERVE] loop <s>: step ...% prefill-first ... prefill-cont ... other ...%` | where the scheduler loop spent its time |
| `[SERVE] device wait X% of loop (N syncs, m ms mean, k per iteration); host Y% (h ms per iteration)` | **the key line.** Device wait is time inside stream syncs; host is everything else on the scheduler thread, while the GPU may have nothing queued. On the L40S the 11 default flags left ~28 ms of host per iteration at C768; the overlap flags turn most of the device wait into hidden time (device wait ~6-7 %), so then host ms per iteration is the whole cost |
| `[SERVE]   sync <file>:<line> calls N (k per iteration) wait ... mean ...` | per-call-site syncs: which sync carries the wait. With L13/L13b a `while queued N` count shows syncs reached while a step or a decode was queued (each one serialises the overlap) |
| `[SERVE] step overlap` / `decode overlap` / `pingpong` | steps queued ahead, launches refused, gangs collected on a poll vs a wait, the host share under the other group (L26) |

During the run, every 160 admissions, the server prints a `[CTX]` line: the
mean cost of building a request context and its parts (`ar_states`,
`codec_setup`, `cuda_backbone`, ...), plus `host_pool: pooled N fresh M` with
A1a, `admit calls: zero-call N fallback M; malloc / free / memset / event`
with the driver-call meter, and `fixed: live L/cap C ...` with A1b. If
`cuda_backbone` is in milliseconds, the driver is paying for allocations and
frees on the admission path: that is what `MYNAH_CUDA_SLOT_FIXED=1` removes.

## 6. What to record

With every result: GPU and driver, CPU model and clock, NUMA pinning, the
container's CPU quota, `ROW_CAP`, the tree or commit, every `MYNAH_*` variable
set, the thermal pre-check, and per level the summary line, the VRAM at the
end, and the `device wait` / host-ms line.
