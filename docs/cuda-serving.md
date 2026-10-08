# Serving PocketTTS on an NVIDIA GPU (CUDA)

How to run the mynah-tts streaming server on one NVIDIA GPU for PocketTTS, the
reference, production-quality engine of this runtime (it also serves on the
CPU): build, model, start, stream, size, tune, monitor and qualify. Everything
here was measured on one NVIDIA L4 (24 GB) on Vast.ai between 2026-09-28 and
2026-10-04, plus 2-minute screens on an NVIDIA L40S and an NVIDIA RTX 6000 Ada
(48 GB each) in October 2026 (sections 6 and 7). The evidence is in
[performance.md](performance.md), the method in
[benchmarking.md](benchmarking.md) and the serving profiles in
[`configs/perf/`](../configs/perf/README.md).

## At a glance

| | PocketTTS small (6 layers) | PocketTTS large (24 layers) |
|---|---:|---:|
| Streams on one L4, qualified (2 x 30 min, 0 stalls) | **256** | **160** |
| Speech produced per second, all streams | 316x realtime | 184x realtime |
| First audio, p95 | 145 ms | 155 ms |
| Streaming realtime factor, p95 | 0.79 | 0.86 |
| GPU memory at that load | ~8.5 GB | ~14.4 GB |
| Profile | `l4-24g-pocket-en-6l-cuda` | `l4-24g-pocket-en-24l-cuda` |

The qualified points above were measured on 2026-09-28 defaults (bf16 KV).
The current defaults add the shared voice prefix, split decode attention, bf16
backbone Linears through cuBLASLt (fused), the fused decoder, one sync per
frame, the per-request memory diet and int8 backbone KV (section 7). With them
the large model screens at **C320 per L4** with `--max-batch 320
--max-inflight 320` (2-min closed-loop knees, v2 corpus, 4 voices, 0 stalls,
0 failures):

| large model, current defaults | C256 | C288 | C320 |
|---|---:|---:|---:|
| stream RTF p95, bf16 KV | 0.745 | 0.827 | |
| stream RTF p95, int8 KV (default) | **0.681** | **0.782** | **0.855** |
| audio-s/s, int8 KV | 345 | 335 | 338 |
| TTFA p95, int8 KV | 118 ms | 134 ms | 146 ms |

Peak GPU memory: 16.3 GB at `--max-batch 320` with int8 KV, 21.3 GB at
`--max-batch 288` with bf16 KV. The 2 x 30-min soak qualification of these
defaults is pending, so C160 stays the qualified figure until it lands.

A 4-vCPU host is enough: on the L4 the server needs about 1.3-1.4 cores, and
pinning it to four cores cost 1.5%. On an L40S, with the large-row defaults
of section 7, a server confined to 4 vCPUs (two cores and their SMT siblings)
held C768 at RTF p95 0.658, ~3 % below the full host (section 6); with the fixed
slot pool (A1b) it held a 10-minute soak at C1024 at RTF p95 0.752 (affinity
experiment, not a real 4-vCPU instance).

## 1. What you need

- Linux with an NVIDIA driver and a CUDA toolkit (nvcc and cuBLAS; tested with
  CUDA 12.8 build / 13.2 runtime, driver 595).
- A GPU with compute capability 8.0 or newer (TF32 tensor cores). Measured:
  L4, L40S and RTX 6000 Ada (`sm_89`; the Vast.ai hosts are listed in
  [benchmarking.md](benchmarking.md#hosts-used-for-the-2026-10-screens-vastai)).
  Built and exercised earlier on Blackwell (`sm_120`).
- Python 3 only for the offline tooling (model conversion, load tests). The
  server itself has no Python.
- A Hugging Face read token: `kyutai/pocket-tts` is a gated repository.

## 2. Build

```bash
make cuda-server cuda CUDA_ARCH=sm_89    # -> build/cuda/mynah-tts-server, build/cuda/mynah-tts
```

| GPU | `CUDA_ARCH` |
|---|---|
| L4, L40S, RTX 40xx | `sm_89` |
| A100 | `sm_80` |
| A10, A40, RTX 30xx | `sm_86` |
| H100, H200 | `sm_90` |
| Blackwell (B200, RTX 50xx) | `sm_120` (consumer) / `sm_100` |

**Rows per process.** `ROW_CAP` (default 384) is the most streams one server
can step together; `--max-batch` and `--max-inflight` stop there. For a 48 GB
GPU build `make cuda-server cuda CUDA_ARCH=sm_89 ROW_CAP=1024`; the build
stamps the value and rebuilds when it changes.

`CUDA_ARCH=native` picks the GPU of the build machine. The CUDA objects live in
`build/cuda/`, separate from the CPU build (`make` / `make server`), and the
CPU binary never turns itself into a CUDA one.

## 3. Get the models

```bash
export HF_TOKEN=hf_...          # or put it in /root/.hf_token (never in the repo)
python3 -c "from huggingface_hub import snapshot_download as d; print(d('kyutai/pocket-tts'))"
python3 tools/convert_pocket.py --language english_2026-04_24l --output models/pocket-english-24l
python3 tools/convert_pocket.py --language english_2026-04     --output models/pocket-english-6l
```

The converter stores the weights in bfloat16 (as Kyutai's own
`switch_to_bf16.py` does); `--dtype source` keeps Kyutai's F32 checkpoint. They
measured equal in quality and speed, because the CUDA path expands weights to
F32 in GPU memory either way. On a fresh box `tools/gpu/provision.sh sm_89` does
build, tooling, download, conversion and a self-check in one step.

Check the GPU path before serving:

```bash
MYNAH_CUDA_KV_DTYPE=bf16 ./build/cuda/mynah-tts --pocket-self-check models/pocket-english-24l --device cuda
# pocket batching self-check: PASS
./build/cuda/mynah-tts --gpu-self-test cuda     # model-free kernel self-tests
```

With the defaults (TF32, bf16 weights and SEANet operands) the backend is not
batch-invariant, so the self-check compares within a tolerance. The pedantic,
batch-invariant setup makes it compare bit for bit (a text pushed in pieces vs
the same text whole); see "Pedantic mode" in section 7.

## 4. Start the server

Let the profile write the command, so nothing is forgotten:

```bash
python3 tools/perf_profile.py command l4-24g-pocket-en-24l-cuda --model models/pocket-english-24l --port 8080
```

which prints the profile's configuration:

```bash
MYNAH_QUANT_GROUPS=none MYNAH_SERVE_PROFILE=1 MYNAH_THREADS=1 \
  ./build/cuda/mynah-tts-server --device cuda -w 8 \
  --max-batch 320 --max-inflight 320 -p 8080 -m models/pocket-english-24l
```

(320 is the screening level of the current defaults; the soak-qualified
level is still C160, so `--max-batch 160 --max-inflight 160` is the
conservative choice until the C320 soaks land.)

(`MYNAH_SERVE_PROFILE=1`, which the profile also prints, only adds a
scheduler report at shutdown.) For the small model use its profile, or the same
line with `--max-batch 256 --max-inflight 256` and `models/pocket-english-6l`
(its profile pins `MYNAH_CUDA_KV_DTYPE=bf16`, what it was qualified with).

| flag | meaning |
|---|---|
| `--device cuda` | run on the GPU; the server refuses to start rather than silently run on the CPU |
| `--max-batch N` | streams batched in one GPU step; set it to the target concurrency |
| `--max-inflight N` | streams admitted at once; same value; up to the build's `ROW_CAP` on CUDA (384 by default, section 2; 128 on CPU) |
| `-w N` | HTTP connection workers (8 was used; the GPU work is one scheduler thread) |
| `--host 0.0.0.0` | expose beyond localhost; there is no authentication or TLS, put a proxy in front |

A healthy start prints lines like:

```
mynah-tts: backbone KV int8 with one float scale per position and head (1088 of 2048 bytes per position and plane); ...
mynah-tts: CUDA quant=bf16 backbone=bf16 flow=f32 mimi=f32 kv=int8 bf16_stages=backbone bf16_fuse=on
mynah-tts: pocket backend=cuda resident=on ... resident{backbone=on flow=on codec_transformer=on}
```

`kv=int8` (or `kv=bf16` when rolled back) and `resident=on` are the two things
to look for. Each default-on
feature of section 7 also prints one line when it is active (shared voice
prefix, split decode attention, one sync per frame, fused decoder, cuBLASLt
bf16 Linears, bf16 tensor-core prefill), so the log shows what ran.

## 5. Send requests

The API is OpenAI-shaped ([server.md](server.md) has every field). Stream raw
PCM as it is produced:

```bash
curl -N -X POST http://localhost:8080/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"pocket","input":"Hello from the GPU.","voice":"alba","response_format":"pcm","stream":true}' \
  | ffplay -f s16le -ar 24000 -ac 1 -nodisp -autoexit -
```

PocketTTS streams 16-bit mono PCM at **24 kHz**, one 80 ms frame at a time; the
headers carry `X-Sample-Rate`. `GET /v1/voices` lists the voices of the pack
(alba, marius, javert, jean, ...). `"seed"` makes a request reproducible; drop
`"stream"` to receive one WAV file.

```python
import http.client, json
c = http.client.HTTPConnection("localhost", 8080)
c.request("POST", "/v1/audio/speech", json.dumps({"input": "Streaming from Python.",
          "voice": "alba", "response_format": "pcm", "stream": True}),
          {"Content-Type": "application/json"})
r = c.getresponse()
while chunk := r.read1(65536):
    pass  # play or forward each chunk as it arrives
```

## 6. How many streams on your GPU

The GPU is the limit, not memory and not the host CPU. Each extra stream slows
every stream a little: the per-stream realtime factor rises linearly with the
number of streams (large model on the L4: 0.71 at 128, 0.86 at 160, 0.99 at
192). Pick the highest concurrency whose **stream RTF p95 stays below 0.90**
with zero stalls; beyond ~1.0 streams fall behind playback.

- Large model: ~75 MB of GPU memory per stream plus ~1.8 GB fixed with the
  2026-09-28 defaults; with the current ones (memory diet, int8 KV) C320 peaks
  at 16.3 GB.
- Small model: ~30 MB per stream plus ~1 GB fixed.
- A different GPU needs its own screen: `tools/gpu/knee.sh <tag> <model> 128,160,192,224,256`
  prints the top level under the gate (section 10).

For 1,000 concurrent listeners on L4s: 4 GPUs for the small model, 7 for the
large one.

**A larger GPU (L40S, RTX 6000 Ada).** On a GPU this fast one scheduler
thread, not the GPU, sets the ceiling, so the build, the host and the
serving-loop defaults matter more than on the L4:

- Build with `ROW_CAP=1024` (section 2) and start with `--max-batch 1024
  --max-inflight 1024`.
- With the eleven serving-loop defaults alone an L40S holds **C768** (867
  audio-s/s, stream RTF p95 0.829); with the large-row defaults of section 7
  (on since 2026-10-07, no variable to set) it holds **C1024 at RTF p95 0.746,
  1283 audio-s/s, TTFA p95 106 ms**, the GPU 95 % busy (2-minute screens on a
  Xeon Gold 6430 host).
- Pin the server to the GPU's NUMA node, prefer a host with fast cores, and
  check the open-file limit ([server.md](server.md#open-file-limit)).
- **A small-core host (4 vCPUs) is enough for C768.** With the server confined
  to two physical cores plus their SMT siblings (the delivery helpers then
  default to 1), the combined large-row configuration measured C512 1131
  audio-s/s / RTF p95 0.445, C640 1135 / 0.549, C768 1126 / 0.658 with the GPU
  87 % busy, ~3 % below the same server on the whole host. C896 (≈ 0.77) and
  C1024 (≈ 0.88, at the gate) are projected from that slope, not measured: on
  4 vCPUs plan for C768-C896 and screen C1024 before relying on it. Fast cores
  matter more than many cores, because one scheduler thread sets the pace.
- Copy-paste commands for all of this: "Quick start: large-row serving" in
  section 7.
  This is a **4-vCPU affinity experiment**: `taskset` on a larger host keeps that host's memory bandwidth, caches, clock and NUMA layout, so it is not a measurement on a real 4-vCPU instance; qualify one before relying on it.

Per-GPU thresholds, configuration by configuration, and the host lessons:
[performance.md, "Pocket CUDA serving thresholds (2026-10)"](performance.md#pocket-cuda-serving-thresholds-2026-10).
No 30-minute qualification has been run on these GPUs yet.

## 7. Feature flags (environment variables)

The defaults are the tuned configuration. Together, the 2026-10-02 rows of the
table below (shared voice through one sync) read on the L4 24L 60-s knees:
stream RTF p95 0.594-0.607 at C208 and 0.710-0.714 at C256, ~303-306 audio-s/s,
TTFA p95 ~102-104 ms at C208, against 1.03 with stalls at C208 for the
2026-09-28 defaults. The 2026-10-07 rows (deferred release through the
delivery helpers, then the host-context pool, the step and decode overlap and
the fixed slot pool) take the serial host loop off the critical path: see
"Serving-loop defaults" below. **Set only the required
variable**; everything else exists to roll a change back while debugging, or
to opt into something that is not a production default. A profile run refuses
to start if a "must be absent" variable is set.

### Required

| variable | value | why |
|---|---|---|
| `MYNAH_THREADS` | `1` | The GPU does the work; one CPU thread avoids a 128-thread pool on a big host |

`MYNAH_QUANT_GROUPS=none` is what the qualified runs set; a CUDA build now
resolves the default to the same thing, so it is harmless and optional. It does
not turn the bf16 backbone off: the groups select int8 encodings, and the f32
groups (`none` = all of them) are what `MYNAH_CUDA_QUANT` upgrades to bf16.
`MYNAH_CUDA_QUANT=f32` is the switch for fp32 weights. The bf16 default is the
Pocket CUDA engine's own (unset `MYNAH_CUDA_QUANT`); an explicit
`MYNAH_CUDA_Q8` (the expert int8 recipe) keeps the f32 base, and CPU builds
never read it.

### On by default (leave unset; the value shown turns the feature off)

| variable | rollback | what it does | measured (L4) |
|---|---|---|---|
| `MYNAH_CUDA_RESIDENT` | `0` | backbone and flow head resident on the GPU | the base of everything |
| `MYNAH_CUDA_FLOW` | `0` | flow head batched on the GPU | |
| `MYNAH_CUDA_POCKET_CODEC` / `MYNAH_CUDA_POCKET_DECODER` | `0` | Mimi codec and SEANet decoder on the GPU | |
| `MYNAH_CUDA_GRAPHS` | `0` | CUDA graphs for the per-step work | |
| `MYNAH_CUDA_MIMI_TILE` | `0` | codec transformer once per frame for all streams | C8: 2.9x -> 10.3x realtime |
| `MYNAH_CUDA_PREFILL_TILE` | `0` | all pending text read in one GPU pass; voice prompt copied on the GPU | C32: 8.5x -> 30.5x, prefill 160 -> 1.5 ms |
| `MYNAH_CUDA_PREFILL_BATCH` | `0` | cross-request text prefill | |
| `MYNAH_CUDA_TF32` | `0` | FP32 matrix products on TF32 tensor cores | C32: +26% |
| `MYNAH_CUDA_TILE_CUBLAS` | `0` | codec tile projections through cuBLAS | C48: 43.9x -> 58.1x |
| `MYNAH_CUDA_TILE_SPLITK` | `0` | split-K in the fixed-order tile GEMM | neutral |
| `MYNAH_CUDA_TILE_ATTN_GROUPED` | `0` | shared-memory tile attention | re-check pending |
| `MYNAH_CUDA_DECODER_CONVTR_GEMM` | `0` | decoder upsampling convolutions as one GEMM | C48: 24.5x -> 48.7x |
| `MYNAH_CUDA_DECODER_BATCH` / `MYNAH_CUDA_DECODER_ARITHMETIC_BATCH` | `0` | decoder batched across streams | |
| `MYNAH_CUDA_DECODER_GRAPHS` | `0` | CUDA graphs for the decoder | |
| `MYNAH_CUDA_DECODER_GRAPH_REUSE` | `0` | one decoder graph per batch width | +2% |
| `MYNAH_CUDA_CODEC_GANG` / `MYNAH_CUDA_CODEC_UPSAMPLE` / `MYNAH_CUDA_CODEC_DEVICE_HANDOFF` | `0` | codec input and PCM output batched, kept on the GPU | C8: +20% |
| `MYNAH_CUDA_BACKBONE_ATTN` | `legacy` | fast decode attention (128 positions at once) | C64: +22% |
| `MYNAH_CUDA_SLOT_POOL` | `0` | GPU buffers of finished streams reused, not freed | +4% |
| `MYNAH_CUDA_KV_GROW` | `0` | attention cache grows with the text instead of being sized for 120 s | GPU memory -48% |
| `MYNAH_CUDA_SEANET_BF16` | `0` | SEANet decoder convolution GEMMs with bf16 operands on tensor cores; fp32 accumulation, causal states and audio | L4 24L: stream RTF p95 -3 to -4 %, +3-4 % audio-s/s; SNR 47.5-49 dB vs fp32 at temperature 0 (`tests/codec_int8_quality.py --mode seanet-bf16`) |
| `MYNAH_CUDA_SLOT_POOL_PREFILL` | `0` | the slot pool is filled at start-up (all `--max-batch` sets) instead of on the first burst | a fresh server's first burst runs like a warm one |
| `MYNAH_CUDA_WIDTH_BUCKETS` | `0` | gang widths rounded up to a few buckets, their graphs captured at start-up | no graph capture during traffic; longer start-up |
| `MYNAH_CUDA_SHARED_VOICE` | `0` | decode attention reads each stream's voice prefix from one shared device copy (stays in L2); with `MYNAH_CUDA_SHARED_VOICE_STRIP` (default on, `0` keeps the per-row copy) the rows no longer store the prefix (~12 MB less per stream on 24L) | L4 24L: stream RTF p95 -10/11 % at C160-C208, +13 % audio-s/s at C208; bit-identical audio |
| `MYNAH_CUDA_ROW_MEM_DIET` | `1` | per-request device memory that is not request state is shared across the decoder gang or right-sized (SEANet scratch 5.1 -> 1.7 MB, Mimi transformer KV 500 -> 265 positions; only placement changes) | L4 24L: start-up -1.3 GB at 256 rows; C288 RTF p95 0.819 with 0 stalls, where it fails for VRAM without it |
| `MYNAH_CUDA_KV_DTYPE` | `bf16` (or `f32`, the parity oracle) | backbone attention cache stored as int8 with one float scale per position and head (1088 instead of 2048 bytes per position and plane on 24L); the shared voice prefix stays bf16. Needs the prefill tile and the shared voice prefix with strip, else bf16. Rows without a shared model-voice KV (a voice that is not an entry of the pack, so there is no voice prefix to skip) run their backbone on the CPU: set `bf16` for such workloads. Built-in voices and voices cloned into the pack are model voices and use int8 | L4 24L: C256 RTF p95 0.745 -> 0.681, 315 -> 345 audio-s/s; C320 0.855 with 0 stalls; peak memory 21.3 GB (C288) -> 16.3 GB (C320); same audio length per request. WER pending (soak) |
| `MYNAH_CUDA_ATTN_SPLIT` | `0` | split (flash-decoding) layout of the batched bf16 decode attention | C208 0.705 -> 0.674, C240 0.787 -> 0.749, +7 % audio-s/s; deterministic and batch-invariant, last-bit differences vs the old kernel |
| `MYNAH_CUDA_QUANT` | `f32` | bf16 weight copies for the backbone Linears (fp32 accumulate; residual stream, norms, flow head, EOS head and codec stay fp32); `MYNAH_CUDA_QUANT_STAGES` defaults to `backbone` | with the two rows below, C208 0.676 -> 0.607, C256 0.788 -> 0.714, 266 -> 303 audio-s/s, TTFA p95 115 -> 104 ms |
| `MYNAH_CUDA_BF16_FUSE` | `0` | bf16 decode layers fused: LayerNorm, GELU and attention write the bf16 GEMM operand, biases folded into the consumers | 456 -> 264 graph nodes per step; bit-identical to the unfused bf16 path |
| `MYNAH_CUDA_BF16_LT` | `0` | bf16 Linears through cuBLASLt with a per-shape heuristic algorithm (created only when bf16 weights are in use) | without it bf16 was 10 % slower than TF32 |
| `MYNAH_CUDA_PREFILL_BF16TC` | `0` | fixed-order text prefill on bf16 tensor cores (bf16 weights only; `=1` explicitly also covers f32 weights) | keeps "text in pieces == text whole" bit for bit at bf16 weight traffic |
| `MYNAH_CUDA_DECODER_FUSE` | `0` | SEANet gang decoder: ELU, causal window, residual add, conv bias and the transposed-conv fold folded into the kernels that read them; `MYNAH_CUDA_DECODER_FUSE_BIAS=0` keeps only the bias on the old path | 36 -> 20 elementwise launches per step; bit-identical audio |
| `MYNAH_CUDA_ONE_SYNC` | `0` | condition, backbone, EOS and flow head share one stream sync per batched frame | 3 fewer syncs per frame; same audio |
| `MYNAH_POCKET_VOICE_CACHE` | `0` (or `all` / `startup` to preload) | voice prompts cached on first use | |
| `MYNAH_CUDA_DEFERRED_RELEASE` | `0` | a finished stream's GPU set is parked behind a stream event, waited on when it is reused, instead of a full stream drain at release | 2026-10-07 serving-loop default; ~2.4 fewer syncs per step |
| `MYNAH_CANCEL_CHECK_EVERY` | `1` (or `0`) | client disconnects are polled every 4 steps instead of every step (`N` sets the interval); a disconnect is noticed up to 3 frames later | 2026-10-07 serving-loop default |
| `MYNAH_STREAM_OUT_WRITEV` | `0` | each HTTP chunk leaves in one `writev()` instead of three `send()` calls; same bytes on the wire | 2026-10-07 serving-loop default |
| `MYNAH_DUP_CHECK_EPOCH` | `0` | the "same request twice in a step" check stamps each request once per step instead of comparing every pair | 2026-10-07 serving-loop default |
| `MYNAH_CUDA_DECODER_TABLE_PATCH` | `0` | a decoder gang change of the same width patches the graph's pointer tables instead of re-recording the decoder (follows `MYNAH_CUDA_DECODER_GRAPH_REUSE`) | decoder re-records per level ~2,000 -> 1-160 |
| `MYNAH_CUDA_ONESYNC_SUBSET` | `0` | when some rows end in a step, the survivors reuse the chained flow-head pass instead of a second one | 2026-10-07 serving-loop default; ~1.6 fewer syncs per step |
| `MYNAH_STREAM_DELIVER_THREADS` | `0` (or `N` helpers) | PCM conversion and hand-off to each stream's writer run on helper threads, not the scheduler thread. CUDA serving only; the count follows the cpus the process may use (affinity mask, capped by a cgroup cpu quota): 1 helper up to 4 cpus, 2 up to 8, 4 above. A full stream queue is found by the helper, one step later | largest single loss when removed on the L4 |
| `MYNAH_CUDA_PCM_DIRECT` | `0` | the gang decode lends each stream a reusable PCM buffer and copies a frame once, straight from the pinned gang rows (the CPU backend keeps the old path unless set to `1`) | 2026-10-07 serving-loop default |
| `MYNAH_CUDA_HIDDEN_LAZY` | `0` | one-sync steps keep the backbone output on the GPU (finite check on the GPU, host copy only on demand) | 2026-10-07 serving-loop default |
| `MYNAH_CUDA_KV_TABLE_CACHE` | `0` | the attention-cache pointer tables are rewritten only for rows whose cache changed | 2026-10-07 serving-loop default |
| `MYNAH_CUDA_DECODER_VALIDATE_ONCE` | `0` | the decoder gang's topology check runs once per decoder, then by compatibility class | 2026-10-07 serving-loop default |
| `MYNAH_CTX_HOST_POOL` (A1a) | `0` (`2` keeps it on and zeroes renewed caches, the leak A/B) | a finished request's host-side state (backbone and codec states, flow head, scratch) is renewed for the next admission instead of rebuilt. CUDA backend only; the CPU backend keeps it opt-in (`=1`) | 2026-10-07 large-row default; L40S C1024 880 -> 1112 audio-s/s, RTF p95 1.076 -> 0.852 |
| `MYNAH_CUDA_STEP_OVERLAP` (L13) | `0` (turns the whole overlap off: L13, L13b and L13d) | dispatch-ahead: the next AR step is queued before the previous step's retire, admission and cancellation run. CUDA backend only | 2026-10-07 large-row default, as one package with the two rows below (alone it gave the same throughput with TTFA p95 +45-60 ms) |
| `MYNAH_CUDA_DECODE_OVERLAP` (L13b, needs L13) | `0` (this half only) | the gang decode of step k runs under AR step k+1 | 2026-10-07 large-row default; L40S C896 passes (RTF p95 0.853) where the eleven defaults fail |
| `MYNAH_CUDA_FIRST_FRAME_FIRST` (L13d, with L13b) | `0` (this part only) | new streams' first frames are decoded and delivered as their own small gang first | 2026-10-07 large-row default; TTFA p95 -50 to -70 ms for ~1-2 % throughput |
| `MYNAH_CUDA_SLOT_FIXED` (A1b; `MYNAH_CUDA_SLOT_FIXED_POSITIONS`, `MYNAH_CUDA_SLOT_FIXED_ROWS`) | `0` | pooled request sets keep a backbone KV of at least a fixed size F, so taking one from the pool makes no allocation or free. F starts at 384 stored positions (19.1 MiB with int8 KV at 24 layers, ~26 s of audio) and moves once, after 1024 served requests, to the 95th percentile of the lengths requests really reached (320..768); `_POSITIONS` fixes it. The cap is provisional at load; the server re-plans it after its start-up walk and again before traffic (`mynah_tts_startup_mark`): it frees the walk's long caches, measures free memory and caps at min(rows + spares, (free - max(2 GiB, total/16)) / F), then fills the parked sets and a small growth reserve (1 in 32 rows, F + 256 positions); when nothing fits it falls back to the plain slot pool. A row that outgrows its cache grows with no device-wide sync (a spare, a parked set's larger cache, or a new allocation within the cap; the old cache is parked, never freed), and a take never frees a parked cache. Needs the slot pool, KV grow and the prefill tile; `MYNAH_SERVE_PROFILE=1` shows zero-call vs fallback takes, the fixed-cache counts, growths by source and the served length percentiles in `[CTX]` | 2026-10-07 large-row default; L40S C1024 1156 -> 1283 audio-s/s, RTF p95 0.843 -> 0.746, VRAM 34 GB at the end; RTX 6000 Ada C768 455 -> 1114 audio-s/s (first version) |

### Opt-in (off by default)

The rows here are not production settings. (The large-row rows that used to be
here, the host-context pool, the step and decode overlap with first frame
first, and the fixed slot pool, are on by default since 2026-10-07; see the
table above.)

| variable | effect | why it is off |
|---|---|---|
| `MYNAH_CUDA_PREFILL_FIXED=0` | text prefill through cuBLAS (bf16 or TF32 by the weights): on f32 weights it measured -7 % RTF p95 | a text sent in pieces no longer gives bit-identical audio to the same text sent whole; a product decision |
| `MYNAH_CUDA_KV_VMM=1` (with `MYNAH_CUDA_KV_DTYPE=bf16`; ignored with int8 KV) | backbone KV rows are VMM ranges that grow in place (no copy, no second allocation, no sync); with `MYNAH_CUDA_KV_VMM_CHUNK=64 MYNAH_CUDA_KV_GROW_INITIAL_STEPS=64` rows start small | L4, `--max-batch 320`, with the diet: start-up 9.6 GB (was 16.0), C320 runs with 0 failures at peak 14.4 GB, but the position-major layout costs ~4 % (C288 0.851 vs 0.819) and C320 is compute-bound (0.926): off on the L4; for GPUs with more compute than memory headroom |
| `MYNAH_CUDA_QUANT_STAGES=all` | bf16 weights for the flow head and the Mimi transformer too | no measurable gain over `backbone` (C256 0.714 -> 0.712) |
| `MYNAH_CUDA_SYNC=blocking` | host thread sleeps while the GPU works: server CPU 117% -> 43% | -9% throughput (each wake-up idles the GPU) |
| `MYNAH_CUDA_STARTUP_WALK=1` | start-up graph walk over the width buckets only (short text, two steps each) and a two-step slot-pool prefill: L4 24L ready 57 -> 17 s at `--max-batch 320`, 210 -> 54 s at `ROW_CAP=1024`; 6L `ROW_CAP=2048` from not ready after 14 min to 21 s; bit-identical, C320 steady state equal | a C320 burst arriving at ready had a worse first 30 s (TTFA p95 994 vs 693 ms, 14 stalls at 250 ms vs 0; one run each). Use it for large `ROW_CAP` builds, where the default walk does not finish |
| `MYNAH_CUDA_TILE_TC=1` | own tensor-core fixed-order GEMM for the f32 prefill | +1-2% only |
| `MYNAH_CUDA_QUANT=int8` | int8 resident weights | diagnostic only |
| `MYNAH_CUDA_FAST_MATH=1` | FP16 GEMMs | not qualified |
| `MYNAH_CUDA_CODEC_BATCH=1` | older multi-row codec path | fails the waveform parity gate |
| `MYNAH_CUDA_ALLOW_CPU_STAGES=1` | lets hot stages run on the CPU | 20-30x slower while reporting CUDA: never in production |
| `MYNAH_CUDA_FAST_FIRST_CHUNK=1` (with `MYNAH_CUDA_FAST_FIRST_CHUNK_WAIT_US=N`, 0..20000, default 0) | late admission: a second admission and prefill pass right before each step, and, when a row retired in the previous step, a hold of up to N µs for a new arrival | L4 C64: with N = 3000, TTFA p50 69 -> 39 ms for -4.6 % throughput; at C160 the hold costs RTF (0.840 -> 0.892) for no TTFA gain. Only for deployments well below their knee; not part of the large-row defaults |
| `MYNAH_CUDA_PINGPONG=2` (L26; `MYNAH_CUDA_PINGPONG_MIN`, default 128 rows) | **experimental.** Two groups of rows in one engine: the scheduler finishes, admits and launches one group while the other group's decode and AR step run | RTX 6000 Ada, `--max-batch 768`, with A1a and A1b: 92 % of the host time hidden, but 7-8 % slower than L13 + L13b (now default) once the GPU is >= 93 % busy (C640 1110 vs 1205, C768 1037 vs 1111 audio-s/s), because of the stream syncs it reaches while an item is queued. Bit-identical. A 1024-row run ran out of device memory, so it was measured at 768 rows |

### Serving-loop defaults (2026-10-07) and large-row serving

At hundreds of streams the GPU waited on the single scheduler thread: host
time per step, not GPU time, set the ceiling. The eleven 2026-10-07 rows of the
"on by default" table cut that host time (fewer stream syncs, no per-step
decoder re-recording, no per-row copies, PCM hand-off on helper threads whose
count follows the usable CPUs). Each one gives bit-identical audio wherever
batch composition is the same (CLI `--batch 32`, CLI `--batch 32 --stream` and
the server at C1) and keeps its variable as a rollback.

**Large-row defaults (2026-10-07).** Five more rows of that table, first
measured as large-row opt-ins on the L40S, are on by default on CUDA serving
since they passed the same L4 regression check: the host-context pool (A1a),
the step and decode overlap with first frame first (L13 + L13b + L13d, one
package) and the fixed slot pool (A1b). Nothing needs to be set for them. Each
keeps a rollback:

```bash
MYNAH_CTX_HOST_POOL=0           # host state rebuilt per request
MYNAH_CUDA_STEP_OVERLAP=0       # the whole overlap package off (L13, L13b, L13d)
MYNAH_CUDA_DECODE_OVERLAP=0     # keep dispatch-ahead, decode at retire as before
MYNAH_CUDA_FIRST_FRAME_FIRST=0  # keep both overlaps, no first-frame gang
MYNAH_CUDA_SLOT_FIXED=0         # plain slot pool (a misfit take frees and reallocates)
```

They are on for the CUDA backend only; on the CPU backend they stay off
(opt-in where they apply), so CPU serving and a CPU CLI run are unchanged. A
CUDA CLI `--batch` run goes through the same serving loop and uses them too,
which is how the identity checks exercise them. The decode overlap
needs dispatch-ahead, so with `MYNAH_CUDA_STEP_OVERLAP=0` it stays off; the
fixed slot pool keeps its VRAM plan and falls back to the plain slot pool
when no fixed cache fits. Each was bit-identical in the same checks as the
eleven (CLI `--batch 32`, `--batch 32 --stream`, server C1; A1b in both its
first version and its v2 sizing; on the L4 32/32, 32/32 and 95/95).
`MYNAH_CUDA_PINGPONG` stays experimental and opt-in; when set it replaces the
overlap package.

Measured on an NVIDIA L40S (Xeon Gold 6430 host, server pinned to the GPU's
NUMA node), `ROW_CAP=1024` build, 2-minute levels:

| configuration | C1024 audio-s/s | stream RTF p95 | TTFA p95 | GPU busy |
|---|---:|---:|---:|---:|
| the eleven defaults | 880 | 1.076 (162,015 stalls) | 181 ms | 64 % |
| + host-context pool (A1a) | 1112 | 0.852 | 147 ms | 84 % |
| + step and decode overlap, first frame first (L13 + L13b + L13d) | 1156 | 0.843 | 127 ms | 88 % |
| + fixed slot pool (A1b): today's defaults | **1283** | **0.746** | **106 ms** | 95 % |

Before the eleven defaults the same GPU did not hold C768 on this host; with
them it holds C768 (0.829), and C896 runs with 0 stalls where it had 80,583. On
the L4 (GPU-bound, `ROW_CAP=384`) the eleven give +4 % audio-s/s and -0.03 RTF
p95 at C288-C320, with no regression.

The five large-row defaults on the NVIDIA L4 (Vast.ai, EPYC 7702 host, default
`ROW_CAP=384` build, 2-minute levels), audio-s/s / stream RTF p95 / TTFA p95:

| level | the eleven defaults | + A1a + A1b (pools only) | + A1a + L13/L13b/L13d + A1b (today's defaults) |
|---|---:|---:|---:|
| C256 | 354 / 0.715 / 117 ms (19 stalls) | 373 / 0.632 / 106 ms | **382 / 0.634 / 100 ms** (0 stalls) |
| C288 | 361 / 0.735 / 128 ms | 373 / 0.702 / 118 ms | **378 / 0.712 / 112 ms** |
| C320 | 367 / 0.788 / 137 ms | 381 / 0.757 / 126 ms | **382 / 0.775 / 121 ms** |

GPU busy goes from 88-90 % to 98 %. So the GPU-bound L4 gains too: up to
+8 % audio-s/s and 16-17 ms less TTFA p95 at every level. Every
configuration and GPU, with TTFA, power and the host lessons:
[performance.md](performance.md#pocket-cuda-serving-thresholds-2026-10).

### Quick start: large-row serving

Ready-to-run commands for one 48 GB GPU (NVIDIA L40S or RTX 6000 Ada) at up
to 1024 streams, as measured. The serving-loop defaults, the large-row ones
included, need no variable: the server line only sets threads and pinning.

**1. Build** with room for 1024 rows:

```bash
make cuda cuda-server CUDA_ARCH=sm_89 ROW_CAP=1024
```

**2. Find the GPU's NUMA node** and its CPU list (skip on a single-node host):

```bash
nvidia-smi topo -m            # "CPU Affinity" column of GPU0, e.g. 32-63,96-127
lscpu | grep 'NUMA node'
ulimit -Hn                    # must exceed the target concurrency (server.md, open-file limit)
```

**3. Start the server**, pinned to that node (L40S / RTX 6000 Ada, C1024):

```bash
MYNAH_THREADS=1 MYNAH_QUANT_GROUPS=none \
taskset -c 32-63,96-127 ./build/cuda/mynah-tts-server --device cuda -w 8 \
  --max-batch 1024 --max-inflight 1024 --max-pending 2048 \
  -p 8080 -m models/pocket-english-24l
```

Add `MYNAH_SERVE_PROFILE=1` for the `[SERVE]` report at shutdown and the
`[CTX]` lines during the run. Each large-row default prints one start-up line
when it is active (`MYNAH_CTX_HOST_POOL (default) ...`, `step overlap ON by
default`, `decode overlap ON by default ... first frames ...`,
`MYNAH_CUDA_SLOT_FIXED (default; =0 to roll back) ...`), each naming its
rollback; a missing line means the feature is off, and the reason is printed
instead. To roll one back for an A/B, put its `=0` from the block above in
front of the command, e.g. `MYNAH_CUDA_STEP_OVERLAP=0` for the whole overlap.

**4-vCPU variant.** The same command with the server confined to two physical
cores and their SMT siblings on the GPU's node (siblings:
`cat /sys/devices/system/cpu/cpu32/topology/thread_siblings_list`). The
delivery helpers default to 1 at ≤ 4 usable CPUs. Measured up to C768;
C1024 is projected at the gate, so start at 768:

```bash
MYNAH_THREADS=1 MYNAH_QUANT_GROUPS=none \
taskset -c 32,33,96,97 ./build/cuda/mynah-tts-server --device cuda -w 8 \
  --max-batch 768 --max-inflight 768 --max-pending 2048 \
  -p 8080 -m models/pocket-english-24l
```

**NVIDIA L4 variant (≤ 384 rows).** The default `ROW_CAP=384` build and the
defaults; the large-row defaults apply here too (C320 367 -> 382 audio-s/s,
TTFA p95 137 -> 121 ms against the eleven alone).

```bash
make cuda cuda-server CUDA_ARCH=sm_89
MYNAH_THREADS=1 MYNAH_QUANT_GROUPS=none ./build/cuda/mynah-tts-server --device cuda -w 8 \
  --max-batch 320 --max-inflight 320 -p 8080 -m models/pocket-english-24l
```

(`--max-batch 160 --max-inflight 160` is the soak-qualified level; 320 the
screened one, section 4.)

**4. Load test.** One level, closed loop, against the running server:

```bash
python3 tools/pocket_ladder.py --port 8080 --mode closed --levels 1024 \
  --warmup 15 --duration 120 --timeout 300 --corpus tools/corpus/pocket_v2_en.jsonl \
  --voices alba,marius,javert,jean --seed 1234 --client-procs 4 --out ladder-out --tag c1024
```

Run the load generator on other cores than the server (another NUMA node, or
another machine). For a full screen, `tools/gpu/knee_closed.sh` starts its own
server, runs the levels and prints one line each
([benchmarking.md](benchmarking.md#4-the-knee)):

```bash
TAG=large LEVELS="768 896 1024" PRE=640 PROCS=4 DUR=120 \
  PIN=32-63,96-127 CLIPIN=0-31,64-95 \
  timeout 1800 tools/gpu/knee_closed.sh
# A/B against a rollback: the same line with ENVS="MYNAH_CUDA_STEP_OVERLAP=0"
```

Pass: `rtf95` ≤ 0.88, `st250 0`, `fail 0` at the level you will serve.
Before the first level, run the 3-minute thermal check of benchmarking.md.

### Serving flag reference

Every variable of the October serving-loop work, as the code reads it.
"Default on" flags are on when unset and off at `=0` (any other value keeps
them on). Opt-ins are off when unset or `=0`. The large-row defaults (A1a,
L13/L13b/L13d, A1b) are on for the CUDA backend only. Effects are 2-minute screens
(performance.md); L40S = Xeon Gold 6430 reference host, L4 = EPYC 7702 host.

| flag | default | what it does | when to use | measured effect | rollback |
|---|---|---|---|---|---|
| `MYNAH_CUDA_DEFERRED_RELEASE` (L6) | on | a finished stream's GPU set is parked behind a stream event instead of a full stream drain | always | L4 ~2.4, L40S ~6 fewer syncs per step; no throughput or VRAM change | `=0` |
| `MYNAH_CANCEL_CHECK_EVERY` (L7) | 4 | client disconnects polled every N steps (1..1000); a disconnect is noticed up to N-1 frames later | always | L4 -3 audio-s/s at C288 without it | `=1` (or `=0`): every step |
| `MYNAH_STREAM_OUT_WRITEV` (L8) | on | one `writev()` per HTTP chunk instead of three `send()`; same bytes | always | a third fewer writer syscalls; neutral on the L4 | `=0` |
| `MYNAH_DUP_CHECK_EPOCH` (L10) | on | the "same request twice in a step" check stamps each request once instead of comparing pairs | always | neutral on the L4 (a C1024-scale saving) | `=0` |
| `MYNAH_CUDA_DECODER_TABLE_PATCH` (L11) | on (follows `MYNAH_CUDA_DECODER_GRAPH_REUSE`) | a decoder gang change patches the graph's pointer tables instead of re-recording | always | decoder re-records per level ~2,000 -> 1-160 | `=0` |
| `MYNAH_CUDA_ONESYNC_SUBSET` (L12) | on | survivors of a step where rows end reuse the chained flow-head pass | always | ~1.6 fewer syncs per step; L4 -5 audio-s/s without it | `=0` |
| `MYNAH_STREAM_DELIVER_THREADS` (L19) | auto, CUDA serving only: 1 helper at ≤ 4 usable CPUs, 2 at ≤ 8, 4 above | PCM conversion and hand-off on helper threads, not the scheduler; a full stream queue is seen one step later | always; `N` (1..64) to override | the largest single flag: L40S -5 % (C768 867 -> 822) without it, L4 -6 audio-s/s | `=0` |
| `MYNAH_CUDA_PCM_DIRECT` (L20) | on for the CUDA backend | the gang decode lends each stream a PCM buffer, one copy per frame | always | L4 -3 to -4 audio-s/s without it | `=0` |
| `MYNAH_CUDA_HIDDEN_LAZY` (L21) | on | one-sync steps keep the backbone output on the GPU; host copy on demand | always | L40S ~-2 % without it | `=0` |
| `MYNAH_CUDA_KV_TABLE_CACHE` (L22) | on | KV pointer tables rewritten only for rows whose cache changed | always | neutral on the L4 | `=0` |
| `MYNAH_CUDA_DECODER_VALIDATE_ONCE` (L24) | on | decoder gang topology check once per decoder, then by compatibility class | always | neutral on the L4 | `=0` |
| `MYNAH_CTX_HOST_POOL` (A1a) | on for the CUDA backend (CPU: off) | a retired request's host state is renewed for the next admission instead of rebuilt; `=2` also zeroes renewed caches (leak A/B) | always | L40S C1024 880 -> 1112 audio-s/s, RTF p95 1.076 -> 0.852; L4 with A1b C256 354 -> 373, C320 367 -> 381 | `=0` |
| `MYNAH_CUDA_STEP_OVERLAP` (L13) | on for the CUDA backend (CPU: off), with L13b and L13d as one package | the next AR step is queued before retire, admission and cancellation | always (as the package) | the package, on top of A1a + A1b on the L4: +1 to +9 audio-s/s and TTFA p95 -5 to -6 ms (C320 381 -> 382, 126 -> 121 ms) for RTF p95 +0.002 to +0.018; on top of A1a on the L40S C1024: 1112 -> 1156 audio-s/s, RTF p95 0.852 -> 0.843, TTFA p95 147 -> 127 ms. Alone it gave the same throughput with TTFA p95 +45-60 ms | `=0`: the whole package off |
| `MYNAH_CUDA_DECODE_OVERLAP` (L13b) | on with L13; off when L13 is off | the gang decode of step k runs under step k+1 | always | L40S C896 passes (0.853) where the eleven defaults fail | `=0`: this half only |
| `MYNAH_CUDA_FIRST_FRAME_FIRST` (L13d) | on with L13b | new streams' first frames decoded and delivered first, as their own small gang | always | TTFA p95 -50 to -70 ms for ~1-2 % throughput | `=0`: this part only |
| `MYNAH_CUDA_FAST_FIRST_CHUNK` + `_WAIT_US` | off; wait 0 µs (max 20000) | late admission pass before each step, optional hold for an arrival | low load, well below the knee | L4 C64 TTFA p50 69 -> 39 ms at 3000 µs, -4.6 % throughput; hurts RTF near the knee. `_WAIT_US` alone does nothing | unset or `=0` |
| `MYNAH_CUDA_SLOT_FIXED` (A1b) | on for CUDA serving; needs the slot pool, KV grow and prefill tile (all default on); the VRAM cap and its auto-cap stay, and the plain pool is used when nothing fits | pooled sets keep a fixed-size backbone KV: an admission makes no `cudaFree` / `cudaMalloc`; growth without a device-wide sync | always; most visible on hosts where `[CTX] cuda_backbone` is in ms | L40S C1024 1156 -> 1283 audio-s/s, RTF p95 0.843 -> 0.746; RTX 6000 Ada C768 455 -> 1114 | `=0` |
| `MYNAH_CUDA_SLOT_FIXED_POSITIONS` | 384, then the served p95 after 1024 requests | fixed cache size F (64..65536, rounded up to 64) | only to pin F | | unset |
| `MYNAH_CUDA_SLOT_FIXED_ROWS` | the build's `ROW_CAP` | upper bound on fixed caches (1..`ROW_CAP`) | to cap A1b's VRAM | | unset |
| `MYNAH_CUDA_PINGPONG` (L26) | off; only `=2` accepted | two row groups in one engine, one's host work under the other's GPU work; replaces L13 / L13b when set | experimental | 7-8 % slower than L13 + L13b once the GPU is ≥ 93 % busy | unset or `=0` |
| `MYNAH_CUDA_PINGPONG_MIN` | 128 rows | below this many rows the loop does not split | with L26 only | | unset |
| `MYNAH_SERVE_HOST_THREADS` | off (opt-in) | the serving loop's per-row host loops (one-sync row staging, gang landing, backbone row preparation, Mimi window advance, decoder table patch) run on a private team of N threads, scheduler included; rows claimed in fixed chunks, no CUDA call on a worker, order-dependent work merged serially in row order. `=auto`: off up to 8 usable CPUs, 2 up to 16, 4 above | host-bound GPUs (L40S class); not below 8 usable CPUs | bit-identical (CLI 32/32, stream 32/32, C1 95/95); L4 C320: gang landing 0.51 -> 0.28 ms, row staging 0.25 -> 0.15 ms per iteration (`.work/cuda-host-mt-2026-10-08.md`) | unset or `=0` |
| `MYNAH_SERVE_HOST_MIN_ROWS` / `MYNAH_SERVE_HOST_SPIN_US` | 64 rows / 50 µs | regions below this many rows run inline; a worker spins this long after a region before parking | with `MYNAH_SERVE_HOST_THREADS` | | unset |
| `MYNAH_CUDA_PREFILL_PINNED` | **on** (`=0` rolls back) | the prefill tile stages the text embeddings in one pinned buffer and uploads them with one copy, instead of a pageable `cudaMemcpyAsync` per row, which first waited for the decode gang queued just before it | always | L4 C320: the upload was 9.3 ms of host time per iteration (12.7 ms per call); same bytes, bit-identical; L4 C320 A/B, two rounds: 376-378 vs 376-386 audio-s/s, TTFA p95 123-124 vs 120-123 ms (GPU-bound there, so neutral) | unset or `=0` |
| `MYNAH_CUDA_MIMI_STALE_WINDOW` | off (opt-in, `=1`) | a row owned by the Mimi tile advances its host codec window without copying its stale host K/V (never read again: the K/V lives in the device ring) | always, once screened on a host-bound GPU | L4 C320: the copy was ~9 µs per row and frame, 2.0 ms per iteration; bit-identical | unset or `=0` |
| `MYNAH_CUDA_STARTUP_WALK` | off (opt-in, CUDA serving with `MYNAH_CUDA_WIDTH_BUCKETS`) | `=1`: the start-up graph walk visits each width bucket for two steps on a short text (`4 + 2 x (buckets + 2)` steps) instead of stepping every width from `--max-batch` down to 1 on a long text, and the slot-pool prefill runs two steps per request. The server prints `start-up phases: ...` when ready, either way | large `ROW_CAP` builds and fast restarts, where start-up time matters more than the first 30 s after ready; the full-width walk grows with the square of `--max-batch` and its long-text caches can fill the GPU | L4 24L: `--max-batch 320` ready 57 -> 17 s, VRAM at ready 16.0 -> 11.0 GB; `ROW_CAP=1024` 210 -> 54 s (and the walk captures its graphs, where the old one captured none); 6L `ROW_CAP=2048` not ready after 14 min -> 21 s. Bit-identical; C320 steady state equal. First 30 s of a C320 burst at ready: TTFA p95 994 vs 693 ms, 14 stalls vs 0 (one run each), hence opt-in | unset or `=0` |
| `MYNAH_SERVE_PROFILE` | off | `[SERVE]` report at shutdown, `[CTX]` lines, per-feature counters; `=2` adds the `[HOSTP]` per-phase host table (wall, device wait, host, rows and µs per row for each loop and engine phase) | diagnostics, benchmarks | | **unset**: any value, `=0` included, turns it on |

`MYNAH_CUDA_PINGPONG` warns when deferred release is rolled back
(`MYNAH_CUDA_DEFERRED_RELEASE=0`) and when the fixed slot pool is rolled back
(`MYNAH_CUDA_SLOT_FIXED=0`) without `MYNAH_CUDA_KV_VMM=1`, because a KV growth
may then drain the device (`MYNAH_CUDA_KV_VMM` applies only to bf16 KV).

### Pedantic mode (rollback to fp32, batch-invariant)

Every default above that changes numerics can be switched off. The fp32,
batch-invariant setup (what the bitwise self-checks and golden comparisons
assume) is:

```bash
MYNAH_CUDA_TF32=0 MYNAH_CUDA_SEANET_BF16=0 MYNAH_CUDA_QUANT=f32 MYNAH_CUDA_ATTN_SPLIT=0 \
  MYNAH_CUDA_KV_DTYPE=f32 ./build/cuda/mynah-tts --pocket-self-check models/pocket-english-24l --device cuda
```

`MYNAH_CUDA_QUANT=f32` restores fp32 weights everywhere (prefill included);
`MYNAH_CUDA_KV_DTYPE=f32` the fp32 attention cache (the CPU/GPU parity oracle);
`MYNAH_CUDA_ATTN_SPLIT=0` restores the summation order of the old decode
attention (the split kernel is batch-invariant too, but its last bits differ).
The other 2026-10-02 defaults (shared voice, decoder fusion, one sync, bf16
fusion) and the 2026-10-07 serving-loop defaults give bit-identical audio and
need no switch. Without
`MYNAH_CUDA_QUANT=f32` the bf16 backbone makes the self-check fall back to its
tolerance comparison.

### Diagnostics and tests

| variable | use |
|---|---|
| `MYNAH_SERVE_PROFILE=1` | per-batch-width step times and loop breakdown printed at shutdown |
| `MYNAH_SERVE_PROFILE=2` | the same plus `[HOSTP]`: where the scheduler thread's host time goes, per phase |
| `MYNAH_COST_MAP=2 MYNAH_NVTX=1` | NVTX ranges for Nsight Systems |
| `MYNAH_CUDA_KV_GROW_LOG=1` | log each attention-cache growth |
| `MYNAH_CUDA_KV_GROW_INITIAL_STEPS=N` | force growth early (bit-identity tests) |
| `MYNAH_CUDA_SLOT_POOL_ZERO_KV=1` | zero reused caches (leak A/B: audio must not change) |
| `MYNAH_CUDA_GRAPH_TRACE=1` | log each pipeline CUDA graph capture (key, record and instantiate time) |

## 8. Monitoring

- `GET /health`: liveness, limits, active streams.
- `GET /metrics`: Prometheus text. Useful CUDA counters: `mynah_backend_graph_replays_total`
  vs `..._graph_captures_total` (captures should stay flat after warm-up),
  `mynah_backend_decoder_graph_*`, the batch-width histograms, `..._sync_calls_total`,
  H2D/D2H bytes. Polling it does not synchronise the GPU.
- `nvidia-smi`: memory should reach a plateau and stay there; utilisation
  85-95% at the qualified levels on an L4; 87-95% `dmon sm` on an L40S at
  C1024 with the large-row variables of section 7 (~65% with the defaults
  alone, where the GPU waits on the host).
- The server process using 100-125% of a CPU core is normal: the scheduler
  thread spins while it waits for the GPU (that spin is faster than sleeping).

## 9. Troubleshooting

| symptom | cause and fix |
|---|---|
| server refuses to start mentioning CPU stages | a configuration would put a hot stage on the CPU; remove `MYNAH_QUANT_GROUPS`/`MYNAH_CUDA_QUANT` overrides |
| out of GPU memory | concurrency too high for the card, or `MYNAH_CUDA_KV_GROW=0` / FP32 cache set; check `kv=int8` (or `kv=bf16`) at start-up |
| some requests are much slower than the rest | with int8 KV (the default) a row without a shared model-voice KV (a voice that is not an entry of the pack) runs its backbone on the CPU, logged once at the first such row; set `MYNAH_CUDA_KV_DTYPE=bf16` |
| stalls or RTF p95 above 0.9 | too many streams for this GPU: lower `--max-batch/--max-inflight` or re-screen (section 6) |
| slow first request per voice | the voice prompt is loaded on first use; `MYNAH_POCKET_VOICE_CACHE=startup` preloads |
| the first seconds after start are slower | CUDA graphs are captured per batch width on first use |
| a voice sounds rough or noisy on every clip | the built-in voice's own timbre (Pocket clones its reference clip, noise included): marius and javert are the roughest, alba the cleanest ([pocket-voices.md](pocket-voices.md)); use a clean reference clip for custom voices |
| speech starts ~1 s after the first audio | javert (and less so jean) open with silence copied from the reference clip; TTFA counts that silence |
| metallic timbre on short sentences | model behaviour of the rougher voices (marius, javert) on one-liners, worst on the 24-layer model; not load-related. Prefer alba; count it with `tools/pocket_audio_noise.py` |

## 10. Qualifying a new GPU box

The same procedure produced the numbers above; all scripts run in tmux and
finish without an ssh session:

```bash
echo "hf_..." > /root/.hf_token && chmod 600 /root/.hf_token
tools/gpu/detach.sh prov tools/gpu/provision.sh sm_89
tools/gpu/detach.sh knee bash -c 'tools/gpu/wait_done.sh prov; tools/gpu/knee.sh knee24 models/pocket-english-24l 128,144,160,176,192'
tools/gpu/detach.sh qual bash -c 'tools/gpu/wait_done.sh knee; MYNAH_Q_CONTROL=1 tools/gpu/qualify.sh pocket-24l-c160 models/pocket-english-24l 160'
```

`qualify.sh` runs two independent 30-minute saturated soaks on the v2 corpus
(`tools/corpus/pocket_v2_en.jsonl`), captures the audio actually streamed,
transcribes it, optionally re-runs the same requests unloaded (C8) and packs an
evidence bundle with capped listening ZIPs. Record the result as a profile
(`tools/perf_profile.py new <id> --like l4-24g-pocket-en-24l-cuda`). Judge
throughput only from 30-minute soaks: short screens under-count long
utterances. Report ASR quality per voice and per length class, against the
unloaded control.

For 2-minute screens and A/B comparisons (one server per configuration,
closed-loop levels, NUMA pinning, the thermal pre-check and the audio-identity
checks), use `tools/gpu/knee_closed.sh` as described in
[benchmarking.md](benchmarking.md).

WER does not hear timbre: a metallic clip that reads correctly scores 0%.
After listening, run `tools/pocket_audio_noise.py` on the unzipped listening
sets to count noisy/metallic outliers per voice, sentence kind and length.

## Related

- [server.md](server.md): the HTTP API and the CPU server.
- [pocket-voices.md](pocket-voices.md): which voice to serve (alba), measured quality and licences.
- [performance.md](performance.md): the measured results, CPU and GPU.
- [benchmarking.md](benchmarking.md): how the GPU serving screens are run (knee script, identity checks, profile).
- [`configs/perf/`](../configs/perf/README.md): serving profiles and their validator.
- [`.work/pocket-cuda-l4-host-cpu.md`](../.work/pocket-cuda-l4-host-cpu.md) and
  [`.work/pocket-cuda-c60-l4.md`](../.work/pocket-cuda-c60-l4.md): how each change was measured.
