# Serving PocketTTS on an NVIDIA GPU (CUDA)

How to run the mynah-tts streaming server on one NVIDIA GPU for PocketTTS, the
reference, production-quality engine of this runtime (it also serves on the
CPU): build, model, start, stream, size, tune, monitor and qualify. Everything
here was measured on one NVIDIA L4 (24 GB) on Vast.ai between 2026-09-28 and
2026-10-04, plus a 2-minute screen on one NVIDIA L40S (48 GB) on 2026-10-04
(section 6). The evidence is in [performance.md](performance.md) and the
serving profiles in [`configs/perf/`](../configs/perf/README.md).

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

A 4-vCPU host is enough: the server needs about 1.3-1.4 cores, and pinning it
to four cores cost 1.5%.

## 1. What you need

- Linux with an NVIDIA driver and a CUDA toolkit (nvcc and cuBLAS; tested with
  CUDA 12.8 build / 13.2 runtime, driver 595).
- A GPU with compute capability 8.0 or newer (TF32 tensor cores). Measured:
  L4 (`sm_89`). Built and exercised earlier on Blackwell (`sm_120`).
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
| `--max-inflight N` | streams admitted at once; same value; up to 384 on CUDA (128 on CPU) |
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

**A larger GPU (L40S).** The same `sm_89` build with the current defaults runs
the large model at **C384 with stream RTF p95 0.49**, 740 audio-s/s and TTFA
p95 87 ms. That was a 2-minute screen with 0 stalls and 0 failures, where the
L4 needs 0.86 at C320. The ceiling there is not the GPU:

- **384 is the build's row cap.** `--max-batch` and `--max-inflight` stop at
  384, and requests beyond it queue.
- **Raising the caps does not raise throughput.** An experimental build with
  the compile-time caps raised to 768 holds C640 at 0.83, but throughput stays
  at ~720 audio-s/s with the GPU at ~60% SM. One engine runs one batched step
  at a time, and on a GPU this size one step does not fill the SMs.

So on an L40S, plan with C384 per GPU today. Expect more from a second engine
per GPU rather than from more rows; that is not built yet. Detail:
[performance.md](performance.md), section "2026-10-04 · PocketTTS 24L on CUDA —
one NVIDIA L40S screen".

## 7. Feature flags (environment variables)

The defaults are the tuned configuration. Together, the 2026-10-02 rows of the
table below (shared voice through one sync) read on the L4 24L 60-s knees:
stream RTF p95 0.594-0.607 at C208 and 0.710-0.714 at C256, ~303-306 audio-s/s,
TTFA p95 ~102-104 ms at C208, against 1.03 with stalls at C208 for the
2026-09-28 defaults. The 2026-10-07 rows (deferred release through the
delivery helpers) take the serial host loop off the critical path: see
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

### Opt-in (off by default; not production settings)

| variable | effect | why it is off |
|---|---|---|
| `MYNAH_CUDA_PREFILL_FIXED=0` | text prefill through cuBLAS (bf16 or TF32 by the weights): on f32 weights it measured -7 % RTF p95 | a text sent in pieces no longer gives bit-identical audio to the same text sent whole; a product decision |
| `MYNAH_CUDA_KV_VMM=1` (with `MYNAH_CUDA_KV_DTYPE=bf16`; ignored with int8 KV) | backbone KV rows are VMM ranges that grow in place (no copy, no second allocation, no sync); with `MYNAH_CUDA_KV_VMM_CHUNK=64 MYNAH_CUDA_KV_GROW_INITIAL_STEPS=64` rows start small | L4, `--max-batch 320`, with the diet: start-up 9.6 GB (was 16.0), C320 runs with 0 failures at peak 14.4 GB, but the position-major layout costs ~4 % (C288 0.851 vs 0.819) and C320 is compute-bound (0.926): off on the L4; for GPUs with more compute than memory headroom |
| `MYNAH_CUDA_QUANT_STAGES=all` | bf16 weights for the flow head and the Mimi transformer too | no measurable gain over `backbone` (C256 0.714 -> 0.712) |
| `MYNAH_CUDA_SYNC=blocking` | host thread sleeps while the GPU works: server CPU 117% -> 43% | -9% throughput (each wake-up idles the GPU) |
| `MYNAH_CUDA_TILE_TC=1` | own tensor-core fixed-order GEMM for the f32 prefill | +1-2% only |
| `MYNAH_CUDA_QUANT=int8` | int8 resident weights | diagnostic only |
| `MYNAH_CUDA_FAST_MATH=1` | FP16 GEMMs | not qualified |
| `MYNAH_CUDA_CODEC_BATCH=1` | older multi-row codec path | fails the waveform parity gate |
| `MYNAH_CUDA_ALLOW_CPU_STAGES=1` | lets hot stages run on the CPU | 20-30x slower while reporting CUDA: never in production |
| `MYNAH_CTX_HOST_POOL=1` | a finished request's host-side state (backbone and codec states, flow head, scratch) is renewed for the next admission instead of rebuilt | NVIDIA L40S C1024: 880 -> 1112 audio-s/s, RTF p95 1.076 -> 0.852; pending a soak |
| `MYNAH_CUDA_STEP_OVERLAP=1` | dispatch-ahead: the next AR step is queued before the previous step's retire, admission and cancellation run | L4: +1 % audio-s/s, RTF p95 -0.01, TTFA p95 +50-60 ms; pending a soak |
| `MYNAH_CUDA_DECODE_OVERLAP=1` (needs `MYNAH_CUDA_STEP_OVERLAP=1`) | the gang decode of step k runs under AR step k+1 | with the row below and the host pool, see "Serving-loop defaults"; pending a soak |
| `MYNAH_CUDA_FIRST_FRAME_FIRST=1` (with `MYNAH_CUDA_DECODE_OVERLAP=1`) | new streams' first frames are decoded and delivered as their own small gang first | keeps TTFA down under the decode overlap; pending a soak |

### Serving-loop defaults (2026-10-07)

At hundreds of streams the GPU waited on the single scheduler thread: host
time per step, not GPU time, set the ceiling. The eleven 2026-10-07 rows of the
"on by default" table cut that host time (fewer stream syncs, no per-step
decoder re-recording, no per-row copies). Each one gives bit-identical audio
wherever batch composition is the same (CLI `--batch 32` and the server at C1)
and keeps its variable as a rollback. Measured, `ROW_CAP=1024` build:

| configuration | NVIDIA L40S C1024: audio-s/s | stream RTF p95 |
|---|---|---|
| the eleven defaults | 880 | 1.076 |
| + `MYNAH_CTX_HOST_POOL=1` | 1112 | 0.852 |
| + `MYNAH_CUDA_STEP_OVERLAP=1 MYNAH_CUDA_DECODE_OVERLAP=1 MYNAH_CUDA_FIRST_FRAME_FIRST=1` + `MYNAH_CTX_HOST_POOL=1` | 1157 | 0.846 (TTFA p95 130 ms) |

Before them, the same GPU held the 0.88 RTF gate at C640; with them, at C768
(+6 % audio-s/s), and C896 ran with 0 stalls where it had 46,640. On the L4
(GPU-bound, `ROW_CAP=384`) they give +4 % audio-s/s and -0.03 RTF p95 at
C288-C320, with no regression. The opt-in rows stay off until a soak at those
widths.

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
| `MYNAH_COST_MAP=2 MYNAH_NVTX=1` | NVTX ranges for Nsight Systems |
| `MYNAH_CUDA_KV_GROW_LOG=1` | log each attention-cache growth |
| `MYNAH_CUDA_KV_GROW_INITIAL_STEPS=N` | force growth early (bit-identity tests) |
| `MYNAH_CUDA_SLOT_POOL_ZERO_KV=1` | zero reused caches (leak A/B: audio must not change) |

## 8. Monitoring

- `GET /health`: liveness, limits, active streams.
- `GET /metrics`: Prometheus text. Useful CUDA counters: `mynah_backend_graph_replays_total`
  vs `..._graph_captures_total` (captures should stay flat after warm-up),
  `mynah_backend_decoder_graph_*`, the batch-width histograms, `..._sync_calls_total`,
  H2D/D2H bytes. Polling it does not synchronise the GPU.
- `nvidia-smi`: memory should reach a plateau and stay there; utilisation
  85-95% at the qualified levels on an L4 (an L40S runs at ~60% SM, see
  section 6).
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

WER does not hear timbre: a metallic clip that reads correctly scores 0%.
After listening, run `tools/pocket_audio_noise.py` on the unzipped listening
sets to count noisy/metallic outliers per voice, sentence kind and length.

## Related

- [server.md](server.md): the HTTP API and the CPU server.
- [pocket-voices.md](pocket-voices.md): which voice to serve (alba), measured quality and licences.
- [performance.md](performance.md): the measured results, CPU and GPU.
- [`configs/perf/`](../configs/perf/README.md): serving profiles and their validator.
- [`.work/pocket-cuda-l4-host-cpu.md`](../.work/pocket-cuda-l4-host-cpu.md) and
  [`.work/pocket-cuda-c60-l4.md`](../.work/pocket-cuda-c60-l4.md): how each change was measured.
