# Serving PocketTTS on an NVIDIA GPU (CUDA)

How to run the mynah-tts streaming server on one NVIDIA GPU for PocketTTS, the
most tested engine of this runtime: build, model, start, stream, size, tune,
monitor and qualify. Everything here was measured on an NVIDIA L4 on
2026-09-28; the evidence is in [performance.md](performance.md) and the
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

A 4-vCPU host (AWS g6.xlarge class) is enough: the server needs about 1.3-1.4
cores, and pinning it to four cores cost 1.5%.

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
```

## 4. Start the server

Let the profile write the command, so nothing is forgotten:

```bash
python3 tools/perf_profile.py command l4-24g-pocket-en-24l-cuda --model models/pocket-english-24l --port 8080
```

which prints the qualified configuration:

```bash
MYNAH_THREADS=1 MYNAH_CUDA_KV_DTYPE=bf16 MYNAH_QUANT_GROUPS=none \
  ./build/cuda/mynah-tts-server --device cuda -w 8 \
  --max-batch 160 --max-inflight 160 -p 8080 -m models/pocket-english-24l
```

(`MYNAH_SERVE_PROFILE=1`, which the profile also prints, only adds a
scheduler report at shutdown.) For the small model use its profile, or the same
line with `--max-batch 256 --max-inflight 256` and `models/pocket-english-6l`.

| flag | meaning |
|---|---|
| `--device cuda` | run on the GPU; the server refuses to start rather than silently run on the CPU |
| `--max-batch N` | streams batched in one GPU step; set it to the target concurrency |
| `--max-inflight N` | streams admitted at once; same value; up to 384 on CUDA (128 on CPU) |
| `-w N` | HTTP connection workers (8 was used; the GPU work is one scheduler thread) |
| `--host 0.0.0.0` | expose beyond localhost; there is no authentication or TLS, put a proxy in front |

A healthy start prints lines like:

```
mynah-tts: CUDA quant=f32 backbone=f32 flow=f32 mimi=f32 kv=bf16
mynah-tts: pocket backend=cuda resident=on ... resident{backbone=on flow=on codec_transformer=on}
```

`kv=bf16` and `resident=on` are the two things to look for.

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

- Large model: ~75 MB of GPU memory per stream plus ~1.8 GB fixed.
- Small model: ~30 MB per stream plus ~1 GB fixed.
- A different GPU needs its own screen: `tools/gpu/knee.sh <tag> <model> 128,160,192,224,256`
  prints the top level under the gate (section 10).

For 1,000 concurrent listeners on L4s: 4 GPUs for the small model, 7 for the
large one.

## 7. Feature flags (environment variables)

The defaults are the tuned configuration. **Set only the two required
variables**; everything else exists to roll a change back while debugging, or
to opt into something that is not a production default. A profile run refuses
to start if a "must be absent" variable is set.

### Required

| variable | value | why |
|---|---|---|
| `MYNAH_CUDA_KV_DTYPE` | `bf16` | Stores the attention cache in 16 bits: twice the streams in memory. The default is FP32 (the parity reference). Quality-checked: different but equally valid samples, WER equal or better |
| `MYNAH_THREADS` | `1` | The GPU does the work; one CPU thread avoids a 128-thread pool on a big host |

`MYNAH_QUANT_GROUPS=none` is what the qualified runs set; a CUDA build now
resolves the default to the same thing, so it is harmless and optional.

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
| `MYNAH_POCKET_VOICE_CACHE` | `0` (or `all` / `startup` to preload) | voice prompts cached on first use | |

### Opt-in (off by default; not production settings)

| variable | effect | why it is off |
|---|---|---|
| `MYNAH_CUDA_PREFILL_FIXED=0` | text prefill through cuBLAS: +4-9% throughput | a text sent in pieces no longer gives bit-identical audio to the same text sent whole; a product decision |
| `MYNAH_CUDA_SYNC=blocking` | host thread sleeps while the GPU works: server CPU 117% -> 43% | -9% throughput (each wake-up idles the GPU) |
| `MYNAH_CUDA_TILE_TC=1` | own tensor-core fixed-order GEMM for the prefill | +1-2% only |
| `MYNAH_CUDA_QUANT=bf16\|int8` | bfloat16 / int8 resident weights | bf16 -4%, int8 diagnostic only |
| `MYNAH_CUDA_FAST_MATH=1` | FP16 GEMMs | not qualified |
| `MYNAH_CUDA_CODEC_BATCH=1` | older multi-row codec path | fails the waveform parity gate |
| `MYNAH_CUDA_ALLOW_CPU_STAGES=1` | lets hot stages run on the CPU | 20-30x slower while reporting CUDA: never in production |

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
  85-95% at the qualified levels.
- The server process using 100-125% of a CPU core is normal: the scheduler
  thread spins while it waits for the GPU (that spin is faster than sleeping).

## 9. Troubleshooting

| symptom | cause and fix |
|---|---|
| server refuses to start mentioning CPU stages | a configuration would put a hot stage on the CPU; remove `MYNAH_QUANT_GROUPS`/`MYNAH_CUDA_QUANT` overrides |
| out of GPU memory | concurrency too high for the card, or `MYNAH_CUDA_KV_GROW=0` / FP32 cache set; check `kv=bf16` at start-up |
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
- [`.work/pocket-cuda-g6-host-cpu.md`](../.work/pocket-cuda-g6-host-cpu.md) and
  [`.work/pocket-cuda-c60-l4.md`](../.work/pocket-cuda-c60-l4.md): how each change was measured.
