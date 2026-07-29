# Server

An OpenAI-compatible HTTP server over the same engine the CLI uses. No
framework, no dependencies — plain sockets in C, one binary.

```bash
make server
./build/cpu/mynah-tts-server -m models/magpie-v2607-pack -p 8080
```

| flag | meaning |
|---|---|
| `-m, --model DIR` | model pack directory (required) |
| `-p, --port N` | listen port (default 8080) |
| `--host ADDR` | bind address (default `127.0.0.1`; use `0.0.0.0` to expose) |
| `-w, --workers N` | connection workers (default 4) — see Concurrency |
| `--device cpu\|metal\|cuda` | backend, same rules as the CLI |

It binds to loopback by default. Exposing it means `--host 0.0.0.0`, which is a
deliberate act: **there is no authentication, no TLS and no rate limiting**. Put
it behind a reverse proxy if it faces anything but localhost.

## POST /v1/audio/speech

The OpenAI speech shape.

```bash
curl -X POST http://localhost:8080/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"magpie-v2607","input":"hello from mynah","voice":"Sofia"}' \
  -o speech.wav
```

| field | type | notes |
|---|---|---|
| `input` | string | **required**, the text to speak |
| `voice` | string | voice name (`"Sofia"`, case-insensitive) or numeric id (`"4"`); defaults to the first voice |
| `response_format` | string | `wav` (default) or `pcm` |
| `stream` | bool | stream PCM as it is produced |
| `language` | string | language code, default `en` — an extension, not OpenAI |
| `temperature` | number | sampling temperature; defaults to the pack's value |
| `top_k` | number | top-k sampling; defaults to the pack's value |
| `max_steps` | number | cap on decoder steps; 0 (default) means the model decides |
| `seed` | number | RNG seed, default 42 |
| `model` | string | accepted and ignored — one pack is served per process |

`POST /v1/tts` is the same handler with native field names: `text` instead of
`input`, `speaker` instead of `voice`. Use whichever reads better; the audio is
identical and the test suite asserts that.

`response_format` supports `wav` and `pcm` only. `mp3`, `opus`, `aac` and
`flac` return 400 rather than silently handing back WAV under another name:
this runtime embeds no audio encoder.

### Streaming

```bash
curl -N -X POST http://localhost:8080/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"streamed speech","voice":"Sofia","stream":true}' \
  -o speech.pcm
```

Streaming responses are **chunked raw PCM**, not WAV: a WAV header must declare
the total length up front, which is unknown while generating. The format is in
the response headers:

```
Content-Type: audio/pcm
X-Sample-Rate: 22050
X-Bits-Per-Sample: 16
X-Channels: 1
Transfer-Encoding: chunked
```

Play it as it arrives, rather than writing bytes to disk:

```bash
# macOS — ffplay
curl -sN -X POST http://localhost:8080/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"hello from mynah","voice":"Sofia","stream":true}' | \
  ffplay -f s16le -ar 22050 -ac 1 -nodisp -autoexit -

# macOS — sox
curl -sN -X POST http://localhost:8080/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"hello from mynah","voice":"Sofia","stream":true}' | \
  play -t raw -r 22050 -e signed -b 16 -c 1 -

# Linux — aplay
curl -sN -X POST http://localhost:8080/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"hello from mynah","voice":"Sofia","stream":true}' | \
  aplay -f S16_LE -r 22050 -c 1

# Save raw PCM, convert later
curl -sN ... -o out.raw && ffmpeg -f s16le -ar 22050 -ac 1 -i out.raw out.wav
```

Each chunk is a stable causal prefix from the same streaming API the C library
exposes, so the concatenated stream is sample-identical to the non-streamed
response for the same request — `make stream-test` asserts exactly that.

One consequence worth knowing: once the 200 header is sent, a mid-stream failure
cannot be reported as an HTTP status. The body simply terminates. Check that the
byte count you received matches what you expected.

## GET /v1/voices

Not an OpenAI route — it exists because `"voice": 4` is meaningless without it.

```bash
curl http://localhost:8080/v1/voices
```
```json
{"voices":[{"id":0,"name":"Aria"},{"id":1,"name":"Jason"},{"id":2,"name":"John"},
           {"id":3,"name":"Leo"},{"id":4,"name":"Sofia"}],"default":0}
```

## GET /v1/models

OpenAI-shaped listing of the single pack this process serves.

## GET /health

```json
{"status":"ok","model":"magpie-v2607","engine":"magpie","sample_rate":22050,"voices":5}
```

## Concurrency

Connections are handled by a **fixed worker pool draining a bounded queue**
(`-w`, default 4; queue depth 256). Thread-per-connection was rejected: it has
no ceiling, so a client that merely opens sockets grows threads until the
process dies. When the queue is full the server answers **503 with
`Retry-After`** and closes — shedding load beats unbounded growth.

Accepted sockets carry a 30 s send/receive timeout, so a client that connects
and never sends cannot pin a worker forever.

**Synthesis runs on one thread — and batches.** Connections are accepted and
parsed concurrently, but only the scheduler thread ever touches the model.

The single thread is a correctness bound, not a tuning decision. A
`mynah_tts_model` carries mutable caches — quantized weight cache, codec
filters, the local projection cache — that synthesis populates and reuses. Two
threads synthesizing on one model would race on them.

What changed is that one thread no longer means one request at a time. A decode
step is bound by the weight bytes it reads, not by arithmetic, so requests
taking turns each paid their own trip to memory for the same weights. The
scheduler hands everything queued to `mynah_tts_synthesize_batch`, which reads
those weights once and serves every waiting request from cache. Requests that
arrive while a batch is running are grouped into the next one, so queue depth
does the batching; `--max-batch` caps the group (default 8, hard limit 16).

Measured on M1 (int8, warm, same text, one request per seed):

| in flight | serial | batched | speedup |
|---|---|---|---|
| 2 | 3.26 s | 2.52 s | 1.30x |
| 4 | 6.48 s | 4.62 s | 1.40x |
| 8 | 12.94 s | 7.94 s | **1.63x** |

**Every batched request received byte-identical audio to the same request run
alone.** That is the property that makes batching safe to enable by default: an
identical request cannot get different audio depending on who it shared a batch
with. It holds because batching reorders independent work and never a reduction
— see `docs/quantization.md` for how the kernels keep each row's accumulation
order fixed, and `mynah_qmat_self_test` for the check that asserts it.

The ceiling is what the phase split predicts rather than what the microbenchmark
promised: batching covers the decoder and the local transformer, roughly half of
wall time, so the gain tops out near 1.6x rather than the ~4x that batching a
single matvec shows in isolation. Raising it further means batching the codec
too.

**Streaming still runs alone.** It needs its callback interleaved with
generation, so a streaming request takes the same lock the scheduler does and
never overlaps a batch.

### Streaming cost

A streamed chunk is decoded with a bounded window of history rather than by
re-running the codec over everything generated so far. The audio decoder is
causal end to end, so an output depends only on inputs at or before it: decoding
from 32 frames before the first unsent frame reproduces the sent frames exactly.
That bound is the receptive field walked through the decoder stack (~25 frames;
32 leaves margin), and `make stream-test` fails at 24 and passes at 32, which is
where the derivation puts it.

Re-decoding the whole prefix every step was quadratic — a 15 s utterance
decoded about 26k frames instead of 320, and streaming ran roughly 30x slower
than the same request offline. Chunks are also emitted every 16 frames rather
than every step, so the fixed window cost is paid once per chunk instead of
sixteen times for the same audio; the added latency to the first chunk is well
under a second and the total is unchanged.

Streamed audio remains byte-identical to the offline result for the same
request — asserted by `make stream-test` and by the server's `stream==batch`
check.

A stream that stops early is logged with how far it got and why:

```
stream aborted after 196608 bytes: Broken pipe
```

This used to be silent when the socket write was what failed, which made a
truncated response indistinguishable from a short utterance — worth knowing if
you ever see a stream come back shorter than the same request offline.

### Memory

A server holds the whole model resident: about 180 MB before its first request
and ~2 GB after, once the weight pages are touched and the quantized copies are
built. That is per process, not per request — batching adds roughly 30 MB per
slot in flight.

The practical consequence is that **leaked servers are what runs a machine out
of memory**, not load. `tests/test_server.sh` refuses to start when something is
already serving on its port, and kills its own server on any exit path
including SIGINT.

Set decode precision with the same environment variable the CLI uses:

```bash
MYNAH_QUANT=int8 ./build/cpu/mynah-tts-server -m models/magpie-v2607-pack
```

## Errors

Errors are OpenAI-shaped:

```json
{"error":{"message":"missing or empty 'input'","type":"invalid_request_error"}}
```

| status | when |
|---|---|
| 400 | missing `input`, unknown `voice`, unsupported `response_format`, tokenizer failure |
| 404 | unknown route |
| 413 | request body over 1 MiB |
| 500 | synthesis failure |
| 503 | connection queue full (with `Retry-After: 1`) |

`make server-test MODEL_DIR=...` covers every route plus three properties worth
stating: streamed audio equals batch audio byte for byte, three identical
requests return identical audio (guarding against state leaking through the
model's mutable caches), two concurrent clients both complete without
perturbing each other, and four concurrent requests are byte-identical to the
same four run one at a time while not being slower.

CORS is open (`Access-Control-Allow-Origin: *`) and `OPTIONS` preflight is
answered, so a browser page can call it directly.
