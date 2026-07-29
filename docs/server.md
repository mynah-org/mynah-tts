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
| `seed` | number | RNG seed, default 42 |
| `model` | string | accepted and ignored — one pack is served per process |

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

**Synthesis is serialized.** Connections are accepted and parsed concurrently,
but one mutex guards the synthesis itself.

That is a correctness bound, not a tuning decision. A `mynah_tts_model` carries
mutable caches — quantized weight cache, codec filters, the local projection
cache — that synthesis populates and reuses. Two threads synthesizing on one
model would race on them. Serving concurrent requests in parallel means moving
that scratch into a per-request context first; until then, extra threads would
corrupt output rather than speed it up.

Practically: throughput is one stream at a time. With RTF ~0.36 on an M1 at
int8, a single process still generates faster than real time, so a small number
of queued clients stay ahead of playback.

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

CORS is open (`Access-Control-Allow-Origin: *`) and `OPTIONS` preflight is
answered, so a browser page can call it directly.
