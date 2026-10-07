#!/usr/bin/env python3
"""A stand-in streaming server for measuring the load generator, not the engine.

It speaks just enough of the Pocket HTTP API for tools/pocket_ladder.py:

- POST /v1/audio/speech answers with chunked 16-bit PCM, one 80 ms frame per
  emission at a fixed pace (--rtf 0.5 emits a frame every 40 ms), for an
  utterance length derived from the text (--chars-per-s);
- GET /health and GET /metrics answer with stubs.

There is no model and no GPU, so the server costs almost nothing per stream and
can be spread over --procs processes that share one listening socket. Run the
ladder against it at the concurrencies of a real campaign. Every stream is
paced the same, so the client should measure stream RTF ~= --rtf and
audio-s/s ~= C / --rtf. The level where it stops doing so is the client's own
ceiling on this host, and no server measured through it can look faster.

    python3 tools/gpu/stub_stream_server.py --port 18090 --procs 4 --rtf 0.5 &
    python3 tools/pocket_ladder.py --port 18090 --levels 64,128,256 \\
        --warmup 5 --duration 30 --client-procs 1 --tag stub1
    python3 tools/pocket_ladder.py --port 18090 --levels 64,128,256 \\
        --warmup 5 --duration 30 --client-procs 4 --tag stub4

Stdlib only.
"""

import argparse
import asyncio
import json
import os
import signal
import socket
import sys

SAMPLE_RATE = 24000
FRAME_S = 0.08
FRAME_BYTES = int(SAMPLE_RATE * FRAME_S) * 2


def utterance_frames(text, chars_per_s):
    seconds = max(FRAME_S, len(text) / chars_per_s)
    return max(1, int(round(seconds / FRAME_S)))


async def handle(reader, writer, a, frame):
    try:
        while True:
            head = await reader.readuntil(b"\r\n\r\n")
            lines = head.decode("latin-1").split("\r\n")
            method, path = lines[0].split(" ")[:2]
            length = 0
            for line in lines[1:]:
                if line.lower().startswith("content-length:"):
                    length = int(line.split(":", 1)[1])
            body = await reader.readexactly(length) if length else b""
            if method == "GET":
                if path == "/metrics":
                    payload = b"mynah_stub_up 1\n"
                    ctype = b"text/plain"
                else:
                    payload = json.dumps({"status": "ok", "engine": "stub",
                                          "sample_rate": SAMPLE_RATE}).encode()
                    ctype = b"application/json"
                writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: " + ctype +
                             b"\r\nContent-Length: " + str(len(payload)).encode() +
                             b"\r\n\r\n" + payload)
                await writer.drain()
                continue
            try:
                text = json.loads(body or b"{}").get("input", "")
            except ValueError:
                text = ""
            frames = utterance_frames(text, a.chars_per_s)
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: audio/pcm\r\n"
                         b"X-Sample-Rate: 24000\r\nTransfer-Encoding: chunked\r\n\r\n")
            chunk = b"%x\r\n" % len(frame) + frame + b"\r\n"
            loop = asyncio.get_running_loop()
            t0 = loop.time()
            for k in range(frames):
                # Paced against the start, not the previous write, so a late
                # wake-up does not push every later frame back.
                delay = t0 + k * FRAME_S * a.rtf - loop.time()
                if delay > 0:
                    await asyncio.sleep(delay)
                writer.write(chunk)
                await writer.drain()
            writer.write(b"0\r\n\r\n")
            await writer.drain()
    except (asyncio.IncompleteReadError, ConnectionError, ValueError):
        pass
    finally:
        writer.close()


def serve(sock, a):
    frame = bytes(FRAME_BYTES)

    async def main():
        server = await asyncio.start_server(
            lambda r, w: handle(r, w, a, frame), sock=sock, backlog=4096)
        async with server:
            await server.serve_forever()

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18090)
    ap.add_argument("--procs", type=int, default=4,
                    help="server processes sharing the listening socket")
    ap.add_argument("--rtf", type=float, default=0.5,
                    help="emission pace: seconds of wall per second of audio")
    ap.add_argument("--chars-per-s", type=float, default=15.0,
                    help="audio length per character of input text")
    a = ap.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((a.host, a.port))
    sock.listen(4096)
    sock.setblocking(False)

    # Pre-fork accept: every child accepts on the one listening socket, which
    # spreads connections on Linux and macOS alike (SO_REUSEPORT does not
    # balance on macOS).
    children = []
    for _ in range(max(1, a.procs)):
        pid = os.fork()
        if pid == 0:
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            serve(sock, a)
            os._exit(0)
        children.append(pid)
    print("stub: %d procs on %s:%d, rtf %.2f" % (len(children), a.host, a.port, a.rtf),
          file=sys.stderr, flush=True)

    def stop(*_):
        for pid in children:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        sys.exit(0)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    for pid in children:
        os.waitpid(pid, 0)


if __name__ == "__main__":
    main()
