#!/bin/sh
# A drop-in stand-in for the server binary that always runs preforked.
#
#   SERVER=tests/prefork_server.sh sh tests/test_server_concurrency.sh
#   SERVER=tests/prefork_server.sh PREFORK=4 sh tests/test_server.sh
#
# The test scripts take SERVER= and invoke it as `"$SERVER" -m ... -p ...`.
# Pointing that at this file runs the same binary with --prefork appended, so
# the whole gate re-runs against a router-plus-workers topology without a
# second copy of the test. (tests/test_server_concurrency.sh also accepts
# SERVER_ARGS="--prefork 4" directly; the wrapper exists because a plain
# SERVER= override is the one knob every harness already has.)
#
# exec, not a background child: the caller kills $! at cleanup, and a wrapper
# that forked would leave the real server orphaned holding the whole pack.
set -eu

REAL="${REAL_SERVER:-build/cpu/mynah-tts-server}"
PREFORK="${PREFORK:-4}"

[ -x "$REAL" ] || { echo "prefork_server.sh: no server binary at $REAL (set REAL_SERVER=)" >&2; exit 2; }

exec "$REAL" "$@" --prefork "$PREFORK"
