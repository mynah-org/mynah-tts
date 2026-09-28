#!/bin/bash
. "$(dirname "$0")/compat.sh"
# Block until a detach.sh job has written its DONE marker.
# usage: wait_done.sh <name> [poll-seconds]
ev="${MYNAH_GPU_EVIDENCE:-/root/evidence}"
while ! grep -q "^$1-DONE" "$ev/$1.out" 2>/dev/null; do sleep "${2:-30}"; done
