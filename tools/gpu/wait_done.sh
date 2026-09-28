#!/bin/bash
# Block until a detach.sh job has written its DONE marker.
# usage: wait_done.sh <name> [poll-seconds]
ev="${MYNAH_L4_EVIDENCE:-/root/evidence}"
while ! grep -q "^$1-DONE" "$ev/$1.out" 2>/dev/null; do sleep "${2:-30}"; done
