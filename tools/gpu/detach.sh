#!/bin/bash
# Run a command in its own tmux session so it survives a dropped ssh line.
# usage: detach.sh <name> <command...>
# Output goes to $MYNAH_L4_EVIDENCE/<name>.out (default /root/evidence) and the
# last line is "<name>-DONE rc=<exit code>", so a later step or a short-lived
# check can wait on it:  grep -q "<name>-DONE" /root/evidence/<name>.out
# Chain steps by making the next command wait for the previous marker, e.g.
#   detach.sh q24 tools/l4/qualify.sh q24 models/pocket-english-24l 160
#   detach.sh q6  bash -c 'tools/l4/wait_done.sh q24; tools/l4/qualify.sh ...'
set -u
[ $# -ge 2 ] || { echo "usage: $0 <name> <command...>" >&2; exit 2; }
name=$1; shift
ev="${MYNAH_L4_EVIDENCE:-/root/evidence}"; mkdir -p "$ev"
cmd=$(printf '%q ' "$@")
tmux new-session -d -s "$name" "cd $(printf '%q' "$(pwd)") && { $cmd; } > $ev/$name.out 2>&1; echo \"$name-DONE rc=\$?\" >> $ev/$name.out"
echo "started tmux session '$name', log $ev/$name.out"
