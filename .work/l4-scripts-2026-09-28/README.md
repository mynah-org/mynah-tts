# L4 orchestration scripts, 2026-09-28 (archive)

Copied verbatim from `/root/evidence/` on the vast.ai L4 used for
`.work/pocket-cuda-g6-host-cpu.md`. They chain the day's runs through tmux
sessions and DONE markers (so work finishes without an ssh session) and call
the maintained harness in `tools/l4/`. Kept as the record of exactly what ran;
new runs should start from `tools/l4/` instead.

Main ones: `qual.sh` (24L C160 qualification: tooling check, two 30-min soaks
with capture, ASR, bundle), `takeover.sh` / `takeover6.sh` (faster ASR and the
bundle after the soaks), `ctrl-now.sh` (unloaded C8 control on the same request
ids), `disc6l.sh` / `disc6lb.sh` / `qual6l.sh` (6L discovery and C256
qualification), `ab-*.sh` (the A/B queues), `leak.sh` / `growtest2.sh` (slot
pool and KV-growth bit-identity tests), `nsys-c64*.sh` (Nsight profiles),
`soak96.sh` (C96 screening soak), `models*.sh` (pack download/conversion; the
HF token is read from `/root/.hf_token` on the box and is not in any script).
