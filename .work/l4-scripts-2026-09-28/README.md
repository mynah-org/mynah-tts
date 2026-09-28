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

## Maintained, generic versions (use these on a new box)

`tools/l4/provision.sh` (build, tooling, Kyutai download with the token from
`/root/.hf_token`, 24L/6L packs, self-checks), `tools/l4/detach.sh` +
`tools/l4/wait_done.sh` (tmux job with a DONE marker), `tools/l4/knee.sh`
(ladder + automatic level pick under the RTF gate), `tools/l4/qualify.sh`
(two soaks with capture and VRAM log, ASR, optional C8 control, bundle),
`tools/l4/vram_log.sh`. Verified end to end on the L4 on 2026-09-28
(provision -> knee -> qualify, `tooltest-DONE rc=0`). A fresh box:

    git clone ... /root/mynah-head && cd /root/mynah-head   # or a git bundle
    echo "<hf read token>" > /root/.hf_token && chmod 600 /root/.hf_token
    tools/l4/detach.sh prov tools/l4/provision.sh sm_89
    tools/l4/detach.sh knee24 bash -c 'tools/l4/wait_done.sh prov; tools/l4/knee.sh knee24 models/pocket-english-24l 128,144,160,176,192'
    tools/l4/detach.sh q24 bash -c 'tools/l4/wait_done.sh knee24; MYNAH_Q_CONTROL=1 tools/l4/qualify.sh pocket-24l-c160 models/pocket-english-24l 160'
