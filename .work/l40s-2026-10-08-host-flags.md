# Host flags on an L40S -- 2026-10-08

Vast.ai NVIDIA L40S 46 GB (driver 575), Intel Xeon Gold 6430, 2 NUMA nodes, server pinned to the GPU's node
(0-31,64-95), clients on the other node, cgroup quota ~30.7 CPUs. main a12e4ab, `ROW_CAP=1024`, 24L, `--max-batch 1024`,
`tools/gpu/knee_closed.sh`, `PRE=768`, 2-minute levels, one round (round 2 cut: the answer was already clear).

| arm | C | audio-s/s | RTF p95 | TTFA p95 | gap p95 | stalls@250 | W | SM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| defaults | 1024 | 1281 | 0.743 | 107 ms | 117 | 0 | 339 | 95% |
| `MYNAH_CUDA_MIMI_STALE_WINDOW=1` | 1024 | 1295 | 0.738 | 114 | 116 | 0 | 342 | 97% |
| `MYNAH_SERVE_HOST_THREADS=auto` | 1024 | 1294 | 0.738 | 111 | 116 | 0 | 343 | 97% |
| both | 1024 | 1292 | 0.738 | 114 | 116 | 0 | 342 | 97% |
| defaults | 896 | 1263 | 0.665 | 142 | 105 | 0 | 342 | 95% |
| stale window | 896 | 1276 | 0.661 | 146 | 105 | 0 | 337 | 97% |
| host threads | 896 | 1276 | 0.660 | 145 | 105 | 0 | 337 | 97% |
| both | 896 | 1276 | 0.660 | 148 | 105 | 0 | 337 | 97% |

Host time per iteration (`[SERVE]`): defaults 20.72 ms (46.5% of the loop), stale window 17.09, host threads 18.23,
both 16.50 (36.9%). The freed host time turns into device wait: the GPU is 95-97% busy at its 350 W power cap
(throttle reason: software power cap; SM clock mostly 2220-2520 MHz with brief dips). So ~+1% throughput, within the
noise of one round, for +3-7 ms TTFA p95 -- the same picture as on the L4.

Identity, both flags vs defaults: CLI `--batch 32` 32/32, `--batch 32 --stream` 32/32, server C1 167/167 identical.

Start-up with the default start-up walk at ROW_CAP 1024: ready in 31.5-32.8 s, 34.3 GB at ready (35.9 GB at the end),
1066 graphs, 0 fallbacks; width walk 3.7 s, slot-pool prefill 26-28 s (the next start-up target).
`--max-batch 16`: ready in 2.1 s.

Decision: both flags stay opt-in. They are for hosts where the scheduler thread, not the GPU, is the limit (slow
or few cores); on a power-capped L40S or L4 they only move time from host to device wait.
