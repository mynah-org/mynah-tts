# Captured `/proc/cpuinfo` feature lines

These exist because this project has **no x86 host**. `tools/simd-auto.sh`
resolves `SIMD=auto` from the kernel's feature list, and without fixtures that
logic could only ever be tested on the one machine we happen to own — which is
ARM. Each file is trimmed to the first processor block, because that is all the
resolver reads.

`neoverse-v2-gcp-c4a.txt` was captured from the project's own Linux box
(GCP Axion, 32 cores, 2026-09-13). The x86 files are the feature lines of the
named parts, transcribed rather than captured; they are fixtures for the
resolution TABLE, not evidence about any host. Nothing here is a performance
claim and nothing here says a kernel ran.
