#!/bin/bash
# Soaks with every win on (11 defaults + A1a + L13/L13b/L13d + A1b v2), tree /root/ma1b2:
# 30 min at C1024 on the full GPU node; then a 4-vCPU check at C896/C1024 and a 10-min soak at the best passing level.
exec >> /root/res/soak.log 2>&1
WIN="MYNAH_CTX_HOST_POOL=1 MYNAH_CUDA_STEP_OVERLAP=1 MYNAH_CUDA_DECODE_OVERLAP=1 MYNAH_CUDA_FIRST_FRAME_FIRST=1 MYNAH_CUDA_SLOT_FIXED=1"
K=/root/jobs/knee.sh; T=/root/ma1b2
detail() { /venv/main/bin/python - "$1" <<"PY"
import json,sys
s=json.loads(open(sys.argv[1]).read().strip().splitlines()[-1])
f=lambda k:"p50 %.3f p95 %.3f p99 %.3f max %.3f"%tuple(s[k][q] for q in ("p50","p95","p99","max"))
print("    completed %d failed %d timeouts %d | stalls250 %d (%d req) stalls500 %d (%d req)"%(s["completed"],s["failed"],s["timeouts"],s["stalls_250"],s["req_with_stall_250"],s["stalls_500"],s["req_with_stall_500"]))
print("    rtf_stream "+f("rtf_stream")); print("    ttfa       "+f("ttfa")); print("    max_gap    "+f("max_gap"))
print("    prebuffer  "+f("required_prebuffer")+" | safe_start "+f("safe_start"))
print("    gpu power %.0f W, sm clock min %s MHz, vram max %s MiB"%(s["gpu_power_mean_w"],s["gpu_sm_mhz_min"],s["gpu_mem_max_mib"]))
PY
}
echo "== soak start $(date -u +%T)"
PIN=32-63,96-127 CLIPIN=0-31,64-95 TREE=$T TAG=soak30 LEVELS="1024" PRE=768 PROCS=4 DUR=1800 ENVS="$WIN" timeout 3000 $K
detail /root/res/soak30/c1024-summary.jsonl
grep -h "\[CTX\]" /root/res/soak30/server.log | tail -1 | cut -c1-400
PIN=32,33,96,97 CLIPIN=0-31,64-95 TREE=$T TAG=v4w LEVELS="896 1024" PRE=768 PROCS=4 DUR=120 ENVS="$WIN" timeout 1500 $K
B=896; r=$(/venv/main/bin/python -c "import json;print(json.loads(open(\"/root/res/v4w/c1024-summary.jsonl\").read().strip().splitlines()[-1])[\"rtf_stream\"][\"p95\"])" 2>/dev/null); awk -v r="$r" "BEGIN{exit !(r!=\"\" && r<0.85)}" && B=1024
echo "== 4-vCPU soak level: C$B (C1024 rtf95=$r)"
PIN=32,33,96,97 CLIPIN=0-31,64-95 TREE=$T TAG=v4soak LEVELS="$B" PRE=768 PROCS=4 DUR=600 ENVS="$WIN" timeout 2000 $K
detail /root/res/v4soak/c$B-summary.jsonl
echo "== soak done $(date -u +%T)"
