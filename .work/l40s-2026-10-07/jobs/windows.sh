#!/bin/bash
# Soak time windows: first 5 min vs last 5 min (audio-s/s, RTF p95, TTFA p95, stalls) plus GPU power/clock/temp per window.
R=$1; T=$2
/venv/main/bin/python - "$R/$T-$T.jsonl" "$R/dmon-$T.log" "$R/$T-summary.jsonl" <<"PY"
import json,sys
rows=[json.loads(l) for l in open(sys.argv[1]) if l.strip()]
s=json.loads(open(sys.argv[3]).read().strip().splitlines()[-1])
f=lambda k:"p50 %.3f p95 %.3f p99 %.3f max %.3f"%tuple(s[k][q] for q in ("p50","p95","p99","max"))
print("    completed %d failed %d timeouts %d | stalls250 %d (%d req) stalls500 %d (%d req)"%(s["completed"],s["failed"],s["timeouts"],s["stalls_250"],s["req_with_stall_250"],s["stalls_500"],s["req_with_stall_500"]))
for k in ("rtf_stream","ttfa","max_gap","required_prebuffer"): print("    %-18s %s"%(k,f(k)))
ok=[r for r in rows if r.get("status")==200 and r.get("t_done")]
if not ok: sys.exit()
t0=min(r["t_send"] for r in ok); t1=max(r["t_done"] for r in ok)
def win(a,b,name):
    w=[r for r in ok if a<=r["t_done"]-t0<b]
    if not w: return
    au=sum(r.get("audio_s",0) for r in w); dur=b-a
    q=lambda k:sorted(r[k] for r in w if r.get(k) is not None)
    p95=lambda v:v[int(0.95*(len(v)-1))] if v else float("nan")
    st=sum(1 for r in w if r.get("max_gap_s",0)>0.25)
    print("    %-12s req %6d  audio-s/s %7.1f  rtf95 %.3f  ttfa95 %.0f ms  gap95 %.0f ms  req>250ms-gap %d"%(name,len(w),au/dur,p95(q("rtf_stream")),1e3*p95(q("ttfa_s")),1e3*p95(q("max_gap_s")),st))
span=t1-t0; win(60,360,"first 5 min"); win(span-300,span,"last 5 min")
d=[l.split() for l in open(sys.argv[2]) if l.strip() and not l.startswith("#")]
d=[x for x in d if len(x)>4 and x[4].replace(".","").isdigit() and float(x[4])>5]
if d:
    n=len(d); a=d[:min(300,n//2)]; b=d[-min(300,n//2):]
    g=lambda xs,i:sum(float(x[i]) for x in xs)/len(xs)
    print("    gpu first 5 min: %.0f W %.0f C sm %.0f%% | last 5 min: %.0f W %.0f C sm %.0f%%"%(g(a,1),g(a,2),g(a,4),g(b,1),g(b,2),g(b,4)))
PY
