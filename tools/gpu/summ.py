#!/usr/bin/env python3
"""One line per ladder level from a pocket_ladder summary JSONL."""
import json
import sys

for line in open(sys.argv[1]):
    s = json.loads(line)
    d = s["metrics_delta"]

    def width(stage):
        c = d.get("mynah_backend_%s_calls_total" % stage)
        i = d.get("mynah_backend_%s_items_total" % stage)
        return round(i / c, 1) if c and i else None

    def f(x):
        return "-" if x is None else "%.3f" % x

    print("C%-3d sent %4d ok %4d fail %d srvfail %d act %d | aud/s %6.2f | ttfa p50 %s p95 %s | "
          "rtfS p50 %s p95 %s | st250 %d(%d req) st500 %d | gpu %s%% mem %s pw %s | cpu %s%%" % (
              s["offered"], s["sent"], s["completed"], s["failed"],
              int(s.get("server_jobs_failed", 0)), s["max_client_active"], s["audio_s_per_s"],
              f(s["ttfa"]["p50"]), f(s["ttfa"]["p95"]), f(s["rtf_stream"]["p50"]),
              f(s["rtf_stream"]["p95"]), s["stalls_250"], s["req_with_stall_250"],
              s["stalls_500"], f(s["gpu_util_mean"]), s["gpu_mem_max_mib"],
              f(s["gpu_power_mean_w"]), f(s["server_cpu_pct_mean"])))
    print("    widths bb=%s dec=%s codec=%s" % (width("backbone_batch"), width("decoder_batch"),
                                                width("codec_transformer_batch")),
          {k.replace("mynah_backend_", "").replace("_total", ""): round(v)
           for k, v in d.items()
           if any(x in k for x in ("graph", "sync", "h2d", "d2h")) and "width" not in k})
