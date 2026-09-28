#!/bin/bash
. "$(dirname "$0")/compat.sh"
# Collect a qualification evidence bundle from one or more soaks.
# usage: bundle.sh <bundle-name> <soak-tag> [<soak-tag>...]
#
# Output: $MYNAH_GPU_EVIDENCE/bundles/<bundle-name>/ (default /root/evidence) with
#   provenance.txt      git HEAD of the checkout, bundle tree hashes, host, GPU
#   binaries.sha256     the built binaries (the checkout may be rsynced sources,
#   sources.sha256      so the hashes are what pins the build, not only git)
#   profile/            serve.sh, ab.sh, soak.sh as run
#   model/              the model pack's source.json and model.json
#   corpus/             every corpus named in the soak summaries
#   <tag>/              summary JSONL, per-request JSONL, per-window report,
#                       server log, MYNAH_* env of the server, ladder output,
#                       VRAM log, quality/ (CSV + summary), audio-<tag>.zip
#                       (a listening selection capped at MYNAH_GPU_LISTEN_MB,
#                       default 100 MB per soak; all captures stay on the box)
#   REPORT.md           headline numbers parsed from the above, plus the
#                       sections a human still has to fill in
# Only bash and python3 stdlib. Nothing is recomputed from scratch except the
# per-window report when the soak did not keep one.
set -u
[ $# -ge 2 ] || { echo "usage: $0 <bundle-name> <soak-tag> [<soak-tag>...]" >&2; exit 2; }
name=$1; shift
root="${MYNAH_GPU_ROOT:-/root/mynah-head}"; ev="${MYNAH_GPU_EVIDENCE:-/root/evidence}"
model="${MYNAH_GPU_MODEL:-models/pocket-english-24l}"
case "$model" in /*) mdir="$model" ;; *) mdir="$root/$model" ;; esac
B="$ev/bundles/$name"
mkdir -p "$B/profile" "$B/model" "$B/corpus"

# --- provenance -------------------------------------------------------------
(
  cd "$root" || exit 1
  ls build/cuda/mynah-tts-server build/cuda/mynah-tts build/mynah-tts-server \
     build/mynah-tts 2>/dev/null | xargs -r sha256sum
) > "$B/binaries.sha256"
(
  cd "$root" || exit 1
  find src server cli gpu tools/gpu Makefile tools/pocket_ladder.py \
       tools/pocket_quality.py tools/pocket_soak_report.py -type f \
       \( -name '*.c' -o -name '*.h' -o -name '*.cu' -o -name '*.cuh' \
          -o -name '*.m' -o -name '*.metal' -o -name '*.sh' -o -name '*.py' \
          -o -name Makefile \) 2>/dev/null | LC_ALL=C sort | xargs -r sha256sum
) > "$B/sources.sha256"
{
  echo "bundle:   $name"
  echo "created:  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "host:     $(hostname) | $(uname -srm)"
  echo "soaks:    $*"
  echo "root:     $root"
  echo "model:    $mdir"
  echo
  echo "== git (the checkout may be rsynced; the hashes below are authoritative)"
  git -C "$root" log -1 --format='commit %H%nauthor %an <%ae>%ndate   %cI%n%n    %s' 2>&1 \
    || echo "(not a git checkout)"
  echo "-- git status --short (first 50 lines)"
  git -C "$root" status --short 2>&1 | head -50
  echo
  echo "== tree hashes"
  echo "sources.sha256  $(sha256sum < "$B/sources.sha256" | cut -d' ' -f1)  ($(wc -l < "$B/sources.sha256" | tr -d " ") files)"
  echo "binaries.sha256 $(sha256sum < "$B/binaries.sha256" | cut -d' ' -f1)"
  cat "$B/binaries.sha256"
  echo
  echo "== GPU"
  nvidia-smi --query-gpu=name,driver_version,memory.total,pci.bus_id --format=csv 2>&1
  echo
  echo "== CPU / memory"
  (lscpu 2>/dev/null | grep -E 'Model name|^CPU\(s\)|Thread|Socket|NUMA node\(s\)') || true
  (free -g 2>/dev/null | head -2) || true
  echo
  echo "== toolchain"
  (nvcc --version 2>/dev/null | tail -1) || true
  (cc --version 2>/dev/null | head -1) || true
} > "$B/provenance.txt" 2>&1

cp "$root/tools/gpu/serve.sh" "$root/tools/gpu/ab.sh" "$root/tools/gpu/soak.sh" "$B/profile/" 2>/dev/null
for f in source.json model.json; do
  cp "$mdir/$f" "$B/model/" 2>/dev/null || echo "missing $mdir/$f" >> "$B/model/MISSING.txt"
done

# --- per soak ---------------------------------------------------------------
for tag in "$@"; do
  T="$B/$tag"; mkdir -p "$T"
  if [ ! -f "$ev/$tag/$tag-summary.jsonl" ]; then
    echo "WARN: $ev/$tag/$tag-summary.jsonl missing" >&2
  fi
  cp "$ev/$tag/$tag-summary.jsonl" "$ev/$tag/$tag-health.json" "$T/" 2>/dev/null
  for j in "$ev/$tag/$tag"-c*.jsonl; do [ -f "$j" ] && cp "$j" "$T/"; done
  cp "$ev/$tag-server.log" "$ev/$tag-ladder.out" "$T/" 2>/dev/null
  [ -f "$ev/$tag-vram.log" ] && cp "$ev/$tag-vram.log" "$T/"
  grep -E '^MYNAH_|affinity:' "$ev/$tag-server.log" > "$T/server-env.txt" 2>/dev/null
  if [ -f "$ev/$tag/$tag-report.txt" ]; then
    cp "$ev/$tag/$tag-report.txt" "$T/"
  else
    for j in "$ev/$tag/$tag"-c*.jsonl; do
      [ -f "$j" ] && python3 "$root/tools/pocket_soak_report.py" "$j" --windows 10 >> "$T/$tag-report.txt"
    done
  fi
  [ -d "$ev/$tag/quality" ] && { mkdir -p "$T/quality"; cp "$ev/$tag/quality"/quality* "$T/quality/" 2>/dev/null; }
done

# --- corpus, audio ZIPs and REPORT.md ----------------------------------------
python3 - "$B" "$ev" "$@" <<'PY'
import csv, glob, io, json, os, re, shutil, sys, zipfile

B, ev, tags = sys.argv[1], sys.argv[2], sys.argv[3:]


def load_jsonl(p):
    try:
        with open(p) as f:
            return [json.loads(l) for l in f if l.strip()]
    except OSError:
        return []


def fmt(v, scale=1.0, nd=3):
    if v is None:
        return "-"
    return ("%%.%df" % nd) % (v * scale)


summaries, quality, vram, report_lines, stream_failed, zips = {}, {}, {}, {}, {}, {}
for tag in tags:
    T = os.path.join(B, tag)
    s = load_jsonl(os.path.join(T, "%s-summary.jsonl" % tag))
    summaries[tag] = s[-1] if s else None
    for row in s:
        c = row.get("corpus")
        if c and os.path.isfile(c):
            shutil.copy(c, os.path.join(B, "corpus", os.path.basename(c)))
    try:
        quality[tag] = json.load(open(os.path.join(T, "quality", "quality-summary.json")))
    except (OSError, ValueError):
        quality[tag] = None
    # VRAM: nvidia-smi CSV (memory.used column) or any "<n> MiB" per line.
    vals, col = [], None
    try:
        for line in open(os.path.join(T, "%s-vram.log" % tag)):
            if "memory.used" in line:
                col = [x.strip() for x in line.split(",")].index(
                    next(x.strip() for x in line.split(",") if "memory.used" in x))
                continue
            if col is not None:
                parts = line.split(",")
                if len(parts) > col:
                    m = re.search(r"[\d.]+", parts[col])
                    if m:
                        vals.append(float(m.group(0)))
                continue
            m = re.search(r"([\d.]+)\s*MiB", line)
            if m:
                vals.append(float(m.group(1)))
    except OSError:
        pass
    vram[tag] = (min(vals), max(vals), len(vals)) if vals else None
    try:
        report_lines[tag] = [l.rstrip() for l in open(os.path.join(T, "%s-report.txt" % tag))
                             if l.startswith(("whole soak", "drift"))]
    except OSError:
        report_lines[tag] = []
    try:
        stream_failed[tag] = sum(1 for l in open(os.path.join(T, "%s-server.log" % tag),
                                                 errors="replace") if "stream failed" in l)
    except OSError:
        stream_failed[tag] = None

    # Audio ZIP: exactly the WAVs the ladder captured, plus a manifest naming
    # each utterance and joining the ASR verdict when it exists.
    qrows = {}
    try:
        for r in csv.DictReader(open(os.path.join(T, "quality", "quality.csv"))):
            qrows[r["wav"]] = r
    except OSError:
        pass
    recs = []
    for j in sorted(glob.glob(os.path.join(T, "%s-c*.jsonl" % tag))):
        recs += [r for r in load_jsonl(j) if r.get("wav")]
    if recs:
        # The listening ZIP is capped (MYNAH_GPU_LISTEN_MB per soak, default 100,
        # so two soaks stay near 200 MB for chat attachments): every flagged or
        # high-WER utterance first (up to a quarter of the budget), then a
        # round-robin over kind x voice x soak third (start/middle/end), so a
        # slow degradation is audible too.  All captured WAVs stay on the box;
        # manifest-all.csv lists every one of them with its ASR verdict.
        budget = float(os.environ.get("MYNAH_GPU_LISTEN_MB", "100")) * 1048576.0
        rows, missing = [], 0
        recs.sort(key=lambda r: (r.get("t_send") or 0.0, r.get("id") or 0))
        for k, r in enumerate(recs):
            p = r["wav"]
            if not os.path.isfile(p):
                missing += 1
                continue
            q = qrows.get(p, {})
            bad = bool(q.get("flags")) or float(q.get("wer") or 0.0) > 0.3
            third = min(2, 3 * k // max(1, len(recs)))
            rows.append((r, q, p, os.path.getsize(p), bad, third))

        def mrow(r, q, p):
            return [os.path.basename(p), r.get("id"), r.get("measured"), r.get("kind"),
                    r.get("corpus_id"), r.get("voice"), r.get("seed"),
                    "%.2f" % (r.get("samples", 0) / 24000.0), fmt(r.get("ttfa_s")),
                    r.get("stalls_250"), q.get("wer", ""), q.get("flags", ""),
                    r.get("text"), q.get("hyp", "")]

        picked, used = [], 0.0
        for row in sorted((x for x in rows if x[4]),
                          key=lambda x: -float(x[1].get("wer") or 0.0)):
            if used + row[3] > budget / 4.0:
                break
            picked.append(row)
            used += row[3]
        groups = {}
        for row in rows:
            if row[4]:
                continue
            groups.setdefault((row[0].get("kind"), row[0].get("voice"), row[5]), []).append(row)
        queues = [groups[g] for g in sorted(groups, key=lambda g: tuple(str(x) for x in g))]
        for qu in queues:  # spread picks across each group, deterministically
            step = max(1, len(qu) // 8)
            qu[:] = qu[::step] + [x for i, x in enumerate(qu) if i % step]
        progress = True
        while progress and used < budget:
            progress = False
            for qu in queues:
                if qu and used + qu[0][3] <= budget:
                    row = qu.pop(0)
                    picked.append(row)
                    used += row[3]
                    progress = True
        header = ["file", "id", "measured", "kind", "corpus_id", "voice", "seed",
                  "seconds", "ttfa_s", "stalls_250", "wer", "flags", "text", "hyp"]
        zp = os.path.join(T, "audio-%s.zip" % tag)
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_STORED, allowZip64=True) as z:
            for r, q, p, size, bad, third in picked:
                z.write(p, "wav/" + os.path.basename(p))
            for name, sel in (("manifest.csv", picked), ("manifest-all.csv", rows)):
                buf = io.StringIO()
                w = csv.writer(buf)
                w.writerow(header)
                out = [mrow(r, q, p) for r, q, p, size, bad, third in sel]
                # Listen to the suspicious ones first: flagged, then by WER.
                out.sort(key=lambda m: (not m[11], -float(m[10] or 0), m[1] or 0))
                w.writerows(out)
                z.writestr(name, buf.getvalue())
        zips[tag] = (len(picked), missing, os.path.getsize(zp) / 1048576.0)
    else:
        zips[tag] = None

out = ["# Pocket CUDA qualification: %s" % os.path.basename(B.rstrip("/")), ""]
try:
    prov = open(os.path.join(B, "provenance.txt")).read().splitlines()
    out += ["Provenance (full detail in `provenance.txt`):", "", "```"]
    out += [l for l in prov if l.startswith(("created", "host", "commit", "sources.sha256",
                                              "binaries.sha256", "    "))][:8]
    out += ["```", ""]
except OSError:
    pass

rows = [
    ("offered concurrency (closed loop)", lambda s, q, t: str(s["offered"])),
    ("measured window, s", lambda s, q, t: "%.0f" % s["duration_s"]),
    ("requests sent / completed", lambda s, q, t: "%d / %d" % (s["sent"], s["completed"])),
    ("failed (client) / server_jobs_failed", lambda s, q, t: "%d / %d" % (s["failed"], s.get("server_jobs_failed", 0))),
    ("rejected HTTP / timeouts", lambda s, q, t: "%d / %d" % (s["rejected_http"], s["timeouts"])),
    ("`stream failed` lines in server log", lambda s, q, t: str(stream_failed.get(t))),
    ("throughput, audio-s/s", lambda s, q, t: "%.1f" % s["audio_s_per_s"]),
    ("TTFA p50 / p95 / p99, ms", lambda s, q, t: "%s / %s / %s" % tuple(fmt(s["ttfa"][k], 1000, 0) for k in ("p50", "p95", "p99"))),
    ("stream RTF p50 / p95 / max", lambda s, q, t: "%s / %s / %s" % tuple(fmt(s["rtf_stream"][k]) for k in ("p50", "p95", "max"))),
    ("e2e RTF p95", lambda s, q, t: fmt(s["rtf_e2e"]["p95"])),
    ("max inter-chunk gap p95 / max, ms", lambda s, q, t: "%s / %s" % (fmt(s["max_gap"]["p95"], 1000, 0), fmt(s["max_gap"]["max"], 1000, 0))),
    ("stalls @250 ms prebuffer (requests)", lambda s, q, t: "%d (%d)" % (s["stalls_250"], s["req_with_stall_250"])),
    ("stalls @500 ms prebuffer (requests)", lambda s, q, t: "%d (%d)" % (s["stalls_500"], s["req_with_stall_500"])),
    ("VRAM min / max, MiB (vram log)", lambda s, q, t: "%.0f / %.0f (%d samples)" % vram[t] if vram[t] else "no vram log"),
    ("VRAM max, MiB (ladder sampler)", lambda s, q, t: fmt(s.get("gpu_mem_max_mib"), 1, 0)),
    ("GPU util mean %, power mean W", lambda s, q, t: "%s, %s" % (fmt(s.get("gpu_util_mean"), 1, 1), fmt(s.get("gpu_power_mean_w"), 1, 1))),
    ("server CPU mean %, RSS max MiB", lambda s, q, t: "%s, %s" % (fmt(s.get("server_cpu_pct_mean"), 1, 0), fmt(s.get("server_rss_max_mib"), 1, 0))),
    ("captured WAVs (every Nth request)", lambda s, q, t: "%s (N=%s, %.0f MB)" % (s.get("saved_wavs"), s.get("save_every"), s.get("saved_mb", 0)) if "saved_wavs" in s else "not captured"),
    ("WER on captured audio", lambda s, q, t: fmt(q.get("wer"), 100, 2) + "%" if q and q.get("wer") is not None else "not run"),
    ("utterances > 30% WER", lambda s, q, t: "%s of %s" % (q.get("utt_over_bad_wer"), q.get("analysed")) if q and q.get("asr") else "not run"),
    ("auto-flagged audio", lambda s, q, t: "%d: %s" % (q["flagged"], ", ".join("%s=%d" % kv for kv in sorted(q["flag_counts"].items())) or "none") if q else "not run"),
]
out += ["## Headline", "", "| metric | " + " | ".join("`%s`" % t for t in tags) + " |",
        "|---|" + "---:|" * len(tags)]
for label, fn in rows:
    cells = []
    for t in tags:
        s = summaries.get(t)
        try:
            cells.append(fn(s, quality.get(t), t) if s else "missing")
        except Exception as e:
            cells.append("n/a")
    out.append("| %s | %s |" % (label, " | ".join(cells)))
out.append("")

out += ["## WER by kind and voice", ""]
for t in tags:
    q = quality.get(t)
    if q and q.get("wer_by_kind"):
        out.append("- `%s` kind: %s" % (t, ", ".join("%s %s%%" % (k, fmt(v, 100, 2)) for k, v in q["wer_by_kind"].items())))
        out.append("- `%s` voice: %s" % (t, ", ".join("%s %s%%" % (k, fmt(v, 100, 2)) for k, v in q["wer_by_voice"].items())))
    else:
        out.append("- `%s`: quality not run (`tools/pocket_quality.py --from-jsonl`)" % t)
out.append("")

out += ["## Stability over the soak (per-window report)", ""]
for t in tags:
    out += ["`%s`:" % t, "", "```"] + (report_lines.get(t) or ["(no report)"]) + ["```", ""]

out += ["## Files", ""]
for t in tags:
    z = zips.get(t)
    out.append("- `%s/`: summary, per-request JSONL, report, server log, env, quality/%s" % (
        t, (", `audio-%s.zip` (%d WAVs, %.0f MB%s)" % (t, z[0], z[2], ", %d missing" % z[1] if z[1] else ""))
        if z else ", no captured audio"))
out += ["- `corpus/`: %s" % (", ".join(sorted(os.listdir(os.path.join(B, "corpus")))) or "none"),
        "- `model/`, `profile/`, `provenance.txt`, `binaries.sha256`, `sources.sha256`", ""]

out += ["## Human listening", "",
        "Listener, date, WAVs heard (by `manifest.csv` row), artefacts found:", "",
        "TODO", "",
        "## Verdict", "",
        "Gates (throughput, TTFA p95, stream RTF p95, 0 stalls, 0 failures, WER, no",
        "drift between halves, both soaks agree): TODO", ""]
with open(os.path.join(B, "REPORT.md"), "w") as f:
    f.write("\n".join(out))
print("\n".join(out[:40]))
PY
echo
echo "bundle: $B"
du -sh "$B" 2>/dev/null
