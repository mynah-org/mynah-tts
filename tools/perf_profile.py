#!/usr/bin/env python3
"""Load, validate and resolve a serving profile.

A profile answers "given this hardware and this engine, how should the server be run,
and what has to be true before the answer may be called qualified?" -- and turns that
into an exact argv, an exact environment and an exact set of gates.

    tools/perf_profile.py validate                       # every profile in configs/perf
    tools/perf_profile.py show        recommended
    tools/perf_profile.py command     recommended --model models/pocket-en --port 8080
    tools/perf_profile.py server-env  recommended
    tools/perf_profile.py forbidden-env recommended
    tools/perf_profile.py soak        recommended --model models/pocket-en   # the soak command
    tools/perf_profile.py best        recommended
    tools/perf_profile.py new  my-box --like axion-c4a-32c-pocket-en

The point of the file is that a configuration cannot be got wrong by hand. The reason
that matters here is specific and was paid for: for six days every capacity number in
docs/ was measured with `codec_convtr:int8` exported by a shell line, while the binary
shipped it OFF -- a documented operating point the product could not reach by itself.
A profile makes that class of gap a validation failure instead of a footnote.
"""
import argparse, json, os, re, shlex, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PERF = os.path.join(ROOT, "configs", "perf")
COMPLETE_SECTIONS = ("hardware", "build", "server", "runtime", "gates", "qualification")


class Bad(Exception):
    pass


# -- a small JSON-Schema subset: enough for this schema, readable in one sitting --------
def _resolve(schema, root):
    while "$ref" in schema:
        ref = schema["$ref"]
        if not ref.startswith("#/"):
            raise Bad("unsupported $ref %s" % ref)
        node = root
        for part in ref[2:].split("/"):
            node = node[part]
        schema = node
    return schema


def check(node, schema, root, path="$"):
    schema = _resolve(schema, root)
    if "const" in schema:
        if node != schema["const"]:
            raise Bad("%s: expected %r, got %r" % (path, schema["const"], node))
        return
    if "enum" in schema:
        if node not in schema["enum"]:
            raise Bad("%s: %r is not one of %s" % (path, node, schema["enum"]))
        return

    t = schema.get("type")
    types = t if isinstance(t, list) else ([t] if t else [])
    if types:
        ok = any((ty == "object" and isinstance(node, dict))
                 or (ty == "array" and isinstance(node, list))
                 or (ty == "string" and isinstance(node, str))
                 or (ty == "boolean" and isinstance(node, bool))
                 or (ty == "integer" and isinstance(node, int) and not isinstance(node, bool))
                 or (ty == "number" and isinstance(node, (int, float))
                     and not isinstance(node, bool))
                 or (ty == "null" and node is None)
                 for ty in types)
        if not ok:
            raise Bad("%s: expected %s, got %s" % (path, "/".join(types), type(node).__name__))

    if isinstance(node, str) and "pattern" in schema:
        if not re.match(schema["pattern"], node):
            raise Bad("%s: %r does not match %s" % (path, node, schema["pattern"]))
    if isinstance(node, (int, float)) and not isinstance(node, bool):
        if "minimum" in schema and node < schema["minimum"]:
            raise Bad("%s: %s < minimum %s" % (path, node, schema["minimum"]))

    if isinstance(node, dict):
        for req in schema.get("required", []):
            if req not in node:
                raise Bad("%s: missing required field %r" % (path, req))
        props = schema.get("properties", {})
        extra = schema.get("additionalProperties", True)
        for k, v in node.items():
            if k in props:
                check(v, props[k], root, "%s.%s" % (path, k))
            elif isinstance(extra, dict):
                check(v, extra, root, "%s.%s" % (path, k))
            elif extra is False:
                raise Bad("%s: unknown field %r" % (path, k))
    if isinstance(node, list):
        if "minItems" in schema and len(node) < schema["minItems"]:
            raise Bad("%s: needs at least %d items" % (path, schema["minItems"]))
        if "maxItems" in schema and len(node) > schema["maxItems"]:
            raise Bad("%s: allows at most %d items" % (path, schema["maxItems"]))
        item = schema.get("items")
        if item:
            for i, v in enumerate(node):
                check(v, item, root, "%s[%d]" % (path, i))


# -- loading ---------------------------------------------------------------------------
def load_raw(name):
    path = os.path.join(PERF, name + ".json") if not name.endswith(".json") else name
    if not os.path.exists(path):
        raise Bad("no such profile: %s" % path)
    with open(path, encoding="utf-8") as f:
        return json.load(f), path


def load(name, _seen=None):
    """Resolve aliases to the profile they point at, refusing cycles."""
    _seen = _seen or []
    if name in _seen:
        raise Bad("alias cycle: %s" % " -> ".join(_seen + [name]))
    raw, path = load_raw(name)
    target = raw["profile"].get("alias_of")
    if target:
        return load(target, _seen + [name])
    return raw, path


def semantic(prof, path):
    """What a schema cannot say: the rules that make a profile mean something."""
    errs = []
    p, hw = prof["profile"], prof.get("hardware", {})
    alias = p.get("alias_of")
    present = [s for s in COMPLETE_SECTIONS if s in prof]
    if alias:
        if present:
            errs.append("an alias must carry no other section, but has: %s" % ", ".join(present))
        return errs
    for s in COMPLETE_SECTIONS:
        if s not in prof:
            errs.append("a complete profile needs section %r" % s)
    if errs:
        return errs

    if os.path.basename(path)[:-5] != p["id"] and not alias:
        errs.append("profile.id %r does not match the file name" % p["id"])

    srv = prof["server"]
    used = srv["prefork_workers"] * srv["threads_per_worker"]
    if hw.get("logical_cpus") and used > hw["logical_cpus"]:
        errs.append("server asks for %d threads (%dx%d) on a %d-CPU host"
                    % (used, srv["prefork_workers"], srv["threads_per_worker"],
                       hw["logical_cpus"]))

    obj = p.get("objective", {})
    lo, hi = (obj.get("concurrency_range") or [None, None])
    pref = obj.get("preferred_concurrency")
    if pref is not None and lo is not None and not (lo <= pref <= hi):
        errs.append("preferred_concurrency %d is outside concurrency_range %d-%d"
                    % (pref, lo, hi))

    q = prof["qualification"]
    op = q.get("operating_point")
    if q["status"] == "qualified":
        if not op:
            errs.append("status is 'qualified' with no operating_point: nothing was measured")
        else:
            if op["verdict"] != "GOOD":
                errs.append("status is 'qualified' but the operating point is %s"
                            % op["verdict"])
            need = prof["gates"]["soak"]["seconds"]
            if op.get("soak_seconds", 0) < need:
                errs.append("status is 'qualified' from a %ss soak; gates.soak.seconds "
                            "requires %ds -- a screen may not promote"
                            % (op.get("soak_seconds", "?"), need))
            if op.get("stalls_250ms") or op.get("stalls_500ms"):
                errs.append("status is 'qualified' with stalls recorded at the operating "
                            "point")
            if pref is not None and op["concurrency"] != pref:
                errs.append("preferred_concurrency is %d but the operating point is C%d"
                            % (pref, op["concurrency"]))
    if op and q.get("history"):
        if q["history"][-1] != op:
            errs.append("qualification.history must end with the current operating_point, "
                        "so that 'the best' is one place and not two")
    for pt in q.get("ceiling", []):
        if pt["verdict"] == "GOOD":
            errs.append("C%d sits in `ceiling` with verdict GOOD: a GOOD level above the "
                        "operating point is either the new operating point or needs "
                        "why_not_promoted" % pt["concurrency"])
    bank = prof["gates"]["soak"]["bank"]
    if not os.path.exists(os.path.join(ROOT, bank)):
        errs.append("gates.soak.bank %r does not exist" % bank)
    return errs


# -- resolution ------------------------------------------------------------------------
def environ(prof):
    """The variables a run must EXPORT (value not null)."""
    return {k: s["value"] for k, s in prof["runtime"]["environment"].items()
            if s["value"] is not None}


def forbidden_env(prof):
    """Variables declared null: they must be ABSENT, not merely unset by us.

    This is the anti-footgun. MYNAH_QUANT_GROUPS is the case that matters: a profile that
    claims to match the shipped default is only measuring the default if that variable is
    not in the environment at all -- an export left over in the shell silently turns the
    run into a measurement of something else that will still look like a pass.
    """
    return sorted(k for k, s in prof["runtime"]["environment"].items() if s["value"] is None)


def server_args(prof):
    srv = prof["server"]
    a = ["--prefork", str(srv["prefork_workers"]),
         "--prefork-threads", str(srv["threads_per_worker"]),
         "--max-batch", str(srv["max_batch"])]
    for key, flag in (("max_pending", "--max-pending"),
                      ("request_timeout_ms", "--request-timeout-ms")):
        if isinstance(srv.get(key), int):
            a += [flag, str(srv[key])]
    return a + srv.get("extra_arguments", [])


def gate_args(prof):
    """The verdict thresholds, as serving_profile.py flags."""
    g = prof["gates"]
    m, p, d, s = g["mandatory"], g.get("preferred", {}), g.get("drift", {}), g["soak"]
    a = ["--rtf-hard", str(m["rtf_p95_below"]),
         "--stall-mandatory-ms", str(m["stall_ms"]),
         "--soak-seconds", str(s["seconds"]),
         "--warmup-seconds", str(s["warmup_seconds"]),
         "--window-seconds", str(s["window_seconds"]),
         "--bank", s["bank"]]
    for key, flag in (("rtf_p95", "--rtf-pref"), ("ttfb_p95_ms", "--ttfb-pref-ms"),
                      ("ttfa_p95_ms", "--ttfa-pref-ms"),
                      ("prebuffer_p95_ms", "--prebuffer-pref-ms"),
                      ("safe_start_p95_ms", "--safe-start-pref-ms"),
                      ("stall_ms", "--stall-gate-ms")):
        if key in p:
            a += [flag, str(p[key])]
    for key, flag in (("rtf_tolerance", "--drift-rtf-tol"),
                      ("prebuffer_tolerance_ms", "--drift-prebuffer-tol-ms"),
                      ("min_windows", "--drift-min-windows")):
        if key in d:
            a += [flag, str(d[key])]
    return a


def env_line(prof):
    return " ".join("%s=%s" % (k, shlex.quote(v)) for k, v in sorted(environ(prof).items()))


# -- commands --------------------------------------------------------------------------
def cmd_validate(a):
    files = sorted(f for f in os.listdir(PERF) if f.endswith(".json") and f != "schema.json")
    with open(os.path.join(PERF, "schema.json"), encoding="utf-8") as f:
        schema = json.load(f)
    bad = 0
    for fn in files:
        name = fn[:-5]
        try:
            raw, path = load_raw(name)
            check(raw, schema, schema, "$")
            prof, ppath = load(name)
            errs = semantic(prof, ppath)
            if errs:
                bad += 1
                print("FAIL %s" % fn)
                for e in errs:
                    print("     %s" % e)
            else:
                print("ok   %s%s" % (fn, " (alias)" if raw["profile"].get("alias_of") else ""))
        except Bad as e:
            bad += 1
            print("FAIL %s\n     %s" % (fn, e))
    print("\n%d/%d profiles valid" % (len(files) - bad, len(files)))
    return 1 if bad else 0


def cmd_show(a):
    print(json.dumps(load(a.name)[0], indent=2, ensure_ascii=False))
    return 0


def cmd_command(a):
    prof = load(a.name)[0]
    line = env_line(prof)
    argv = ["./build/cpu/mynah-tts-server", "-m", a.model, "-p", str(a.port)] + server_args(prof)
    print((line + " " if line else "") + " ".join(argv))
    return 0


def cmd_server_env(a):
    print(server_env(load(a.name)[0]))
    return 0


def cmd_forbidden_env(a):
    for k in forbidden_env(load(a.name)[0]):
        print(k)
    return 0


def cmd_soak(a):
    """The exact soak that would qualify this profile, as one copy-pasteable line."""
    prof = load(a.name)[0]
    c = a.level or (prof["profile"].get("objective", {}).get("preferred_concurrency"))
    if c is None:
        print("no --level and no objective.preferred_concurrency", file=sys.stderr)
        return 2
    line = env_line(prof)
    argv = ["python3", "tools/serving_profile.py", "--profile", prof["profile"]["id"],
            "--model", a.model, "--levels", str(c)]
    print((line + " " if line else "") + " ".join(argv))
    return 0


def cmd_best(a):
    """What this profile has qualified, and what stopped the level above it."""
    prof = load(a.name)[0]
    q = prof["qualification"]
    op = q.get("operating_point")
    print("%s -- %s" % (prof["profile"]["id"], q["status"]))
    if not op:
        print("  nothing measured on this hardware yet")
        return 0
    print("  operating point   C%d  %s   %s  (%ds soak, %d requests)"
          % (op["concurrency"], op["verdict"], op["date"], op.get("soak_seconds", 0),
             op.get("requests", 0)))
    for k, label, unit in (("ttfa_p95_ms", "TTFA p95", "ms"), ("rtf_p95", "RTF p95", ""),
                           ("stalls_250ms", "stalls@250", ""),
                           ("throughput_audio_s_per_s", "throughput", "audio-s/s")):
        if k in op:
            print("    %-12s %s%s" % (label, op[k], (" " + unit) if unit else ""))
    if op.get("evidence"):
        print("    evidence     %s" % op["evidence"])
    for pt in q.get("ceiling", []):
        print("  above it          C%d  %s -- %s"
              % (pt["concurrency"], pt["verdict"], pt.get("why_not_promoted", "")))
    hist = q.get("history", [])
    if len(hist) > 1:
        print("  history")
        for pt in hist:
            print("    %s  C%-4d %-8s %s" % (pt["date"], pt["concurrency"], pt["verdict"],
                                             pt.get("version", pt.get("commit", ""))))
    return 0


# -- "what is this box, and which profile do I start from?" ----------------------------
def host_facts():
    """What the machine says about itself. No model, no weights, no measurement."""
    import platform, subprocess
    f = {"architecture": {"aarch64": "arm64", "arm64": "arm64",
                          "x86_64": "x86_64", "amd64": "x86_64"}.get(platform.machine(),
                                                                     platform.machine()),
         "logical_cpus": os.cpu_count() or 0, "cpu_family": "unknown",
         "physical_cores": 0, "numa_nodes": 1, "ram_gib": 0, "isa_features_used": []}
    try:
        if sys.platform == "linux":
            txt = open("/proc/cpuinfo").read()
            for key in ("model name", "Model name", "CPU part"):
                for line in txt.splitlines():
                    if line.startswith(key):
                        f["cpu_family"] = line.split(":", 1)[1].strip()
                        break
                if f["cpu_family"] != "unknown":
                    break
            flags = set()
            for line in txt.splitlines():
                if line.startswith(("flags", "Features")):
                    flags |= set(line.split(":", 1)[1].split())
            for name in ("asimddp", "i8mm", "bf16", "sve", "sve2", "avx2", "avx512f",
                         "avx512_vnni", "amx_tile"):
                if name in flags:
                    f["isa_features_used"].append(name)
            f["ram_gib"] = int(round(os.sysconf("SC_PAGE_SIZE")
                                     * os.sysconf("SC_PHYS_PAGES") / (1 << 30)))
            f["numa_nodes"] = len([d for d in os.listdir("/sys/devices/system/node")
                                   if d.startswith("node")]) or 1
            out = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
            for line in out.splitlines():
                if line.startswith("Core(s) per socket"):
                    per = int(line.split(":")[1])
                elif line.startswith("Socket(s)"):
                    f["physical_cores"] = per * int(line.split(":")[1])
        elif sys.platform == "darwin":
            def sysctl(k):
                return subprocess.run(["sysctl", "-n", k], capture_output=True,
                                      text=True).stdout.strip()
            f["cpu_family"] = sysctl("machdep.cpu.brand_string") or "Apple Silicon"
            f["physical_cores"] = int(sysctl("hw.physicalcpu") or 0)
            f["ram_gib"] = int(int(sysctl("hw.memsize") or 0) / (1 << 30))
    except Exception:
        pass
    f["physical_cores"] = f["physical_cores"] or f["logical_cpus"]
    f["threads_per_core"] = max(1, f["logical_cpus"] // max(1, f["physical_cores"]))
    return f


def rank_profiles(facts):
    """Closest first. Architecture is a wall, not a weight: an ARM profile on an x86 box
    is not 'a bit off', it is a different set of kernels and a different answer."""
    out = []
    for fn in sorted(os.listdir(PERF)):
        if not fn.endswith(".json") or fn == "schema.json":
            continue
        raw, _ = load_raw(fn[:-5])
        if raw["profile"].get("alias_of"):
            continue
        prof, path = load(fn[:-5])
        hw = prof.get("hardware", {})
        if hw.get("architecture") != facts["architecture"]:
            continue
        diffs = []
        score = 0.0
        if hw.get("logical_cpus"):
            ratio = facts["logical_cpus"] / hw["logical_cpus"]
            score += abs(1.0 - ratio)
            if facts["logical_cpus"] != hw["logical_cpus"]:
                diffs.append("%d logical CPUs here against %d there"
                             % (facts["logical_cpus"], hw["logical_cpus"]))
        if hw.get("threads_per_core") and hw["threads_per_core"] != facts["threads_per_core"]:
            score += 0.5
            diffs.append("SMT: %d thread(s) per core here against %d there"
                         % (facts["threads_per_core"], hw["threads_per_core"]))
        missing = [x for x in hw.get("isa_features_used", [])
                   if facts["isa_features_used"] and x not in facts["isa_features_used"]]
        if missing:
            score += 0.25 * len(missing)
            diffs.append("the profile uses %s, not reported here" % ", ".join(missing))
        out.append((score, prof, path, diffs))
    out.sort(key=lambda t: t[0])
    return out


def cmd_detect(a):
    """Say what this machine is and which profile is the honest starting point."""
    f = host_facts()
    print("THIS HOST")
    print("  %s, %s" % (f["architecture"], f["cpu_family"]))
    print("  %d logical CPUs / %d physical cores (%d thread(s) per core), %d NUMA node(s), "
          "%d GiB" % (f["logical_cpus"], f["physical_cores"], f["threads_per_core"],
                      f["numa_nodes"], f["ram_gib"]))
    if f["isa_features_used"]:
        print("  ISA: %s" % ", ".join(f["isa_features_used"]))
    if sys.platform == "darwin":
        print("  PLATFORM  macOS is a development platform, not a target. Accelerate and")
        print("            BNNS do not exist on Linux and the P-core thread default is a")
        print("            macOS concept, so a profile qualified here would describe a")
        print("            machine we do not ship on. Screen here; qualify on Linux.")
    print()

    ranked = rank_profiles(f)
    if not ranked:
        print("NO PROFILE FOR THIS ARCHITECTURE (%s)." % f["architecture"])
        print("  Start one: tools/perf_profile.py new <id> --like <closest>")
        print("  There is no default to fall back on, and that is deliberate: a topology")
        print("  carried over from another architecture is a guess wearing a number.")
        return 0

    score, prof, path, diffs = ranked[0]
    q = prof["qualification"]
    op = q.get("operating_point") or {}
    srv = prof["server"]
    exact = not diffs
    print("CLOSEST PROFILE   %s%s" % (prof["profile"]["id"],
                                      "   (exact hardware match)" if exact else ""))
    print("  %s" % prof["profile"]["description"])
    print("  topology        %d workers x %d threads, max-batch %d"
          % (srv["prefork_workers"], srv["threads_per_worker"], srv["max_batch"]))
    print("  build           BLAS=%s  SIMD=%s" % (prof["build"]["blas"], prof["build"]["simd"]))
    print("  environment     %s" % (server_env(prof) or "(none -- the shipped default)"))
    if op:
        print("  it qualified    C%d %s over %ds on ITS host"
              % (op["concurrency"], op["verdict"], op.get("soak_seconds", 0)))
    for d in diffs:
        print("  DIFFERS         %s" % d)
    print()

    if exact:
        print("WHAT TO DO")
        print("  Reproduce it before believing it:")
        print("    python3 tools/serving_profile.py --profile %s --model models/pocket-en"
              % prof["profile"]["id"])
        print("  Same silicon is not the same machine: kernel, BLAS and neighbours differ.")
    else:
        print("WHAT TO DO")
        print("  1. tools/perf_profile.py new <id> --like %s" % prof["profile"]["id"])
        print("  2. fill hardware from what THIS box reports (the block above)")
        print("  3. re-screen the topology -- do NOT inherit %dx%d."
              % (srv["prefork_workers"], srv["threads_per_worker"]))
        print("     The optimum follows the model's T_frame(B) = a + b*B ratio: PocketTTS")
        print("     (109.5M) wants many narrow workers, qwen-tts (1.7B) wants few wide ones,")
        print("     on the SAME 32-core host. Scale the count with the cores you have and")
        print("     screen one step either side of it.")
        print("  4. soak 30 minutes at the candidate level, then write what it earned")
    if len(ranked) > 1:
        print()
        print("OTHER PROFILES FOR THIS ARCHITECTURE")
        for sc, pr, _, df in ranked[1:]:
            print("  %-32s %s" % (pr["profile"]["id"], df[0] if df else "exact match"))
    return 0


def server_env(prof):
    return ",".join("%s=%s" % kv for kv in sorted(environ(prof).items()))


def cmd_new(a):
    """A skeleton, with everything unmeasured marked as such.

    Copying a profile turns a number measured on another machine into a claim about this
    one. Starting from 'unspecified' makes filling a field a decision.
    """
    ref = load(a.like)[0]
    out = json.loads(json.dumps(ref))
    out["profile"]["id"] = a.name
    out["profile"]["description"] = ("FILL IN. Every value below must come from a "
                                     "measurement on THIS machine.")
    for k in ("cpu_family", "canonical_topology", "smt_policy"):
        if k in out.get("hardware", {}):
            out["hardware"][k] = "unspecified"
    out["hardware"]["notes"] = ("PLACEHOLDER counts copied for shape only. Replace with what "
                                "this machine reports before quoting anything here.")
    out["qualification"] = {
        "status": "unqualified",
        "model": out["qualification"]["model"],
        "workload": "NOTHING MEASURED HERE YET.",
        "notes": ("Run `tools/perf_profile.py soak <id>`, then write the operating point it "
                  "earned and set status to 'qualified'. The topology is a property of the "
                  "model's a/b ratio as much as of the machine: re-screen it, do not inherit "
                  "it."),
    }
    dest = os.path.join(PERF, a.name + ".json")
    if os.path.exists(dest) and not a.force:
        print("%s exists; pass --force to overwrite" % dest, file=sys.stderr)
        return 2
    with open(dest, "w", encoding="utf-8") as f:
        f.write(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print("wrote %s\nnext: edit it, then `tools/perf_profile.py validate`" % dest)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("validate").set_defaults(fn=cmd_validate)
    s = sub.add_parser("show"); s.add_argument("name"); s.set_defaults(fn=cmd_show)
    c = sub.add_parser("command")
    c.add_argument("name"); c.add_argument("--model", required=True)
    c.add_argument("--port", type=int, default=8080); c.set_defaults(fn=cmd_command)
    e = sub.add_parser("server-env"); e.add_argument("name"); e.set_defaults(fn=cmd_server_env)
    f = sub.add_parser("forbidden-env"); f.add_argument("name")
    f.set_defaults(fn=cmd_forbidden_env)
    k = sub.add_parser("soak"); k.add_argument("name"); k.add_argument("--model", required=True)
    k.add_argument("--level", type=int, default=0); k.set_defaults(fn=cmd_soak)
    b = sub.add_parser("best"); b.add_argument("name"); b.set_defaults(fn=cmd_best)
    sub.add_parser("detect").set_defaults(fn=cmd_detect)
    n = sub.add_parser("new"); n.add_argument("name"); n.add_argument("--like", required=True)
    n.add_argument("--force", action="store_true"); n.set_defaults(fn=cmd_new)
    a = ap.parse_args(argv)
    try:
        return a.fn(a)
    except Bad as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
