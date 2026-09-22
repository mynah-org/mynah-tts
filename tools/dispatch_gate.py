#!/usr/bin/env python3
"""dispatch_gate.py - expected vs observed dispatch, as a gate.

    tools/dispatch_gate.py <dispatch.json> [--expect tools/dispatch_expect.json]
    tools/dispatch_gate.py --binary build/cpu/mynah-tts
    tools/dispatch_gate.py --selftest

WHY THIS EXISTS. The dispatch report is the most carefully written document in
this repository and nothing checks it. On 2026-09-22 an EPYC 9254 ran a bf16
kernel while `make simd-auto` called it NOT-IMPLEMENTED, and the report called
a unit that two kernels require "idle hardware". Both were read by a person,
neither broke a test, and both had been true for exactly one day.

The generic rules below need no expectation table and are the valuable half:
they describe shapes that are wrong whatever the host is.

  SUSPICIOUS  compiled=yes, supported=yes, resolved=OFF, and no env asked for
              that -- the unit is there, the kernel is there, and neither runs.
              This is the shape of a fallback nobody chose.
  ORPHAN      source=unknown -- a row no module owns, so its `resolved` is a
              guess printed in a table whose purpose is to not guess.
  GUARD       isa_guard_ok=false -- the binary should not have started here.
  DRIFT       gate_drift=true -- a compile gate disagrees with the module that
              owns it, which is the class of bug the drift canary exists for.

IDLE is reported, not failed: hardware we have no kernel for is a roadmap item,
not a defect, and `idle_hardware_count` is the number to watch rather than to
gate on.
"""
import argparse, json, os, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_EXPECT = os.path.join(HERE, "dispatch_expect.json")


def load_report(path=None, binary=None):
    if binary:
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "dispatch.json")
            env = dict(os.environ, MYNAH_DISPATCH_JSON=out)
            subprocess.run([binary, "--dispatch-map", "--json"], env=env,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           check=False)
            if os.path.exists(out):
                with open(out, encoding="utf-8") as fh:
                    return json.load(fh)
            r = subprocess.run([binary, "--dispatch-map", "--json"],
                               capture_output=True, text=True, check=True)
            return json.loads(r.stdout)
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def rows_by_id(report):
    return {r["id"]: r for r in report.get("features", [])}


def generic_findings(report):
    """Wrong shapes, whatever the host is."""
    out = []
    if report.get("isa_guard_ok") is False:
        out.append(("GUARD", "-", report.get("isa_guard_detail", "")))
    if report.get("gate_drift") is True:
        out.append(("DRIFT", "-", report.get("gate_drift_detail", "")))
    for r in report.get("features", []):
        rid = r.get("id", "?")
        if r.get("source") == "unknown":
            out.append(("ORPHAN", rid,
                        "no module registered a predicate for this row"))
            continue
        # An env that was SET is a human choice: not suspicious, whatever it did.
        asked = (r.get("env_value") or "unset") not in ("unset", "")
        if (r.get("compiled") == "yes" and r.get("supported") == "yes"
                and r.get("resolved") == "OFF" and not asked):
            out.append(("SUSPICIOUS", rid,
                        "compiled and supported here, and it resolved OFF with "
                        "nothing in the environment asking for that"))
    return out


def expected_findings(report, expect):
    cls = report.get("isa_class", "")
    table = expect.get(cls)
    if table is None:
        return [], cls, 0
    rows = rows_by_id(report)
    out = []
    for rid, want in table.items():
        if rid.startswith("_"):
            continue
        row = rows.get(rid)
        if row is None:
            out.append(("NOTE", rid, "expected %r, and this binary has no such "
                                     "row" % want))
            continue
        got = row.get("resolved", "")
        if got == want or (want not in ("ON", "OFF") and got.startswith(want)):
            continue
        # An env that was SET is a human choice, and the expectation table
        # describes a DEFAULT host. Without this, `MYNAH_KERNELS_X86=scalar`
        # -- which exists so both halves of a dispatch can be exercised on one
        # machine -- makes the gate cry about the thing it was asked to do.
        # Caught by this file's own self-test before it was ever run for real.
        if (row.get("env_value") or "unset") not in ("unset", ""):
            print("  ASKED       %-26s resolved %r, and %s=%s asked for that"
                  % (rid, got, row.get("env") or "an env",
                     row.get("env_value")))
            continue
        if (want == "ON" and got == "OFF" and row.get("compiled") == "yes"
                and row.get("supported") == "yes"):
            out.append(("SUSPICIOUS", rid,
                        "expected ON, compiled and supported here, resolved OFF"))
        else:
            out.append(("MISMATCH", rid, "expected %r, got %r" % (want, got)))
    return out, cls, len([k for k in table if not k.startswith("_")])


def report_findings(report, expect, warn_only=False):
    findings = generic_findings(report)
    exp, cls, n_exp = expected_findings(report, expect)
    findings += exp
    print("dispatch gate: isa_class=%s build=%s rows=%d, %d expectations"
          % (cls or "?", report.get("build", "?"),
             len(report.get("features", [])), n_exp))
    idle = report.get("idle_hardware_count", 0)
    if idle:
        print("  IDLE   %d unit(s) this CPU has and this binary has no kernel "
              "for: %s" % (idle, report.get("idle_hardware", "")))
    fatal = 0
    for kind, rid, why in findings:
        print("  %-11s %-26s %s" % (kind, rid, why))
        if kind in ("SUSPICIOUS", "MISMATCH", "ORPHAN", "GUARD", "DRIFT"):
            fatal += 1
    if not findings:
        print("  clean -- every expectation met and no generic rule fired")
    if fatal and warn_only:
        print("dispatch gate: %d finding(s), --warn-only" % fatal)
        return 0
    if fatal:
        print("dispatch gate: FAIL (%d)" % fatal)
        return 1
    print("dispatch gate: PASS")
    return 0


def selftest():
    """The gate has to be able to FAIL, or its PASS says nothing."""
    base = {"isa_class": "x86_avx512_vnni", "build": "test",
            "isa_guard_ok": True, "gate_drift": False, "idle_hardware_count": 0,
            "features": []}

    def row(rid, compiled="yes", supported="yes", resolved="ON",
            source="predicate", env_value="unset"):
        return {"id": rid, "compiled": compiled, "supported": supported,
                "resolved": resolved, "source": source, "env_value": env_value,
                "env": "", "reason": ""}

    cases = []
    ok = dict(base, features=[
        row("isa.x86.avx2"), row("isa.x86.fma"), row("isa.x86.avx512f"),
        row("isa.x86.avx512bw"), row("isa.x86.avx512vl"),
        row("isa.x86.avx512vnni"),
        row("quant.int8_kernel", resolved="avx512vnni"),
        row("sgemm.provider", resolved="mynah")])
    cases.append(("a host meeting its class expectation", ok, 0))

    bad = json.loads(json.dumps(ok))
    for r in bad["features"]:
        if r["id"] == "isa.x86.avx512bf16" or r["id"] == "isa.x86.avx512vnni":
            r["resolved"] = "OFF"
    cases.append(("have the unit, have the kernel, run neither", bad, 1))

    orphan = dict(base, features=[row("isa.x86.avx2", source="unknown")])
    cases.append(("a row no module owns", orphan, 1))

    guard = dict(base, isa_guard_ok=False, features=[row("isa.x86.avx2")])
    cases.append(("a binary that should not have started", guard, 1))

    drift = dict(base, gate_drift=True, features=[row("isa.x86.avx2")])
    cases.append(("a compile gate disagreeing with its module", drift, 1))

    asked = dict(base, features=[
        row("isa.x86.avx2", resolved="OFF", env_value="scalar")])
    cases.append(("OFF because a human asked for it", asked, 0))

    with open(DEFAULT_EXPECT, encoding="utf-8") as fh:
        expect = json.load(fh)
    fails = 0
    for name, rep, want in cases:
        got = report_findings(rep, expect, warn_only=False)
        mark = "ok  " if got == want else "FAIL"
        if got != want:
            fails += 1
        print("  %s %-46s want exit %d, got %d\n" % (mark, name, want, got))
    print("dispatch-gate self-test: %s (%d cases, %d failed)"
          % ("PASS" if fails == 0 else "FAIL", len(cases), fails))
    return 1 if fails else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("report", nargs="?")
    ap.add_argument("--expect", default=DEFAULT_EXPECT)
    ap.add_argument("--binary")
    ap.add_argument("--warn-only", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.report and not a.binary:
        ap.error("give a dispatch.json, or --binary to produce one")
    with open(a.expect, encoding="utf-8") as fh:
        expect = json.load(fh)
    return report_findings(load_report(a.report, a.binary), expect, a.warn_only)


if __name__ == "__main__":
    sys.exit(main())
