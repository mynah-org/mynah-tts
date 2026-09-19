#!/usr/bin/env python3
"""Self-test for the serving-profile loader.

Two halves, and the second is the one that earns its keep:

  1. every committed profile validates;
  2. the refusals actually refuse -- a profile that can be silently contradicted by a
     shell export or a command-line flag would qualify nothing, which is the exact
     failure this file exists to prevent.
"""
import copy, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import perf_profile as PP                                    # noqa: E402

CASES = []


def check(label, cond):
    CASES.append((bool(cond), label))
    if not cond:
        print("  FAIL  %s" % label)


def load_reference():
    prof, _ = PP.load("axion-c4a-32c-pocket-en")
    return copy.deepcopy(prof)


def errors_for(prof):
    return PP.semantic(prof, os.path.join(PP.PERF, prof["profile"]["id"] + ".json"))


def main():
    # -- 1. the committed set ----------------------------------------------------------
    check("every committed profile validates", PP.main(["validate"]) == 0)

    raw, _ = PP.load_raw("recommended")
    check("recommended is an alias, not a second copy",
          raw["profile"].get("alias_of") and
          not any(s in raw for s in PP.COMPLETE_SECTIONS))
    resolved, path = PP.load("recommended")
    check("the alias resolves to a complete profile",
          resolved["profile"]["id"] != "recommended" and "qualification" in resolved)

    # -- 2. resolution -----------------------------------------------------------------
    prof = load_reference()
    argv = PP.server_args(prof)
    check("server args carry the pinned topology",
          argv[:6] == ["--prefork", "16", "--prefork-threads", "2", "--max-batch", "8"])
    gates = dict(zip(PP.gate_args(prof)[0::2], PP.gate_args(prof)[1::2]))
    check("the TTFA gate comes from the profile", gates["--ttfa-pref-ms"] == "500.0")
    check("the soak length comes from the profile", gates["--soak-seconds"] == "1800")
    check("the bank comes from the profile",
          gates["--bank"] == "tests/load_texts_en_v2.txt")
    check("a profile that matches the shipped default exports nothing",
          PP.environ(prof) == {} and prof["runtime"]["matches_shipped_default"] is True)
    check("MYNAH_QUANT_GROUPS is declared as must-be-absent",
          "MYNAH_QUANT_GROUPS" in PP.forbidden_env(prof))

    # -- 3. the semantic rules, each one broken on purpose -----------------------------
    p = load_reference(); p["server"]["threads_per_worker"] = 8
    check("refuses a topology wider than the host",
          any("threads" in e for e in errors_for(p)))

    p = load_reference(); p["qualification"]["operating_point"]["soak_seconds"] = 600
    p["qualification"]["history"][-1]["soak_seconds"] = 600
    check("refuses 'qualified' earned by a ten-minute screen",
          any("may not promote" in e for e in errors_for(p)))

    p = load_reference(); p["qualification"]["operating_point"]["verdict"] = "MARGINAL"
    p["qualification"]["history"][-1]["verdict"] = "MARGINAL"
    check("refuses 'qualified' on a MARGINAL point",
          any("operating point is MARGINAL" in e for e in errors_for(p)))

    p = load_reference(); p["qualification"]["operating_point"]["stalls_250ms"] = 3
    p["qualification"]["history"][-1]["stalls_250ms"] = 3
    check("refuses 'qualified' with stalls at the operating point",
          any("with stalls recorded" in e for e in errors_for(p)))

    p = load_reference(); p["qualification"]["ceiling"][0]["verdict"] = "GOOD"
    check("refuses a GOOD level parked in `ceiling`",
          any("either the new operating point" in e for e in errors_for(p)))

    p = load_reference(); p["qualification"]["history"].append(
        dict(p["qualification"]["history"][-1], concurrency=99))
    check("refuses a history that does not end at the operating point",
          any("history must end" in e for e in errors_for(p)))

    p = load_reference()
    p["profile"]["objective"]["preferred_concurrency"] = \
        p["qualification"]["operating_point"]["concurrency"] + 1
    check("refuses a preferred concurrency that is not the measured one",
          any("operating point is C" in e for e in errors_for(p)))

    p = load_reference(); p["gates"]["soak"]["bank"] = "tests/no_such_bank.txt"
    check("refuses a bank that is not there", any("does not exist" in e for e in errors_for(p)))

    # -- 4. the refusals that protect a RUN --------------------------------------------
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import serving_profile as SP                              # noqa: E402

    class Args:
        pass

    def apply(env=None, argv=()):
        a = Args()
        for k, v in (("profile", "axion-c4a-32c-pocket-en"), ("server_args", ""),
                     ("levels", "1"), ("mode", "wave"), ("bank", ""),
                     ("soak_seconds", 300.0), ("warmup_seconds", 30.0),
                     ("window_seconds", 60.0), ("rtf_hard", 1.0), ("rtf_pref", 0.9),
                     ("ttfb_pref_ms", 100.0), ("ttfa_pref_ms", 500.0),
                     ("prebuffer_pref_ms", 500.0), ("safe_start_pref_ms", 1000.0),
                     ("stall_gate_ms", 250), ("stall_mandatory_ms", 500),
                     ("drift_rtf_tol", 0.05), ("drift_prebuffer_tol_ms", 150.0),
                     ("drift_min_windows", 3)):
            setattr(a, k, v)
        saved_env, saved_argv = dict(os.environ), list(sys.argv)
        try:
            for k in PP.forbidden_env(PP.load(a.profile)[0]):
                os.environ.pop(k, None)
            os.environ.update(env or {})
            sys.argv = ["serving_profile.py"] + list(argv)
            SP.apply_profile(a, None)
            return a, None
        except SP.PB.Refusal as exc:
            return None, exc.reasons
        finally:
            os.environ.clear(); os.environ.update(saved_env)
            sys.argv = saved_argv

    a, why = apply()
    check("a clean environment applies the profile", a is not None)
    if a:
        check("the run becomes a soak", a.mode == "soak")
        want = str(PP.load(a.profile)[0]["profile"]["objective"]["preferred_concurrency"])
        check("the level comes from the profile", a.levels == want)
        check("the topology reaches --server-args",
              a.server_args == "--prefork 16 --prefork-threads 2 --max-batch 8")
        check("the 30-minute soak length is applied", a.soak_seconds == 1800.0)
        check("gate thresholds are floats, not strings",
              isinstance(a.ttfa_pref_ms, float) and a.ttfa_pref_ms == 500.0)

    a, why = apply(env={"MYNAH_QUANT_GROUPS": "backbone:f16"})
    check("a leftover MYNAH_QUANT_GROUPS export refuses the run", a is None)
    check("and the refusal names the variable",
          why and any("MYNAH_QUANT_GROUPS" in r for r in why))

    a, why = apply(argv=["--levels", "120"])
    check("a command-line level that fights the profile refuses", a is None)
    check("and the refusal says the profile pins it",
          why and any("--levels" in r for r in why))

    a, why = apply(argv=["--server-args", "--prefork 4"])
    check("a command-line topology that fights the profile refuses", a is None)

    a, why = apply(argv=["--port", "9000"])
    check("a flag the profile does not pin is still allowed", a is not None)

    bad = [c for c in CASES if not c[0]]
    print("  %d checks, %d failed -- %s"
          % (len(CASES), len(bad), "PASSED" if not bad else "FAILED"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
