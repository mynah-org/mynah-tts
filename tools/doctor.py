#!/usr/bin/env python3
"""doctor.py -- what is this machine, what will this binary choose on it, and
how should the server probably be run here.

Runs in under a minute, needs no model pack, and downloads nothing.

THE RULE THIS FILE EXISTS FOR
-----------------------------
Every value carries a label, and the label is part of the value:

  [MEASURED]     read from this machine, in this process, just now
  [CACHED]       read from a file this tool wrote on this host earlier
  [TRANSFERRED]  a number from ANOTHER host, carried here by a human.
                 It is printed because it is the only number we have and
                 hiding it would not make it less influential -- but it is
                 never arithmetic input, and the host it came from is named.
  [PREDICTED]    derived from a rule of thumb, +/-10-25%. A PREDICTION IS A
                 STARTING POINT FOR A MEASUREMENT, NEVER A CLAIM. The output
                 says so itself, every time, because a number in a terminal
                 outlives the caveat that was spoken next to it.
  [UNKNOWN]      nothing here supports a number, and the doctor says so
                 instead of printing a plausible one.

There is no sixth label and there is no unlabelled number.

THE DISTINCTION THAT ALREADY COST US A WRONG ANSWER
---------------------------------------------------
"How many cores does this machine have" and "how many cores may this process
use" are different questions, and in a container, under taskset, or inside a
cgroup they have different answers. A capacity number derived from the machine
while the process ran on a slice of it is wrong in a way that looks right. This
tool always reports both and shouts when they differ.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

MEASURED = "[MEASURED]"
CACHED = "[CACHED]"
TRANSFERRED = "[TRANSFERRED]"
PREDICTED = "[PREDICTED]"
UNKNOWN = "[UNKNOWN]"

PREDICTION_DISCLAIMER = (
    "A PREDICTED value is a starting point for a measurement, never a claim. "
    "Nothing below marked [PREDICTED] has been observed on this host."
)


class Report:
    """Collects labelled rows and prints them in sections."""

    def __init__(self) -> None:
        self.sections: list[tuple[str, list[tuple[str, str, str, str]]]] = []
        self.warnings: list[str] = []
        self.refusals: list[str] = []

    def section(self, title: str) -> str:
        self.sections.append((title, []))
        return title

    def add(self, key: str, value, label: str, note: str = "") -> None:
        if value is None or value == "":
            value, label = "-", UNKNOWN
        self.sections[-1][1].append((key, str(value), label, note))

    def unknown(self, key: str, why: str) -> None:
        """Nothing supports a number. Say that, do not guess one."""
        self.sections[-1][1].append((key, "-", UNKNOWN, why))

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def refuse(self, message: str) -> None:
        self.refusals.append(message)

    def render(self, out) -> None:
        for title, rows in self.sections:
            if not rows:
                continue
            print(f"\n{title}", file=out)
            print("-" * len(title), file=out)
            width = max(len(r[0]) for r in rows)
            vwidth = min(max((len(r[1]) for r in rows), default=0), 34)
            for key, value, label, note in rows:
                line = f"  {key:<{width}}  {value:<{vwidth}}  {label:<13}"
                if note:
                    line += f" {note}"
                print(line.rstrip(), file=out)

    def to_json(self) -> dict:
        return {
            "sections": [
                {
                    "title": t,
                    "rows": [
                        {"key": k, "value": v, "label": la.strip("[]"), "note": n}
                        for (k, v, la, n) in rows
                    ],
                }
                for (t, rows) in self.sections
            ],
            "warnings": self.warnings,
            "refusals": self.refusals,
        }


# ----------------------------------------------------------------- helpers


def run(cmd: list[str], timeout: float = 10.0) -> str | None:
    try:
        p = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if p.returncode != 0:
        return None
    return p.stdout.strip()


def read_text(path: str) -> str | None:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def sysctl(name: str) -> str | None:
    return run(["sysctl", "-n", name])


def human_bytes(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024 or unit == "GiB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n / 1:.0f} {unit}"
        n /= 1024.0
    return str(n)


def kib(n: int) -> str:
    if n >= 1024 * 1024:
        return f"{n / (1024 * 1024):.0f} MiB"
    if n >= 1024:
        return f"{n / 1024:.0f} KiB"
    return f"{n} B"


# ------------------------------------------------------------------- CPU


def cpu_section(rep: Report) -> dict:
    rep.section("CPU")
    facts: dict = {}
    system = platform.system()
    rep.add("os", f"{system} {platform.release()}", MEASURED)
    rep.add("arch", platform.machine(), MEASURED)

    if system == "Darwin":
        brand = sysctl("machdep.cpu.brand_string")
        rep.add("model", brand, MEASURED if brand else UNKNOWN)
        facts["model"] = brand
        # Apple Silicon reports feature availability through hw.optional.
        feats = []
        for name, flag in (
            ("hw.optional.AdvSIMD", "neon"),
            ("hw.optional.arm.FEAT_DotProd", "dotprod"),
            ("hw.optional.arm.FEAT_I8MM", "i8mm"),
            ("hw.optional.arm.FEAT_BF16", "bf16"),
            ("hw.optional.arm.FEAT_SME", "sme"),
            ("hw.optional.arm.FEAT_SVE", "sve"),
            ("hw.optional.arm.FEAT_FP16", "fp16"),
        ):
            v = sysctl(name)
            if v == "1":
                feats.append(flag)
        rep.add(
            "isa features",
            " ".join(feats) if feats else "-",
            MEASURED if feats else UNKNOWN,
            "sysctl hw.optional.*",
        )
        facts["features"] = feats

    elif system == "Linux":
        info = read_text("/proc/cpuinfo") or ""
        m = re.search(r"^model name\s*:\s*(.+)$", info, re.M)
        if not m:
            # aarch64 /proc/cpuinfo has no model name; use the part number.
            impl = re.search(r"^CPU implementer\s*:\s*(\S+)$", info, re.M)
            part = re.search(r"^CPU part\s*:\s*(\S+)$", info, re.M)
            known = {"0xd4f": "Neoverse-V2", "0xd40": "Neoverse-V1",
                     "0xd49": "Neoverse-N2", "0xd0c": "Neoverse-N1"}
            if part:
                name = known.get(part.group(1).lower())
                model = name or f"aarch64 impl={impl.group(1) if impl else '?'} part={part.group(1)}"
                rep.add("model", model, MEASURED if name else PREDICTED,
                        "/proc/cpuinfo CPU part" +
                        ("" if name else "; part number not in this tool's table"))
                facts["model"] = model
            else:
                rep.unknown("model", "/proc/cpuinfo has neither model name nor CPU part")
        else:
            rep.add("model", m.group(1), MEASURED, "/proc/cpuinfo")
            facts["model"] = m.group(1)

        flags = re.search(r"^(?:flags|Features)\s*:\s*(.+)$", info, re.M)
        if flags:
            allf = flags.group(1).split()
            interesting = [
                f for f in allf
                if f in {
                    "asimd", "asimddp", "i8mm", "bf16", "sve", "sve2", "svei8mm",
                    "svebf16", "fphp", "asimdhp", "sha3", "avx2", "fma", "f16c",
                    "avx512f", "avx512bw", "avx512vl", "avx512_vnni", "avx_vnni",
                    "amx_int8", "amx_bf16", "avx512_bf16",
                }
            ]
            rep.add("isa features", " ".join(interesting) or "-",
                    MEASURED if interesting else UNKNOWN,
                    f"/proc/cpuinfo ({len(allf)} flags total)")
            facts["features"] = interesting
        else:
            rep.unknown("isa features", "/proc/cpuinfo exposes no flags/Features line")
    else:
        rep.unknown("model", f"no probe for {system}")
        rep.unknown("isa features", f"no probe for {system}")

    return facts


# ------------------------------------------------- affinity vs the machine


def affinity_section(rep: Report) -> dict:
    """The distinction that has already cost a wrong answer.

    machine cpus  = what the hardware has
    allowed cpus  = what THIS process may actually run on

    They differ under taskset, in a cgroup, in a container with a cpuset, and
    on a shared box where someone else pinned us. A capacity or a thread count
    derived from the first while the process lives on the second is wrong, and
    it is wrong in the direction that looks generous.
    """
    rep.section("AFFINITY -- the machine vs THIS PROCESS")
    facts: dict = {}
    system = platform.system()

    machine = os.cpu_count()
    rep.add("machine cpus", machine, MEASURED if machine else UNKNOWN,
            "os.cpu_count(): what the hardware reports")
    facts["machine_cpus"] = machine

    allowed = None
    if hasattr(os, "sched_getaffinity"):
        try:
            mask = sorted(os.sched_getaffinity(0))
            allowed = len(mask)
            rep.add("allowed cpus", allowed, MEASURED,
                    "sched_getaffinity(0): what this process may use")
            rep.add("allowed mask", _compact_mask(mask), MEASURED)
            facts["allowed_cpus"] = allowed
            facts["allowed_mask"] = mask
        except OSError:
            allowed = None
    if allowed is None:
        if system == "Darwin":
            rep.unknown(
                "allowed cpus",
                "macOS has no sched_getaffinity and no CPU pinning API: the "
                "affinity question cannot be answered here, and a number "
                "copied from the machine row would be an assumption, not a "
                "measurement",
            )
        else:
            rep.unknown("allowed cpus", "sched_getaffinity is unavailable")

    # A cgroup quota is a third answer again: it caps CPU TIME, not placement,
    # so a process can be allowed 32 cpus and still only get 4 cpus' worth.
    quota = _cgroup_quota()
    if quota is None:
        rep.unknown("cgroup cpu quota",
                    "no cgroup v2 cpu.max / v1 cfs quota visible from here")
    elif quota == "max":
        rep.add("cgroup cpu quota", "unlimited", MEASURED, "cgroup cpu.max")
    else:
        rep.add("cgroup cpu quota", f"{quota:.2f} cpus", MEASURED,
                "cgroup cpu.max: a TIME cap, independent of the mask above")
        facts["cgroup_cpus"] = quota

    if machine and allowed and allowed != machine:
        rep.warn(
            f"AFFINITY DIFFERS FROM THE MACHINE: this process may use "
            f"{allowed} of {machine} cpus. Every capacity, thread-count and "
            f"per-core number below is about the {allowed}-cpu slice, NOT "
            f"about this host. Do not compare it with a number taken on the "
            f"whole machine -- that is the division this toolchain refuses."
        )
    if isinstance(quota, float) and allowed and quota < allowed:
        rep.warn(
            f"CGROUP QUOTA IS BELOW THE MASK: {quota:.2f} cpus of cpu time "
            f"across {allowed} allowed cpus. Threads will be throttled rather "
            f"than descheduled, which shows up as latency, not as idle cores."
        )
    return facts


def _compact_mask(cpus: list[int]) -> str:
    if not cpus:
        return "-"
    out, start, prev = [], cpus[0], cpus[0]
    for c in cpus[1:] + [None]:
        if c is not None and c == prev + 1:
            prev = c
            continue
        out.append(str(start) if start == prev else f"{start}-{prev}")
        if c is not None:
            start = prev = c
    return ",".join(out)


def _cgroup_quota():
    v2 = read_text("/sys/fs/cgroup/cpu.max")
    if v2:
        parts = v2.split()
        if parts[0] == "max":
            return "max"
        try:
            return int(parts[0]) / int(parts[1])
        except (ValueError, IndexError, ZeroDivisionError):
            return None
    q = read_text("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
    p = read_text("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    if q and p:
        try:
            qi, pi = int(q), int(p)
            return "max" if qi < 0 else qi / pi
        except (ValueError, ZeroDivisionError):
            return None
    return None


# ------------------------------------------------------ cache and NUMA


def topology_section(rep: Report) -> dict:
    rep.section("CACHE AND NUMA")
    facts: dict = {}
    system = platform.system()

    if system == "Darwin":
        for key, name in (
            ("L1 data", "hw.perflevel0.l1dcachesize"),
            ("L2", "hw.perflevel0.l2cachesize"),
            ("L3", "hw.l3cachesize"),
        ):
            v = sysctl(name)
            if v and v.isdigit() and int(v) > 0:
                rep.add(key, kib(int(v)), MEASURED, f"sysctl {name}")
            else:
                rep.unknown(key, f"sysctl {name} reports nothing")
        p = sysctl("hw.perflevel0.logicalcpu")
        e = sysctl("hw.perflevel1.logicalcpu")
        if p:
            rep.add("core classes", f"{p}P" + (f" + {e}E" if e else ""), MEASURED,
                    "asymmetric: the pool defaults to P cores only")
            facts["p_cores"] = int(p)
        rep.add("NUMA nodes", 1, MEASURED, "Apple Silicon is single-node by construction")
        facts["numa_nodes"] = 1
        return facts

    if system != "Linux":
        rep.unknown("cache", f"no probe for {system}")
        rep.unknown("NUMA nodes", f"no probe for {system}")
        return facts

    base = Path("/sys/devices/system/cpu/cpu0/cache")
    seen = False
    if base.is_dir():
        for idx in sorted(base.glob("index*")):
            level = read_text(str(idx / "level"))
            ctype = read_text(str(idx / "type"))
            size = read_text(str(idx / "size"))
            shared = read_text(str(idx / "shared_cpu_list"))
            if not (level and size):
                continue
            seen = True
            label = f"L{level}" + ("" if ctype in (None, "Unified") else f" {ctype.lower()}")
            rep.add(label, size, MEASURED, f"shared by cpus {shared}" if shared else "")
    if not seen:
        rep.unknown("cache", "/sys/devices/system/cpu/cpu0/cache is absent "
                             "(common in a container without a host sysfs mount)")

    nodes = sorted(Path("/sys/devices/system/node").glob("node[0-9]*")) \
        if Path("/sys/devices/system/node").is_dir() else []
    if nodes:
        rep.add("NUMA nodes", len(nodes), MEASURED, "/sys/devices/system/node")
        facts["numa_nodes"] = len(nodes)
        for n in nodes:
            cpus = read_text(str(n / "cpulist"))
            if cpus:
                rep.add(f"  {n.name} cpus", cpus, MEASURED)
        if len(nodes) > 1:
            rep.warn(
                f"{len(nodes)} NUMA nodes. A thread pool spanning nodes pays "
                f"remote-memory latency on the weight stream, which is the "
                f"bandwidth-bound half of this engine. Pin per node and run "
                f"one process per node before reading any scaling number."
            )
    else:
        rep.unknown("NUMA nodes", "/sys/devices/system/node is absent")
    return facts


# ------------------------------------------------------ the dispatch table


def dispatch_section(rep: Report, binary: str | None) -> dict:
    """The resolved dispatch table, from the binary itself.

    This tool does NOT re-derive any dispatch decision. It runs the binary and
    prints what the binary says, because a second implementation of the same
    guess is exactly how a report and a README came to agree and both be wrong
    (src/dispatch.h, commit 724d677).
    """
    rep.section("DISPATCH -- what this binary resolves on this host")
    facts: dict = {}
    if binary is None or not os.access(binary, os.X_OK):
        rep.unknown(
            "dispatch table",
            f"no runnable binary at {binary or '(none given)'}; build it with "
            f"`make` and re-run, or pass --binary. This tool will not guess a "
            f"dispatch decision the binary can answer for itself",
        )
        rep.refuse(
            "the dispatch table could not be collected, so no statement about "
            "which kernels this host will run is supported"
        )
        return facts

    raw = run([binary, "--dispatch-map", "--json"], timeout=45.0)
    if not raw:
        rep.unknown("dispatch table", f"{binary} --dispatch-map --json failed")
        rep.refuse("the dispatch table could not be collected")
        return facts
    try:
        doc = json.loads(raw)
    except json.JSONDecodeError as exc:
        rep.unknown("dispatch table", f"unparseable JSON from --dispatch-map: {exc}")
        rep.refuse("the dispatch table could not be parsed")
        return facts

    facts["dispatch"] = doc
    rep.add("build", doc.get("build"), MEASURED, "git revision the binary was built from")
    rep.add("simd profile", doc.get("simd_profile"), MEASURED)
    rep.add("isa class", doc.get("isa_class"), MEASURED)
    rep.add("pool threads", doc.get("threads"), MEASURED, "mynah_num_threads()")

    counts = doc.get("counts", {})
    rep.add(
        "row provenance",
        f"{counts.get('predicate', 0)} predicate, {counts.get('runtime', 0)} runtime, "
        f"{counts.get('gate', 0)} gate, {counts.get('unknown', 0)} UNKNOWN",
        MEASURED,
        "UNKNOWN rows are predicates nobody exports yet, not failures",
    )
    if counts.get("unknown", 0):
        rep.warn(f"{counts['unknown']} dispatch rows resolved UNKNOWN.")

    if doc.get("gate_drift"):
        rep.warn(f"GATE DRIFT: {doc.get('gate_drift_detail')}")
    if not doc.get("isa_guard_ok", True):
        rep.refuse(f"ISA GUARD FAILED: {doc.get('isa_guard_detail')}")

    # The rows that decide what the hot loop runs.
    interesting = [
        "quant.requested", "quant.int8_kernel", "quant.f16", "quant.groups",
        "quant.argmax_mt", "cpu.matvec_policy", "sgemm.provider", "sgemm.kernel",
        "pool.threads", "pool.cpu_topology", "pool.decoder_lane", "pool.spin",
        "backend.cpu", "backend.metal", "backend.cuda",
        "isa.arm.dotprod", "isa.arm.i8mm", "isa.x86.avx512vnni", "isa.x86.avxvnni",
        "codec.seanet_blas",
    ]
    by_id = {r["id"]: r for r in doc.get("features", [])}
    for rid in interesting:
        row = by_id.get(rid)
        if row is None:
            continue
        # A row's provenance IS its label.
        #
        #   predicate/runtime -> the binary answered by CALLING something here
        #   gate              -> a compile-time fact about the binary. Still
        #                        MEASURED: the kernel really is or is not in
        #                        this binary. It is NOT a prediction, and it is
        #                        also NOT evidence the kernel ever ran -- that
        #                        question belongs to the census.
        #   a reason that says TRANSFERRED carries a number from another host,
        #   and that beats the row's own provenance for labelling purposes.
        src = row.get("source", "")
        reason = row.get("reason") or ""
        if "TRANSFERRED" in reason:
            label = TRANSFERRED
            note = "the reason names another host; re-measure before trusting it here"
        elif src in ("predicate", "runtime"):
            label, note = MEASURED, ""
        elif src == "gate":
            label = MEASURED
            note = ("compile gate: a fact about the binary, not evidence the "
                    "kernel executed -- run the census for that")
        else:
            label, note = UNKNOWN, "no predicate exports this decision"
        if row.get("env_value") and row["env_value"] != "unset":
            note = (note + "; " if note else "") + f"{row['env']}={row['env_value']}"
        rep.add(rid, row.get("resolved"), label, note)

    # A feature that is present and idle is a finding, not a blank.
    idle = doc.get("idle_hardware")
    if idle:
        rep.add("idle hardware", idle, MEASURED,
                "present on this CPU, no kernel in this build uses it")
    return facts


# ------------------------------------------------------ suggested topology


def suggestion_section(rep: Report, cpu: dict, aff: dict, topo: dict,
                       disp: dict) -> None:
    """How should the server probably be run here.

    Every number in this section is [PREDICTED] unless it came from the binary.
    The word "probably" in the heading is load-bearing.
    """
    rep.section("SUGGESTED TOPOLOGY -- a starting point, not a configuration")

    usable = aff.get("allowed_cpus") or aff.get("machine_cpus")
    if usable is None:
        rep.unknown("everything below", "no usable cpu count could be determined")
        return
    scope = "allowed mask" if aff.get("allowed_cpus") else "whole machine"
    quota = aff.get("cgroup_cpus")
    if isinstance(quota, float) and quota < usable:
        usable = int(quota)
        scope = "cgroup quota"

    rep.add("sizing scope", f"{usable} cpus ({scope})", MEASURED,
            "every suggestion below is derived from THIS number")

    # Threads per worker. The engine is bandwidth-bound in the codec and
    # latency-bound in the AR step; the reference's shape is a small number of
    # threads per worker and several workers, not one wide pool.
    doc = disp.get("dispatch") or {}
    resolved_threads = doc.get("threads")
    if resolved_threads:
        rep.add("pool threads now", resolved_threads, MEASURED,
                "what the binary would use in this environment today")

    per_worker = 4 if usable >= 8 else max(1, usable // 2)
    workers = max(1, usable // per_worker)
    rep.add("threads / worker", per_worker, PREDICTED,
            "rule of thumb: the AR step stops scaling past ~4 threads on a "
            "1024-wide backbone; MEASURE with a quantum sweep")
    rep.add("workers", workers, PREDICTED,
            f"{usable} cpus / {per_worker} threads, no oversubscription")

    if topo.get("numa_nodes", 1) > 1:
        rep.add("processes", topo["numa_nodes"], PREDICTED,
                "one per NUMA node, each pinned to its node's cpulist")
    else:
        rep.add("processes", 1, PREDICTED, "single NUMA node")

    # Concurrency. This is the number people most want and the one we least
    # have: it is a measured property of the model, the text length and the
    # host, and the reference showed a wave screen passing at C16 while a
    # 30-minute soak failed at the same C.
    rep.unknown(
        "safe concurrency (C)",
        "NOT PREDICTED HERE. .work/serving-design.md 10: a wave screen passed "
        "C16 at STREAM_RTF 0.919-0.974 and a 30-minute soak at the same C "
        "failed with p95 1.004, 596 rejects and 111 broken pipes. Only a SOAK "
        "promotes, so this tool refuses to print a C it did not soak. Run "
        "`make serving-wave` to screen, then `make serving-soak` to qualify",
    )

    # Anything carried from another host is named as such and never multiplied.
    carried = [r for r in doc.get("features", [])
               if "TRANSFERRED" in (r.get("reason") or "")]
    for r in carried:
        rep.add(f"{r['id']} (carried)", r.get("resolved"), TRANSFERRED,
                "this default came from a measurement on ANOTHER host; "
                "src/dispatch.c says so itself. It is not arithmetic input here")

    rep.add("first measurement to take", "make bench + make serving-wave", MEASURED,
            "in that order; the bench pins RTF, the wave screens concurrency")


# ----------------------------------------------------------------- main


def find_binary(explicit: str | None) -> str | None:
    if explicit:
        return explicit if os.access(explicit, os.X_OK) else None
    here = Path(__file__).resolve().parent.parent
    for candidate in (here / "build/cpu/mynah-tts", here / "build/metal/mynah-tts",
                      here / "build/cuda/mynah-tts"):
        if os.access(candidate, os.X_OK):
            return str(candidate)
    found = shutil.which("mynah-tts")
    return found


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--binary", help="path to mynah-tts (default: build/cpu/mynah-tts)")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    ap.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero if the doctor had to refuse (see REFUSALS)",
    )
    args = ap.parse_args()

    t0 = time.time()
    rep = Report()
    binary = find_binary(args.binary)

    cpu = cpu_section(rep)
    aff = affinity_section(rep)
    topo = topology_section(rep)
    disp = dispatch_section(rep, binary)
    suggestion_section(rep, cpu, aff, topo, disp)

    elapsed = time.time() - t0

    if args.json:
        doc = rep.to_json()
        doc["elapsed_seconds"] = round(elapsed, 3)
        doc["binary"] = binary
        doc["prediction_disclaimer"] = PREDICTION_DISCLAIMER
        json.dump(doc, sys.stdout, indent=2)
        print()
    else:
        host = platform.node()
        print(f"[DOCTOR] mynah-tts, host={host}, binary={binary or '(none)'}")
        print(f"  Labels: {MEASURED} here, now  {CACHED} from a file this host "
              f"wrote  {TRANSFERRED} from ANOTHER host  {PREDICTED} rule of "
              f"thumb +/-10-25%  {UNKNOWN} nothing supports a number")
        print(f"  {PREDICTION_DISCLAIMER}")
        rep.render(sys.stdout)

        if rep.warnings:
            print("\nWARNINGS")
            print("--------")
            for w in rep.warnings:
                print(f"  ! {w}")
        print("\nREFUSALS")
        print("--------")
        if rep.refusals:
            for r in rep.refusals:
                print(f"  X {r}")
        else:
            print("  none -- every value above is either measured or labelled "
                  "as not measured.")
        print(f"\n  collected in {elapsed:.1f}s, no model pack, nothing downloaded.")

    return 1 if (args.strict and rep.refusals) else 0


if __name__ == "__main__":
    sys.exit(main())
