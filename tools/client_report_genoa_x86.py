#!/usr/bin/env python3
"""PocketTTS on x86 - the Genoa mini report.

    uv run --with reportlab python3 tools/client_report_genoa_x86.py OUT.pdf

WHY THIS IS A "MINI" REPORT AND SAYS SO ON ITS FACE.

The Arm report certifies C126 on thirty-minute soaks at five concurrency
levels, taken over three days. This one rests on ONE ten-minute soak at ONE
level, and two things it cannot state are stated as missing rather than
omitted: the cadence percentiles are not quotable from this host, and audio
quality was not scored at all.

The style helpers come from client_report_pocket.py rather than being copied,
so the two documents cannot drift apart typographically while claiming to be
the same family of deliverable.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.platypus import (BaseDocTemplate, Frame, PageTemplate, Paragraph,
                                Spacer, KeepTogether)

from client_report_pocket import (ACCENT, MUTE, RULE, GOOD, MARG, BAD,
                                  BODY, SMALL, H1, H2, H3, SUB, MONO, LEAD,
                                  Rule, Callout, table)

HOST = "AMD EPYC 9254 (Genoa, Zen 4), 24 cores, Ubuntu 24.04, gcc 13.3"
DATE = "22 September 2026"


def build(path):
    doc = BaseDocTemplate(path, pagesize=A4,
                          leftMargin=20*mm, rightMargin=18*mm,
                          topMargin=17*mm, bottomMargin=17*mm,
                          title="PocketTTS on x86 - Genoa mini report",
                          author="Gabriele Mastrapasqua")
    fw = doc.width
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="n")

    def decorate(canv, d):
        canv.saveState()
        canv.setFont("Helvetica", 7.2); canv.setFillColor(MUTE)
        canv.drawString(doc.leftMargin, 11*mm,
                        "PocketTTS on x86 - AMD EPYC 9254 - %s - MINI REPORT" % DATE)
        canv.drawRightString(doc.leftMargin+doc.width, 11*mm, "page %d" % d.page)
        canv.setStrokeColor(RULE); canv.setLineWidth(0.4)
        canv.line(doc.leftMargin, 13.5*mm, doc.leftMargin+doc.width, 13.5*mm)
        canv.restoreState()

    doc.addPageTemplates([PageTemplate(id="all", frames=[frame], onPage=decorate)])
    F = []
    A = F.append

    A(Paragraph("PocketTTS on x86", H1))
    A(Paragraph("Streaming capacity on a single 24-core AMD server &mdash; "
                "EPYC 9254 &middot; %s &middot; <b>mini report</b>" % DATE, SUB))
    A(Spacer(1, 4))
    A(Rule(fw, 1.2, ACCENT))
    A(Spacer(1, 8))

    A(Callout(fw, [
        ("80", ["concurrent streams", "10 minutes, every gate met"]),
        ("0", ["stalls in 33,899", "requests"]),
        ("184 ms", ["to first audio", "95th percentile*"]),
        ("88x", ["realtime aggregate", "audio seconds per second"]),
    ]))
    A(Spacer(1, 8))

    A(Paragraph(
        "One 24-core x86 server, CPU only, no GPU and no Python in the serving path, "
        "sustained <b>80 simultaneous speech streams</b> for ten minutes and passed "
        "every mandatory and preferred gate the harness applies. 33,899 requests were "
        "launched and 33,899 completed. Not one listener would have heard a gap.", LEAD))
    A(Paragraph(
        "* This is a <b>mini</b> report and the asterisk is the reason. The Arm "
        "report for this engine rests on thirty-minute soaks at five levels; this "
        "rests on one ten-minute soak at one level. And the time-to-first-audio "
        "above is marked because the harness itself declines to certify it on this "
        "host &mdash; section 5 says why, and why the zero-stall figure beside it is "
        "not affected.", BODY))

    # ------------------------------------------------------------------ 1
    A(Paragraph("1. What was measured", H2))
    A(table([
        ["Host", HOST],
        ["ISA in use", "AVX2 + FMA, AVX-512 F/BW/VL, <b>AVX-512 VNNI</b> (int8 dot), "
                       "<b>AVX512-BF16</b> (VDPBF16PS). Selected at runtime by CPUID, "
                       "no build flag"],
        ["Model", "kyutai/pocket-tts rev 4925226, English pack, 24 kHz"],
        ["Server", "12 worker processes &times; 2 threads, pinned one core-slice each; "
                   "96 admission slots"],
        ["Build", "<font face=\"Courier\" size=\"8\">make</font> defaults, "
                  "BLAS=none (the Linux default), gcc 13.3"],
        ["Load", "80 concurrent streaming clients, 10 minutes, warm-up discarded, "
                 "generator on the same host"],
        ["Box state", "quiet &mdash; the disk array finished resyncing before this run"],
    ], [28*mm, fw-28*mm], header=False))

    # ------------------------------------------------------------------ 2
    A(Paragraph("2. The gates, and every one of them", H2))
    A(Paragraph("A level is not qualified by its average. It is qualified by a list of "
                "conditions fixed before the run, and the harness prints each one with "
                "its threshold so a reader can disagree with the threshold rather than "
                "guess at it.", BODY))
    A(table([
        ["Gate", "Measured", "Threshold", "Result"],
        ["completed == launched", "33,899 == 33,899", "exact", "PASS"],
        ["STREAM_RTF p95", "0.880", "&lt; 1.00 (mandatory)", "PASS"],
        ["STREAM_RTF p95", "0.880", "&le; 0.90 (preferred)", "PASS"],
        ["stall rate @ 500 ms buffer", "0 of 33,899", "== 0", "PASS"],
        ["stall rate @ 250 ms buffer", "0 of 33,899", "== 0", "PASS"],
        ["time to first byte p95", "70.5 ms", "&le; 100 ms", "PASS"],
        ["time to first audio p95", "183.8 ms", "&le; 500 ms", "PASS"],
        ["required prebuffer p95", "13.3 ms", "&le; 500 ms", "PASS"],
        ["safe play start p95", "185.9 ms", "&le; 1000 ms", "PASS"],
    ], [52*mm, 30*mm, 42*mm, fw-124*mm]))

    # ------------------------------------------------------------------ 3
    A(Paragraph("3. It did not drift", H2))
    A(Paragraph("Ten consecutive one-minute windows. A configuration that is fast for "
                "the first minute and degrades is not a configuration, and this is the "
                "only evidence that separates the two.", BODY))
    A(table([
        ["window", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
        ["STREAM_RTF p95", "0.881", "0.874", "0.896", "0.886", "0.875",
         "0.872", "0.880", "0.879", "0.876", "0.875"],
        ["first audio p95 (ms)", "182", "185", "186", "189", "183",
         "180", "181", "183", "187", "187"],
        ["stalls", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"],
    ], [34*mm] + [ (fw-34*mm)/10.0 ]*10, size=7.4))
    A(Paragraph("Last window against best: <b>+0.0030</b> on capacity (tolerance +0.050) "
                "and <b>+10 ms</b> on prebuffer (tolerance +150 ms).", BODY))

    # ------------------------------------------------------------------ 4
    A(Paragraph("4. Where the ceiling is", H2))
    A(Paragraph("C80 is the level that met every gate. It is not the maximum, and the "
                "difference matters when sizing.", BODY))
    A(table([
        ["Level", "Duration", "Requests", "STREAM_RTF p95", "Stalls", "Verdict"],
        ["C48", "4 min", "14,980", "0.515", "0", "every gate met"],
        ["<b>C80</b>", "<b>10 min</b>", "<b>33,899</b>", "<b>0.880</b>", "<b>0</b>",
         "<b>every gate met</b>"],
        ["C96", "4 min", "12,838", "0.992", "0", "mandatory met, capacity "
                                                 "gate 0.992 &gt; 0.90"],
    ], [18*mm, 20*mm, 24*mm, 30*mm, 16*mm, fw-108*mm]))
    A(Paragraph("At C96 the server still lost no request and caused no stall, but "
                "STREAM_RTF p95 reached 0.992: it was emitting audio at very nearly the "
                "rate it is consumed, with no margin for a slow minute. That is a "
                "ceiling, not an operating point.", BODY))
    A(Paragraph("A second limit is arithmetic, not silicon: admission is "
                "<b>8 slots per worker process</b>, so a 12-worker server admits 96 "
                "requests and refuses the 97th regardless of how much CPU is idle. Every "
                "refusal count observed in testing matched "
                "<font face=\"Courier\" size=\"8\">(C &minus; slots) &times; waves</font> "
                "exactly. Sizing a fleet on cores alone will therefore mispredict it.", BODY))

    # ------------------------------------------------------------------ 5
    A(Paragraph("5. What this report does NOT establish", H2))
    A(Paragraph("Listed because a reader would otherwise reasonably assume it.", BODY))
    A(table([
        ["Not established", "Why"],
        ["Time-to-first-audio as a certified number",
         "The load generator runs on the same 24 cores as the server. 19% of its socket "
         "reads returned data that had already arrived, above the harness's own 15% "
         "refusal threshold, so every cadence percentile here is an <b>upper bound on "
         "server lateness</b> rather than the server's true emission. The true figure is "
         "at or below 184 ms; it is not measured. Moving the generator to a second host "
         "resolves this."],
        ["Stall rate is <i>not</i> affected by the above",
         "A stall is a gap large enough to empty a player's buffer. Coalesced reads make "
         "the measured gaps LARGER, never smaller, so a zero-stall result taken under "
         "coalescing remains zero when the coalescing is removed. This is the one "
         "cadence-derived number that survives."],
        ["Audio quality",
         "Not scored. No MOS, no listening panel, no objective metric. The bundle "
         "accompanying this report exists so a human can judge; nothing here claims a "
         "quality result."],
        ["Thirty-minute behaviour",
         "The longest run here is ten minutes. The Arm report qualifies at thirty, and a "
         "level that passes a short soak has failed a long one before on this project."],
        ["Other text lengths and languages",
         "One English pack, one utterance length (1.65 s mean). Longer utterances amortize "
         "fixed cost better and would likely improve the aggregate; that is an "
         "expectation, not a measurement."],
        ["Cost per stream",
         "No pricing is stated because none was measured."],
    ], [45*mm, fw-45*mm]))

    # ------------------------------------------------------------------ 6
    A(Paragraph("6. The listening bundle", H2))
    A(Paragraph("48 WAV files accompany this report, captured through the same streaming "
                "route a real client uses, <b>while 79 other streams were in flight on "
                "the same machine</b> &mdash; short, medium, conversational and long "
                "sentences, 223 seconds of audio in total, with a manifest naming every "
                "sentence and the load conditions.", BODY))
    A(Paragraph("The first attempt at this capture was discarded. The load generator had "
                "refused to start, the capture script did not check, and it produced 48 "
                "perfectly good files on an <i>idle</i> box while printing that the box "
                "was loaded. Audio rendered quiet and labelled &ldquo;under load&rdquo; is "
                "indistinguishable to a listener from the real thing, so the script now "
                "refuses to proceed unless the machine's load average confirms it. The "
                "bundle shipped here was captured at a confirmed load average of 32 on 24 "
                "cores.", BODY))

    # ------------------------------------------------------------------ 7
    A(Paragraph("7. Against the Arm measurement", H2))
    A(Paragraph("Both are this engine; neither is a benchmark of the other. Read the "
                "caveat in the last column before the ratio.", BODY))
    A(table([
        ["", "Arm &mdash; Axion c4a", "x86 &mdash; EPYC 9254", "Comparable?"],
        ["Cores", "32 Neoverse-V2", "24 Zen 4", "&mdash;"],
        ["Certified concurrency", "126", "80", "No: 30-minute soaks at five "
                                                "levels vs one 10-minute soak"],
        ["Stalls", "0 in 65,039", "0 in 33,899", "Yes"],
        ["Aggregate throughput", "157x realtime", "88x realtime", "No: different "
                                                                  "utterance mix"],
        ["Per core", "3.9 streams", "3.3 streams", "Indicative only"],
    ], [34*mm, 34*mm, 34*mm, fw-102*mm]))
    A(Paragraph("The honest summary is that x86 is <b>in the same class</b> as the Arm "
                "host per core, and that the x86 figure is the conservative one: it comes "
                "from a shorter run, at a level chosen to keep margin, on a host where the "
                "load generator competes for the same cores. What has been established is "
                "that the x86 kernels are not a fallback &mdash; AVX-512 VNNI and "
                "VDPBF16PS execute, and this is the first time this engine has been "
                "measured on a machine that can run them.", BODY))

    # ------------------------------------------------------------------ 8
    A(Paragraph("8. Reproducing this", H2))
    A(Paragraph(
        "python3 tools/serving_profile.py --mode soak --model models/pocket-en \\<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;--levels 80 --soak-seconds 600 --warmup-seconds 30 "
        "--window-seconds 60 \\<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;--server-args &quot;--prefork 12 --prefork-threads 2&quot;<br/>"
        "<br/>"
        "PREFORK_W=12 PREFORK_T=2 bash tools/capture_bundle.sh 80 9200 600",
        MONO))
    A(Spacer(1, 6))
    A(Rule(fw, 0.6))
    A(Paragraph("Measured 22 September 2026 on a rented AMD EPYC 9254. Model "
                "kyutai/pocket-tts revision 4925226, CC-BY-4.0. Runtime mynah-tts, C11, "
                "open source. Prepared by Gabriele Mastrapasqua.", SMALL))
    doc.build(F)
    print("wrote", path)


if __name__ == "__main__":
    build(sys.argv[1])
