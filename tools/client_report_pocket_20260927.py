#!/usr/bin/env python3
"""PocketTTS on Arm, small (6L) and large (24L) - the client-facing PDF of the
25-27 September 2026 campaign. Generated with reportlab.

Reuses the drawing helpers of tools/client_report_pocket.py (the 21 September
report). Every number below is copied from a soak log of the campaign; the
logs are benchmark dumps under reports/ (gitignored), the generator is source.

    uv run --with reportlab python3 tools/client_report_pocket_20260927.py OUT.pdf
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reportlab.lib import colors                                   # noqa: E402
from reportlab.lib.pagesizes import A4                             # noqa: E402
from reportlab.lib.units import mm                                 # noqa: E402
from reportlab.platypus import (BaseDocTemplate, Frame, PageTemplate,  # noqa: E402
                                Paragraph, Spacer, Flowable, KeepTogether)

from client_report_pocket import (INK, MUTE, RULE, GOOD, MARG, BAD, BAND,  # noqa: E402
                                  ACCENT, BODY, SMALL, H1, H2, H3, SUB, MONO,
                                  table, Rule, Callout, LevelChart)

# ------------------------------------------------------------------ data
# 30-minute soaks of 27 September, shipped default, nothing exported, 16x2.
# (C, slots, requests, TTFB p95, TTFA p95, TTFA p50, RTF p95, RTF p50,
#  prebuffer p95, max_gap p95, stalls@250, stalls@500, audio-s/s, drift, verdict)
S6_SOAKS = [
    (152, 12, 77088, 78.5, 175, 129, 0.816, 0.773, 7, 158, 0, 0, 191.4, 0.0000, "GOOD"),
    (160, 12, 77555, 80.0, 178, 134, 0.836, 0.794, 10, 165, 0, 0, 192.5, 0.0129, "GOOD"),
    (164, 12, 77819, 81.5, 179, 136, 0.881, 0.799, 13, 169, 0, 0, 193.3, 0.0000, "GOOD"),
    (168, 12, 78235, 82.8, 182, 138, 0.891, 0.814, 15, 172, 1, 0, 194.3, 0.0001, "MARGINAL"),
]
L24_SOAKS = [
    (88, 8, 41458, 79.0, 278, 187, 0.756, 0.681, 11, 275, 0, 0, 121.4, 0.0024, "GOOD"),
    (96, 8, 42310, 82.7, 292, 202, 0.776, 0.726, 15, 298, 1, 0, 123.8, 0.0016, "MARGINAL"),
]
KEYS = ("level", "mb", "requests", "ttfb", "ttfa", "ttfa50", "rtf", "rtf50", "prebuf", "gap",
        "s250", "s500", "tput", "drift", "verdict")


def certified(soaks):
    """The highest level with every gate met and zero stalls: the bar is zero."""
    ok = [r for r in soaks if r[14] == "GOOD" and r[10] == 0 and r[11] == 0]
    return dict(zip(KEYS, max(ok)))


S6 = certified(S6_SOAKS)
L24 = certified(L24_SOAKS)
# 3-minute screens of 27 September, 6L, shipped default (segmentation on).
S6_SCREENS = [
    # C, mb, TTFA p95, TTFB p95, RTF p95, s250, s500, audio-s/s, verdict
    (120, 8, 142.6, 63.8, 0.647, 0, 0, 173.3, "GOOD"),
    (136, 12, 154.5, 69.1, 0.743, 0, 0, 174.0, "GOOD"),
    (152, 12, 173.8, 77.9, 0.809, 0, 0, 174.1, "GOOD"),
    (168, 12, 182.5, 82.9, 0.887, 0, 0, 173.0, "GOOD"),
    (176, 12, 193.7, 87.1, 0.906, 0, 0, 176.3, "MARGINAL"),
]


def need(d, name):
    missing = [k for k, v in d.items() if v is None]
    if missing:
        raise SystemExit("%s: fill in %s" % (name, ", ".join(missing)))


def n(v):
    return "{:,}".format(v)


class TtfaLadder(Flowable):
    """TTFA p95 per level for both models, against the 500 ms target."""
    def __init__(self, width, series, lo=0.0, hi=560.0):
        self.width, self.series, self.lo, self.hi = width, series, lo, hi
        self.height = 58*mm

    def draw(self):
        c = self.canv
        L, R, B = 30, 70, 24
        pw = self.width - L - R
        ph = self.height - B - 18
        lo, hi = self.lo, self.hi
        xs = [p[0] for _, _, pts in self.series for p in pts]
        xlo, xhi = min(xs) - 6, max(xs) + 6

        def xp(v): return L + pw * (v - xlo) / (xhi - xlo)
        def yp(v): return B + ph * (v - lo) / (hi - lo)
        c.setFont("Helvetica-Bold", 8.5); c.setFillColor(INK)
        c.drawString(0, self.height - 8,
                     "Time to first audio, 95th percentile, by concurrency (27 September build)")
        c.setFont("Helvetica", 7); c.setFillColor(MUTE)
        for v in range(0, int(hi) + 1, 100):
            c.setStrokeColor(RULE); c.setLineWidth(0.3); c.line(L, yp(v), L + pw, yp(v))
            c.drawRightString(L - 4, yp(v) - 2.5, "%d" % v)
        c.saveState(); c.translate(8, B + ph / 2); c.rotate(90)
        c.setFont("Helvetica", 7.2); c.drawCentredString(0, 0, "TTFA p95 (ms)")
        c.restoreState()
        c.setStrokeColor(BAD); c.setLineWidth(1); c.setDash(3, 2)
        c.line(L, yp(500), L + pw, yp(500)); c.setDash()
        c.setFillColor(BAD); c.setFont("Helvetica-Bold", 6.8)
        c.drawString(L + pw + 4, yp(500) - 2.5, "500 ms target")
        for label, col, pts in self.series:
            pts = sorted(pts)
            c.setStrokeColor(col); c.setLineWidth(1)
            for i in range(1, len(pts)):
                c.line(xp(pts[i-1][0]), yp(pts[i-1][1]), xp(pts[i][0]), yp(pts[i][1]))
            for cc, v in pts:
                c.setFillColor(colors.white); c.circle(xp(cc), yp(v), 3.6, stroke=0, fill=1)
                c.setFillColor(col); c.circle(xp(cc), yp(v), 2.6, stroke=0, fill=1)
                c.setFillColor(INK); c.setFont("Helvetica", 6.4)
                c.drawCentredString(xp(cc), yp(v) + 5, "%.0f" % v)
            c.setFillColor(col); c.setFont("Helvetica-Bold", 7.2)
            c.drawString(L + pw + 4, yp(pts[-1][1]) - 2.5, label)
        c.setFont("Helvetica", 7); c.setFillColor(MUTE)
        for cc in sorted(set(xs)):
            c.drawCentredString(xp(cc), B - 10, "C%d" % cc)
        c.setStrokeColor(MUTE); c.setLineWidth(0.6); c.line(L, B, L + pw, B)


def build(path):
    doc = BaseDocTemplate(path, pagesize=A4,
                          leftMargin=20*mm, rightMargin=18*mm,
                          topMargin=17*mm, bottomMargin=17*mm,
                          title="PocketTTS on Arm - small and large, streaming capacity report",
                          author="Gabriele Mastrapasqua")
    fw = doc.width
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="n")

    def decorate(canv, d):
        canv.saveState()
        canv.setFont("Helvetica", 7.2); canv.setFillColor(MUTE)
        canv.drawString(doc.leftMargin, 11*mm,
                        "PocketTTS on Arm, small and large - GCP Axion c4a - 27 September 2026")
        canv.drawRightString(doc.leftMargin + doc.width, 11*mm, "page %d" % d.page)
        canv.setStrokeColor(RULE); canv.setLineWidth(0.4)
        canv.line(doc.leftMargin, 13.5*mm, doc.leftMargin + doc.width, 13.5*mm)
        canv.restoreState()

    doc.addPageTemplates([PageTemplate(id="all", frames=[frame], onPage=decorate)])
    F = []
    A = F.append
    s6, l24 = S6, L24
    gain6 = 100.0 * (s6["level"] - 126) / 126.0

    # ---------------------------------------------------------------- title
    A(Paragraph("PocketTTS on Arm: small and large", H1))
    A(Paragraph("Streaming capacity of both English PocketTTS models on a single 32-core Arm "
                "server &mdash; GCP Axion c4a &middot; 25-27 September 2026", SUB))
    A(Spacer(1, 4)); A(Rule(fw, 1.2, ACCENT)); A(Spacer(1, 8))
    A(Callout(fw, [
        ("%d" % s6["level"], ["small model (6 layers):", "streams certified, 30 min"]),
        ("%d" % l24["level"], ["large model (24 layers):", "streams certified, 30 min"]),
        ("0", ["stalls in %s requests" % n(s6["requests"] + l24["requests"]),
               "across both certifications"]),
        ("%.0f / %.0f ms" % (s6["ttfa"], l24["ttfa"]),
         ["first audio, 95th percentile", "small / large"]),
    ]))
    A(Spacer(1, 8))
    A(Paragraph(
        "One 32-core Arm virtual machine, CPU only, no GPU and no Python in the serving path, "
        "now sustains <b>%d simultaneous speech streams on the small PocketTTS model</b> and "
        "<b>%d on the large one</b>, each for thirty minutes without a single listener hearing "
        "a gap. Six days ago the small model was certified at 126 streams and the large one "
        "had not been served at all. Everything described here is now the <b>shipped default</b>: "
        "the numbers below were measured with nothing tuned by hand for the run." %
        (s6["level"], l24["level"]), BODY))
    A(Spacer(1, 4))
    A(Paragraph("Where it started and where it is now", H3))
    A(table([
        ["", "Small, 21 Sep", "Small, 27 Sep", "Change", "Large, 25 Sep", "Large, 27 Sep"],
        ["Certified streams (30 min)", "126", "%d" % s6["level"], "+%.0f%%" % gain6,
         "none (~48 screened)", "%d" % l24["level"]],
        ["First audio, 95th pct", "330 ms", "%.0f ms" % s6["ttfa"],
         "%.0f%% faster" % (100.0 * (330 - s6["ttfa"]) / 330),
         "743 ms at C48", "%.0f ms" % l24["ttfa"]],
        ["Speech per second", "157x", "%.0fx" % s6["tput"],
         "+%.0f%%" % (100.0 * (s6["tput"] - 157) / 157), "~67x", "%.0fx" % l24["tput"]],
        ["Stalls at the certified level", "0", "%d" % s6["s250"], "\u2014", "\u2014",
         "%d" % l24["s250"]],
    ], [44*mm, 23*mm, 23*mm, 20*mm, 28*mm, fw-138*mm],
        align={1: "RIGHT", 2: "RIGHT", 3: "RIGHT", 4: "RIGHT", 5: "RIGHT"}, size=8.2))
    A(Spacer(1, 4))
    A(Paragraph("Same virtual machine, same model files, same test corpus as the 21 September "
                "report. Nothing comes from better hardware or from lowering audio quality: the "
                "audio of every kernel change is byte-identical to what it replaced, and the one "
                "change that alters the audio (text segmentation, section 5) follows what the "
                "model's authors do in their own reference implementation.", SMALL))
    A(Spacer(1, 8))

    # ---------------------------------------------------------------- 1
    A(Paragraph("1. What is being measured", H2))
    A(Paragraph("<b>Two models</b>, both Kyutai PocketTTS English packs from the same public "
                "repository and revision, used unmodified:", BODY))
    A(table([
        ["", "Small", "Large"],
        ["Pack", "english (default)", "english_2026-04_24l"],
        ["Parameters", "~109.5 M", "~336 M"],
        ["Backbone", "6 transformer layers", "24 transformer layers"],
        ["Shared parts", "flow-matching head, Mimi/SEANet causal decoder, 24 kHz, one 80 ms "
                         "frame per step", "identical"],
        ["Precision (shipped default)", "int8 codec, bfloat16 backbone, f16 flow head",
         "int8 codec, <b>int8 backbone</b>, f16 flow head"],
    ], [38*mm, (fw-38*mm)/2, (fw-38*mm)/2], size=8.4))
    A(Spacer(1, 4))
    A(table([
        ["Model source", "https://huggingface.co/kyutai/pocket-tts, revision 4925226 (CC-BY-4.0)"],
        ["Runtime", "mynah-tts, open-source C11; memory-mapped weights, no allocation in the "
                    "decode loop, its own matrix kernels (BLAS=none)"],
        ["Host", "GCP Axion c4a-highcpu-32: 32 x Arm Neoverse-V2, 62 GiB, Ubuntu, gcc 15.2"],
        ["Server", "16 worker processes x 2 threads; %d request slots each for the small model, "
                   "8 for the large" % s6["mb"]],
    ], [30*mm, fw-30*mm], header=False))
    A(Spacer(1, 3))
    A(Paragraph("The large model has three times the parameters and four times the backbone "
                "depth. On a quiet machine that costs little (one request: realtime factor 0.41 "
                "against 0.385); under a hundred concurrent requests it costs a lot, and that "
                "difference is what the first half of this campaign was about.", BODY))

    # ---------------------------------------------------------------- 2
    A(Paragraph("2. What &ldquo;qualified&rdquo; means here", H2))
    A(Paragraph("Unchanged from the 21 September report, and applied identically to both "
                "models. Every level is judged by a simulated player, not by throughput:", BODY))
    A(table([
        ["Gate", "Threshold", "What it protects"],
        ["Stall rate", "0 requests, with a 250 ms client buffer (and 500 ms)",
         "Nobody hears a gap. Zero is the requirement"],
        ["Time to first audio", "500 ms at the 95th percentile", "The pause before the voice"],
        ["Time to first byte", "100 ms at the 95th percentile", "The server answers at once"],
        ["Streaming realtime factor", "below 0.90 at the 95th percentile",
         "Each stream stays ahead of its playback"],
        ["Completion", "every request served", "No silent failures"],
        ["Drift", "no upward trend across ten 3-minute windows", "Not slowly falling behind"],
    ], [38*mm, 62*mm, fw-100*mm], size=8.4))
    A(Spacer(1, 3))
    A(Paragraph("Load is sustained, not burst, and <b>only a thirty-minute run certifies a "
                "level</b>; the 3-minute runs in this report are screens that choose which level "
                "gets a thirty-minute run. The corpus is the same fixed bank of 277 English "
                "sentences, 1 to 19 seconds of speech, in five length classes.", BODY))

    # ---------------------------------------------------------------- 3
    A(Paragraph("3. Results: the small model", H2))
    screens = S6_SCREENS
    soaks6 = sorted(S6_SOAKS)
    bars = [("C126", 30, "GOOD", "21 Sep cert.")]
    soaked = set(r[0] for r in S6_SOAKS)
    for (cc, mb, ttfa, ttfb, rtf, a, b, tp, v) in screens:
        if cc not in soaked:
            bars.append(("C%d" % cc, 3, v, "screen"))
    for r in soaks6:
        tag = "CERTIFIED" if r[0] == s6["level"] else "%d stall / %s" % (r[10], n(r[2])) \
            if r[10] else "30 min"
        bars.append(("C%d" % r[0], 30, r[14], tag))
    A(LevelChart(fw, bars))
    A(Spacer(1, 4))
    rows = [["Level", "Slots", "Soak", "Requests", "First audio p95", "Header p95",
             "RTF p95", "Stalls 250/500", "Speech/s", "Verdict"]]
    rows.append(["C126", "8", "30 min", "65,039", "329.5 ms", "79.1 ms", "\u2014", "0/0",
                 "157x", "GOOD (21 Sep)"])
    rows.append(["C120", "8", "3 min", "7,766", "246.2 ms", "65.3 ms", "0.668", "0/0", "170x",
                 "GOOD, segmentation off"])
    for (cc, mb, ttfa, ttfb, rtf, a, b, tp, v) in screens:
        rows.append(["C%d" % cc, "%d" % mb, "3 min", "", "%.1f ms" % ttfa, "%.1f ms" % ttfb,
                     "%.3f" % rtf, "%d/%d" % (a, b), "%.0fx" % tp, v])
    for r in soaks6:
        cert = r[0] == s6["level"]
        B = (lambda x: "<b>%s</b>" % x) if cert else (lambda x: x)
        rows.append([B("C%d" % r[0]), "%d" % r[1], B("30 min"), n(r[2]), B("%.0f ms" % r[4]),
                     "%.1f ms" % r[3], "%.3f" % r[6], B("%d/%d" % (r[10], r[11])),
                     "%.0fx" % r[12], B("GOOD (certified)") if cert else r[14]])
    A(table(rows, [13*mm, 12*mm, 14*mm, 18*mm, 19*mm, 16*mm, 13*mm, 17*mm, 18*mm,
                   fw-140*mm], align={3: "RIGHT", 4: "RIGHT", 5: "RIGHT", 6: "RIGHT",
                                      8: "RIGHT"}, size=7.8))
    A(Spacer(1, 3))
    A(Paragraph("Two things moved the small model. <b>Faster arithmetic</b> shared with the "
                "large model (section 5): at the same 120 streams and the same run length, "
                "throughput 139x &rarr; 171x and the realtime factor 0.81 &rarr; 0.67, with "
                "byte-identical audio. <b>Text segmentation</b> then halved first audio at the "
                "same load: 246 ms without it, 143 ms with it, same build, back to back. The old "
                "ceiling at 128 streams was never speed: 16 workers x 8 slots is 128 places, so "
                "12 slots per worker remove it.", BODY))
    A(Paragraph("Above the certified level the processor, not the queue, is the limit: the "
                "3-minute screens hold a flat ~173x from C120 upward (the 30-minute runs read "
                "higher, ~191-194x, because warm-up weighs less; compare like with like), and "
                "what a higher level buys is more simultaneous listeners at a slower but still "
                "realtime per-stream pace &mdash; exactly what the realtime-factor gate "
                "measures. At 176 streams it crosses 0.90.", BODY))

    # ---------------------------------------------------------------- 4
    A(Paragraph("4. Results: the large model", H2))
    A(LevelChart(fw, [
        ("C48", 10, "NOT STREAMABLE", "25 Sep, before"),
        ("C64", 30, "MARGINAL", "+segmentation"),
        ("C80", 30, "MARGINAL", "+wide int8 kernel"),
        ("C96", 30, "GOOD", "+pooled attn."),
        ("C112", 3, "MARGINAL", "screen"),
        ("C120", 3, "NOT STREAMABLE", "screen"),
        ("C96", 30, "MARGINAL", "default, 1 stall"),
        ("C%d" % l24["level"], 30, "GOOD", "CERTIFIED"),
    ]))
    A(Spacer(1, 4))
    rows = [
        ["Level", "Build", "Soak", "Requests", "First audio p95", "RTF p95",
         "Stalls 250/500", "Speech/s", "Verdict"],
        ["C48", "25 Sep", "10 min", "7,779", "670 ms", "0.746", "133/18", "64x",
         "NOT STREAMABLE"],
        ["C64", "+ segmentation", "30 min", "28,328", "341 ms", "0.780", "20/0", "83x",
         "MARGINAL"],
        ["C80", "+ wide kernel", "30 min", "35,175", "312 ms", "0.771", "1/0", "103x",
         "MARGINAL"],
        ["C96", "+ pooled attention", "30 min", "43,113", "282 ms", "0.763", "0/0", "126x",
         "GOOD"],
        ["C104", "same", "3 min", "4,321", "310 ms", "0.857", "0-1/0", "111x", "GOOD/MARGINAL"],
        ["C112", "same", "3 min", "4,364", "321 ms", "0.871", "0-3/0", "113x", "GOOD/MARGINAL"],
        ["C120", "same", "3 min", "4,433", "344 ms", "0.951", "7/1", "114x", "NOT STREAMABLE"],
    ]
    for r in sorted(L24_SOAKS, key=lambda r: -r[0]):
        cert = r[0] == l24["level"]
        B = (lambda x: "<b>%s</b>" % x) if cert else (lambda x: x)
        rows.append([B("C%d" % r[0]), B("27 Sep default"), B("30 min"), n(r[2]),
                     B("%.0f ms" % r[4]), "%.3f" % r[6], B("%d/%d" % (r[10], r[11])),
                     "%.0fx" % r[12], B("GOOD (certified)") if cert else r[14]])
    A(table(rows, [13*mm, 25*mm, 14*mm, 18*mm, 20*mm, 13*mm, 18*mm, 18*mm, fw-139*mm],
            align={3: "RIGHT", 4: "RIGHT", 5: "RIGHT", 7: "RIGHT"}, size=7.8))
    A(Spacer(1, 3))
    A(Paragraph("The large model went from not streamable at 48 streams to a clean 96 in two "
                "days, one measured change at a time; each thirty-minute row is a run of the "
                "build that added it. On 27 September the same configuration became the "
                "shipped default &mdash; nothing exported, chosen by the runtime from the model "
                "itself, audio byte-identical &mdash; and was measured again. At 96 the repeat "
                "had <b>one</b> brief interruption in 42,310 requests (the first run had none in "
                "43,113): 96 is a rare-event edge, roughly one in 40,000-80,000. Our bar is "
                "zero, so the certified figure is <b>%d</b>, one step lower, clean over "
                "%s requests. 120 is the ceiling: there the processor is saturated at ~114x."
                % (l24["level"], n(l24["requests"])), BODY))
    A(Spacer(1, 4))
    A(TtfaLadder(fw, [
        ("small", ACCENT, [(cc, t) for (cc, mb, t, *_r) in screens]),
        ("large", MARG, [(96, 289), (104, 304), (112, 326), (120, 344)]),
    ]))
    A(Paragraph("Both models stay far inside the 500 ms first-audio target at every level "
                "measured (3-minute screens of the final build).", SMALL))

    # ---------------------------------------------------------------- 5
    A(Paragraph("5. What changed, and what each change was worth", H2))
    A(Paragraph("Six changes, all measured on this machine against a control run back to back, "
                "all now the default. Four of them change no audio at all: the output is "
                "byte-for-byte what it was before, checked by hash.", BODY))
    A(table([
        ["Change", "What it is", "Measured effect", "Audio"],
        ["Text segmentation",
         "Long text is split at sentence boundaries into chunks of at most 50 tokens (24 for "
         "the first) and each chunk is generated in turn, as the model's reference "
         "implementation does. Only the first chunk's reading sits in front of the first audio.",
         "Large, C48: first audio 743 &rarr; 275 ms, stalls 9 &rarr; 0, +10% throughput. "
         "Small, C120: 246 &rarr; 143 ms.",
         "Changes long texts, as upstream does; ASR-checked (section 7)"],
        ["Wide int8 kernel",
         "The Arm matrix-multiply instruction now serves eight requests per pass over the "
         "weights instead of two.",
         "Large: C64 &rarr; C80 certified; backbone step at 8 requests 50 &rarr; 36 ms",
         "Byte-identical"],
        ["Pooled attention",
         "Attention was computed on one of each worker's two threads while the other waited; "
         "it is now split across both.",
         "Large: C80 &rarr; C96 certified, +17-24% throughput at C80-C96. Small: with the wide kernel, +22% throughput at C120 (139x &rarr; 171x)",
         "Byte-identical"],
        ["int8 backbone for the large model",
         "The large backbone's weights are read as 8-bit. It lost on the small model (20 Sep) "
         "and wins on the large one, so the runtime picks it from the model's depth.",
         "Large, C48: +16% throughput, first audio 830 &rarr; 752 ms",
         "ASR-checked: indistinguishable from full precision"],
        ["Prefill slice sized per model",
         "The text reading is interleaved with audio in slices; the slice is now sized by "
         "what one token costs for the model (32 tokens small, 16 large).",
         "Large: worst gap 209 &rarr; 134 ms and no stalls at 500 ms", "Byte-identical"],
        ["12 slots per worker (small)",
         "16 workers x 12 slots = 192 places instead of 128.",
         "Removes the 128-stream wall of the 21 Sep report", "Byte-identical"],
    ], [30*mm, 58*mm, 52*mm, fw-140*mm], size=7.8))

    # ---------------------------------------------------------------- 6
    A(Paragraph("6. What did not work", H2))
    A(Paragraph("Reported because a capacity figure is only as credible as the ideas that "
                "were tried against it and failed. Each of these was measured, not argued, and "
                "each was removed or left disabled:", BODY))
    A(table([
        ["Tried", "Result", "Why it was dropped"],
        ["Faster attention inner loop (NEON, four keys per pass)",
         "11% faster in isolation; in serving, realtime factor p95 0.872/0.879 &rarr; "
         "0.896/0.896 at C112", "A local win that turned into a serving loss. Removed"],
        ["Wide int8 kernel applied to the audio decoder's convolutions",
         "2.47 &rarr; 2.45 ms per frame", "No effect: the cost there is not in the multiply. "
         "Reverted"],
        ["8-bit for the decoder's 32-channel stage",
         "Spectral correlation 0.9976 &rarr; 0.9785", "Audible quality risk for ~2% speed"],
        ["More or fewer workers for the large model (4x8, 32x1)",
         "4x8: 31.5x vs 69x realtime; 32x1: worse on every gate",
         "16 x 2 stays the layout for both models"],
        ["A time cap on the text reading (large model)",
         "20, 30, 40 and 60 ms measured identical", "One slice already exceeds any cap"],
        ["Slices smaller than 16 tokens",
         "8, 16 and 24 measured identical", "16 tokens is the kernel's tile; smaller is the "
         "same work"],
        ["Further decoder optimisation",
         "Best remaining candidate worth &le; 3.4% of the wall",
         "Below the agreed 3% serving-gain bar at acceptable risk: stopped"],
    ], [48*mm, 62*mm, fw-110*mm], size=7.8))
    A(Spacer(1, 3))
    A(Paragraph("One apparent quality defect was also chased and closed: segmented audio "
                "seemed to lose its last clause. It was the speech recogniser used for the check, "
                "which drops the tail of inputs around 20 seconds long; the audio was proved "
                "complete by bit-identity at temperature 0 and by transcribing the tail alone.",
                SMALL))

    # ---------------------------------------------------------------- 7
    A(Paragraph("7. Audio quality", H2))
    A(Paragraph("Changes that alter the audio were checked by transcribing it with an "
                "independent speech recogniser and comparing with the input text (word error "
                "rate, lower is better; long texts cut at pauses into pieces of at most 12 "
                "seconds). Large model, four seeds, 33 long texts (8,200 words) and 30 medium "
                "(1,424 words):", BODY))
    A(table([
        ["Configuration", "WER, long texts", "WER, medium texts"],
        ["full-precision backbone (reference)", "0.87%", "0.49%"],
        ["bfloat16 backbone", "0.73%", "0.56%"],
        ["int8 backbone", "0.83%", "0.56%"],
        ["int8 backbone + segmentation (the shipped default)", "0.94%", "0.56%"],
    ], [80*mm, 40*mm, fw-120*mm], align={1: "RIGHT", 2: "RIGHT"}, size=8.4))
    A(Spacer(1, 3))
    A(Paragraph("The int8 backbone is indistinguishable from full precision. Segmentation "
                "costs about a tenth of a point on long texts and adds 0.6-0.8 s of natural "
                "pause per long text at chunk boundaries; medium texts are a single chunk and are "
                "unchanged. The owner listened to the large model's output under load and "
                "judged it very good.", BODY))
    A(Paragraph("The same check on the small model, where only segmentation changed (two seeds, "
                "66 long-text runs of 4,100 words, 60 medium of 712):", BODY))
    A(table([
        ["Small model", "WER, long texts", "WER, medium texts"],
        ["segmentation off (the 21 September default)", "0.78%", "0.42%"],
        ["segmentation on (the shipped default)", "1.07%", "0.42%"],
    ], [80*mm, 40*mm, fw-120*mm], align={1: "RIGHT", 2: "RIGHT"}, size=8.4))
    A(Spacer(1, 3))
    A(Paragraph("The same pattern as the large model: about three tenths of a point on long "
                "texts, nothing on the others.", BODY))

    # ---------------------------------------------------------------- 8
    A(Paragraph("8. The operating points in detail", H2))
    A(table([
        ["Thirty minutes, shipped default", "Small, C%d" % s6["level"], "Large, C%d" % l24["level"]],
        ["Requests completed", "%s of %s" % (n(s6["requests"]), n(s6["requests"])),
         "%s of %s" % (n(l24["requests"]), n(l24["requests"]))],
        ["Stalls, 250 ms / 500 ms client buffer", "%d / %d" % (s6["s250"], s6["s500"]),
         "%d / %d" % (l24["s250"], l24["s500"])],
        ["Time to first byte, p95", "%.1f ms" % s6["ttfb"], "%.1f ms" % l24["ttfb"]],
        ["Time to first audio, p95 (median)", "%.0f ms (%.0f)" % (s6["ttfa"], s6["ttfa50"]),
         "%.0f ms (%.0f)" % (l24["ttfa"], l24["ttfa50"])],
        ["Streaming realtime factor, p95 (median)", "%.3f (%.3f)" % (s6["rtf"], s6["rtf50"]),
         "%.3f (%.3f)" % (l24["rtf"], l24["rtf50"])],
        ["Client buffer needed, p95", "%.0f ms" % s6["prebuf"], "%.0f ms" % l24["prebuf"]],
        ["Longest gap between chunks, p95", "%.0f ms" % s6["gap"], "%.0f ms" % l24["gap"]],
        ["Speech produced per second", "%.0f s" % s6["tput"], "%.0f s" % l24["tput"]],
        ["Realtime-factor drift over ten windows", "%+.4f" % s6["drift"], "%+.4f" % l24["drift"]],
    ], [70*mm, (fw-70*mm)/2, (fw-70*mm)/2], size=8.4))

    # ---------------------------------------------------------------- 9
    A(Paragraph("9. What to run on this machine", H2))
    A(table([
        ["", "Small model", "Large model"],
        ["Recommended concurrency", "%d streams" % s6["level"], "%d streams" % l24["level"]],
        ["Server layout", "16 workers x 2 threads, %d slots" % s6["mb"],
         "16 workers x 2 threads, 8 slots"],
        ["Expected first audio, p95", "%.0f ms" % s6["ttfa"], "%.0f ms" % l24["ttfa"]],
        ["Expected throughput", "%.0fx realtime" % s6["tput"], "%.0fx realtime" % l24["tput"]],
        ["Machines for 1,000 concurrent listeners", "%d" % -(-1000 // s6["level"]),
         "%d" % -(-1000 // l24["level"])],
        ["Configuration to export", "none", "none"],
    ], [60*mm, (fw-60*mm)/2, (fw-60*mm)/2], size=8.4))
    A(Spacer(1, 3))
    A(Paragraph("The large model costs about %.1fx the machines of the small one for the same "
                "audience, for three times the parameters. Both recommendations are the "
                "certified levels; the levels above them are listed in sections 3 and 4 with "
                "exactly what they missed." % (s6["level"] / float(l24["level"])), BODY))

    # ---------------------------------------------------------------- 10
    A(Paragraph("10. Scope of these numbers", H2))
    A(table([
        ["One language", "English only; other packs and fine-tunes not yet measured for serving."],
        ["Arm measured", "Every figure is from Arm Neoverse-V2. The x86 build passes the same "
                         "correctness suites in CI but has not been through this campaign."],
        ["One host class", "32 cores. The worker layout follows the model's cost structure and "
                           "must be re-screened per host."],
        ["Load generator on the same box", "The client shares the 32 cores with the server "
                                           "(about 11% of its reads coalesce), so the figures "
                                           "are conservative."],
        ["CPU only", "A GPU path exists separately and is not part of this report."],
    ], [45*mm, fw-45*mm], header=False, size=8.4))

    A(Paragraph("11. What comes next", H2))
    A(table([
        ["Move the load generator off the host", "So the measurement stops sharing the cores it "
                                                 "measures."],
        ["The small model's exact ceiling", "Screens above the certified level are listed in "
                                            "section 3; each candidate is a thirty-minute run."],
        ["The large model's 96-112 edge", "Per-request records in the load tool, to attribute "
                                           "a rare stall to a chunk transition or a text read."],
        ["The same campaign on x86", "The identical protocol on AMD and Intel hosts."],
    ], [60*mm, fw-60*mm], header=False, size=8.4))

    A(Paragraph("12. Reproducing this", H2))
    A(Paragraph("The runtime, the harness and both qualified configurations are open source and "
                "committed as profiles; the harness refuses to run if the environment "
                "contradicts them.", BODY))
    A(Paragraph("python3 tools/serving_profile.py --profile axion-c4a-32c-pocket-en \\<br/>"
                "&nbsp;&nbsp;&nbsp;&nbsp;--model models/pocket-en --server-bin "
                "build/cpu/mynah-tts-server<br/>"
                "python3 tools/serving_profile.py --profile axion-c4a-32c-pocket-en-24l \\<br/>"
                "&nbsp;&nbsp;&nbsp;&nbsp;--model models/pocket-en-24l --server-bin "
                "build/cpu/mynah-tts-server", MONO))
    A(Spacer(1, 6))
    A(Rule(fw))
    A(Paragraph("Measurements taken 25-27 September 2026 on GCP Axion c4a. Model "
                "kyutai/pocket-tts revision 4925226, CC-BY-4.0. Runtime mynah-tts, C11, open "
                "source. Prepared by Gabriele Mastrapasqua.", SMALL))
    doc.build(F)


if __name__ == "__main__":
    build(sys.argv[1])
