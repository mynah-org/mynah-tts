#!/usr/bin/env python3
"""PocketTTS on Arm - the client-facing PDF. Generated with reportlab.

Lives in tools/ and not in reports/ because reports/ is gitignored: the logs it
reads are benchmark dumps that must never be committed, but this generator is
source for a deliverable that has now been reissued three times. Data in, code
tracked.

    uv run --with reportlab python3 tools/client_report_pocket.py OUT.pdf
"""
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (BaseDocTemplate, Frame, PageTemplate, Paragraph,
                                Spacer, Table, TableStyle, Flowable, KeepTogether)

INK   = colors.HexColor("#1a1a1a")
MUTE  = colors.HexColor("#6b6b6b")
RULE  = colors.HexColor("#d8d8d8")
GOOD  = colors.HexColor("#2f7d4f")
MARG  = colors.HexColor("#b07d1c")
BAD   = colors.HexColor("#a33a2e")
BAND  = colors.HexColor("#f4f4f2")
ACCENT= colors.HexColor("#1f4e79")

S = getSampleStyleSheet()
def st(name, **kw):
    base = dict(fontName="Helvetica", fontSize=9.5, leading=14, textColor=INK)
    base.update(kw)
    return ParagraphStyle(name, **base)

BODY  = st("body", spaceAfter=6)
SMALL = st("small", fontSize=8.2, leading=11.5, textColor=MUTE)
H1    = st("h1", fontName="Helvetica-Bold", fontSize=17, leading=21, spaceAfter=2)
H2    = st("h2", fontName="Helvetica-Bold", fontSize=11.5, leading=15,
           spaceBefore=13, spaceAfter=5, textColor=ACCENT)
H3    = st("h3", fontName="Helvetica-Bold", fontSize=9.5, leading=13,
           spaceBefore=8, spaceAfter=3)
SUB   = st("sub", fontSize=10, leading=14, textColor=MUTE)
MONO  = st("mono", fontName="Courier", fontSize=8, leading=11)
LEAD  = st("lead", fontSize=11, leading=16)

CELL  = st("cell", fontSize=8.6, leading=11.4)
CELLB = st("cellb", fontName="Helvetica-Bold", fontSize=8.6, leading=11.4)

def _wrap(rows, header):
    """Every cell becomes a Paragraph, so a long one wraps instead of running off
    the page and so the entities in it are rendered rather than printed."""
    out = []
    for r, row in enumerate(rows):
        style = CELLB if (header and r == 0) else CELL
        out.append([c if not isinstance(c, str) else Paragraph(c, style) for c in row])
    return out


def table(rows, widths, header=True, align=None, size=8.6):
    t = Table(_wrap(rows, header), colWidths=widths, hAlign="LEFT")
    cmds = [
        ("FONT", (0,0), (-1,-1), "Helvetica", size),
        ("TEXTCOLOR", (0,0), (-1,-1), INK),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("TOPPADDING", (0,0), (-1,-1), 3.5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 3.5),
        ("LEFTPADDING", (0,0), (-1,-1), 5),
        ("LINEBELOW", (0,0), (-1,-2), 0.3, RULE),
    ]
    if header:
        cmds += [("FONT", (0,0), (-1,0), "Helvetica-Bold", size),
                 ("BACKGROUND", (0,0), (-1,0), BAND),
                 ("LINEBELOW", (0,0), (-1,0), 0.6, MUTE)]
    for col, a in (align or {}).items():
        cmds.append(("ALIGN", (col,0), (col,-1), a))
        cmds.append(("RIGHTPADDING", (col,0), (col,-1), 6))
    t.setStyle(TableStyle(cmds))
    return t


class Rule(Flowable):
    def __init__(self, w, thick=0.6, color=RULE, pad=3):
        self.width, self.thick, self.color, self.pad = w, thick, color, pad
        self.height = thick + pad*2
    def draw(self):
        self.canv.setStrokeColor(self.color); self.canv.setLineWidth(self.thick)
        self.canv.line(0, self.pad, self.width, self.pad)


class Callout(Flowable):
    """The headline numbers, in a band."""
    def __init__(self, width, items):
        self.width, self.items = width, items
        self.height = 30*mm
    def draw(self):
        c = self.canv
        c.setFillColor(BAND); c.rect(0, 0, self.width, self.height, stroke=0, fill=1)
        c.setStrokeColor(ACCENT); c.setLineWidth(2)
        c.line(0, 0, 0, self.height)
        n = len(self.items)
        col = self.width / n
        for i, (big, small) in enumerate(self.items):
            x = 8 + i*col
            c.setFillColor(ACCENT); c.setFont("Helvetica-Bold", 19)
            c.drawString(x, self.height - 15*mm, big)
            c.setFillColor(MUTE); c.setFont("Helvetica", 7.6)
            for j, line in enumerate(small):
                c.drawString(x, self.height - 20*mm - j*9, line)


class LevelChart(Flowable):
    """Every concurrency level measured, how long for, and the verdict."""
    def __init__(self, width, data):
        self.width, self.data = width, data
        self.height = 66*mm
    def draw(self):
        c = self.canv
        L, R, B = 26, 12, 30
        plot_w = self.width - L - R
        plot_h = self.height - B - 20
        c.setFont("Helvetica-Bold", 8.5); c.setFillColor(INK)
        c.drawString(0, self.height - 8, "Concurrency levels measured, soak duration and verdict")
        # y axis: minutes
        c.setFont("Helvetica", 7); c.setFillColor(MUTE)
        for m in (0, 10, 20, 30):
            y = B + plot_h * m/32.0
            c.setStrokeColor(RULE); c.setLineWidth(0.3)
            c.line(L, y, L+plot_w, y)
            c.drawRightString(L-4, y-2.5, "%d" % m)
        c.saveState(); c.translate(7, B+plot_h/2); c.rotate(90)
        c.setFont("Helvetica", 7.2); c.drawCentredString(0, 0, "soak minutes")
        c.restoreState()
        n = len(self.data)
        slot = plot_w / n
        for i, (label, mins, verdict, note) in enumerate(self.data):
            x = L + i*slot + slot*0.22
            w = slot*0.56
            h = plot_h * mins/32.0
            col = {"GOOD": GOOD, "MARGINAL": MARG, "NOT STREAMABLE": BAD}[verdict]
            c.setFillColor(col); c.rect(x, B, w, h, stroke=0, fill=1)
            c.setFillColor(colors.white); c.setFont("Helvetica-Bold", 7)
            if h > 14:
                c.drawCentredString(x+w/2, B+h-9, "%d min" % mins)
            c.setFillColor(col); c.setFont("Helvetica-Bold", 7.2)
            c.drawCentredString(x+w/2, B+h+4, verdict if verdict!="NOT STREAMABLE" else "NOT STREAM.")
            c.setFillColor(INK); c.setFont("Helvetica-Bold", 8.4)
            c.drawCentredString(x+w/2, B-11, label)
            c.setFillColor(MUTE); c.setFont("Helvetica", 6.4)
            c.drawCentredString(x+w/2, B-19, note)
        c.setStrokeColor(MUTE); c.setLineWidth(0.6)
        c.line(L, B, L+plot_w, B)


class TtfaChart(Flowable):
    """First audio against the gate: why the ceiling sits where it does."""
    def __init__(self, width, points):
        self.width, self.points = width, points
        self.height = 50*mm
    def draw(self):
        c = self.canv
        L, R, B = 30, 12, 24
        pw = self.width - L - R
        ph = self.height - B - 18
        lo, hi = 300.0, 620.0
        def ypix(v): return B + ph * (v-lo)/(hi-lo)
        xs = [p[0] for p in self.points]
        xlo, xhi = min(xs)-3, max(xs)+3
        def xpix(v): return L + pw * (v-xlo)/(xhi-xlo)
        c.setFont("Helvetica-Bold", 8.5); c.setFillColor(INK)
        c.drawString(0, self.height-8, "Time to first audio, 95th percentile, against the 500 ms target")
        c.setFont("Helvetica", 7); c.setFillColor(MUTE)
        for v in (300, 350, 400, 450, 500, 550, 600):
            y = ypix(v)
            c.setStrokeColor(RULE); c.setLineWidth(0.3); c.line(L, y, L+pw, y)
            c.drawRightString(L-4, y-2.5, "%d" % v)
        c.saveState(); c.translate(8, B+ph/2); c.rotate(90)
        c.setFont("Helvetica", 7.2); c.drawCentredString(0, 0, "TTFA p95 (ms)")
        c.restoreState()
        y500 = ypix(500)
        c.setStrokeColor(BAD); c.setLineWidth(1); c.setDash(3,2)
        c.line(L, y500, L+pw, y500); c.setDash()
        c.setFillColor(BAD); c.setFont("Helvetica-Bold", 6.8)
        c.drawRightString(L+pw, y500+5, "500 ms target")
        pts = sorted(self.points, key=lambda p: p[0])
        for i, (cc, val, verdict, tag) in enumerate(pts):
            if i:
                px, py = xpix(pts[i-1][0]), ypix(pts[i-1][1])
                c.setStrokeColor(RULE); c.setLineWidth(0.8)
                c.line(px, py, xpix(cc), ypix(val))
        for cc, val, verdict, tag in pts:
            x, y = xpix(cc), ypix(val)
            col = {"GOOD": GOOD, "MARGINAL": MARG}[verdict]
            c.setFillColor(colors.white); c.circle(x, y, 4.2, stroke=0, fill=1)
            c.setFillColor(col); c.circle(x, y, 3.1, stroke=0, fill=1)
            c.setFillColor(INK); c.setFont("Helvetica-Bold", 6.8)
            # above the marker when the point sits below the target line, below it
            # when above, so a label never lands on the dashed rule
            c.drawCentredString(x, y + (-11 if val < 505 else 7), "%.0f" % val)
            c.setFillColor(MUTE); c.setFont("Helvetica", 6.4)
            c.drawCentredString(x, B-17, tag)
            c.setFillColor(INK); c.setFont("Helvetica-Bold", 8)
            c.drawCentredString(x, B-9, "C%d" % cc)
        c.setStrokeColor(MUTE); c.setLineWidth(0.6); c.line(L, B, L+pw, B)


def build(path):
    doc = BaseDocTemplate(path, pagesize=A4,
                          leftMargin=20*mm, rightMargin=18*mm,
                          topMargin=17*mm, bottomMargin=17*mm,
                          title="PocketTTS on Arm - streaming capacity report",
                          author="Gabriele Mastrapasqua")
    fw = doc.width
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="n")

    def decorate(canv, d):
        canv.saveState()
        canv.setFont("Helvetica", 7.2); canv.setFillColor(MUTE)
        canv.drawString(doc.leftMargin, 11*mm,
                        "PocketTTS on Arm - GCP Axion c4a - 19 September 2026")
        canv.drawRightString(doc.leftMargin+doc.width, 11*mm, "page %d" % d.page)
        canv.setStrokeColor(RULE); canv.setLineWidth(0.4)
        canv.line(doc.leftMargin, 13.5*mm, doc.leftMargin+doc.width, 13.5*mm)
        canv.restoreState()

    doc.addPageTemplates([PageTemplate(id="all", frames=[frame], onPage=decorate)])
    F = []
    A = F.append

    # ---------------------------------------------------------------- title
    A(Paragraph("PocketTTS on Arm", H1))
    A(Paragraph("Streaming capacity on a single 32-core Arm server &mdash; "
                "GCP Axion c4a &middot; 19-21 September 2026", SUB))
    A(Spacer(1, 4))
    A(Rule(fw, 1.2, ACCENT))
    A(Spacer(1, 8))

    A(Callout(fw, [
        ("126", ["concurrent streams", "certified over 30 minutes"]),
        ("0", ["stalls in 65,039", "requests"]),
        ("330 ms", ["to first audio", "95th percentile"]),
        ("157x", ["realtime aggregate", "audio seconds per second"]),
    ]))
    A(Spacer(1, 8))

    A(Paragraph(
        "One 32-core Arm virtual machine, CPU only, no GPU and no Python in the "
        "serving path, sustains <b>120 simultaneous speech streams</b> for thirty "
        "minutes without a single listener hearing a gap. This document states what "
        "was measured, how a level is allowed to be called qualified, and what has "
        "<i>not</i> been measured yet &mdash; this engine is a week old and the "
        "numbers below cover one model, one language and one class of host.", LEAD))

    A(Spacer(1, 6))
    A(Paragraph("Where it started and where it is now", H3))
    A(table([
        ["", "19 Sep\nstart", "19 Sep\nend", "20 Sep", "21 Sep", "Change"],
        ["Certified streams", "96", "120", "120", "126", "+31%"],
        ["Highest level tested", "96", "130", "128", "128", "\u2014"],
        ["First audio, 95th pct", "492 ms", "447 ms", "336 ms", "330 ms", "33% faster"],
        ["Speech per second", "130x", "155x", "158x", "157x", "+21%"],
        ["Interruptions, certified level", "0", "0", "0", "0", "\u2014"],
    ], [44*mm, 21*mm, 21*mm, 21*mm, 21*mm, fw-128*mm],
       align={1:"RIGHT",2:"RIGHT",3:"RIGHT",4:"RIGHT",5:"RIGHT"}, size=8.2))
    A(Spacer(1, 4))
    A(Paragraph("Same virtual machine, same model, same test corpus throughout. Nothing in "
                "the table comes from better hardware or from lowering audio quality; the "
                "changes behind it are described in section 5. The final column compares the "
                "best figure available at the end against the starting point.", SMALL))
    A(Spacer(1, 10))

    # ---------------------------------------------------------------- 1
    A(Paragraph("1. What is being measured", H2))
    A(Paragraph("<b>The model</b> is Kyutai PocketTTS, the public English pack, used "
                "unmodified.", BODY))
    A(table([
        ["Model", "kyutai/pocket-tts (Hugging Face)"],
        ["Link", "https://huggingface.co/kyutai/pocket-tts"],
        ["Revision", "492522650173a0653b7575cdc25ae09810e5d741"],
        ["Licence", "CC-BY-4.0 (weights). Per-voice licences vary; see the pack's VOICES.md"],
        ["Language", "English"],
        ["Architecture", "Continuous-latent autoregressive, ~109.5 M parameters: a 6-layer "
                         "backbone, a flow-matching head, a Mimi/SEANet causal decoder"],
        ["Audio", "24 000 Hz, one 80 ms frame per step (12.5 frames per second)"],
    ], [26*mm, fw-26*mm], header=False))
    A(Spacer(1, 5))
    A(Paragraph("<b>The runtime</b> is mynah-tts, an open-source C11 inference engine. "
                "Weights are memory-mapped, there are no allocations in the decode loop, "
                "and nothing Python-shaped runs at serving time. The matrix kernels are "
                "its own: this host runs with <font face='Courier' size='8'>BLAS=none</font>, "
                "so no external BLAS is involved.", BODY))
    A(table([
        ["Host", "GCP Axion c4a, 32 x Arm Neoverse-V2, 62 GiB, Ubuntu, gcc 15.2"],
        ["Build", "BLAS=none, SIMD=native (ASIMD/dotprod), CPU only"],
        ["Server", "16 worker processes x 2 threads, 8 request slots each"],
        ["Precision", "int8 for the audio codec, bfloat16 for the backbone, f16 for the "
                      "flow head and conditioner &mdash; the shipped default, nothing "
                      "hand-tuned for this test"],
    ], [26*mm, fw-26*mm], header=False))

    # ---------------------------------------------------------------- 2
    A(Paragraph("2. What &ldquo;qualified&rdquo; means here", H2))
    A(Paragraph("A text-to-speech server can be fast on average and still be unusable, "
                "because a listener does not experience an average &mdash; they experience "
                "the one moment the audio ran out. Every level below is therefore judged "
                "by a simulated player, not by throughput.", BODY))
    A(KeepTogether(table([
        ["Gate", "Threshold", "What it protects"],
        ["Stall rate", "0 requests, with a 250 ms client buffer",
         "Nobody hears a gap. Zero is the requirement, not a small number"],
        ["Time to first audio", "500 ms at the 95th percentile",
         "The pause before the voice starts"],
        ["Streaming realtime factor", "below 0.90 at the 95th percentile",
         "Each stream stays ahead of its own playback"],
        ["Completion", "every request served, none dropped", "No silent failures under load"],
        ["Drift", "no upward trend across ten 3-minute windows",
         "The server is not slowly falling behind"],
    ], [34*mm, 44*mm, fw-78*mm])))
    A(Spacer(1, 5))
    A(Paragraph("Two rules of method are worth stating, because they are what makes the "
                "headline number trustworthy:", BODY))
    A(Paragraph("<b>Load is sustained, not burst.</b> Requests are kept continuously in "
                "flight for the whole run, so the server never gets an idle moment to catch "
                "up in. <b>Thirty minutes, not ten.</b> A ten-minute run of this exact "
                "configuration reported a clean result at C96 and a thirty-minute run of the "
                "same configuration found one stall. Only the long run is allowed to promote "
                "a level, and the short runs in the chart below are marked as screens.", BODY))
    A(Paragraph("The text corpus is a fixed bank of 277 English sentences spanning five "
                "length classes, from 1.0 to 19.0 seconds of speech &mdash; short "
                "confirmations, conversational replies and long passages mixed exactly as a "
                "real service would see them. In the qualifying run every one of the 277 was "
                "spoken about 195 times.", BODY))

    # ---------------------------------------------------------------- 3
    A(Paragraph("3. Results", H2))
    A(LevelChart(fw, [
        ("C96", 30, "GOOD",     "19 Sep build"),
        ("C110", 30, "GOOD",    "19 Sep build"),
        ("C120", 30, "GOOD",    "19 Sep build"),
        ("C126", 30, "GOOD",    "CERTIFIED"),
        ("C128", 30, "MARGINAL", "3 stalls / 65,295"),
        ("C130", 10, "MARGINAL", "queue, not speed"),
    ]))
    A(Spacer(1, 3))
    A(Paragraph("Green cleared every target; amber cleared all the essential ones and missed "
                "a single stricter target. Only thirty-minute runs are allowed to promote a "
                "level. <b>128 streams is the headline of the second day</b>: thirty minutes, "
                "65,295 requests delivered, and three brief interruptions in the whole run "
                "&mdash; <b>99.995% of requests played without one</b>. Our bar for promotion "
                "is zero, so 128 is recorded as not-yet-qualified rather than as a "
                "capacity claim; but it is 8 streams above the certified figure and it took "
                "three interruptions, not a wall, to hold it back.", SMALL))
    A(Spacer(1, 8))
    A(table([
        ["Level", "Soak", "Requests", "First audio p95", "Header p95",
         "Stalls 250ms", "Build", "Verdict"],
        ["C96",  "30 min", "53,886", "492.5 ms", "\u2014",  "0", "19 Sep", "GOOD"],
        ["C110", "30 min", "63,120", "482.3 ms", "\u2014",  "0", "19 Sep", "GOOD"],
        ["C120", "30 min", "64,205", "447.4 ms", "75.1 ms",  "0", "19 Sep", "GOOD"],
        ["C126", "30 min", "65,039", "329.5 ms", "79.1 ms",  "0", "21 Sep",
         "GOOD (certified)"],
        ["C128", "30 min", "65,295", "336.3 ms", "77.4 ms",  "3", "20 Sep", "MARGINAL"],
        ["C130", "10 min", "21,742", "402.8 ms", "203.5 ms", "0", "20 Sep", "MARGINAL"],
        ["C130", "10 min", "21,620", "580.9 ms", "200.7 ms", "2", "19 Sep", "MARGINAL"],
    ], [14*mm, 14*mm, 19*mm, 24*mm, 20*mm, 19*mm, 15*mm, fw-125*mm],
       align={1:"CENTER",2:"RIGHT",3:"RIGHT",4:"RIGHT",5:"CENTER",6:"CENTER"}, size=7.8))
    A(Spacer(1, 4))
    A(Paragraph("<b>The build column matters and is not decoration.</b> Levels measured on "
                "different days are not directly comparable, and reading down the "
                "first-audio column across builds would credit the wrong thing. The two "
                "C130 rows are the same level on consecutive builds: first audio improved "
                "from 580.9&nbsp;ms to 402.8&nbsp;ms overnight, which is the build and not "
                "the concurrency. Within a single build the comparison is sound; across "
                "them it is not.", SMALL))
    A(Spacer(1, 3))
    A(Paragraph("C96 is the starting point of 19 September and is shown so the progression "
                "is visible: three changes described in section 5 took the qualified level "
                "from 96 to 120 in a single day, on the same machine and the same corpus.",
                SMALL))

    A(Spacer(1, 10))
    A(TtfaChart(fw, [
        (96, 492.5, "GOOD", "30 min"), (110, 482.3, "GOOD", "30 min"),
        (120, 447.4, "GOOD", "30 min"), (126, 329.5, "GOOD", "certified"),
        (128, 336.3, "MARGINAL", "30 min"), (130, 402.8, "MARGINAL", "10 min"),
    ]))
    A(Spacer(1, 3))
    A(Paragraph("<b>First audio gets faster as the load goes up</b>, which is the opposite "
                "of what usually happens and is the direct result of the changes in section "
                "5. It improved again overnight: at 130 streams it went from 581&nbsp;ms to "
                "403&nbsp;ms between the two builds, and at 128 streams it now sits at "
                "336&nbsp;ms &mdash; a third faster than the certified 120-stream figure and "
                "comfortably inside the 500&nbsp;ms target. The one number that does rise "
                "past 128 streams is how long a caller waits to be accepted, which is the "
                "queue described in section 6, not the speech engine.", SMALL))

    # ---------------------------------------------------------------- 4
    A(Paragraph("4. The operating point in detail", H2))
    A(Paragraph("C120, thirty minutes, sustained load, shipped configuration with "
                "nothing tuned by hand for the run:", BODY))
    A(table([
        ["Requests completed", "64,205 of 64,205 (none dropped, none timed out)"],
        ["Stalls, 250 ms client buffer", "0"],
        ["Stalls, 500 ms client buffer", "0"],
        ["Time to first byte, p95", "75.1 ms"],
        ["Time to first audio, p95", "447.4 ms  (median 177 ms)"],
        ["Streaming realtime factor, p95", "0.794  (median 0.734)"],
        ["Buffering a client needed, p95", "2.8 ms  (against a 250 ms contract)"],
        ["Longest gap between audio chunks", "121 ms at p95 (a frame lasts 80 ms)"],
        ["Aggregate throughput", "155.3 seconds of speech per second of wall clock"],
        ["Stability", "flat across ten 3-minute windows; realtime factor drift +0.002"],
    ], [58*mm, fw-58*mm], header=False))
    A(Spacer(1, 5))
    A(Paragraph("In plainer terms: the machine produces about two and a half hours of speech "
                "per minute of its own time, a hundred and twenty people can be listening at once, and the "
                "voice starts in about a fifth of a second for a typical request.", BODY))

    # ---------------------------------------------------------------- 5
    A(Paragraph("5. Why the gains are this large", H2))
    A(Paragraph("A 25% capacity increase in one day, on unchanged hardware and an unchanged "
                "model, is a number that deserves an explanation rather than a celebration. "
                "The explanation is that <b>none of it came from making the model smaller or "
                "the audio worse</b>. Three of the four changes below cost nothing at all "
                "measurable, and the fourth is a quality trade that was judged by listening "
                "before it was accepted.", BODY))
    A(Paragraph("The reason such gains were still available is that this engine had been "
                "optimised for <b>one request at a time</b>, and was then asked to serve a "
                "hundred. Those are different problems. A single request wants the fastest "
                "possible arithmetic; a hundred concurrent requests want the work shared so "
                "that no one waits behind everyone else. Every finding below is of the "
                "second kind &mdash; the machine was never short of compute. At the "
                "qualified level each stream is still produced about 25% faster than it is "
                "played, and at the level above it that headroom is still there while the "
                "server fails for an unrelated reason. What was scarce was never throughput. "
                "It was <b>order</b>.", BODY))
    A(Paragraph("Three of these were found by instrumenting the server and reading the "
                "result, and each one contradicted a written expectation &mdash; including, "
                "twice, a rule this team had itself written down a few hours earlier. That "
                "is the method, and it is why the numbers in section 3 are soak "
                "measurements rather than estimates.", BODY))
    A(Paragraph("5.1 The four changes", H3))
    A(Paragraph("Every one measured rather than reasoned, and three of them in a single "
                "day:", BODY))
    A(Paragraph("<b>Admitting a request used to freeze the ones already running.</b> Reading "
                "a new request's text was done in one uninterrupted piece inside the serving "
                "loop, so every listener already being served waited for it &mdash; up to "
                "707 ms against an 80 ms frame. That work is now sliced and interleaved with "
                "audio generation. Before the change no concurrency level on this mixed "
                "corpus could be qualified at all.", BODY))
    A(Paragraph("<b>The slice budget was a guess, then a rule, then a measurement.</b> "
                "Timing the loop showed a typical generation step costing 50 ms of the 80 ms "
                "frame, so the budget was set to the 30 ms of slack that left. That moved "
                "the qualified level from C90 to C96 &mdash; and then stopped being right. "
                "Making the arithmetic faster (below) made the step cheaper, and the same "
                "rule then pointed the wrong way: measured directly, 40 ms beats 30 by sixty "
                "milliseconds of first audio. The lesson is written into the code: this is a "
                "measured quantity and has to be re-measured whenever a kernel changes what "
                "a step costs.", BODY))
    A(Paragraph("<b>The weights were already in a format the processor can multiply "
                "natively, and the engine was not using it.</b> The model ships in bfloat16; "
                "the runtime converted it to a wider format at load and a different narrow "
                "one before every multiply &mdash; three representations to arrive back "
                "where it started. Arm processors of this generation multiply bfloat16 "
                "directly. Using it, with a kernel that keeps eight independent "
                "accumulations in flight, runs the relevant arithmetic <b>14.7x faster</b> "
                "than the path it replaced, and took first audio from 526 ms to 387 ms at "
                "the same concurrency.", BODY))
    A(Paragraph("<b>The largest single gain came from changing no arithmetic at all.</b> "
                "When several requests are waiting to have their text read, the server used "
                "to share the time between them evenly. That sounds fair and it is the worst "
                "choice: everyone finishes at roughly the moment the last one would have. "
                "Serving them one at a time, oldest first, took <b>127 ms</b> off first "
                "audio &mdash; more than any kernel change did. Every category of text "
                "improved, and the short ones, which are most of the traffic, improved most.",
                BODY))

    # ---------------------------------------------------------------- 6
    A(Paragraph("5.2 What each one was worth", H3))
    A(table([
        ["Change", "Kind", "Effect"],
        ["Slicing the text read", "scheduling", "made ANY level qualifiable; before it, none was"],
        ["Slice budget re-measured", "one constant", "C90 &rarr; C96, then C110 &rarr; C120"],
        ["bfloat16 arithmetic", "kernel", "C96 &rarr; C110; first audio 526 &rarr; 387 ms at fixed load"],
        ["Serving waits in order", "scheduling", "127 ms off first audio; touches no arithmetic at all"],
    ], [46*mm, 26*mm, fw-72*mm]))
    A(Spacer(1, 4))
    A(Paragraph("The last row is the one worth reading twice. The kernel change is 14.7x on "
                "the arithmetic it replaces and is the most technically involved thing here. "
                "Serving waiting requests one at a time instead of sharing time between them "
                "changes no arithmetic whatsoever, and took more off first audio than any "
                "other single change. Sharing time evenly sounds fair; it is the policy under "
                "which everyone finishes at roughly the moment the last one would have.",
                SMALL))
    A(Spacer(1, 6))
    A(Paragraph("6. What to run on this machine", H2))
    A(Paragraph("For a 32-core Arm server of this class &mdash; GCP Axion c4a-highcpu-32, "
                "or equivalent &mdash; running the English PocketTTS pack, the "
                "recommendation is:", BODY))
    A(Spacer(1, 4))
    A(table([
        ["Recommended concurrency", "126 simultaneous streams",
         "Certified: 30 minutes, 65,039 requests, zero interruptions"],
        ["Server layout", "16 workers &times; 2 threads, batch 8",
         "128 request places; measured better than 8&times;4 at equal threads"],
        ["Expected first audio", "330 ms at the 95th percentile",
         "Target is 500 ms; median is 132 ms"],
        ["Expected throughput", "157&times; realtime",
         "157 seconds of speech produced per second of wall clock"],
        ["Hard ceiling of this layout", "128 streams",
         "16 &times; 8 request places; beyond it callers queue"],
        ["Headroom above the recommendation", "2 streams",
         "Deliberately small: 128 was measured and did not qualify"],
    ], [42*mm, 40*mm, fw-82*mm], header=False, size=8.2))
    A(Spacer(1, 5))
    A(Paragraph("<b>Why 126 and not 128.</b> 128 was measured over a full thirty minutes "
                "and delivered 65,295 requests with three brief interruptions &mdash; "
                "99.995% clean. Our bar for a certified figure is zero, so the recommended "
                "number is the highest one that met it. If your own tolerance is "
                "\u201cunder one interruption in ten thousand requests\u201d rather than "
                "zero, 128 is available and measured, and the difference is 1.6% more "
                "capacity.", BODY))
    A(Spacer(1, 4))
    A(Paragraph("<b>Sizing by demand.</b> At 126 streams per machine, a service expecting "
                "a peak of 1,000 concurrent listeners needs 8 machines of this class; "
                "2,500 needs 20. Because each stream is produced about 22% faster than it "
                "plays, a machine at its recommended load still has compute in reserve for "
                "bursts &mdash; what it does not have is spare request places, which is the "
                "subject of the rest of this section.", BODY))
    A(Spacer(1, 10))

    A(Paragraph("6.1 Where the limit is, and what it costs to move it", H2))
    A(Paragraph("Knowing the capacity of a server is worth less than knowing <i>what sets</i> "
                "it. Two days of measurement have now answered that, and the answer is a "
                "good one: <b>the limit is a setting, not the hardware.</b>", BODY))

    A(Paragraph("The server is configured to run 16 worker processes, each handling up to 8 "
                "requests at a time. That is 128 places. Below 128 simultaneous callers, "
                "everybody is served the moment they arrive. Above it, the next caller waits "
                "for a place to free up. The measurement follows the arithmetic precisely:",
                BODY))
    A(Spacer(1, 3))
    A(table([
        ["Simultaneous streams", "Time to accept a request", "What is happening"],
        ["120",  "75 ms",  "8 places spare"],
        ["126  (recommended)",  "79 ms",  "2 places spare &mdash; certified"],
        ["128",  "77 ms",  "exactly full &mdash; still immediate"],
        ["130",  "204 ms", "two callers waiting for a place"],
    ], [38*mm, 40*mm, fw-78*mm], align={1:"RIGHT"}))
    A(Spacer(1, 4))
    A(Paragraph("Acceptance time is flat at 75&ndash;77&nbsp;ms all the way to the 128th "
                "stream, then rises sharply with two more. Crucially the machine is <b>not "
                "short of compute</b> at any point: even at 130 streams each one is produced "
                "about 20% faster than it is played. The queue is full; the processor is not. "
                "<b>Raising the ceiling is a configuration change &mdash; more workers, or a "
                "larger batch each &mdash; rather than new engineering</b>, which makes it "
                "the cheapest improvement still available.", BODY))
    A(Spacer(1, 6))

    A(Paragraph("6.2 How efficient the system already is", H3))
    A(Paragraph("A further optimisation was tested on 20 September: storing the largest part "
                "of the model in 8-bit instead of 16-bit, halving how much data is read from "
                "memory for every frame of audio. On a quiet machine it does exactly what "
                "you would expect:", BODY))
    A(Spacer(1, 3))
    A(table([
        ["Streams sharing the work", "Time per frame", "Faster by"],
        ["1  (quiet server)",   "2.43 &rarr; 2.16 ms", "11%"],
        ["2",                   "3.34 &rarr; 3.17 ms", "5%"],
        ["4",                   "5.25 &rarr; 4.89 ms", "7%"],
        ["8  (busy server)",    "8.29 &rarr; 8.26 ms", "0.3%"],
    ], [46*mm, 34*mm, fw-80*mm], align={1:"CENTER",2:"RIGHT"}))
    A(Spacer(1, 4))
    A(Paragraph("The gain shrinks as the server fills, and that is <b>good news about the "
                "current design</b>. The model is read once per batch and shared across "
                "everyone in it, so with eight streams sharing a read the cost was already "
                "divided by eight before the optimisation was applied. In other words the "
                "batching is doing its job so well that there is very little memory traffic "
                "left to save. The configuration shipping today is already at that point, "
                "and was left unchanged.", BODY))
    A(Spacer(1, 3))
    A(Paragraph("This is a useful general result for anyone sizing a deployment: a model "
                "half the size is not half the cost when the cost is shared between "
                "concurrent users. Compression buys memory footprint and helps a lightly "
                "loaded machine; on a busy one, good batching has already collected that "
                "saving.", SMALL))
    A(Spacer(1, 10))

    A(Paragraph("7. Voice cloning", H2))
    A(Paragraph("PocketTTS supports zero-shot voice cloning from a short recording, and this "
                "runtime now exposes it end to end. It is cheap on this architecture because "
                "the voice condition and the generated audio live in the same latent space: "
                "a voice is simply the model's own state after it has listened to a "
                "reference. There is no speaker encoder and no adapter.", BODY))
    A(Paragraph("Verified on the production host by cloning the pack's own <i>alba</i> voice "
                "(CC-BY-4.0) from 10.3 seconds of audio and then having the clone speak a "
                "<b>different</b> sentence, compared against the real voice and against an "
                "unrelated voice as a control:", BODY))
    A(table([
        ["", "Mean pitch", "Median pitch", "Timbre match vs source"],
        ["Cloned voice (this host)", "127.4 Hz", "121.8 Hz", "0.986"],
        ["The original voice", "127.0 Hz", "121.5 Hz", "&mdash;"],
        ["An unrelated voice (control)", "86.3 Hz", "77.7 Hz", "0.922"],
    ], [52*mm, 26*mm, 28*mm, fw-106*mm], align={1:"RIGHT",2:"RIGHT",3:"RIGHT"}))
    A(Spacer(1, 4))
    A(Paragraph("Building the voice took 6.7 seconds for 10.3 seconds of reference. The "
                "result is an ordinary voice file in the model pack's own format, so a cloned "
                "voice is served by exactly the same path as a built-in one, streaming "
                "included. Consent is a required argument, checked before any audio is read "
                "and recorded inside the resulting file &mdash; the runtime will not clone a "
                "voice without one.", BODY))
    A(Paragraph("Samples 00-03 in the accompanying archive are this comparison. Listen to 01 "
                "against 02, then against 03.", SMALL))

    # ---------------------------------------------------------------- 7
    A(Paragraph("8. The audio in the accompanying archive", H2))
    A(Paragraph("Every clip was produced by the runtime described here. The streaming clips "
                "were captured <b>from the live server while it was carrying the full "
                "concurrent load</b>, not generated quietly on an idle machine &mdash; they "
                "are what a customer would have heard.", BODY))
    A(table([
        ["Folder", "What it contains"],
        ["streaming-under-load/", "Twelve clips taken from the HTTP streaming endpoint during "
                                  "a 30-minute run at full concurrency: three short, three "
                                  "medium, three conversational and three long, drawn at "
                                  "random from the 277-sentence bank. SENTENCES.txt has the "
                                  "text of each."],
        ["voice-cloning/", "The four clips behind section 6: the reference recording, the "
                           "cloned voice on a new sentence, the original voice on the same "
                           "sentence, and a different voice as a control."],
    ], [38*mm, fw-38*mm]))

    # ---------------------------------------------------------------- 8
    A(Paragraph("9. Maturity, and the scope of these numbers", H2))
    A(Paragraph("The runtime is an established open-source project: some two months of work, "
                "287 commits, seven tagged releases, and a CI that runs sanitisers and "
                "correctness suites on both Arm and x86 on every change. This particular "
                "engine is the newer part &mdash; the PocketTTS path landed on 12 September "
                "and the figures here were measured in the week since. Everything in this "
                "report is a real measurement on real hardware; what follows is simply the "
                "boundary of what has been measured so far, so a reader knows exactly what "
                "they are being told:", BODY))
    A(table([
        ["One language, one model", "English PocketTTS only. No other language pack, "
                                    "fine-tune or model family has been measured for serving."],
        ["Arm measured; x86 next", "Every number here is from Arm Neoverse-V2. The x86 "
                                  "build exists and passes the same correctness and sanitiser "
                                  "suites in CI on every change, and carries AVX-512 and VNNI "
                                  "code paths &mdash; it simply has not been through this "
                                  "concurrency campaign yet. That is the first item in "
                                  "section 10."],
        ["One host class", "A 32-core machine. Nothing here predicts a 16-core or 64-core "
                           "host: the right worker layout follows the model's own cost "
                           "structure and has to be re-screened per host."],
        ["The load generator shares the box", "The client that measured these numbers ran on "
                                              "the same 32 cores as the server, which costs the "
                                              "server some capacity and makes the true figure "
                                              "conservative rather than optimistic."],
        ["The exact figure between 120 and 128", "120 is certified over thirty minutes and "
                                   "128 was held for thirty minutes with three interruptions. "
                                   "The certifiable number is somewhere in between and each "
                                   "candidate is a thirty-minute run. We know where the limit "
                                   "comes from (section 6) and we have not yet pinned the "
                                   "precise figure."],
        ["Audio quality is not certified", "The codec runs in int8 for speed. That is a "
                                           "measured trade (roughly 29-33 dB signal-to-noise "
                                           "against 36-38 dB, with the spectral envelope "
                                           "essentially unchanged) and it is why the archive "
                                           "exists: judge it by ear."],
    ], [46*mm, fw-46*mm], header=False))

    # ---------------------------------------------------------------- 9
    A(Paragraph("10. What comes next", H2))
    A(table([
        ["Other languages and fine-tunes", "Run the same qualification on the other PocketTTS "
                                           "language packs and on dedicated fine-tunes."],
        ["Move the load generator off the host", "A second machine, so the measurement stops "
                                                 "competing with the thing it measures."],
        ["Settle 127", "126 is certified and 128 missed by three interruptions, so only "
                       "one number is still open. One thirty-minute run answers it, and "
                       "would also tell us whether 128's three were bad luck."],
        ["The same campaign on x86", "Every figure in this report is from an Arm server. The "
                                 "identical thirty-minute protocol on AMD and Intel hosts "
                                 "would tell a customer which processor to buy, and how many. "
                                 "The engine already carries the AVX-512 and VNNI code paths "
                                 "those machines need; they have never been put under this "
                                 "test."],
        ["Raise the slot count", "Section 6 shows the ceiling is 128 request places, not a "
                                 "speed limit. More workers, or a larger batch in each, is "
                                 "the next capacity increase and it is a configuration "
                                 "change rather than new engineering."],
        ["Reduce the cost of reading text", "The largest remaining lever on first-audio "
                                            "latency, and the only one that would raise the "
                                            "concurrency ceiling as well. It has never been "
                                            "optimised."],
    ], [56*mm, fw-56*mm], header=False))

    # ---------------------------------------------------------------- 10
    A(Paragraph("11. Reproducing this", H2))
    A(Paragraph("The runtime, the benchmark harness and the qualified configuration are all "
                "open source. The configuration is a committed file rather than a set of "
                "instructions, and the harness refuses to run if the environment contradicts "
                "it, so a repeat of this measurement cannot quietly become a measurement of "
                "something else.", BODY))
    A(Paragraph(
        "python3 tools/perf_profile.py best recommended<br/>"
        "python3 tools/serving_profile.py --profile axion-c4a-32c-pocket-en \\<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;--model models/pocket-en --server-bin build/cpu/mynah-tts-server",
        MONO))
    A(Spacer(1, 6))
    A(Rule(fw, 0.6))
    A(Paragraph("Measurements taken 18-19 September 2026 on GCP Axion c4a. Model "
                "kyutai/pocket-tts revision 4925226, CC-BY-4.0. Runtime mynah-tts, C11, "
                "open source. Prepared by Gabriele Mastrapasqua.", SMALL))
    doc.build(F)
    print("wrote", path)


if __name__ == "__main__":
    import sys
    build(sys.argv[1])
