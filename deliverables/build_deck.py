"""Builds the presentation: deck.html -> deck.pdf (Chrome headless) ->
per-slide PNGs (PyMuPDF) -> deck.pptx (full-bleed slide images + speaker
notes). Charts are the SVGs in figs/deck, inlined so they use the deck fonts.

  python build_deck.py
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figs", "deck")
OUT = os.path.join(HERE, "out")
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
TITLE = "Hardware-aware autotuning of a CUDA GEMM kernel with response surface methodology"


def svg(name):
    import re
    s = open(os.path.join(FIGS, name + ".svg"), encoding="utf-8").read()
    # matplotlib sizes SVGs in pt; one SVG unit was designed as one CSS px
    head, rest = s.split(">", 1)
    head = re.sub(r'\b(width|height)="([0-9.]+)pt"', r'\1="\2px"', head)
    return (head + ">" + rest).replace("<svg ", '<svg class="chart" ', 1)


CSS = """
@page { size: 1920px 1080px; margin: 0 }
:root { --paper:#F2EFE6; --paper2:#E8E3D5; --ink:#141414; --ink2:#3A3833; --muted:#8A857A;
        --rule:#141414; --hair:#CFC8B8; --accent:#E4472B; --blue:#2F4B66 }
* { box-sizing: border-box; margin: 0; padding: 0 }
html, body { background: var(--paper); -webkit-print-color-adjust: exact; print-color-adjust: exact }
body { font-family: 'Geist', Arial, sans-serif; color: var(--ink) }
.slide { width: 1920px; height: 1080px; position: relative; overflow: hidden; background: var(--paper);
         padding: 150px 120px 120px; break-after: page; display: flex; flex-direction: column }
.slide.dark { background: var(--ink); color: var(--paper) }
.run { position: absolute; left: 120px; right: 120px; top: 56px; display: flex; justify-content: space-between;
       font: 500 18px 'Geist Mono', monospace; letter-spacing: .08em; text-transform: uppercase; color: var(--ink);
       padding-bottom: 14px; border-bottom: 1.5px solid var(--rule) }
.run .sec { color: var(--accent) }
.dark .run { color: var(--paper); border-color: #5A574F }
.foot { position: absolute; left: 120px; right: 120px; bottom: 48px; display: flex; justify-content: space-between;
        font: 400 17px 'Geist Mono', monospace; letter-spacing: .06em; text-transform: uppercase; color: var(--muted) }
h1, h2 { font-family: 'Instrument Serif', Georgia, serif; font-weight: 400; letter-spacing: -0.01em }
h2 { font-size: 84px; line-height: 1.02; max-width: 1560px }
h2 em { font-style: italic; color: var(--accent) }
.dark h2 em { color: #F08A6E }
.lede { font-size: 30px; line-height: 1.45; color: var(--ink2); max-width: 1400px }
.label { font: 500 17px 'Geist Mono', monospace; letter-spacing: .08em; text-transform: uppercase; color: var(--muted) }
.num { font-family: 'Instrument Serif', Georgia, serif; font-size: 112px; line-height: .95; letter-spacing: -0.02em }
.num.md { font-size: 84px }
.num.acc { color: var(--accent) }
.small { font-size: 24px; line-height: 1.45; color: var(--ink2) }
.dark .small, .dark .lede { color: #CFCABE }
.row { display: flex; gap: 0 }
.col { display: flex; flex-direction: column }
.stack > * + * { border-top: 1.5px solid var(--hair); padding-top: 26px; margin-top: 26px }
.vrules > * + * { border-left: 1.5px solid var(--hair); padding-left: 48px; margin-left: 48px }
.chart { display: block; overflow: visible }
.cap { font: 400 16px 'Geist Mono', monospace; letter-spacing: .06em; text-transform: uppercase; color: var(--muted); margin-top: 10px }
table.ds { border-collapse: collapse; width: 100% }
table.ds th { font: 500 17px 'Geist Mono', monospace; letter-spacing: .08em; text-transform: uppercase; color: var(--muted);
              text-align: left; padding: 0 20px 14px 0; border-bottom: 1.5px solid var(--rule); font-weight: 500 }
table.ds td { font-size: 28px; padding: 18px 20px 18px 0; border-bottom: 1px solid var(--hair); color: var(--ink) }
table.ds td.m, table.ds th.m { font-family: 'Geist Mono', monospace; text-align: right; font-size: 26px }
table.ds th.m { font-size: 17px }
table.ds tr.hl td { color: var(--accent) }
table.ds.big td { font-size: 32px; padding-top: 26px; padding-bottom: 26px }
table.ds.big td.m { font-size: 30px }
.dark table.ds th { color: #A9A497; border-color: #F2EFE6 }
.dark table.ds td { color: var(--paper); border-color: #3E3C37 }
.box { border: 1.5px solid var(--rule); padding: 30px 32px }
.tag { font: 500 16px 'Geist Mono', monospace; letter-spacing: .08em; color: var(--accent) }
.sq { display: inline-block; width: 14px; height: 14px; background: var(--accent); margin-right: 14px; vertical-align: 2px }
"""

TOTAL = 18


def slide(n, sec, body, notes, dark=False, foot_left="V. Bharadwaj · 230957088 · MIT Manipal"):
    return {
        "n": n, "notes": notes,
        "html": f"""<section class="slide{' dark' if dark else ''}">
<div class="run"><span>DOE–RSM · CUDA GEMM autotuning · H200</span><span class="sec">{sec}</span></div>
{body}
<div class="foot"><span>{foot_left}</span><span>{n:02d} / {TOTAL:02d}</span></div>
</section>"""}


def chart_side(n, sec, head, chart, cap, stats, notes, chart_w=None):
    side = "".join(f'<div class="col" style="gap:10px"><span class="label">{l}</span>'
                   f'<span class="num md{" acc" if a else ""}">{v}</span><span class="small">{t}</span></div>'
                   for l, v, t, a in stats)
    body = f"""<h2>{head}</h2>
<div class="row" style="margin-top:44px;gap:64px;align-items:flex-start">
  <div class="col"><div>{svg(chart)}</div><div class="cap">{cap}</div></div>
  <div class="col stack" style="flex:1;padding-top:8px">{side}</div>
</div>"""
    return slide(n, sec, body, notes)


S = []

# 01 cover -------------------------------------------------------------------
S.append({"n": 1, "notes": "Introduce the project in one line: I took one GPU kernel, matrix multiplication, and asked whether classical statistics, Design of Experiments and response surface methodology, can find its fastest configuration without trying every option. Everything was measured on an NVIDIA H200, and every claim is checked against an exhaustive search of all 1,396 valid configurations.",
          "html": f"""<section class="slide" style="padding:120px">
<div class="run"><span>Minor project · Department of Mathematics</span><span class="sec">Manipal Institute of Technology · 2026–27</span></div>
<div style="flex:1"></div>
<div class="tag" style="font-size:20px"><span class="sq"></span>DESIGN OF EXPERIMENTS × GPU KERNELS</div>
<h1 style="font-size:132px;line-height:.98;margin-top:28px;max-width:1640px">Hardware-aware autotuning of a CUDA GEMM kernel with <em style="font-style:italic;color:#E4472B">response surface methodology</em></h1>
<div style="flex:1"></div>
<div class="row vrules" style="border-top:1.5px solid #141414;padding-top:26px">
  <div class="col" style="gap:8px"><span class="label">Author</span><span style="font-size:28px">Vaidheesh Bharadwaj</span></div>
  <div class="col" style="gap:8px"><span class="label">Roll no.</span><span style="font:400 28px 'Geist Mono'">230957088</span></div>
  <div class="col" style="gap:8px"><span class="label">Guide</span><span style="font-size:28px">Prof. Sunita Sharma</span></div>
  <div class="col" style="gap:8px"><span class="label">Hardware</span><span style="font-size:28px">NVIDIA H200 · CUDA 13.0</span></div>
</div>
</section>"""})

# 02 problem -----------------------------------------------------------------
S.append(slide(2, "§01 Context", """<h2>Same code, six numbers, <em>2.6×</em> apart in speed</h2>
<div class="row vrules" style="margin-top:96px">
  <div class="col" style="gap:18px;flex:1"><span class="label">Untuned kernel · TFLOPS</span><span class="num" style="font-size:200px">13.2</span><span class="small">textbook tiles<br><span style="font-family:'Geist Mono'">(64, 64, 8, 4, 4, 1)</span></span></div>
  <div class="col" style="gap:18px;flex:1"><span class="label">Tuned by DOE–RSM · TFLOPS</span><span class="num acc" style="font-size:200px">34.3</span><span class="small">selected configuration<br><span style="font-family:'Geist Mono'">(128, 128, 32, 8, 8, 8)</span></span></div>
  <div class="col" style="gap:18px;flex:1"><span class="label">Legal configurations</span><span class="num" style="font-size:200px">1,396</span><span class="small">of 2,304; each needs its own<br>compile and benchmark</span></div>
</div>
<div style="flex:1"></div>
<p class="lede">Exhaustive search is slow and grows with every new factor; machine-learning autotuners need many samples. <b style="color:#141414;font-weight:600">Can classical statistics find a near-best configuration from a small sample?</b></p>""",
                 "A GEMM kernel has tunable compile-time parameters: tile sizes, register tiling and loop unrolling. The same source code runs at 13.2 TFLOPS with textbook settings and 34.3 with tuned ones, a factor of 2.6. Of 2,304 combinations, 1,396 obey the hardware limits, and every one needs its own compile and benchmark. The project asks whether Design of Experiments can find a near-best configuration from a small sample."))

# 03 questions ---------------------------------------------------------------
rq = [("RQ1", "Screening", "Can a fractional factorial design separate the parameters that matter from noise?"),
      ("RQ2", "Model", "Does a second-order response surface capture the GPU’s non-linear behaviour?"),
      ("RQ3", "Efficiency", "With the same number of trials, does DOE–RSM beat random search?"),
      ("RQ4", "Optimality", "How close does it get to the exhaustive-search optimum?"),
      ("RQ5", "Context", "What share of cuBLAS throughput does the tuned kernel reach?")]
rows = "".join(f'<tr><td style="width:150px;font:500 26px \'Geist Mono\';color:#E4472B">{a}</td><td style="width:260px;font:500 20px \'Geist Mono\';letter-spacing:.06em;text-transform:uppercase;color:#8A857A">{b}</td><td>{c}</td></tr>' for a, b, c in rq)
S.append(slide(3, "§01 Context", f"""<h2>Five questions, each answered by measurement</h2>
<table class="ds big" style="margin-top:64px">{rows}</table>
<div style="flex:1"></div>
<p class="lede"><span class="sq"></span>The main claim is <b style="color:#141414;font-weight:600">RQ3, sample efficiency</b>. RQ5 is context, not the goal.</p>""",
               "The first two questions come straight from the synopsis: does screening work, and does a quadratic surface model the hardware. The third and fourth test whether the method is actually efficient, against random search and against the true optimum. The fifth puts the result in context against NVIDIA’s own library. The main claim of the project is RQ3, not RQ5."))

# 04 approach ----------------------------------------------------------------
stages = [("01", "Screen", "2⁶⁻¹ fractional factorial · 32 configs · ANOVA"),
          ("02", "Model", "central composite design · 48 points · quadratic OLS"),
          ("03", "Optimise", "constrained SQP + enumeration of the fitted model"),
          ("04", "Validate", "15 candidates measured · fastest is chosen")]
cells = []
for i, (k, t, d) in enumerate(stages):
    cells.append(f'<div class="box col" style="flex:1;gap:14px;min-height:380px;padding:40px 36px{";background:#141414;color:#F2EFE6" if i == 3 else ""}">'
                 f'<span class="tag">{k}</span><span style="font-family:\'Instrument Serif\';font-size:80px;line-height:1">{t}</span>'
                 f'<span class="small" style="{"color:#CFCABE" if i == 3 else ""}">{d}</span></div>')
    if i < 3:
        cells.append('<div style="align-self:center;font:400 40px \'Geist Mono\';color:#8A857A;padding:0 14px">→</div>')
base = [("Exhaustive search", "all 1,396 configurations: the true optimum"),
        ("Random search", "same budget, repeated 10,000 times"),
        ("cuBLAS", "NVIDIA’s library, FP32, interleaved")]
bcells = "".join(f'<div class="col" style="gap:8px;flex:1"><span style="font-size:30px;font-weight:500">{a}</span><span class="small">{b}</span></div>' for a, b in base)
S.append(slide(4, "§02 Method", f"""<h2>A four-stage pipeline, judged against three baselines</h2>
<div class="row" style="margin-top:56px;align-items:stretch">{''.join(cells)}</div>
<div style="flex:1"></div><div class="label" style="margin-bottom:18px">Judged against</div>
<div class="row vrules" style="border-top:1.5px solid #141414;padding-top:24px">{bcells}</div>""",
               "The pipeline follows the synopsis objectives. Screening finds which factors matter. A central composite design fits a quadratic response surface. SQP and an exhaustive pass over the model propose the best configurations. Then the proposals are measured and the fastest measured one is chosen. The bottom row is how the method is judged: the true optimum from exhaustive search, random search with the same budget, and cuBLAS as an outside reference."))

# 05 space -------------------------------------------------------------------
fac = [("BM", "block tile rows · shared memory", "32 64 128 256"),
       ("BN", "block tile columns · shared memory", "32 64 128 256"),
       ("BK", "K-depth of each tile", "8 16 32 64"),
       ("TM", "rows per thread · registers", "2 4 8"),
       ("TN", "columns per thread · registers", "2 4 8"),
       ("UNROLL", "inner-loop unroll factor", "1 2 4 8")]
frows = "".join(f'<tr><td style="font:500 26px \'Geist Mono\'">{a}</td><td>{b}</td><td class="m">{c}</td></tr>' for a, b, c in fac)
S.append(slide(5, "§02 Method", f"""<h2>Six compile-time factors, fenced in by the hardware</h2>
<div class="row" style="margin-top:52px;gap:72px;align-items:flex-start">
  <table class="ds" style="width:1060px"><tr><th>Factor</th><th>Meaning</th><th class="m">Levels</th></tr>{frows}</table>
  <div class="col" style="flex:1;gap:12px">
    <span class="label">Feasible</span><span class="num acc">1,396</span><span class="small">of 2,304 combinations</span>
    <div class="col" style="gap:6px;margin-top:24px;border-top:1.5px solid #CFC8B8;padding-top:22px">
      <span class="small">over 1,024 threads · <b style="font-family:'Geist Mono';font-weight:500">528</b></span>
      <span class="small">over 48 KiB shared memory · <b style="font-family:'Geist Mono';font-weight:500">168</b></span>
      <span class="small">tile-load overflow · <b style="font-family:'Geist Mono';font-weight:500">140</b></span>
      <span class="small">fractional load stride · <b style="font-family:'Geist Mono';font-weight:500">72</b></span>
    </div>
    <span class="small" style="margin-top:20px">Filter checked against the compiler on all 2,304: no disagreements.</span>
  </div>
</div>""",
               "The kernel is a standard 2D block-tiled matrix multiply in FP32 on CUDA cores. Six numbers are injected at compile time. Not every combination is legal: the GPU caps threads per block and shared memory, and the tile-loading code needs the thread count to fit the tiles. That constraint set is objective 1 of the synopsis. I checked the Python version of the rules against the real compiler on all 2,304 combinations, and they agree on every one."))

# 06 rigor -------------------------------------------------------------------
rig = [("Clock locked", "1830", "MHz on all 2,001 measurements; block-to-block variation 0.033%"),
       ("Randomised blocks", "1.86", "Durbin–Watson on run order: no drift (a fixed-order pilot gave 0.60)"),
       ("Shared GPU", "37", "measurements overlapped other users’ jobs, were detected and re-run"),
       ("Every output checked", "0", "configurations disagree with cuBLAS; a planted bug was caught at 0.044")]
rc = "".join(f'<div class="col" style="gap:12px;padding:40px 0 44px;border-top:1.5px solid #141414"><span class="label">{a}</span><span class="num" style="font-size:130px">{b}</span><span class="small">{c}</span></div>' for a, b, c in rig)
S.append(slide(6, "§02 Method", f"""<h2>Rules fixed in advance; every number checked</h2>
<div style="display:grid;grid-template-columns:1fr 1fr;column-gap:96px;margin-top:40px">{rc}</div>""",
               "Statistics are only as good as the measurements. Before collecting final data I wrote a contract fixing every rule: tolerances, the factor-retention rule, the model adequacy test, the random-search budget. The GPU clock was locked. Runs were randomised, which removed the drift a pilot run had shown. The H200 is shared, so I detected other users’ activity during each measurement and repeated contested runs. And every configuration’s output was checked against cuBLAS, with a deliberately broken kernel proving the check catches errors of the expected size."))

# 07 screening ---------------------------------------------------------------
S.append(chart_side(7, "§03 Screening · RQ1", "All six factors matter; <em>unrolling</em> matters most", "effects",
                    "Fig. 1 — Effects at 4096³, 2⁶⁻¹ design, 3 randomised blocks",
                    [("Model fit", "R² 0.959", "32 configurations × 3 blocks; replicate noise 0.04%", False),
                     ("Largest effect", "+7.9", "TFLOPS from unrolling 1 → 8; 54% of all variance", True),
                     ("Retained", "6 of 6", "by the pre-set rule p &lt; 0.05; weakest BM at p = 0.0026", False)],
                    "The answer to RQ1 is yes. A 32-run half-fraction design, each run replicated three times in random order, separates the factors cleanly: R-squared 0.96 and replicate noise around 0.04 percent. Every factor is significant. Unrolling the inner loop is by far the largest effect, adding 7.9 TFLOPS on average, then register tiling in the M direction. Interactions matter too: the biggest one, BN times UNROLL, is negative. If asked: residual variance is not constant, but refitting on log latency gives the same significant factors."))

# 08 shapes ------------------------------------------------------------------
S.append(chart_side(8, "§03 Screening · shape", "What matters changes for <em>small</em> matrices", "shapes",
                    "Fig. 2 — Share of variance per factor at three matrix sizes",
                    [("4096³ vs 2048³", "τ = 1.0", "identical factor ranking", False),
                     ("4096³ vs 1024³", "τ = −0.07", "BM rises 1.9% → 23.6%; TM falls 22.7% → 3.2%", True),
                     ("Likely cause", "64 / 132", "thread blocks vs SMs with 128 × 128 tiles at 1024³: tile size sets GPU fill", False)],
                    "I repeated the screening at two smaller sizes. At 2048 cubed the factor ranking is identical to 4096, Kendall tau of 1. At 1024 cubed it is unrelated, tau near zero: the block-tile height BM becomes the second most important factor and the register tile TM drops to last. My explanation, stated as likely rather than proven: with big tiles a 1024 matrix produces only 64 thread blocks, fewer than the H200’s 132 SMs, so tile size decides how much of the GPU is busy. Tuned settings do not transfer to small problems."))

# 09 rsm gate ----------------------------------------------------------------
mrows = [("Quadratic · pre-registered", "0.956", "7.8%", "fails", "10.6%", True),
         ("Quadratic on log latency", "0.961", "7.0%", "fails", "11.4%", False),
         ("Quadratic + 3-way terms", "0.982", "10.1%", "fails", "32.6%", False)]
mr = "".join(f'<tr{" class=hl" if h else ""}><td>{a}</td><td class="m">{b}</td><td class="m">{c}</td><td class="m">{d}</td><td class="m">{e}</td></tr>' for a, b, c, d, e, h in mrows)
S.append(slide(9, "§04 Response surface · RQ2", f"""<h2>A quadratic surface is close, but <em>statistically inadequate</em></h2>
<table class="ds big" style="margin-top:64px"><tr><th>Model · 6-factor CCD · 144 runs</th><th class="m">R²</th><th class="m">Lack of fit</th><th class="m">Gate · limit 5%</th><th class="m">Held-out error</th></tr>{mr}</table>
<p class="lede" style="margin-top:72px">Replicate noise is about 0.05%, so even small systematic misfit is detectable. The fitted quadratic is a saddle with no interior maximum.</p>""",
               "This is the honest answer to RQ2. A six-factor central composite design gave a quadratic model with R-squared 0.956, which looks good. But I fixed an adequacy test in advance: lack of fit had to be under 5% of the model mean square. It came in at 7.8%, so it fails, and so does the log-latency fallback. Held-out error is the median error predicting a configuration the model never saw. Because the measurements are so precise, the test sees systematic misfit a noisier experiment would hide. I report this rather than quietly optimising the model anyway."))

# 10 rsm why -----------------------------------------------------------------
S.append(slide(10, "§04 Response surface · why", f"""<h2>Three-way interactions and compiler <em>jumps</em></h2>
<div class="row" style="margin-top:44px;gap:80px;align-items:flex-start">
  <div class="col stack" style="width:760px">
    <div class="col" style="gap:10px"><span class="label">3-way interactions</span><span class="num md">6 of 10</span><span class="small">estimable pairs significant; strongest BM·BN·BK = TM·TN·UNROLL, p&nbsp;=&nbsp;10⁻¹³</span></div>
    <div class="col" style="gap:10px"><span class="label">Adding them</span><span class="num md acc">10.6 → 32.6%</span><span class="small">held-out error: explains 59% of the misfit but overfits, with 40 parameters on 45 configurations</span></div>
  </div>
  <div class="col"><div>{svg('unroll')}</div><div class="cap">Fig. 3 — Unrolling by 4 is slower than by 2</div></div>
</div>""",
               "Where does the misfit come from? Eighty percent of it is inside the main factorial region, not at the extremes. Two causes. First, three-way interactions: a large register tile, a deep K tile and full unrolling together give a boost no pair of factors explains. Adding those terms, as a follow-up after the gate failed, confirms they are significant, but the model then has 40 parameters for 45 configurations and overfits badly. Second, some behaviour is not smooth at all: here unrolling by 4 is slower than by 2. That is a compiler code-generation effect, and no polynomial can follow it."))

# 11 extrapolation -----------------------------------------------------------
S.append(slide(11, "§04 Response surface · ground truth", f"""<h2>Trustworthy only inside the region it was fitted on</h2>
<div style="margin-top:30px">{svg('pred_vs_measured')}</div>
<div class="row vrules" style="margin-top:22px">
  <div class="col" style="gap:4px;flex:1"><span class="label">Inside fitted region</span><span style="font:400 40px 'Instrument Serif'">≈ 5% median error</span></div>
  <div class="col" style="gap:4px;flex:1"><span class="label">Outside</span><span style="font:400 40px 'Instrument Serif'">≈ 22%, some predictions negative</span></div>
  <div class="col" style="gap:4px;flex:1"><span class="label">Best guess outside</span><span style="font:400 40px 'Instrument Serif';color:#E4472B">37.1 predicted, 30.5 measured</span></div>
</div>""",
               "Because I measured every configuration, I can check the models everywhere. Blue points are inside the region the design actually tested together; there both models are off by about 5 percent. Grey points are outside; there the error is around 22 percent and the model even predicts negative throughput. Its favourite configuration outside the region was predicted at 37.1 TFLOPS and measured at 30.5. That is why optimisation was restricted to the tested region, a decision fixed before seeing this data."))

# 12 validation --------------------------------------------------------------
vrows = [("(128, 128, 32, 8, 8, 8) · selected", "34.32", "1", "3", "11", True),
         ("(128, 128, 32, 8, 8, 4) · augmented #1", "33.08", "2", "1", "3", False),
         ("(64, 128, 32, 8, 4, 4) · quadratic #1", "31.17", "8", "8", "1", False)]
vr = "".join(f'<tr{" class=hl" if h else ""}><td style="font-family:\'Geist Mono\';font-size:25px">{a}</td><td class="m">{b}</td><td class="m">{c}</td><td class="m">{d}</td><td class="m">{e}</td></tr>' for a, b, c, d, e, h in vrows)
S.append(slide(12, "§05 Results · validation", f"""<h2>Measurement, not the model, <em>picked the winner</em></h2>
<p class="lede" style="margin-top:28px">SQP and enumeration proposed 15 candidates from both models; each was measured in 5 randomised blocks.</p>
<table class="ds big" style="margin-top:44px"><tr><th>Configuration</th><th class="m">Measured TFLOPS</th><th class="m">Measured rank</th><th class="m">Augmented rank</th><th class="m">Quadratic rank</th></tr>{vr}</table>
<p class="lede" style="color:#141414;margin-top:72px;font-size:36px">Trusting either model’s own first choice would have cost <b style="color:#E4472B;font-weight:600">3.6%</b> (augmented) or <b style="color:#E4472B;font-weight:600">9.2%</b> (quadratic).</p>""",
               "After the gate failed, I used the models only to propose candidates: constrained SQP on the continuous surface, plus exhaustive evaluation of the model over the 192 legal configurations in the tested region. Fifteen candidates from both models were then measured five times each. The fastest, 128, 128, 32, 8, 8, 8 at 34.3 TFLOPS, was ranked only third by the augmented model and eleventh by the quadratic. Each model’s own favourite measured 4 to 9 percent slower. In this space a response surface is a good proposal generator, but the decision has to be measured."))

# 13 efficiency --------------------------------------------------------------
S.append(chart_side(13, "§05 Results · RQ3, RQ4", "59 evaluations, <em>0.17%</em> from the optimum", "sample_efficiency",
                    "Fig. 4 — Best configuration found vs evaluations, 4096³",
                    [("Rank", "#2 of 1,396", "0.17% slower than the exhaustive optimum, 34.40 TFLOPS", True),
                     ("Random search", "424", "median evaluations to match it: about 7× the budget", False),
                     ("Same budget", "7.8%", "of 10,000 random trials did as well", False)],
                    "This is the main result, RQ3 and RQ4. DOE–RSM used 59 distinct configurations and its choice is the second-best of all 1,396, 0.17 percent behind the true optimum. With exactly the same 59-configuration budget, repeated 10,000 times, random search is 7.3 percent behind at the median and matches DOE–RSM in only 7.8 percent of trials. To match it, random search needs a median of 424 evaluations. One honest note: the winning configuration is a corner of the screening design and was first measured at evaluation 7. The surface and validation confirmed it rather than discovered it, so the credit goes to the design as a whole."))

# 14 cublas ------------------------------------------------------------------
S.append(chart_side(14, "§05 Results · RQ5", "72% of cuBLAS, <em>2.6×</em> the untuned kernel", "final_comparison",
                    "Fig. 5 — Median of 900 interleaved launches each, pedantic FP32",
                    [("Share of cuBLAS", "72.0%", "34.35 vs 47.70 TFLOPS", True),
                     ("Like for like", "FP32", "TF32 off; outputs identical bit for bit", False),
                     ("Ceiling", "72.2%", "even the best of all 1,396 configurations: the gap is the kernel design", False)],
                    "RQ5. The tuned kernel reaches 72 percent of cuBLAS and is 2.6 times faster than the untuned default. The comparison is fair: cuBLAS ran in pedantic FP32 mode with TF32 disabled, the outputs are bit-identical, and the timings come from 900 launches of each, interleaved in the same process. The remaining 28 percent is not a tuning failure: even the exhaustive best stops at 72.2 percent. cuBLAS uses techniques this kernel does not have, such as vectorised loads, asynchronous double buffering and warp-level tiling. The project never claims to beat cuBLAS."))

# 15 spilling ----------------------------------------------------------------
S.append(chart_side(15, "§05 Results · registers", "Register spilling is rare here and <em>not</em> what limits speed", "registers",
                    "Fig. 6 — Registers per thread vs throughput, all 1,396 configurations",
                    [("Spilling", "28", "of 1,396 configurations spill registers at all", False),
                     ("Median TFLOPS", "21.6 vs 19.0", "spilling vs not: spillers are not slower", True),
                     ("Rank correlation", "ρ +0.47", "more registers, more speed: tile reuse beats occupancy", False)],
                    "The synopsis expected the response surface to capture register spilling. I recorded the compiler’s register and spill report for every configuration. Spilling is rare, 28 of 1,396, and those configurations are not slower than the rest. The real pattern is that more registers per thread mean more speed, a rank correlation of 0.47, because a bigger register tile reuses each loaded value more often, which outweighs running fewer threads."))

# 16 limitations -------------------------------------------------------------
lim = [("One deviation", "After the gate failed, the models only proposed candidates; measurement chose."),
       ("Winner from screening", "The selected configuration is a corner of the screening design; later stages confirmed it."),
       ("Optimum out of reach", "The true best (BK = 64) lies outside the region the models were allowed to search."),
       ("Narrow scope", "One GPU, one kernel family, FP32 on CUDA cores, tuned at 4096³ only."),
       ("Shared, clamped GPU", "Other users’ jobs were detected and re-run; the clock held at 1830, not 1980 MHz."),
       ("Discrete lattice", "Power-of-two levels forced design points to snap; the CCD centre is not the true centre.")]
lc = "".join(f'<div class="col" style="gap:14px;padding:44px 0 60px;border-top:1.5px solid #141414"><span class="tag">{i + 1:02d}</span><span style="font-size:38px;font-weight:500">{a}</span><span class="small">{b}</span></div>' for i, (a, b) in enumerate(lim))
S.append(slide(16, "§06 Discussion", f"""<h2>What the results do not show</h2>
<div style="display:grid;grid-template-columns:1fr 1fr 1fr;column-gap:64px;margin-top:40px">{lc}</div>""",
               "Six limits worth stating before anyone asks. One deviation from the pre-registered plan, made after the adequacy test failed and documented. The winning configuration came from the screening design, so the later stages confirmed rather than found it. The true optimum used BK of 64, outside the region the models were allowed to search, a restriction chosen to avoid extrapolation. The scope is one GPU, one kernel and one main matrix size. The GPU was shared, handled by detection and retries. And the power-of-two lattice meant design points had to snap to legal values."))

# 17 conclusions (dark) ------------------------------------------------------
crow = [("RQ1", "Screening", "Yes: all six factors separated (R² 0.959); importance shifts at 1024³"),
        ("RQ2", "Quadratic", "Close but inadequate: lack of fit 7.8% against a 5% limit"),
        ("RQ3", "Efficiency", "Yes: 59 evaluations against a median of 424 for random search"),
        ("RQ4", "Optimality", "0.17% from the exhaustive optimum, #2 of 1,396"),
        ("RQ5", "cuBLAS", "72.0% of FP32 throughput; 2.6× the untuned kernel")]
cr = "".join(f'<tr><td style="width:120px;font:500 24px \'Geist Mono\';color:#F08A6E">{a}</td><td style="width:240px;font:500 19px \'Geist Mono\';letter-spacing:.06em;text-transform:uppercase;color:#A9A497">{b}</td><td>{c}</td></tr>' for a, b, c in crow)
S.append(slide(17, "§06 Conclusions", f"""<h2 style="color:#F2EFE6">Few experiments, a near-best kernel, <em>an honest model</em></h2>
<table class="ds big" style="margin-top:56px">{cr}</table>
<div style="flex:1"></div>
<p class="lede">Use the response surface to propose; let measurement decide.</p>""",
               "One line per question. Screening works and separates all six factors, though their importance shifts for small matrices. A quadratic surface gets close but fails a strict adequacy test, because the GPU has three-way interactions and abrupt compiler effects. The full design is still very sample-efficient: 59 evaluations to land 0.17 percent from the true optimum, where random search needs about 424. And the tuned kernel reaches 72 percent of cuBLAS. The practical message: use the response surface to propose candidates, and let measurement decide.",
               dark=True))

# 18 closing -----------------------------------------------------------------
nxt = [("Larger design", "A full 2⁶ core to estimate 3-way interactions without overfitting"),
       ("Per-shape tuning", "Importance shifts below 2048³, so tune each matrix size"),
       ("Stronger baseline", "Head-to-head with Bayesian optimisation, as modern autotuners use"),
       ("Richer kernel", "Vectorised loads and double buffering as categorical factors")]
nc = "".join(f'<div class="col" style="gap:10px;flex:1"><span class="tag">{i + 1:02d}</span><span style="font-size:30px;font-weight:500">{a}</span><span class="small">{b}</span></div>' for i, (a, b) in enumerate(nxt))
S.append(slide(18, "§06 Next", f"""<h2>Where this goes next</h2>
<div class="row vrules" style="margin-top:48px;border-top:1.5px solid #141414;padding-top:30px">{nc}</div>
<div style="flex:1"></div>
<div style="font-family:'Instrument Serif';font-size:150px;line-height:1">Thank you<span style="color:#E4472B">.</span></div>
<div class="label" style="margin-top:14px">Questions welcome</div>""",
               "Four natural extensions. A full 2 to the 6 factorial core would let the three-way interactions be estimated without the overfitting we saw. Because factor importance changes for small matrices, tuning should be done per shape. A comparison with Bayesian optimisation, which modern autotuners use, would test the efficiency claim against a stronger baseline than random search. And adding kernel features such as vectorised loads as factors would raise the 72 percent ceiling. Thank you, happy to take questions."))


def build():
    os.makedirs(OUT, exist_ok=True)
    html = ('<!doctype html><html><head><meta charset="utf-8"><title>' + TITLE + '</title>'
            '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1'
            '&family=Geist:wght@300..700&family=Geist+Mono:wght@400;500&display=block">'
            f'<style>{CSS}</style></head><body>' + "".join(s["html"] for s in S) + '</body></html>')
    hp = os.path.join(OUT, "deck.html")
    open(hp, "w", encoding="utf-8").write(html)
    pdf = os.path.join(OUT, "Vaidheesh_Bharadwaj_DOE-RSM_CUDA_Autotuning_Slides.pdf")
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    "--virtual-time-budget=20000", f"--print-to-pdf={pdf}",
                    "file:///" + hp.replace("\\", "/")], capture_output=True)
    import fitz
    doc = fitz.open(pdf)
    pngdir = os.path.join(OUT, "deck_png")
    os.makedirs(pngdir, exist_ok=True)
    for i, page in enumerate(doc):
        page.get_pixmap(matrix=fitz.Matrix(2 * 1920 / page.rect.width, 2 * 1920 / page.rect.width)).save(
            os.path.join(pngdir, f"slide_{i + 1:02d}.png"))
    print(f"pdf pages: {len(doc)} (expected {len(S)}), page size {doc[0].rect}")

    from pptx import Presentation
    from pptx.util import Emu
    prs = Presentation()
    prs.slide_width, prs.slide_height = Emu(12192000), Emu(6858000)
    blank = prs.slide_layouts[6]
    for i, s in enumerate(S):
        sl = prs.slides.add_slide(blank)
        sl.shapes.add_picture(os.path.join(pngdir, f"slide_{i + 1:02d}.png"), 0, 0,
                              width=prs.slide_width, height=prs.slide_height)
        sl.notes_slide.notes_text_frame.text = s["notes"]
    prs.save(os.path.join(OUT, "Vaidheesh_Bharadwaj_DOE-RSM_CUDA_Autotuning_Slides.pptx"))
    print("pptx written")


if __name__ == "__main__":
    build()
