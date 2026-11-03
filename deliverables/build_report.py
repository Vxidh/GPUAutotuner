"""Builds the written report: report.html -> A4 PDF (Chrome headless) and
per-page PNGs for inspection. Figures are the SVGs in figs/report, inlined
so they use the report fonts.

  python build_report.py
"""
import os
import re
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figs", "report")
OUT = os.path.join(HERE, "out")
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
PDF_NAME = "Vaidheesh_Bharadwaj_DOE-RSM_CUDA_Autotuning_Report.pdf"


def svg(name):
    s = open(os.path.join(FIGS, name + ".svg"), encoding="utf-8").read()
    head, rest = s.split(">", 1)
    head = re.sub(r'\b(width|height)="([0-9.]+)pt"', r'\1="\2px"', head)
    return (head + ">" + rest).replace("<svg ", '<svg class="chart" ', 1)


FIG_N = [0]


def fig(name, caption):
    FIG_N[0] += 1
    return (f'<figure>{svg(name)}<figcaption><span class="fn">Figure {FIG_N[0]}</span>{caption}</figcaption></figure>')


TAB_N = [0]


def table(header, rows, caption, num_cols=(), hl_first=False, widths=None):
    TAB_N[0] += 1
    th = "".join(f'<th class="{"m" if i in num_cols else ""}"{f" style=\"width:{widths[i]}\"" if widths else ""}>{h}</th>'
                 for i, h in enumerate(header))
    trs = ""
    for r, row in enumerate(rows):
        tds = "".join(f'<td class="{"m" if i in num_cols else ""}">{c}</td>' for i, c in enumerate(row))
        trs += f'<tr{" class=hl" if hl_first and r == 0 else ""}>{tds}</tr>'
    return (f'<div class="tbl"><div class="tcap"><span class="fn">Table {TAB_N[0]}</span>{caption}</div>'
            f'<table><tr>{th}</tr>{trs}</table></div>')


CSS = """
@page { size: A4; margin: 24mm 19mm 22mm 19mm;
  @top-left { content: "DOE–RSM · CUDA GEMM AUTOTUNING"; font: 500 7pt 'Geist Mono', monospace; letter-spacing: .08em; color: #8A857A; vertical-align: bottom; padding-bottom: 4mm }
  @top-right { content: "V. BHARADWAJ · 230957088"; font: 500 7pt 'Geist Mono', monospace; letter-spacing: .08em; color: #8A857A; vertical-align: bottom; padding-bottom: 4mm }
  @bottom-right { content: counter(page); font: 500 8pt 'Geist Mono', monospace; color: #141414 } }
@page :first { margin: 0; @top-left { content: none } @top-right { content: none } @bottom-right { content: none } }
:root { --ink:#141414; --ink2:#2E2C28; --muted:#8A857A; --hair:#D3CCBD; --accent:#E4472B; --blue:#2F4B66; --paper:#F2EFE6 }
* { box-sizing: border-box; margin: 0; padding: 0 }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact }
body { font-family: 'Geist', Arial, sans-serif; font-size: 9.6pt; line-height: 1.55; color: var(--ink2); font-weight: 400 }
.cover { width: 210mm; height: 297mm; background: var(--paper); padding: 22mm 20mm 20mm; display: flex; flex-direction: column; break-after: page }
.cover .run { display: flex; justify-content: space-between; font: 500 7.5pt 'Geist Mono'; letter-spacing: .08em; text-transform: uppercase; border-bottom: 1px solid var(--ink); padding-bottom: 3mm; color: var(--ink) }
.cover .run span:last-child { color: var(--accent) }
.cover h1 { font-family: 'Instrument Serif', Georgia, serif; font-weight: 400; font-size: 40pt; line-height: 1.0; letter-spacing: -0.01em; color: var(--ink); margin-top: 6mm }
.cover h1 em { color: var(--accent) }
.cover .tag { font: 500 8pt 'Geist Mono'; letter-spacing: .1em; color: var(--accent); text-transform: uppercase }
.cover .meta { display: grid; grid-template-columns: repeat(4, 1fr); border-top: 1px solid var(--ink); padding-top: 3.5mm; gap: 4mm }
.cover .meta div span { display: block }
.k { font: 500 6.8pt 'Geist Mono'; letter-spacing: .1em; text-transform: uppercase; color: var(--muted); margin-bottom: 1mm }
.abstract { margin-top: 10mm; font-size: 10pt; line-height: 1.6; color: var(--ink2); max-width: 150mm }
.abstract b { color: var(--ink); font-weight: 600 }
.keyfigs { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0; margin-top: 9mm; border-top: 1px solid var(--hair) }
.keyfigs div { padding: 4mm 4mm 0 0 }
.keyfigs .v { font-family: 'Instrument Serif', Georgia, serif; font-size: 30pt; line-height: 1; color: var(--ink) }
.keyfigs .v.acc { color: var(--accent) }
.keyfigs .d { font-size: 8pt; line-height: 1.4; color: var(--muted); margin-top: 1.5mm }
h2 { font-family: 'Instrument Serif', Georgia, serif; font-weight: 400; font-size: 21pt; line-height: 1.1; color: var(--ink);
     margin: 9mm 0 3mm; break-after: avoid; display: flex; gap: 4mm; align-items: baseline }
h2 .sn { font: 500 9pt 'Geist Mono'; color: var(--accent); letter-spacing: .06em }
h2.first { margin-top: 0 }
h3 { font-size: 10pt; font-weight: 600; color: var(--ink); margin: 4.5mm 0 1.5mm; break-after: avoid }
p { margin: 0 0 2.6mm; text-align: left; hyphens: auto }
p.lead { font-size: 10.6pt; line-height: 1.5; color: var(--ink); border-left: 2px solid var(--accent); padding-left: 3.5mm; margin-bottom: 4mm }
b { font-weight: 600; color: var(--ink) }
ul, ol { margin: 0 0 3mm 5mm } li { margin-bottom: 1.2mm; padding-left: 1mm }
code, .mono { font-family: 'Geist Mono', monospace; font-size: 8.6pt }
figure { margin: 4mm 0 5mm; break-inside: avoid }
figure .chart { display: block; max-width: 100%; height: auto; overflow: visible }
figcaption, .tcap { font: 400 7.6pt 'Geist Mono'; letter-spacing: .03em; color: var(--muted); margin-top: 2mm; line-height: 1.45 }
.fn { color: var(--accent); font-weight: 500; text-transform: uppercase; letter-spacing: .08em; margin-right: 2.5mm }
.tbl { margin: 4mm 0 5mm; break-inside: avoid }
.tcap { margin: 0 0 2mm }
table { border-collapse: collapse; width: 100%; font-size: 8.8pt }
th { font: 500 6.9pt 'Geist Mono'; letter-spacing: .08em; text-transform: uppercase; color: var(--muted); text-align: left;
     padding: 0 3mm 1.6mm 0; border-bottom: 1px solid var(--ink); vertical-align: bottom }
td { padding: 1.7mm 3mm 1.7mm 0; border-bottom: 0.5pt solid var(--hair); color: var(--ink); vertical-align: top }
td.m, th.m { text-align: right; font-family: 'Geist Mono', monospace; font-size: 8.3pt }
th.m { font-size: 6.9pt }
tr.hl td { color: var(--accent) }
.note { font-size: 8.6pt; color: var(--muted) }
.refs li { font-size: 8.8pt; margin-bottom: 1.6mm }
.pb { break-before: page }
"""

BODY = []
A = BODY.append

# ------------------------------------------------------------------ cover --
A(f"""<section class="cover">
<div class="run"><span>Minor project · Department of Mathematics</span><span>Manipal Institute of Technology · 2026–27</span></div>
<div style="height:30mm"></div>
<div class="tag">Design of experiments × GPU kernels</div>
<h1>Hardware-aware autotuning of a CUDA GEMM kernel with <em>response surface methodology</em></h1>
<div class="abstract"><span class="k" style="display:block;margin-bottom:2mm">Abstract</span>
A single parameterised CUDA matrix-multiplication kernel, adapted from a public tutorial kernel, was tuned over six compile-time factors on an NVIDIA H200, using a
2<sup>6−1</sup> fractional-factorial screen, a 6-factor central composite design, a second-order response surface, constrained
sequential quadratic programming and measured validation. Every result is checked against an exhaustive sweep of all 1,396 feasible
configurations. <b>The pipeline selected the second-best configuration of 1,396 (0.17% from the optimum) after measuring 59.</b>
Random search with the same budget did as well in 7.8% of 10,000 trials and needs a median of 424 evaluations to match it.
The pre-registered second-order surface was statistically inadequate (lack of fit 7.8% against a 5% limit), because of
3-factor interactions and abrupt compiler effects, so the final choice was made by measurement. The winner was a corner of the
screening design, so the gain comes from screening and from where its levels were placed, not from the fitted surface. The selected
kernel reaches 72.0% of cuBLAS FP32 throughput and is 1.31× faster than the default tile sizes with the inner loop fully unrolled.</div>
<div class="keyfigs">
<div><div class="v acc">0.17%</div><div class="d">gap to the exhaustive optimum</div></div>
<div><div class="v">59</div><div class="d">configurations measured, of 1,396</div></div>
<div><div class="v">1.31×</div><div class="d">faster than the full-unroll default</div></div>
<div><div class="v">72.0%</div><div class="d">of cuBLAS FP32 throughput</div></div>
</div>
<div style="flex:1"></div>
<div class="meta">
<div><span class="k">Author</span><span>Vaidheesh Bharadwaj</span></div>
<div><span class="k">Roll no.</span><span class="mono">230957088</span></div>
<div><span class="k">Guide</span><span>Prof. Sunita Sharma</span></div>
<div><span class="k">Platform</span><span>NVIDIA H200 · CUDA 13.0</span></div>
</div>
</section>""")

# ---------------------------------------------------------- 1 introduction --
A("""<h2 class="first"><span class="sn">01</span>Introduction and research questions</h2>
<p class="lead">The same kernel source runs at 26.1 TFLOPS with small default tiles, 30.5 TFLOPS with the parameters of the tutorial it comes from and 34.3 TFLOPS with tuned ones: six compile-time numbers decide a 1.31× difference.</p>
<p>Choosing those numbers is usually done by exhaustive search, which grows with every new factor, by search-based autotuners such as OpenTuner, CLTune and Kernel Tuner, or by machine-learning autotuners such as TVM and Ansor, which need many samples. This project tests whether classical Design of Experiments (DOE) and Response Surface Methodology (RSM) can find a near-best configuration from a small, statistically designed sample, as proposed in the synopsis.</p>""")
A(table(["", "Question"],
        [["RQ1", "Can a fractional factorial design separate the parameters that matter from noise?"],
         ["RQ2", "Does a second-order response surface capture the GPU’s non-linear behaviour?"],
         ["RQ3", "With the same number of trials, does DOE–RSM beat random search?"],
         ["RQ4", "How close does DOE–RSM get to the exhaustive-search optimum?"],
         ["RQ5", "What share of cuBLAS throughput does the tuned kernel reach?"]],
        "Research questions. RQ3 is the primary claim; RQ5 is context.", widths=["12%", "88%"]))

# ----------------------------------------------------------- 2 setup --------
A("""<h2><span class="sn">02</span>Setup and measurement protocol</h2>
<p class="lead">One parameterised kernel was tuned over six factors, with every rule of the experiment fixed in a written contract before any final data was collected.</p>
<h3>Kernel and design space</h3>
<p>The kernel is a 2D block-tiled shared-memory SGEMM computing C = A·B (row-major, FP32 inputs and accumulation, CUDA-core FFMA only, no Tensor Cores). It is adapted from kernel 5 (“2D blocktiling”) of Boehm’s public SGEMM worklog; this project added the compile-time parameterisation, the unroll factor, the feasibility assertions and a fault-injection switch for testing the correctness check. The six factors are injected as compile-time macros.</p>""")
A(table(["Factor", "Meaning", "Levels"],
        [["BM", "block output-tile rows (shared memory)", "32 · 64 · 128 · 256"],
         ["BN", "block output-tile columns (shared memory)", "32 · 64 · 128 · 256"],
         ["BK", "K-depth of each shared-memory tile", "8 · 16 · 32 · 64"],
         ["TM", "output rows per thread (register tile)", "2 · 4 · 8"],
         ["TN", "output columns per thread (register tile)", "2 · 4 · 8"],
         ["UNROLL", "unroll factor of the inner dot-product loop", "1 · 2 · 4 · 8"]],
        "Tunable factors and their levels in the full design space", num_cols=(2,), widths=["14%", "56%", "30%"]))
A("""<p>Of the 2,304 combinations, <b>1,396 are feasible</b>. The rest break a hardware or tiling constraint: more than 1,024 threads per block (528), shared memory over 48 KiB (168), a tile-load overflow (140) or a fractional load stride (72). The feasibility filter was checked against the compiler on all 2,304 candidates with no disagreements, and the kernel carries the same rules as compile-time assertions.</p>
<h3>Platform</h3>
<p>NVIDIA H200 (sm_90, 132 SMs), driver 595.71.05, CUDA 13.0.88, cuBLAS 13.1.1.3, <code>nvcc -O3</code>. The GPU is shared with other users’ processes.</p>
<h3>Measurement protocol</h3>
<ul>
<li><b>Timing.</b> Each launch is timed individually with a CUDA event pair; allocation, transfers, verification and compilation are outside the timed region. A measurement block is the median of 30 launches after 10 warm-up launches at 4096³ (80 and 25 at 2048³; 300 and 100 at 1024³).</li>
<li><b>Clock.</b> The SM clock was locked for the whole run; the driver held 1830 MHz (1980 MHz was requested) on all 2,001 measurements. A stability gate measured two configurations in 10 separated blocks: coefficient of variation 0.033% and 0.106%, against a 1% limit.</li>
<li><b>Randomisation.</b> Every experiment ran as replicate blocks, each a fresh random permutation of the design, so replicates of one point never ran back to back.</li>
<li><b>Shared GPU.</b> NVML per-process accounting checked every measurement window; a window in which another process used the SMs was repeated, up to six attempts. 37 measurements were repeated and 1 remained flagged, kept and marked as the contract requires.</li>
<li><b>Correctness.</b> Every configuration was checked against cuBLAS once per shape by relative Frobenius error, with a tolerance of 10⁻⁵ fixed before any data; all passed. A deliberately broken kernel produced an error of 0.044, matching the predicted √(8/4096), and both the custom kernel and cuBLAS agree with a float64 CPU reference to within 6×10⁻⁷.</li>
</ul>""")

# ----------------------------------------------------------- 3 screening ----
A("""<h2><span class="sn">03</span>Screening (RQ1)</h2>
<p class="lead">A 32-run fractional factorial found all six factors significant at 4096³ (R² = 0.959). The unroll factor dominates, with 54% of the variance including its interactions.</p>
<p><b>Design.</b> A 2<sup>6−1</sup> half fraction with generator F = ABCDE (defining relation I = ABCDEF, resolution VI): main effects are aliased only with 5-factor interactions and 2-factor interactions only with 4-factor interactions. Each factor ran at two legal levels (BM, BN 64/128; BK 8/32; TM, TN 4/8; UNROLL 1/8). There are no centre points, because the midpoint of a power-of-two pair is not a legal configuration; pure error comes from 3 replicate blocks (96 measurements per shape).</p>""")
A(fig("effects", "Screening effects at 4096³ with 95% confidence intervals: six main effects and the six largest 2-factor interactions. All are significant."))
A("""<p>Raising UNROLL from 1 to 8 adds 7.9 TFLOPS on average and TM from 4 to 8 adds 4.9; the strongest interaction, BN × UNROLL, is negative (−3.1). UNROLL’s dominance partly reflects its low level: UNROLL = 1 forbids unrolling the inner loop at all, so much of its effect is the cost of switching unrolling off rather than a gain found by tuning. The pre-registered retention rule (keep a factor if its main effect or any interaction containing it has p &lt; 0.05) kept all six; the weakest, BM, still had p = 0.0026.</p>
<p><b>Diagnostics.</b> Randomising the run order removed the autocorrelation that a fixed order produced in a pilot run: the Durbin–Watson statistic on run order is 1.86, against 0.60 in the pilot. There is no drift between blocks (p = 1.0) and replicate noise is small (median coefficient of variation 0.037%). Residual variance is not constant (Breusch–Pagan p &lt; 0.001), but refitting on log latency gives the same significant factors at all three shapes. Because measurement noise is so small, the residuals are mostly real 3-factor interactions the model omits; that inflates the error estimate, so the tests are conservative.</p>
<p><b>Effect of matrix shape.</b> The same screen was repeated at 2048³ and 1024³ (Figure 2). Factor rankings at 4096³ and 2048³ agree exactly (Kendall τ = 1.0); at 1024³ they do not (τ = −0.07): BM rises from 1.9% to 23.6% of the variance and TM falls from 22.7% to 3.2%. A likely cause is GPU fill: at 1024³ a 128 × 128 tile yields only 64 thread blocks for the H200’s 132 SMs, so tile size decides how much of the GPU is used. Tuned settings should not be carried from large to small matrices.</p>""")
A(fig("shapes", "Share of variance per factor (main effect plus its 2-factor interactions) at three matrix sizes."))

# ----------------------------------------------------------- 4 RSM ----------
A("""<h2><span class="sn">04</span>Response surface (RQ2)</h2>
<p class="lead">A second-order surface explains 95.6% of the variance but fails the pre-registered adequacy test: its lack of fit is 7.8% of the model mean square against a 5% limit. The answer to RQ2 is “close, but not adequate”.</p>
<p><b>Design.</b> A 6-factor central composite design in log₂-coded units: the 32-run screening design as its factorial core, 12 axial points at the rotatable α = 2.378 and 4 centre points. Each coded point was mapped to the nearest legal power of two (ties to the lower level) and the model was fitted on the realised coded values, so the centre runs as (64, 64, 16, 4, 4, 2). In total, 48 points × 3 randomised blocks = 144 measurements on 45 distinct configurations.</p>""")
A(table(["Model", "Params", "R²", "Pred. R²", "Lack of fit", "Gate (≤ 5%)", "Held-out error, median / max"],
        [["Second order · pre-registered", "30", "0.956", "0.932", "7.8%", "fails", "10.6% / 46%"],
         ["Second order · log latency (fallback)", "30", "0.961", "0.940", "7.0%", "fails", "11.4% / 38%"],
         ["Second order + 3-factor terms", "40", "0.982", "0.970", "10.1%", "fails", "32.6% / 412%"]],
        "Response-surface models. Lack of fit is the lack-of-fit mean square as a percentage of the model mean square; held-out error leaves out one configuration (all its replicates) at a time.",
        num_cols=(1, 2, 3, 4, 5, 6), hl_first=True, widths=["30%", "8%", "8%", "10%", "11%", "12%", "21%"]))
A("""<p>The lack-of-fit F-test is overwhelming (p &lt; 10⁻¹⁵⁰) because replicate noise is only about 0.05%, so even small systematic errors are detectable. 80% of the misfit lies inside the factorial core rather than at the axial extremes, and canonical analysis shows the fitted quadratic is a saddle with no interior maximum.</p>
<h3>The 3-factor investigation</h3>
<p>After the gate failed, the project owner chose to report the failure and then test the obvious remedy: adding the 10 estimable 3-factor interaction pairs (the half-fraction core aliases each triple with its complement). Six are significant, the strongest being BM·BN·BK = TM·TN·UNROLL (p = 1.4×10⁻¹³), and together they explain 59% of the quadratic’s residual. The augmented model still fails the gate and is overfitted: 40 parameters on 45 distinct configurations make its error on a held-out configuration three times worse. The ordinary predicted R² hides this, because it leaves out one measurement at a time while the other replicates of that configuration stay in the training set.</p>
<p>Part of the misfit is not polynomial at all. For (BM, BN, BK, TM, TN) = (64, 64, 32, 8, 4), throughput is non-monotonic in the unroll factor (Figure 3). A smooth surface cannot follow a compiler code-generation jump like that.</p>""")
A(fig("unroll", "Throughput against unroll factor for configuration (64, 64, 32, 8, 4): unrolling by 4 is slower than by 2."))
A(fig("pred_vs_measured", "Predicted against measured throughput for all 1,396 feasible configurations, for both models. Blue: inside the jointly tested region; grey: outside."))
A("""<p>Checked against the exhaustive sweep (Figure 4), both models are usable inside the region they were fitted on (median error 5.5% for the quadratic, 5.0% augmented) and unreliable outside it (21.8% and 22.7%), sometimes predicting negative throughput. The model’s best configuration outside the fitted region was predicted at 37.1 TFLOPS and measured at 30.5. This is why optimisation was restricted to the fitted region. The residuals (Figure 5) show no run-order structure (Durbin–Watson 2.15) and no influential point (largest Cook’s distance 0.049): the failure is systematic misfit, not noise or outliers.</p>""")
A(fig("residuals", "Residual diagnostics for the pre-registered second-order model."))
A(fig("surfaces", "Fitted surface for the three strongest interactions (augmented model; it fails the adequacy gate and is shown for interpretation only). Other factors are held at the model’s best configuration."))

# ----------------------------------------------------------- 5 validation ---
A("""<h2><span class="sn">05</span>Optimisation and validation</h2>
<p class="lead">Measured validation, not the model, picked the final configuration: (128, 128, 32, 8, 8, 8) at 34.32 TFLOPS, which neither model ranked first.</p>
<p>Optimisation was limited to the region the factorial core tested jointly (every coded factor within [−1, 1]), which leaves 192 feasible lattice configurations. Two methods proposed candidates: sequential quadratic programming (SciPy SLSQP, 30 random starts, all converged) on the continuous relaxation, with the thread-count, shared-memory and tile-load limits as nonlinear constraints, then projected to the lattice with its feasible ±1-level neighbourhood; and exhaustive evaluation of each fitted model over the 192 lattice configurations. The augmented model’s top 5, the quadratic’s top 3 and the SQP neighbourhood gave 15 distinct candidates, each measured in 5 randomised blocks.</p>""")
A(table(["Configuration", "Measured TFLOPS (sd)", "Augmented pred.", "Quadratic pred.", "Rank: meas.", "aug.", "quad."],
        [["(128, 128, 32, 8, 8, 8) · selected", "34.32 (0.04)", "34.07", "31.90", "1", "3", "11"],
         ["(128, 128, 32, 8, 8, 4)", "33.08 (0.01)", "35.83", "34.39", "2", "1", "3"],
         ["(64, 128, 32, 8, 8, 4)", "33.05 (0.03)", "33.23", "34.73", "3", "6", "2"],
         ["(64, 64, 32, 8, 4, 8)", "33.02 (0.02)", "32.57", "32.37", "4", "11", "8"],
         ["(64, 128, 16, 8, 8, 4)", "32.21 (0.02)", "32.95", "33.69", "5", "9", "4"],
         ["(64, 128, 32, 8, 4, 4)", "31.17 (0.03)", "33.07", "35.01", "8", "8", "1"],
         ["(64, 64, 32, 8, 4, 4)", "20.84 (0.01)", "32.15", "32.07", "15", "13", "10"]],
        "Validation candidates: the five fastest, each model’s first choice and the largest miss (15 candidates in total, 5 blocks each).",
        num_cols=(1, 2, 3, 4, 5, 6), hl_first=True, widths=["33%", "17%", "12%", "12%", "10%", "8%", "8%"]))
A("""<p>Within this narrow top band both models rank poorly (Spearman correlation with measured throughput 0.25 augmented, 0.37 quadratic; mean prediction error 9.0% and 8.5%). Trusting a model’s own first choice would have cost 3.6% (augmented) or 9.2% (quadratic); benchmarking a candidate set instead of a single rounded optimum made the difference.</p>
<p><b>Against the exhaustive optimum.</b> The true best configuration, (64, 128, 64, 8, 8, 8) at 34.40 TFLOPS, has BK = 64, outside the region the models were allowed to search, so the restriction chosen to avoid extrapolation also put it out of reach. The top 20 exhaustive configurations were re-measured in 3 more blocks: the selected configuration is second, 0.17% slower, a borderline-significant difference (Welch p = 0.042); the third is 33.22 TFLOPS.</p>""")

# ----------------------------------------------------------- 6 efficiency ---
A("""<h2><span class="sn">06</span>Sample efficiency (RQ3, RQ4)</h2>
<p class="lead">With 59 evaluations DOE–RSM landed 0.17% from the optimum; random search needs a median of 424 evaluations to do as well.</p>
<p>DOE–RSM consumed 59 unique configurations: 32 in screening, 13 new ones in the CCD and 14 new ones in validation. Random search was given exactly that budget, repeated 10,000 times by sampling from the exhaustive measurements of all 1,396 configurations.</p>""")
A(fig("sample_efficiency", "Best throughput found against configurations evaluated (log scale). DOE–RSM in vermilion; random search median and 10th–90th percentile band in blue."))
A(fig("equal_budget", "Distribution of random search’s best result at the same 59-evaluation budget, 10,000 trials."))
A(table(["Method", "Evaluations", "Best TFLOPS", "Gap to optimum"],
        [["DOE–RSM (this project)", "59", "34.34", "0.17%"],
         ["Random search, same budget (median; 10th–90th pct)", "59", "32.07 (30.50–33.23)", "7.26%"],
         ["Exhaustive search", "1,396", "34.40", "0%"],
         ["Tutorial parameters (128, 128, 8, 8, 8, 8), one informed guess", "1", "30.54", "12.7%"],
         ["Full-unroll default (64, 64, 8, 4, 4, 8)", "1", "26.13", "31.7%"],
         ["Unroll-disabled default (64, 64, 8, 4, 4, 1)", "1", "13.15", "161%"]],
        "Equal-budget comparison. Gaps are in latency against the exhaustive optimum (3.995 ms); single-configuration rows are from the exhaustive sweep.", num_cols=(1, 2, 3), hl_first=True,
        widths=["46%", "14%", "22%", "18%"]))
A("""<p>Random search matched or beat DOE–RSM in 7.8% of trials and needs a median of 424 evaluations (10th–90th percentile 77–936), about 7× the budget, to reach the same result.</p>
<p><b>What to credit to which stage.</b> The selected configuration is the all-high corner of the screening design and was first measured at evaluation 7. The response surface and validation confirmed it, by measuring it fastest among 15 model-proposed candidates, but did not discover it. Its presence in the screening design follows from where the screening levels were placed (by engineering judgement, before any measurement). The efficiency gain belongs to the design as a whole, screening included, not to the fitted surface alone.</p>
<p><b>A fairer reference than random search.</b> Random search starts with no knowledge of GPUs, while the screening levels were chosen by someone who has it, so beating random search does not show that the statistical method adds value beyond that knowledge. A better comparison is an informed engineer. Their natural first guess, the tutorial’s own parameters, reaches 30.54 TFLOPS (rank 51 of 1,396) in one evaluation; the all-high screening corner, an equally natural guess, is the selected configuration itself. What this study supports is that a designed screen around sensible levels finds a near-optimal configuration cheaply and explains which factors matter. It does not show that the response surface found it.</p>""")

# ----------------------------------------------------------- 7 cuBLAS -------
A("""<h2><span class="sn">07</span>Comparison with cuBLAS (RQ5)</h2>
<p class="lead">The selected kernel attains 72.0% of cuBLAS FP32 throughput. It is 1.31× faster than the default tile sizes with the inner loop fully unrolled, and 1.12× faster than the tutorial’s parameters.</p>
<p>Each configuration was benchmarked in the same process as cuBLAS, alternating 30-launch blocks of each (10 per process, 3 randomised processes: 900 launches per side). cuBLAS ran in pedantic FP32 mode with TF32 disabled; its default mode gave identical times, and its output matches the custom kernel’s bit for bit, so the comparison is FP32 against FP32.</p>""")
A(table(["Configuration", "Median latency (ms)", "Custom TFLOPS", "cuBLAS TFLOPS", "Share of cuBLAS"],
        [["DOE–RSM selection (128, 128, 32, 8, 8, 8)", "4.001", "34.35", "47.71", "72.0%"],
         ["Exhaustive best (64, 128, 64, 8, 8, 8)", "3.993", "34.42", "47.68", "72.2%"],
         ["Unroll-disabled default (64, 64, 8, 4, 4, 1)", "10.447", "13.16", "47.69", "27.6%"]],
        "Interleaved comparison with cuBLAS, 4096³, pedantic FP32.", num_cols=(1, 2, 3, 4), hl_first=True,
        widths=["44%", "16%", "13%", "13%", "14%"]))
A(fig("final_comparison", "Throughput of the unroll-disabled default, the DOE–RSM selection, the exhaustive best and cuBLAS."))
A("""<p><b>Choice of baseline.</b> The default in the table has UNROLL = 1, which forbids unrolling the inner loop. Against it the selection is 2.61× faster, but most of that gap is the cost of switching unrolling off: the same tiles with UNROLL = 8, which unrolls the loop completely at BK = 8, already reach 26.13 TFLOPS (54.8% of cuBLAS) in the exhaustive sweep. The 1.31× figure against that configuration is the honest measure of what tuning bought.</p>
<p>The remaining 28% is a ceiling of the kernel design, not of the tuning: even the best of all 1,396 configurations stops at 72.2%. cuBLAS uses techniques this kernel does not have (vectorised 128-bit loads, asynchronous double-buffered copies, warp-level tiling); adding them would change the kernel family and lies outside this study.</p>""")

# ----------------------------------------------------------- 8 registers ----
A("""<h2><span class="sn">08</span>Register spilling</h2>
<p class="lead">Register spilling is rare in this kernel family and does not drive its non-linear behaviour: 28 of 1,396 configurations spill, and they are not systematically slower.</p>
<p>The compiler’s resource report was recorded for every configuration. Register use ranges from 20 to 255 per thread. Spilling configurations have a median of 21.6 TFLOPS against 19.0 for the rest; most sit at 64 registers, the per-thread limit a 1,024-thread block imposes. Throughput rises with registers per thread (Spearman 0.47) and falls with occupancy (−0.45): larger register tiles reuse each loaded value more, which outweighs having fewer threads resident. The synopsis expected the surface to reveal register spilling; in this kernel the non-linearity comes instead from register-tile reuse, unroll-dependent code generation and, at small shapes, GPU fill.</p>""")
A(fig("registers", "Registers per thread against throughput for all 1,396 feasible configurations; spilling configurations in vermilion."))

# ----------------------------------------------------------- 9 threats ------
A("""<h2><span class="sn">09</span>Deviations and threats to validity</h2>
<p class="lead">One deviation from the pre-registered contract was made after seeing data; everything else ran as written.</p>
<p><b>Deviation.</b> The contract said not to optimise a surface that fails the adequacy gate. After it failed, the project owner chose to report the failure (Section 4) and continue with the 3-factor-augmented model as a proposal generator, selecting the final configuration only by measured validation. Both models’ recommendations were benchmarked, so the effect of that choice is visible in Table 4.</p>
<ul>
<li><b>Shared GPU.</b> Contested measurement windows were detected through NVML and repeated (37 repeats, 1 still flagged). NVML samples utilisation periodically and can miss very short bursts; the median over 30 launches limits their influence.</li>
<li><b>Clock.</b> The requested 1980 MHz lock was held at 1830 MHz by the driver. Every measurement ran at that clock, so all throughputs, cuBLAS included, are about 7.5% below unlocked boost figures; ratios are unaffected.</li>
<li><b>Scope.</b> One GPU, FP32 on CUDA cores, one kernel family and one primary shape (4096³). The 1024³ screen shows factor importance changes with shape, so the selected configuration should not be carried to other shapes without re-tuning.</li>
<li><b>Power-of-two lattice.</b> Coded design points were snapped to legal levels, so the CCD is not exactly rotatable and its centre runs at (64, 64, 16, 4, 4, 2); models were fitted on the realised levels.</li>
<li><b>Where the winner came from.</b> The selected configuration is a corner of the screening design, so the result depends partly on where the screening levels were placed, and the true optimum lies outside the region the models were allowed to search.</li>
<li><b>Baselines.</b> Random search is the weakest sensible baseline. The study has no Bayesian-optimisation or local-search comparison, and the unroll-disabled default overstates the gain from tuning (Section 7).</li>
<li><b>Ground truth.</b> Random search was scored with the single-block exhaustive measurements; the top 20 were re-measured in 3 more blocks and kept their order.</li>
<li><b>Correctness reference.</b> At 4096³ cuBLAS sums in the same order as this kernel, so their outputs are identical; agreement with an independent float64 CPU reference was checked for the default and selected configurations at 512³ and 1024³.</li>
<li><b>Pilot data excluded.</b> An earlier pilot run without clock locking or randomisation shaped this protocol; none of its measurements are used here.</li>
</ul>""")

# ----------------------------------------------------------- 10 conclusions -
A("""<h2><span class="sn">10</span>Conclusions and future work</h2>""")
A(table(["", "Answer"],
        [["RQ1", "Yes. The fractional factorial separated all six factors (R² 0.959); their importance shifts at 1024³."],
         ["RQ2", "Close but inadequate: lack of fit 7.8% against a 5% limit, caused by 3-factor interactions and compiler jumps."],
         ["RQ3", "Yes against random search: 59 evaluations against a median of 424 for the same result. The gain comes from the screen and its expert-chosen levels, not the fitted surface."],
         ["RQ4", "0.17% from the exhaustive optimum: the second-best of 1,396 configurations."],
         ["RQ5", "72.0% of cuBLAS FP32 throughput; 1.31× the full-unroll default and 1.12× the tutorial’s parameters."]],
        "Answers to the research questions.", widths=["10%", "90%"]))
A("""<p>The practical lesson is to use the response surface to propose candidates and let measurement decide. Four extensions follow: a full 2⁶ factorial core, so 3-factor interactions can be estimated without overfitting; tuning per matrix shape, since importance shifts below 2048³; a head-to-head comparison with Bayesian optimisation, the method modern autotuners use; and kernel features such as vectorised loads and double buffering as categorical factors, to raise the 72% ceiling.</p>""")

# ----------------------------------------------------------- reproducibility -
A("""<h2><span class="sn">A</span>Reproducibility</h2>
<p>Every number in this report comes from run <code>final_20261028</code> and can be regenerated from its raw files. The experimental contract (<code>contract.json</code>, SHA-256 prefix <code>857dc5fbcf29</code>) was frozen before data collection and its hash is stamped on the run.</p>""")
A(table(["File", "Contents"],
        [["<code>results/final_20261028/measurements.csv</code>", "2,001 measurement blocks: configuration, validity, clock, temperature, contention flags, registers, median and IQR latency"],
         ["<code>results/final_20261028/raw_launches.csv</code>", "every individual timed launch"],
         ["<code>results/final_20261028/decisions/</code>", "each automatic decision: retained factors, CCD design, validation candidates, final selection"],
         ["<code>results/final_20261028/analysis/</code>", "ANOVA tables, model coefficients, cross-validation, summaries"],
         ["<code>results/feasibility_audit.csv</code>", "all 2,304 candidates with feasibility status and the compiler’s verdict"],
         ["<code>cuda/gemm_kernel.cuh</code>, <code>cuda/bench.cu</code>", "the kernel and its benchmark harness"]],
        "Data and code.", widths=["44%", "56%"]))
A("""<p>The run is executed phase by phase with <code>python pipeline.py &lt;run&gt; &lt;phase&gt;</code> (stability, screening, ccd, validate, exhaustive, recheck, compare) and analysed with <code>python analysis.py &lt;run&gt; &lt;phase&gt;</code>.</p>
<h2><span class="sn">B</span>References</h2>
<ol class="refs">
<li>G. E. P. Box and K. B. Wilson, “On the Experimental Attainment of Optimum Conditions,” <i>Journal of the Royal Statistical Society, Series B</i>, 1951.</li>
<li>D. C. Montgomery, <i>Design and Analysis of Experiments</i>, Wiley, 2017.</li>
<li>S. Boehm, “How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog,” 2022. Online: siboehm.com/articles/22/CUDA-MMM (kernel 5, source of the kernel used here).</li>
<li>J. Ansel et al., “OpenTuner: An Extensible Framework for Program Autotuning,” PACT, 2014.</li>
<li>C. Nugteren and V. Codreanu, “CLTune: A Generic Auto-Tuner for OpenCL Kernels,” MCSoC, 2015.</li>
<li>B. van Werkhoven, “Kernel Tuner: A Search-Optimizing GPU Code Auto-Tuner,” <i>Future Generation Computer Systems</i>, 2019.</li>
<li>NVIDIA Corporation, <i>CUDA C++ Programming Guide</i>.</li>
<li>NVIDIA Corporation, <i>cuBLAS Library Documentation</i>.</li>
<li>D. B. Kirk and W. W. Hwu, <i>Programming Massively Parallel Processors: A Hands-on Approach</i>, Morgan Kaufmann, 2016.</li>
<li>T. Chen et al., “TVM: An Automated End-to-End Optimizing Compiler for Deep Learning,” OSDI, 2018.</li>
<li>L. Zheng et al., “Ansor: Generating High-Performance Tensor Programs for Deep Learning,” OSDI, 2020.</li>
<li>J. Nocedal and S. J. Wright, <i>Numerical Optimization</i>, Springer, 2006.</li>
</ol>""")


def build():
    os.makedirs(OUT, exist_ok=True)
    html = ('<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Hardware-aware autotuning of a CUDA GEMM kernel '
            'with response surface methodology</title>'
            '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1'
            '&family=Geist:wght@300..700&family=Geist+Mono:wght@400;500&display=block">'
            f'<style>{CSS}</style></head><body>' + "".join(BODY) + '</body></html>')
    hp = os.path.join(OUT, "report.html")
    open(hp, "w", encoding="utf-8").write(html)
    pdf = os.path.join(OUT, PDF_NAME)
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    "--virtual-time-budget=20000", f"--print-to-pdf={pdf}",
                    "file:///" + hp.replace("\\", "/")], capture_output=True)
    import fitz
    doc = fitz.open(pdf)
    pd = os.path.join(OUT, "report_png")
    os.makedirs(pd, exist_ok=True)
    for f in os.listdir(pd):
        os.remove(os.path.join(pd, f))
    for i, page in enumerate(doc):
        page.get_pixmap(dpi=110).save(os.path.join(pd, f"page_{i + 1:02d}.png"))
    print(f"report pages: {len(doc)}, size {doc[0].rect}")


if __name__ == "__main__":
    build()
