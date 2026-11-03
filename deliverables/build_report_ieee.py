"""Builds the IEEE-format version of the report: two-column A4, Times, IEEE
captions and numbered citations -> PDF (Chrome headless) and per-page PNGs.
Same figures (figs/report) and numbers as build_report.py.

  python build_report_ieee.py
"""
import os
import re
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figs", "report")
OUT = os.path.join(HERE, "out")
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
PDF_NAME = "Vaidheesh_Bharadwaj_DOE-RSM_CUDA_Autotuning_Report_IEEE.pdf"

# ------------------------------------------------------------- references --
# IEEE style; numbered in order of first citation by cite().
REFS = {
    "tvm": "T. Chen <i>et al.</i>, “TVM: An automated end-to-end optimizing compiler for deep learning,” in <i>Proc. 13th USENIX Symp. Operating Systems Design and Implementation (OSDI)</i>, Carlsbad, CA, USA, 2018, pp. 578–594.",
    "ansor": "L. Zheng <i>et al.</i>, “Ansor: Generating high-performance tensor programs for deep learning,” in <i>Proc. 14th USENIX Symp. Operating Systems Design and Implementation (OSDI)</i>, 2020, pp. 863–879.",
    "box": "G. E. P. Box and K. B. Wilson, “On the experimental attainment of optimum conditions,” <i>J. Roy. Statist. Soc. Ser. B</i>, vol. 13, no. 1, pp. 1–45, 1951.",
    "montgomery": "D. C. Montgomery, <i>Design and Analysis of Experiments</i>, 9th ed. Hoboken, NJ, USA: Wiley, 2017.",
    "opentuner": "J. Ansel <i>et al.</i>, “OpenTuner: An extensible framework for program autotuning,” in <i>Proc. 23rd Int. Conf. Parallel Architectures and Compilation (PACT)</i>, Edmonton, AB, Canada, 2014, pp. 303–316.",
    "cltune": "C. Nugteren and V. Codreanu, “CLTune: A generic auto-tuner for OpenCL kernels,” in <i>Proc. IEEE 9th Int. Symp. Embedded Multicore/Many-core Systems-on-Chip (MCSoC)</i>, Turin, Italy, 2015, pp. 195–202.",
    "kerneltuner": "B. van Werkhoven, “Kernel Tuner: A search-optimizing GPU code auto-tuner,” <i>Future Gener. Comput. Syst.</i>, vol. 90, pp. 347–358, 2019.",
    "bo": "F.-J. Willemsen, R. van Nieuwpoort, and B. van Werkhoven, “Bayesian optimization for auto-tuning GPU kernels,” in <i>Proc. IEEE/ACM Int. Workshop Performance Modeling, Benchmarking and Simulation of High Performance Computer Systems (PMBS)</i>, 2021, pp. 106–117.",
    "boehm": "S. Boehm, “How to optimize a CUDA matmul kernel for cuBLAS-like performance: A worklog,” 2022. [Online]. Available: https://siboehm.com/articles/22/CUDA-MMM",
    "volkov":"V. Volkov and J. W. Demmel, “Benchmarking GPUs to tune dense linear algebra,” in <i>Proc. ACM/IEEE Conf. Supercomputing (SC)</i>, Austin, TX, USA, 2008, pp. 1–11.",
    "myers": "R. H. Myers, D. C. Montgomery, and C. M. Anderson-Cook, <i>Response Surface Methodology: Process and Product Optimization Using Designed Experiments</i>, 4th ed. Hoboken, NJ, USA: Wiley, 2016.",
    "kirk": "D. B. Kirk and W. W. Hwu, <i>Programming Massively Parallel Processors: A Hands-on Approach</i>, 3rd ed. Cambridge, MA, USA: Morgan Kaufmann, 2016.",
    "cudaguide": "NVIDIA Corporation, <i>CUDA C++ Programming Guide</i>, Release 13.0, 2025. [Online]. Available: https://docs.nvidia.com/cuda/cuda-c-programming-guide/",
    "cublas": "NVIDIA Corporation, <i>cuBLAS Library User Guide</i>, CUDA Toolkit 13.0, 2025. [Online]. Available: https://docs.nvidia.com/cuda/cublas/",
    "dw": "J. Durbin and G. S. Watson, “Testing for serial correlation in least squares regression. II,” <i>Biometrika</i>, vol. 38, no. 1/2, pp. 159–177, 1951.",
    "bp": "T. S. Breusch and A. R. Pagan, “A simple test for heteroscedasticity and random coefficient variation,” <i>Econometrica</i>, vol. 47, no. 5, pp. 1287–1294, 1979.",
    "sw": "S. S. Shapiro and M. B. Wilk, “An analysis of variance test for normality (complete samples),” <i>Biometrika</i>, vol. 52, no. 3/4, pp. 591–611, 1965.",
    "cook": "R. D. Cook, “Detection of influential observation in linear regression,” <i>Technometrics</i>, vol. 19, no. 1, pp. 15–18, 1977.",
    "nocedal": "J. Nocedal and S. J. Wright, <i>Numerical Optimization</i>, 2nd ed. New York, NY, USA: Springer, 2006.",
    "scipy": "P. Virtanen <i>et al.</i>, “SciPy 1.0: Fundamental algorithms for scientific computing in Python,” <i>Nature Methods</i>, vol. 17, pp. 261–272, 2020.",
    "bergstra": "J. Bergstra and Y. Bengio, “Random search for hyper-parameter optimization,” <i>J. Mach. Learn. Res.</i>, vol. 13, pp. 281–305, 2012.",
    "statsmodels": "S. Seabold and J. Perktold, “Statsmodels: Econometric and statistical modeling with Python,” in <i>Proc. 9th Python in Science Conf. (SciPy)</i>, Austin, TX, USA, 2010, pp. 92–96.",
}
CITED = []


def cite(*keys):
    nums = []
    for k in keys:
        if k not in CITED:
            CITED.append(k)
        nums.append(CITED.index(k) + 1)
    nums.sort()
    if len(nums) > 2 and nums[-1] - nums[0] == len(nums) - 1:
        return f"[{nums[0]}]–[{nums[-1]}]"
    return ", ".join(f"[{n}]" for n in nums)


# ------------------------------------------------------- figures / tables --
def svg(name):
    s = open(os.path.join(FIGS, name + ".svg"), encoding="utf-8").read()
    head, rest = s.split(">", 1)
    head = re.sub(r'\b(width|height)="([0-9.]+)pt"', r'\1="\2px"', head)
    return (head + ">" + rest).replace("<svg ", '<svg class="chart" ', 1)


FIG_N = [0]


def fig(name, caption, wide=True, width="100%"):
    FIG_N[0] += 1
    cls = "wide" if wide else ""
    return (f'<figure class="{cls}"><div style="width:{width};margin:0 auto">{svg(name)}</div>'
            f'<figcaption>Fig. {FIG_N[0]}.&ensp;{caption}</figcaption></figure>')


TAB_N = [0]
ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]


def table(header, rows, caption, num_cols=(), wide=False, widths=None, note=""):
    TAB_N[0] += 1
    th = "".join(f'<th class="{"m" if i in num_cols else ""}"{f" style=\"width:{widths[i]}\"" if widths else ""}>{h}</th>'
                 for i, h in enumerate(header))
    trs = ""
    for row in rows:
        trs += "<tr>" + "".join(f'<td class="{"m" if i in num_cols else ""}">{c}</td>' for i, c in enumerate(row)) + "</tr>"
    note = f'<div class="tnote">{note}</div>' if note else ""
    return (f'<div class="tbl{" wide" if wide else ""}"><div class="tcap">Table {ROMAN[TAB_N[0] - 1]}<br>{caption}</div>'
            f'<table><tr>{th}</tr>{trs}</table>{note}</div>')


SEC = [0]
SUB = [0]


def h1(title):
    SEC[0] += 1
    SUB[0] = 0
    return f'<h2>{ROMAN[SEC[0] - 1]}. {title}</h2>'


def h2(title):
    SUB[0] += 1
    return f'<h3>{chr(64 + SUB[0])}. {title}</h3>'


CSS = """
@page { size: A4; margin: 19mm 14.3mm 22mm 14.3mm;
  @bottom-center { content: counter(page); font: 9pt 'Times New Roman', Times, serif } }
* { box-sizing: border-box; margin: 0; padding: 0 }
body { font-family: 'Times New Roman', Times, serif; font-size: 10pt; line-height: 1.17; color: #000 }
.title { text-align: center; font-size: 24pt; line-height: 1.15; margin: 0 6mm 4mm }
.authors { display: flex; justify-content: center; gap: 14mm; text-align: center; margin-bottom: 6mm; font-size: 10pt }
.authors .name { font-size: 11pt }
.authors i, .authors span { display: block }
.cols { column-count: 2; column-gap: 4.22mm; text-align: justify; hyphens: auto }
.abstract, .index { font-size: 9pt; font-weight: bold; margin-bottom: 2.4mm; text-indent: 1em }
.abstract i, .index i { font-weight: bold }
p { text-indent: 1em; margin: 0 }
p.noind { text-indent: 0 }
h2 { font-size: 10pt; font-weight: normal; font-variant: small-caps; text-align: center; margin: 3.4mm 0 1.6mm; break-after: avoid }
h3 { font-size: 10pt; font-weight: normal; font-style: italic; margin: 2.2mm 0 1mm; break-after: avoid }
.runin { font-style: italic }
ol.rq, ol.fut { margin: 1mm 0 1.4mm 5mm } ol.rq li, ol.fut li { margin-bottom: .5mm }
code { font-family: 'Courier New', monospace; font-size: 8.6pt }
figure { display: inline-block; width: 100%; margin: 2.5mm 0 3mm }
figure.wide, .tbl.wide { display: block; break-inside: avoid; column-span: all; margin: 3mm 0 3.5mm }
.chart { display: block; width: 100%; height: auto; overflow: visible }
.chart text { font-family: Arial, Helvetica, sans-serif !important }
figcaption { font-size: 8pt; margin-top: 1.2mm; text-align: justify; line-height: 1.2 }
.tbl { display: inline-block; width: 100%; margin: 2.5mm 0 3mm }
.tcap { font-size: 8pt; text-align: center; font-variant: small-caps; margin-bottom: 1.4mm; line-height: 1.3 }
table { border-collapse: collapse; width: 100%; font-size: 8pt; line-height: 1.2 }
th { font-weight: normal; font-variant: small-caps; text-align: left; padding: .8mm 1.6mm; border-top: .75pt solid #000; border-bottom: .5pt solid #000; vertical-align: bottom }
td { padding: .7mm 1.6mm; vertical-align: top; text-align: left }
tr:last-child td { border-bottom: .75pt solid #000 }
td.m, th.m { text-align: center }
.tnote { font-size: 7.5pt; margin-top: .8mm; text-align: left }
.todo { border: 1pt solid #c00; background: #fff4f4; color: #900; padding: 2mm; margin: 1mm 0 2mm; text-align: left; font-size: 9pt; break-inside: avoid }
.todo ul { margin: 1mm 0 0 4mm }
.refs { font-size: 8pt; line-height: 1.2; text-align: left }
.refs div { display: flex; gap: 1.4mm; margin-bottom: .9mm }
.refs span:first-child { min-width: 5.5mm; text-align: right; flex-shrink: 0 }
"""

B = []
A = B.append

# --------------------------------------------------------------- title -----
A("""<div class="title">Hardware-Aware Autotuning of a CUDA GEMM Kernel with Response Surface Methodology</div>
<div class="authors">
<div><span class="name">Vaidheesh Bharadwaj</span><i>Department of Mathematics</i><i>Manipal Institute of Technology</i>
<i>Manipal Academy of Higher Education</i><span>Manipal, India</span><span>Reg. No. 230957088</span></div>
<div><span class="name">Prof. Sunita Sharma (Guide)</span><i>Department of Mathematics</i><i>Manipal Institute of Technology</i>
<i>Manipal Academy of Higher Education</i><span>Manipal, India</span></div>
</div>
<div class="cols">""")

A("""<p class="abstract"><i>Abstract</i>—A single parameterised CUDA matrix-multiplication kernel, adapted from a public tutorial kernel, was tuned over six compile-time
factors on an NVIDIA H200, using a 2<sup>6−1</sup> fractional-factorial screen, a 6-factor central composite design, a second-order
response surface, constrained sequential quadratic programming and measured validation. Every result is checked against an exhaustive
sweep of all 1,396 feasible configurations. The pipeline selected the second-best configuration of 1,396 (0.17% from the optimum) after
measuring 59. Random search with the same budget did as well in 7.8% of 10,000 trials and needs a median of 424 evaluations to match it.
The pre-registered second-order surface was statistically inadequate (lack of fit 7.8% against a 5% limit), because of 3-factor
interactions and abrupt compiler effects, so the final choice was made by measurement. The winner was a corner of the screening design,
so the gain comes from screening and from where its levels were placed, not from the fitted surface. The selected kernel reaches 72.0% of
cuBLAS FP32 throughput and is 1.31× faster than the default tile sizes with the inner loop fully unrolled.</p>
<p class="index"><i>Index Terms</i>—Autotuning, central composite design, CUDA, design of experiments, fractional factorial design, GEMM,
GPU performance, response surface methodology.</p>""")

# ------------------------------------------------------- I introduction ----
A(h1("Introduction"))
A(f"""<p>The same kernel source runs at 26.1 TFLOPS with small default tiles, 30.5 TFLOPS with the parameters of the tutorial it comes
from and 34.3 TFLOPS with tuned ones: six compile-time numbers decide a 1.31× difference. Choosing those numbers is usually done by exhaustive search, which grows with every new factor, or by
machine-learning autotuners such as TVM {cite("tvm")} and Ansor {cite("ansor")}, which need many samples. This project tests whether
classical Design of Experiments (DOE) and Response Surface Methodology (RSM) {cite("box", "montgomery")} can find a near-best
configuration from a small, statistically designed sample.</p>""")
A(h2("Problem Statement"))
A("""<p>Given a parameterised CUDA SGEMM kernel with six compile-time factors and 1,396 feasible configurations on an NVIDIA H200, find a
configuration close to the fastest one while measuring only a small fraction of the space, and quantify how close it gets compared with
exhaustive search and with random search on the same budget.</p>""")
A(h2("Objectives"))
A("""<p>The study answers five research questions. RQ3 is the primary claim; RQ5 is context.</p>
<ol class="rq">
<li><i>RQ1:</i> Can a fractional factorial design separate the parameters that matter from noise?</li>
<li><i>RQ2:</i> Does a second-order response surface capture the GPU’s non-linear behaviour?</li>
<li><i>RQ3:</i> With the same number of trials, does DOE–RSM beat random search?</li>
<li><i>RQ4:</i> How close does DOE–RSM get to the exhaustive-search optimum?</li>
<li><i>RQ5:</i> What share of cuBLAS throughput does the tuned kernel reach?</li>
</ol>""")

# ------------------------------------------------------- II related work ---
A(h1("Related Work"))
A(f"""<p>Empirical autotuners measure candidate configurations on the target hardware and use a search strategy to decide what to
measure next. OpenTuner {cite("opentuner")} is a general framework that runs an ensemble of search techniques and gives more trials to
whichever is making progress. CLTune {cite("cltune")} and Kernel Tuner {cite("kerneltuner")} target GPU kernels specifically and offer
strategies such as random search, simulated annealing, genetic algorithms and particle swarm optimisation. These tools find fast
configurations, but their output is a configuration, not an account of which parameters matter or how they interact.</p>
<p>Compiler-based autotuners for deep learning replace part of the measurement with a learned cost model. TVM {cite("tvm")} trains a
model on measured schedules and uses it to rank unmeasured ones, and Ansor {cite("ansor")} adds automatic construction of the search
space and evolutionary search guided by such a model. Bayesian optimisation {cite("bo")} is the closest relative of this project: a
Gaussian-process surrogate is fitted to the measurements so far and chooses the next configuration, and it compared favourably with the
other strategies in Kernel Tuner. All of these evaluate configurations one after another, and their models are built to guide the search
rather than to be interpreted.</p>
<p>On the kernel side, Volkov and Demmel {cite("volkov")} showed by hand-tuning dense linear algebra on GPUs that blocking the computation
in registers is central to GEMM performance; the register-tile factors TM and TN used here follow from that observation.</p>
<p>Response surface methodology began with Box and Wilson {cite("box")} as a sequence of designed experiments and fitted low-order
polynomials for locating optimum operating conditions, and factorial, fractional factorial and central composite designs are now standard
tools {cite("montgomery", "myers")}. RSM is widely used to optimise chemical and manufacturing processes, but to the author’s knowledge it
has not been evaluated for GPU kernel tuning against an exhaustive ground truth. This project makes that comparison, and reports where the
second-order model is inadequate as well as where the design succeeds.</p>""")

# ---------------------------------------------- III methodology ------------
A(h1("Methodology and Implementation"))
A("""<p>One parameterised kernel was tuned over six factors, with every rule of the experiment fixed in a written contract before any
final data was collected.</p>""")
A(h2("Kernel and Design Space"))
A(f"""<p>The kernel is a 2D block-tiled shared-memory SGEMM {cite("kirk", "cudaguide")} computing <i>C</i> = <i>A</i>·<i>B</i>
(row-major, FP32 inputs and accumulation, CUDA-core FFMA only, no Tensor Cores). It is adapted from kernel 5 (“2D blocktiling”) of
Boehm’s public SGEMM worklog {cite("boehm")}; this work added the compile-time parameterisation, the unroll factor, the feasibility
assertions and a fault-injection switch for testing the correctness check. The six factors (Table I) are injected as compile-time
macros.</p>""")
A(table(["Factor", "Meaning", "Levels"],
        [["BM", "block output-tile rows (shared memory)", "32, 64, 128, 256"],
         ["BN", "block output-tile columns (shared memory)", "32, 64, 128, 256"],
         ["BK", "K-depth of each shared-memory tile", "8, 16, 32, 64"],
         ["TM", "output rows per thread (register tile)", "2, 4, 8"],
         ["TN", "output columns per thread (register tile)", "2, 4, 8"],
         ["UNROLL", "unroll factor of the inner dot-product loop", "1, 2, 4, 8"]],
        "Tunable Factors and Their Levels", widths=["17%", "53%", "30%"]))
A("""<p>Of the 2,304 combinations, 1,396 are feasible. The rest break a hardware or tiling constraint: more than 1,024 threads per block
(528), shared memory over 48 KiB (168), a tile-load overflow (140) or a fractional load stride (72). The feasibility filter was checked
against the compiler on all 2,304 candidates with no disagreements, and the kernel carries the same rules as compile-time assertions.</p>""")
A(h2("Platform"))
A("""<p>NVIDIA H200 (sm_90, 132 SMs), driver 595.71.05, CUDA 13.0.88, cuBLAS 13.1.1.3, <code>nvcc -O3</code>. The GPU is shared with
other users’ processes.</p>""")
A(h2("Measurement Protocol"))
A("""<p><span class="runin">Timing:</span> Each launch is timed individually with a CUDA event pair; allocation, transfers, verification and
compilation are outside the timed region. A measurement block is the median of 30 launches after 10 warm-up launches at 4096³ (80 and
25 at 2048³; 300 and 100 at 1024³).</p>
<p><span class="runin">Clock:</span> The SM clock was locked for the whole run; the driver held 1830 MHz (1980 MHz was requested) on all
2,001 measurements. A stability gate measured two configurations in 10 separated blocks: coefficient of variation 0.033% and 0.106%,
against a 1% limit.</p>
<p><span class="runin">Randomisation:</span> Every experiment ran as replicate blocks, each a fresh random permutation of the design, so
replicates of one point never ran back to back.</p>
<p><span class="runin">Shared GPU:</span> NVML per-process accounting checked every measurement window; a window in which another process
used the SMs was repeated, up to six attempts.</p>
<p><span class="runin">Correctness:</span> Every configuration was checked against cuBLAS once per shape by relative Frobenius error, with a
tolerance of 10<sup>−5</sup> fixed before any data. A deliberately broken kernel produced an error of 0.044, matching the predicted
√(8/4096), and both the custom kernel and cuBLAS agree with a float64 CPU reference to within 6×10<sup>−7</sup>.</p>""")
A(h2("Screening Design"))
A(f"""<p>A 2<sup>6−1</sup> half fraction {cite("montgomery")} with generator F = ABCDE (defining relation I = ABCDEF, resolution VI): main
effects are aliased only with 5-factor interactions and 2-factor interactions only with 4-factor interactions. Each factor ran at two legal
levels (BM, BN 64/128; BK 8/32; TM, TN 4/8; UNROLL 1/8). There are no centre points, because the midpoint of a power-of-two pair is not
a legal configuration; pure error comes from 3 replicate blocks (96 measurements per shape). The screen was run at 4096³, 2048³ and
1024³. The pre-registered retention rule keeps a factor if its main effect or any interaction containing it has <i>p</i> &lt; 0.05.</p>""")
A(h2("Central Composite Design and Response Surface"))
A(f"""<p>A 6-factor central composite design (CCD) {cite("box", "myers")} in log<sub>2</sub>-coded units: the 32-run screening design as its
factorial core, 12 axial points at the rotatable α = 2.378 and 4 centre points. Each coded point was mapped to the nearest legal power of
two (ties to the lower level) and the model was fitted on the realised coded values, so the centre runs as (64, 64, 16, 4, 4, 2). In
total, 48 points × 3 randomised blocks = 144 measurements on 45 distinct configurations.</p>
<p>Second-order models were fitted by ordinary least squares with a block term, so the full quadratic has 28 polynomial terms plus 2
block effects (30 parameters). The pre-registered adequacy gate requires the lack-of-fit mean square to be at most 5% of the model mean
square. Held-out error leaves out one configuration (all its replicates) at a time. Residuals were checked with the Durbin–Watson
{cite("dw")}, Breusch–Pagan {cite("bp")} and Shapiro–Wilk {cite("sw")} tests and Cook’s distance {cite("cook")}.</p>""")
A(h2("Optimisation and Validation"))
A(f"""<p>Optimisation was limited to the region the factorial core tested jointly (every coded factor within [−1, 1]), which leaves 192
feasible lattice configurations. Two methods proposed candidates: sequential quadratic programming {cite("nocedal")} (SciPy SLSQP
{cite("scipy")}, 30 random starts) on the continuous relaxation, with the thread-count, shared-memory and tile-load limits as nonlinear
constraints, then projected to the lattice with its feasible ±1-level neighbourhood; and exhaustive evaluation of each fitted model over
the 192 lattice configurations. The candidates were measured in 5 randomised blocks each, and the fastest measured candidate was
selected.</p>""")
A(h2("Baselines"))
A(f"""<p>All 1,396 feasible configurations were measured once (the exhaustive sweep), and the top 20 re-measured in 3 more blocks, to
provide ground truth. Random search {cite("bergstra")} was given the same budget as DOE–RSM, repeated 10,000 times by sampling without
replacement from the exhaustive measurements; the evaluations-to-match curve uses 2,000 random orderings of the same measurements. For
cuBLAS {cite("cublas")}, each configuration was benchmarked in the same process, alternating 30-launch blocks of each (10 per process, 3
randomised processes: 900 launches per side), with cuBLAS in pedantic FP32 mode and TF32 disabled.</p>""")
A(h2("Implementation"))
A(f"""<p>Each configuration is compiled by <code>nvcc</code> into its own benchmark binary (<code>cuda/bench.cu</code> with
<code>cuda/gemm_kernel.cuh</code>), cached by a hash of the sources and flags. A Python driver (<code>pipeline.py</code>) generates each
design, randomises the run order, locks the clocks, monitors the GPU through NVML and records every launch; it runs phase by phase
(stability, screening, CCD, validation, exhaustive, recheck, comparison). <code>analysis.py</code> fits the models with statsmodels
{cite("statsmodels")} and runs the optimisation with SciPy.</p>""")

# ---------------------------------------------- IV results -----------------
A(h1("Results and Discussion"))
A(h2("Screening (RQ1)"))
A("""<p>The 32-run fractional factorial found all six factors significant at 4096³ (<i>R</i>² = 0.959). The unroll factor dominates, with
54% of the variance including its interactions. Raising UNROLL from 1 to 8 adds 7.9 TFLOPS on average and TM from 4 to 8 adds 4.9; the
strongest interaction, BN × UNROLL, is negative (−3.1) (Fig. 1). UNROLL’s dominance partly reflects its low level: UNROLL = 1 forbids
unrolling the inner loop at all, so much of its effect is the cost of switching unrolling off rather than a gain found by tuning. The
retention rule kept all six factors; the weakest, BM, still had
<i>p</i> = 0.0026.</p>""")
A(fig("effects", "Screening effects at 4096³ with 95% confidence intervals: six main effects and the six largest 2-factor interactions. All are significant.", width="62%"))
A("""<p><span class="runin">Diagnostics:</span> Randomising the run order removed the autocorrelation that a fixed order produced in a pilot
run: the Durbin–Watson statistic on run order is 1.86, against 0.60 in the pilot. There is no drift between blocks (<i>p</i> = 1.0) and
replicate noise is small (median coefficient of variation 0.037%). Residual variance is not constant (Breusch–Pagan <i>p</i> &lt; 0.001),
but refitting on log latency gives the same significant factors at all three shapes. Because measurement noise is so small, the residuals
are mostly real 3-factor interactions the model omits; that inflates the error estimate, so the tests are conservative.</p>
<p><span class="runin">Effect of matrix shape:</span> Factor rankings at 4096³ and 2048³ agree exactly (Kendall τ = 1.0); at 1024³ they do
not (τ = −0.07): BM rises from 1.9% to 23.6% of the variance and TM falls from 22.7% to 3.2% (Fig. 2). A likely cause is GPU fill: at
1024³ a 128 × 128 tile yields only 64 thread blocks for the H200’s 132 SMs, so tile size decides how much of the GPU is used. Tuned
settings should not be carried from large to small matrices.</p>""")
A(fig("shapes", "Share of variance per factor (main effect plus its 2-factor interactions) at three matrix sizes.", width="58%"))

A(h2("Response Surface (RQ2)"))
A("""<p>A second-order surface explains 95.6% of the variance but fails the pre-registered adequacy test: its lack of fit is 7.8% of the
model mean square against a 5% limit (Table II). The answer to RQ2 is “close, but not adequate”. The lack-of-fit <i>F</i>-test is
overwhelming (<i>p</i> &lt; 10<sup>−150</sup>) because replicate noise is only about 0.05%, so even small systematic errors are
detectable. 80% of the misfit lies inside the factorial core rather than at the axial extremes, and canonical analysis shows the fitted
quadratic is a saddle with no interior maximum.</p>""")
A(table(["Model", "Params", "<i>R</i>²", "Pred. <i>R</i>²", "Lack of fit", "Gate (≤ 5%)", "Held-out error, median / max"],
        [["Second order, pre-registered", "30", "0.956", "0.932", "7.8%", "fails", "10.6% / 46%"],
         ["Second order, log latency (fallback)", "30", "0.961", "0.940", "7.0%", "fails", "11.4% / 38%"],
         ["Second order + 3-factor terms", "40", "0.982", "0.970", "10.1%", "fails", "32.6% / 412%"]],
        "Response-Surface Models", num_cols=(1, 2, 3, 4, 5, 6), wide=True,
        widths=["30%", "8%", "8%", "10%", "11%", "12%", "21%"],
        note="Params include 2 block effects. Lack of fit is the lack-of-fit mean square as a percentage of the model mean square. "
             "Held-out error leaves out one configuration (all its replicates) at a time."))
A("""<p><span class="runin">The 3-factor investigation:</span> After the gate failed, the author chose to report the failure and then test
the obvious remedy: adding the 10 estimable 3-factor interaction pairs (the half-fraction core aliases each triple with its complement).
Six are significant, the strongest being BM·BN·BK = TM·TN·UNROLL (<i>p</i> = 1.4×10<sup>−13</sup>), and together they explain 59% of the
quadratic’s residual. The augmented model still fails the gate and is overfitted: 40 parameters on 45 distinct configurations make its
error on a held-out configuration three times worse. The ordinary predicted <i>R</i>² hides this, because it leaves out one measurement at
a time while the other replicates of that configuration stay in the training set.</p>
<p>Part of the misfit is not polynomial at all. For (BM, BN, BK, TM, TN) = (64, 64, 32, 8, 4), throughput is non-monotonic in the unroll
factor (Fig. 3). A smooth surface cannot follow a compiler code-generation jump like that.</p>""")
A(fig("unroll", "Throughput against unroll factor for configuration (64, 64, 32, 8, 4): unrolling by 4 is slower than by 2.", wide=False))
A("""<p>Checked against the exhaustive sweep (Fig. 4), both models are usable inside the region they were fitted on (median error 5.5%
for the quadratic, 5.0% augmented) and unreliable outside it (21.8% and 22.7%), sometimes predicting negative throughput. The model’s best
configuration outside the fitted region was predicted at 37.1 TFLOPS and measured at 30.5. This is why optimisation was restricted to the
fitted region.</p>""")
A(fig("pred_vs_measured", "Predicted against measured throughput for all 1,396 feasible configurations, for both models. Blue: inside the jointly tested region; grey: outside.", width="85%"))
A("""<p>The residuals of the pre-registered model (Fig. 5) show no run-order structure (Durbin–Watson 2.15) and no influential point
(largest Cook’s distance 0.049). They are, however, not normal (Shapiro–Wilk <i>p</i> = 0.006) and their variance is not constant
(Breusch–Pagan <i>p</i> &lt; 10<sup>−8</sup>); the log-latency refit removes the first problem (<i>p</i> = 0.08) but not the second.
The failure is systematic misfit, not noise or outliers. Fig. 6 shows the fitted surface for three of the largest 2-factor interactions
in Fig. 1.</p>""")
A(fig("residuals", "Residual diagnostics for the pre-registered second-order model.", width="68%"))
A(fig("surfaces", "Fitted surface for three of the largest 2-factor interactions (augmented model; it fails the adequacy gate and is shown for interpretation only). Other factors are held at the model’s best configuration.", width="90%"))

A(h2("Optimisation and Validation"))
A("""<p>Measured validation, not the model, picked the final configuration: (128, 128, 32, 8, 8, 8) at 34.32 TFLOPS, which neither model
ranked first. The augmented model’s top 5, the quadratic’s top 3 and the SQP neighbourhood gave 15 distinct candidates (Table III); all
30 SQP starts converged.</p>""")
A(table(["Configuration", "Measured TFLOPS (SD)", "Augmented pred.", "Quadratic pred.", "Rank: meas.", "Aug.", "Quad."],
        [["(128, 128, 32, 8, 8, 8), selected", "34.32 (0.04)", "34.07", "31.90", "1", "3", "11"],
         ["(128, 128, 32, 8, 8, 4)", "33.08 (0.01)", "35.83", "34.39", "2", "1", "3"],
         ["(64, 128, 32, 8, 8, 4)", "33.05 (0.03)", "33.23", "34.73", "3", "6", "2"],
         ["(64, 64, 32, 8, 4, 8)", "33.02 (0.02)", "32.57", "32.37", "4", "11", "8"],
         ["(64, 128, 16, 8, 8, 4)", "32.21 (0.02)", "32.95", "33.69", "5", "9", "4"],
         ["(64, 128, 32, 8, 4, 4)", "31.17 (0.03)", "33.07", "35.01", "8", "8", "1"],
         ["(64, 64, 32, 8, 4, 4)", "20.84 (0.01)", "32.15", "32.07", "15", "13", "10"]],
        "Validation Candidates", num_cols=(1, 2, 3, 4, 5, 6), wide=True,
        widths=["31%", "17%", "12%", "12%", "11%", "8%", "9%"],
        note="The five fastest, each model’s first choice and the largest miss, of 15 candidates measured in 5 blocks each. "
             "Configurations are (BM, BN, BK, TM, TN, UNROLL)."))
A("""<p>Within this narrow top band both models rank poorly (Spearman correlation with measured throughput 0.25 augmented, 0.37
quadratic; mean prediction error 9.0% and 8.5%). Trusting a model’s own first choice would have cost 3.6% (augmented) or 9.2%
(quadratic) of throughput; benchmarking a candidate set instead of a single rounded optimum made the difference.</p>
<p><span class="runin">Against the exhaustive optimum:</span> The true best configuration, (64, 128, 64, 8, 8, 8) at 34.40 TFLOPS, has
BK = 64, outside the region the models were allowed to search, so the restriction chosen to avoid extrapolation also put it out of reach.
In the re-measurement of the top 20, the selected configuration is second, 0.17% slower, a borderline-significant difference (Welch
<i>p</i> = 0.042); the third is 33.22 TFLOPS.</p>""")

A(h2("Sample Efficiency (RQ3, RQ4)"))
A("""<p>DOE–RSM consumed 59 unique configurations: 32 in screening, 13 new ones in the CCD and 14 new ones in validation. With these it
landed 0.17% from the optimum (Table IV, Fig. 7). Random search with the same budget matched or beat DOE–RSM in 7.8% of 10,000 trials
(Fig. 8) and needs a median of 424 evaluations (10th–90th percentile 77–936), about 7× the budget, to reach the same result.</p>""")
A(fig("sample_efficiency", "Best throughput found against configurations evaluated (log scale). DOE–RSM in vermilion; random-search median of 2,000 random orderings as a dark line and its 10th–90th percentile band in grey.", width="70%"))
A(table(["Method", "Evals.", "Best TFLOPS", "Gap"],
        [["DOE–RSM (this work)", "59", "34.34", "0.17%"],
         ["Random search, same budget: median (10th–90th pct.)", "59", "32.07 (30.50–33.23)", "7.26%"],
         ["Exhaustive search", "1,396", "34.40", "0%"],
         ["Tutorial parameters (128, 128, 8, 8, 8, 8), one informed guess", "1", "30.54", "12.7%"],
         ["Full-unroll default (64, 64, 8, 4, 4, 8)", "1", "26.13", "31.7%"],
         ["Unroll-disabled default (64, 64, 8, 4, 4, 1)", "1", "13.15", "161%"]],
        "Equal-Budget Comparison", num_cols=(1, 2, 3), wide=True, widths=["46%", "14%", "24%", "16%"],
        note="Gaps are in latency against the exhaustive optimum (3.995 ms). TFLOPS from the exhaustive sweep."))
A(fig("equal_budget", "Distribution of random search’s best result at the same 59-evaluation budget, 10,000 trials.", width="72%"))
A(f"""<p><span class="runin">What to credit to which stage:</span> The selected configuration is the all-high corner of the screening
design and was first measured at evaluation 7. The response surface and validation confirmed it, by measuring it fastest among 15
model-proposed candidates, but did not discover it. Its presence in the screening design follows from where the screening levels were
placed (by engineering judgement, before any measurement). The efficiency gain belongs to the design as a whole, screening included, not to
the fitted surface alone.</p>
<p><span class="runin">A fairer reference than random search:</span> Random search starts with no knowledge of GPUs, while the screening
levels were chosen by someone who has it, so beating random search does not show that the statistical method adds value beyond that
knowledge. A better comparison is an informed engineer. Their natural first guess, the tutorial’s own parameters {cite("boehm")}, reaches
30.54 TFLOPS (rank 51 of 1,396) in one evaluation; the all-high screening corner, an equally natural guess, is the selected configuration
itself. What this study supports is that a designed screen around sensible levels finds a near-optimal configuration cheaply and explains
which factors matter. It does not show that the response surface found it.</p>
<p>The selected configuration appears as 34.34 TFLOPS in Table IV (exhaustive sweep), 34.32 in Table III (validation) and 34.35 in
Table V (cuBLAS comparison). The differences, under 0.1%, come from separate measurement sessions.</p>""")

A(h2("Comparison with cuBLAS (RQ5)"))
A("""<p>The selected kernel attains 72.0% of cuBLAS FP32 throughput (Table V, Fig. 9). It is 1.31× faster than the default tile sizes
with the inner loop fully unrolled, and 1.12× faster than the tutorial’s parameters. cuBLAS’s default mode gave identical times to pedantic FP32 mode, and its output matches the custom kernel’s bit for bit, so the comparison
is FP32 against FP32.</p>""")
A(table(["Configuration", "Latency (ms)", "Custom TFLOPS", "cuBLAS TFLOPS", "Share"],
        [["DOE–RSM selection (128, 128, 32, 8, 8, 8)", "4.001", "34.35", "47.71", "72.0%"],
         ["Exhaustive best (64, 128, 64, 8, 8, 8)", "3.993", "34.42", "47.68", "72.2%"],
         ["Unroll-disabled default (64, 64, 8, 4, 4, 1)", "10.447", "13.16", "47.69", "27.6%"]],
        "Interleaved Comparison with cuBLAS, 4096³, Pedantic FP32", num_cols=(1, 2, 3, 4), wide=True,
        widths=["44%", "14%", "14%", "14%", "14%"], note="Median of 900 interleaved launches per side."))
A(fig("final_comparison", "Throughput of the unroll-disabled default, the DOE–RSM selection, the exhaustive best and cuBLAS.", width="66%"))
A("""<p><span class="runin">Choice of baseline:</span> The default in Table V has UNROLL = 1, which forbids unrolling the inner loop.
Against it the selection is 2.61× faster, but most of that gap is the cost of switching unrolling off: the same tiles with UNROLL = 8,
which unrolls the loop completely at BK = 8, already reach 26.13 TFLOPS (54.8% of cuBLAS) in the exhaustive sweep. The 1.31× figure
against that configuration is the honest measure of what tuning bought.</p>
<p>The remaining 28% is a ceiling of the kernel design, not of the tuning: even the best of all 1,396 configurations stops at 72.2%.
cuBLAS uses techniques this kernel does not have (vectorised 128-bit loads, asynchronous double-buffered copies, warp-level tiling); adding
them would change the kernel family and lies outside this study.</p>""")

A(h2("Register Spilling"))
A("""<p>Register spilling is rare in this kernel family and does not drive its non-linear behaviour: 28 of 1,396 configurations spill,
and they are not systematically slower (Fig. 10). The compiler’s resource report was recorded for every configuration. Register use ranges
from 20 to 255 per thread. Spilling configurations have a median of 21.6 TFLOPS against 19.0 for the rest; most sit at 64 registers, the
per-thread limit a 1,024-thread block imposes. Throughput rises with registers per thread (Spearman 0.47) and falls with occupancy
(−0.45): larger register tiles reuse each loaded value more, which outweighs having fewer threads resident. Register spilling was expected
to be a source of non-linearity; in this kernel the non-linearity comes instead from register-tile reuse, unroll-dependent code generation
and, at small shapes, GPU fill.</p>""")
A(fig("registers", "Registers per thread against throughput for all 1,396 feasible configurations; spilling configurations in vermilion.", width="60%"))

A(h2("Deviations and Threats to Validity"))
A("""<p>One deviation from the pre-registered contract was made after seeing data; everything else ran as written.</p>
<p><span class="runin">Deviation:</span> The contract said not to optimise a surface that fails the adequacy gate. After it failed, the
author chose to report the failure (Section IV-B) and continue with the 3-factor-augmented model as a proposal generator, selecting the
final configuration only by measured validation. Both models’ recommendations were benchmarked, so the effect of that choice is visible in
Table III.</p>
<p><span class="runin">Shared GPU:</span> Contested measurement windows were detected through NVML and repeated (37 repeats, 1 still
flagged, kept and marked as the contract requires). NVML samples utilisation periodically and can miss very short bursts; the median over
30 launches limits their influence.</p>
<p><span class="runin">Clock:</span> The requested 1980 MHz lock was held at 1830 MHz by the driver. Every measurement ran at that clock,
so all throughputs, cuBLAS included, are about 7.5% below unlocked boost figures; ratios are unaffected.</p>
<p><span class="runin">Scope:</span> One GPU, FP32 on CUDA cores, one kernel family and one primary shape (4096³). The 1024³ screen shows
factor importance changes with shape, so the selected configuration should not be carried to other shapes without re-tuning.</p>
<p><span class="runin">Power-of-two lattice:</span> Coded design points were snapped to legal levels, so the CCD is not exactly rotatable
and its centre runs at (64, 64, 16, 4, 4, 2); models were fitted on the realised levels.</p>
<p><span class="runin">Where the winner came from:</span> The selected configuration is a corner of the screening design, so the result
depends partly on where the screening levels were placed, and the true optimum lies outside the region the models were allowed to
search.</p>
<p><span class="runin">Baselines:</span> Random search is the weakest sensible baseline. The study has no Bayesian-optimisation or
local-search comparison, and the unroll-disabled default overstates the gain from tuning (Section IV-E).</p>
<p><span class="runin">Ground truth:</span> Random search was scored with the single-block exhaustive measurements; the top 20 were
re-measured in 3 more blocks and kept their order.</p>
<p><span class="runin">Correctness reference:</span> At 4096³ cuBLAS sums in the same order as this kernel, so their outputs are identical;
agreement with an independent float64 CPU reference was checked for the default and selected configurations at 512³ and 1024³.</p>
<p><span class="runin">Pilot data excluded:</span> An earlier pilot run without clock locking or randomisation shaped this protocol; none
of its measurements are used here.</p>""")

# ---------------------------------------------- V conclusion ---------------
A(h1("Conclusion and Future Scope"))
A("""<p>The answers to the research questions are as follows. RQ1: yes; the fractional factorial separated all six factors
(<i>R</i>² = 0.959), though their importance shifts at 1024³. RQ2: close but inadequate; lack of fit was 7.8% against a 5% limit, caused
by 3-factor interactions and compiler jumps. RQ3: yes against random search; 59 evaluations against a median of 424 for the same result, though the gain
comes from the screen and its expert-chosen levels rather than the fitted surface. RQ4: 0.17% from the exhaustive optimum, the
second-best of 1,396 configurations. RQ5: 72.0% of cuBLAS FP32 throughput; 1.31× the full-unroll default and 1.12× the tutorial’s
parameters.</p>
<p>The practical lesson is to use the response surface to propose candidates and let measurement decide. Four extensions follow:</p>
<ol class="fut">
<li>a full 2<sup>6</sup> factorial core, so 3-factor interactions can be estimated without overfitting;</li>
<li>tuning per matrix shape, since importance shifts below 2048³;</li>
<li>a head-to-head comparison with Bayesian optimisation, the method modern autotuners use;</li>
<li>kernel features such as vectorised loads and double buffering as categorical factors, to raise the 72% ceiling.</li>
</ol>""")

# ---------------------------------------------- appendix -------------------
A("""<h2>Appendix<br>Reproducibility</h2>
<p>Every number in this report comes from run <code>final_20261028</code> and can be regenerated from its raw files (Table VI). The
experimental contract (<code>contract.json</code>, SHA-256 prefix <code>857dc5fbcf29</code>) was frozen before data collection and its
hash is stamped on the run. The run is executed with <code>python pipeline.py &lt;run&gt; &lt;phase&gt;</code> and analysed with
<code>python analysis.py &lt;run&gt; &lt;phase&gt;</code>.</p>""")
A(table(["File", "Contents"],
        [["<code>results/final_20261028/measurements.csv</code>", "2,001 measurement blocks: configuration, validity, clock, temperature, contention flags, registers, median and IQR latency"],
         ["<code>results/final_20261028/raw_launches.csv</code>", "every individual timed launch"],
         ["<code>results/final_20261028/decisions/</code>", "each automatic decision: retained factors, CCD design, validation candidates, final selection"],
         ["<code>results/final_20261028/analysis/</code>", "ANOVA tables, model coefficients, cross-validation, summaries"],
         ["<code>results/feasibility_audit.csv</code>", "all 2,304 candidates with feasibility status and the compiler’s verdict"],
         ["<code>cuda/gemm_kernel.cuh</code>, <code>cuda/bench.cu</code>", "the kernel and its benchmark harness"]],
        "Data and Code", wide=True, widths=["44%", "56%"]))


def references():
    unused = [k for k in REFS if k not in CITED]
    assert not unused, f"uncited references: {unused}"
    rows = "".join(f"<div><span>[{i + 1}]</span><span>{REFS[k]}</span></div>" for i, k in enumerate(CITED))
    return f'<h2>References</h2><div class="refs">{rows}</div>'


def build():
    os.makedirs(OUT, exist_ok=True)
    body = "".join(B) + references() + "</div>"
    html = ('<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Hardware-Aware Autotuning of a CUDA GEMM Kernel '
            'with Response Surface Methodology</title>'
            f'<style>{CSS}</style></head><body>{body}</body></html>')
    hp = os.path.join(OUT, "report_ieee.html")
    open(hp, "w", encoding="utf-8").write(html)
    pdf = os.path.join(OUT, PDF_NAME)
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    "--virtual-time-budget=20000", f"--print-to-pdf={pdf}",
                    "file:///" + hp.replace("\\", "/")], capture_output=True)
    import fitz
    doc = fitz.open(pdf)
    pd = os.path.join(OUT, "report_ieee_png")
    os.makedirs(pd, exist_ok=True)
    for f in os.listdir(pd):
        os.remove(os.path.join(pd, f))
    for i, page in enumerate(doc):
        page.get_pixmap(dpi=110).save(os.path.join(pd, f"page_{i + 1:02d}.png"))
    print(f"report pages: {len(doc)}, size {doc[0].rect}")


if __name__ == "__main__":
    build()
