"""Generate spike.ipynb from a list of cells (valid-JSON by construction).

Run once:  python build_notebook.py
Produces spike.ipynb next to this script. Safe to re-run; it overwrites.
"""

from __future__ import annotations

import json
from pathlib import Path

cells: list[tuple[str, str]] = []


def md(source: str) -> None:
    cells.append(("markdown", source.strip("\n")))


def code(source: str) -> None:
    cells.append(("code", source.strip("\n")))


# --------------------------------------------------------------------------- #
md(r"""
# Baidu Unlimited-OCR — Phase 0 spike

Spike (not the product) to answer, with numbers:

1. Does multi-page "whole book in one pass" work, or truncate / OOM / drift?
2. How accurate is it, and does accuracy drop later in long documents (R-SWA drift)?
3. GPU time / VRAM per page, and projected cost for a 300-page book?
4. If single-pass fails, does chunking (N pages + overlap) work as a fallback?

Run order: **0 Install (restart if asked)** → 1 Config → 2 GPU probe → 3 Load + introspect API →
4 Data prep → 5 Exp A (single-page) → 6 Exp B (multi-page scaling) → 7 Exp C (chunked) →
8 Metrics → 9 Report + GO/NO-GO verdict.

All reusable logic lives in the `ocr_spike` package; cells stay thin. Inputs, cached page
images, raw/cleaned outputs and `results.jsonl` are written to Google Drive so a disconnect
does not lose work.
""")

# --------------------------------------------------------------------------- #
md(r"""
## 0. Install first (then restart if prompted)

This cell runs **before** the GPU probe and Drive mount. It checks the NVIDIA driver against the
CUDA 12.9 torch wheels, installs the pinned deps, and tells you if a runtime restart is needed
(because Colab shipped a different torch). If it says RESTART REQUIRED: `Runtime > Restart
session`, then re-run starting at the **next** cell.
""")

code(r"""
# ===== 0. INSTALL FIRST -- then Runtime > Restart session if prompted =====
import re
import subprocess
import sys
from pathlib import Path

REQUIRED_TORCH = "2.10.0"
WHEEL_CUDA = "12.9"


def _run(cmd):
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=False)


# 1) Driver vs CUDA 12.9 wheel compatibility (nvidia-smi reports max supported CUDA).
driver_cuda = "unknown"
try:
    smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout
    m = re.search(r"CUDA Version:\s*([0-9.]+)", smi)
    driver_cuda = m.group(1) if m else "unknown"
except Exception as exc:
    print("nvidia-smi unavailable:", exc)
print("driver max CUDA:", driver_cuda, "| target wheel CUDA:", WHEEL_CUDA)

index_url = None
try:
    dmaj, dmin = (int(x) for x in driver_cuda.split(".")[:2])
    wmaj, wmin = (int(x) for x in WHEEL_CUDA.split(".")[:2])
    if dmaj != wmaj or (dmaj, dmin) < (wmaj, wmin):
        index_url = f"https://download.pytorch.org/whl/cu{wmaj}{min(dmin, wmin)}"
        print("Driver may not support CUDA", WHEEL_CUDA, "-> fallback wheels:", index_url)
except Exception:
    pass

# 2) Record the torch Colab shipped with, to know whether a restart is required.
try:
    import torch as _t
    preinstalled_torch = _t.__version__
except Exception:
    preinstalled_torch = None
print("preinstalled torch:", preinstalled_torch)

# 3) Install pinned deps (prefer requirements-spike.txt if present in CWD).
req = Path("requirements-spike.txt")
pip = [sys.executable, "-m", "pip", "install", "-q"]
if index_url:
    pip += ["--extra-index-url", index_url]
if req.exists():
    _run(pip + ["-r", str(req)])
else:
    print("requirements-spike.txt not found in CWD; installing explicit pins.")
    _run(pip + [
        "torch==2.10.0", "torchvision==0.25.0", "transformers==4.57.1", "Pillow==12.1.1",
        "einops==0.8.2", "addict==2.4.0", "easydict==1.13", "pymupdf==1.27.2.2",
        "psutil==7.2.2", "matplotlib==3.10.8", "jiwer==3.0.4", "edlib==1.3.9",
        "rapidfuzz==3.10.1", "pandas==2.2.3", "huggingface-hub==0.35.3",
        "safetensors==0.4.5", "accelerate==1.1.1", "pytest==8.3.3",
    ])

# 4) Restart guidance.
need_restart = preinstalled_torch not in (None, REQUIRED_TORCH)
print("=" * 72)
if need_restart:
    print("RESTART REQUIRED: torch changed from", preinstalled_torch, "to", REQUIRED_TORCH)
    print(">>> Runtime > Restart session, then run from the NEXT cell (skip this one).")
else:
    print("No torch restart needed. Continue to the next cell.")
print("=" * 72)
""")

# --------------------------------------------------------------------------- #
md(r"""
## 0b. Run the unit tests first

Runs `pytest tests -q` and **raises if anything fails or is skipped**. On Colab `edlib` installs
as a wheel, so the alignment tests must actually run (0 skipped) before any experiment.
""")

code(r"""
# ===== Gate: unit tests must pass with 0 skipped before any experiment =====
import re
import subprocess
import sys

result = subprocess.run(
    [sys.executable, "-m", "pytest", "tests", "-q"], capture_output=True, text=True
)
print(result.stdout[-4000:])
if result.stderr.strip():
    print("STDERR:\n", result.stderr[-2000:])

if result.returncode != 0:
    raise RuntimeError("Unit tests failed -- fix before running any experiment.")

skipped = re.search(r"(\d+) skipped", result.stdout)
if skipped and int(skipped.group(1)) > 0:
    raise RuntimeError(
        f"{skipped.group(1)} test(s) skipped -- edlib must be installed so the alignment "
        "tests run on Colab before experiments. Check the install cell."
    )
print("All unit tests passed (0 skipped).")
""")

# --------------------------------------------------------------------------- #
md(r"""
## 1. Config — every tunable lives here

Set your 3 books in `BOOKS` (schema shown). Also set `GPU_PRICE_PER_HOUR` to your real hourly
price and point `REPO_DIR` (in the next cell) at wherever this repo lives on Colab.
""")

code(r"""
# ===================== CONFIG: every tunable lives here =====================
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(levelname)s %(name)s: %(message)s", force=True
)
log = logging.getLogger("spike")

# --- storage (Google Drive) ---
DRIVE_ROOT = Path("/content/drive/MyDrive/ocr_spike")
INPUT_DIR = DRIVE_ROOT / "inputs"       # put your PDFs + ground-truth .txt files here
PAGES_DIR = DRIVE_ROOT / "pages"        # cached page PNGs (skipped if already rendered)
OUTPUT_DIR = DRIVE_ROOT / "outputs"     # raw/cleaned model outputs per run
RESULTS_PATH = DRIVE_ROOT / "results.jsonl"
REPORT_DIR = DRIVE_ROOT / "report"

# --- model (official repo only; trust_remote_code executes repo code) ---
MODEL_REPO = "baidu/Unlimited-OCR"
MODEL_REVISION = None   # None -> auto-resolve current SHA via HfApi and pin it to every row

# --- prompts / per-config inference kwargs (from the model card) ---
PROMPT_SINGLE = "<image>document parsing."
PROMPT_MULTI = "<image>Multi page parsing."

# Decoding is greedy per the remote code's defaults; no_repeat_ngram_* are recorded below.
GUNDAM_KWARGS = dict(
    base_size=1024, image_size=640, crop_mode=True, max_length=32768,
    no_repeat_ngram_size=35, ngram_window=128, save_results=True,
)
BASE_SINGLE_KWARGS = dict(
    base_size=1024, image_size=1024, crop_mode=False, max_length=32768,
    no_repeat_ngram_size=35, ngram_window=1024, save_results=True,
)
MULTI_KWARGS = dict(
    image_size=1024, max_length=32768,
    no_repeat_ngram_size=35, ngram_window=1024, save_results=True,
)

# --- max_length semantics (CONFIRM in section 3 from the modeling code) ---
# "max_length"     => cap counts prompt + image + output tokens
# "max_new_tokens" => cap counts output tokens only
MAX_LENGTH_SEMANTICS = "max_length"
# Guard: experiments refuse to run until you confirm the section-3 findings and set this True.
MAX_LENGTH_SEMANTICS_CONFIRMED = False
MAX_LENGTH_SLACK = 64            # within this many tokens of the cap still counts as "hit"
IMAGE_TOKENS_PER_PAGE = 256      # read from modeling code; used for the input-token estimate
BASE_MAX_LENGTH = 32768
MAX_LENGTH_ESCALATION = [65536, 131072]   # Exp B retries a truncated N at these caps until OOM

# --- experiment knobs ---
A_NUM_PAGES = 10                 # pages sampled evenly for the single-page baseline
A_BOOK_INDEX = 0                 # which book Experiments A/B/C run on
B_N_VALUES = [1, 5, 10, 25, 50, 100, 200]   # full book is appended automatically
C_K_VALUES = [10, 25, 50]
C_OVERLAP = 1
DPI = 300
NGRAM_N = 8
N_DECILES = 10

# --- cost / verdict thresholds ---
GPU_PRICE_PER_HOUR = 0.80        # <-- set to your actual GPU hourly price
BOOK_PAGES_FOR_COST = 300
MAX_COST_PER_BOOK = 1.00
THRESH_MAX_CER = 0.03
THRESH_MAX_DECILE_RATIO = 2.0
THRESH_MAX_REP_RATIO = 1.5

# --- control ---
FORCE = False                    # True = re-run even if a matching result row exists

# --- books under test (supply 3; pick variety: clean prose, footnotes, tables/2-col) ---
# schema: dict(id=str, pdf_path=str, ground_truth_path=str, notes=str)
BOOKS = [
    # dict(id="clean_prose", pdf_path=str(INPUT_DIR / "book1.pdf"),
    #      ground_truth_path=str(INPUT_DIR / "book1_gutenberg.txt"),
    #      notes="clean single-column prose"),
    # dict(id="footnotes", pdf_path=str(INPUT_DIR / "book2.pdf"),
    #      ground_truth_path=str(INPUT_DIR / "book2_gutenberg.txt"),
    #      notes="heavy footnotes"),
    # dict(id="tables_2col", pdf_path=str(INPUT_DIR / "book3.pdf"),
    #      ground_truth_path=str(INPUT_DIR / "book3_gutenberg.txt"),
    #      notes="tables / two-column"),
]
if not BOOKS:
    log.warning("BOOKS is empty -- add your 3 books before running Experiments A/B/C.")
""")

# --------------------------------------------------------------------------- #
md(r"""
## 1b. Mount Drive and import the package
""")

code(r"""
from google.colab import drive

drive.mount("/content/drive")
for d in (INPUT_DIR, PAGES_DIR, OUTPUT_DIR, REPORT_DIR):
    d.mkdir(parents=True, exist_ok=True)

import sys

# Point this at wherever the repo lives on Colab (e.g. after !git clone ...).
REPO_DIR = Path.cwd()
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from ocr_spike import infer, metrics, pdf, report  # noqa: E402

log.info("ocr_spike imported from %s", REPO_DIR)
""")

# --------------------------------------------------------------------------- #
md(r"""
## 2. GPU / dtype probe

Prints GPU name, VRAM, CUDA/torch build, driver version, and bf16 support. If bf16 is
unsupported (e.g. free-tier T4) it loudly falls back to float16; the chosen dtype is recorded in
every result row.
""")

code(r"""
runtime = infer.detect_runtime()
runtime
""")

# --------------------------------------------------------------------------- #
md(r"""
## 3. Load model at a pinned revision + introspect the API (don't guess)

Loads **only** `baidu/Unlimited-OCR` at the resolved commit SHA, then prints the live
`infer` / `infer_multi` signatures and the path to the cached modeling file.
""")

code(r"""
model, tokenizer, REVISION = infer.load_model(
    MODEL_REPO, runtime.dtype, revision=MODEL_REVISION
)
print("pinned revision:", REVISION)

api = infer.introspect_api(model)
for key, value in api.items():
    print(f"{key}: {value}")
""")

code(r"""
# Skim the modeling file for how max_length reaches generate() and the stop condition.
source_path = api.get("modeling_file", "unknown")
print("modeling file:", source_path)
text = ""
if source_path != "unknown" and Path(source_path).exists():
    text = Path(source_path).read_text(encoding="utf-8", errors="replace")

for kw in ("max_new_tokens", "max_length", ".generate(", "eos_token", "no_repeat_ngram"):
    print(f"\n==== occurrences of {kw!r} ====")
    for i, line in enumerate(text.splitlines()):
        if kw in line:
            print(f"{i:5d}: {line.strip()[:160]}")
""")

md(r"""
### What we found about `max_length` (fill from the grep output above)

Read the lines above and decide:

- If the code calls `self.generate(..., max_length=max_length)` → the cap counts **prompt +
  image + output** tokens. Set `MAX_LENGTH_SEMANTICS = "max_length"` (default).
- If it calls `self.generate(..., max_new_tokens=max_length)` → the cap counts **output only**.
  Set `MAX_LENGTH_SEMANTICS = "max_new_tokens"`.
- Set `IMAGE_TOKENS_PER_PAGE` to a rough per-page estimate; the spike then **measures** the real
  value and logs it, so you can correct the estimate.
- **Confirm, then set `MAX_LENGTH_SEMANTICS_CONFIRMED = True`** in the config — the experiment
  cells refuse to run until you do.

How this spike uses it:

- `hit_max_length` is computed from the chosen semantics. `GenerateCapture` patches `generate` on
  the model **and every submodule that owns it** (so an inner language-model `generate` is still
  caught) and records **every outermost** generate call (nested patched calls are de-duplicated
  via a depth counter), so per-page generation shows up as multiple calls. Each row logs
  `capture_fired`, `capture_target`, `generate_calls`, `per_page_generation`, `input_tokens_first`
  and `image_tokens_per_page_measured` (next to the `input_tokens` estimate), plus
  `output_tokens_total`, `any_hit_max_length` and `stop_reason`. If the capture does not fire, a
  warning is emitted — the fallback is never silent.
- A 300-page book is ~150k+ output tokens, far beyond 32,768 — so at the default cap, "whole book
  in one pass" is expected to truncate. Experiment B measures this directly and then raises the
  cap (`MAX_LENGTH_ESCALATION`) until it stops truncating or OOMs.
""")

# --------------------------------------------------------------------------- #
md(r"""
## 4. Data prep

Rasterize each PDF to 300-DPI PNGs (cached to Drive, skipping pages already converted) and load
+ strip Gutenberg boilerplate from each ground-truth text.
""")

code(r"""
books = {}
for entry in BOOKS:
    book_id = entry["id"]
    images = pdf.pdf_to_images(entry["pdf_path"], PAGES_DIR / book_id, dpi=DPI, force=FORCE)
    gt_raw = Path(entry["ground_truth_path"]).read_text(encoding="utf-8", errors="replace")
    ground_truth = metrics.strip_gutenberg_boilerplate(gt_raw)
    books[book_id] = dict(
        meta=entry, images=images, n_pages=len(images), ground_truth=ground_truth
    )
    log.info("%s: %d pages, %d GT chars (%s)",
             book_id, len(images), len(ground_truth), entry.get("notes", ""))


def make_evaluator(ground_truth, *, slice_to_span):
    # Return an execute_run evaluator that scores an outcome against ground truth.
    def _evaluate(outcome):
        ev = metrics.evaluate(
            outcome.cleaned_text, ground_truth,
            slice_to_span=slice_to_span, ngram_n=NGRAM_N, n_deciles=N_DECILES,
        )
        return ev.as_row()
    return _evaluate
""")

# --------------------------------------------------------------------------- #
md(r"""
## 5. Experiment A — single-page baseline

10 pages sampled evenly from one book, both **gundam** and **base** single-page configs.
Per-page accuracy and speed with no long-context effects. Each partial-page output is scored
against its matching GT span (infix alignment).
""")

code(r"""
if not MAX_LENGTH_SEMANTICS_CONFIRMED:
    raise RuntimeError(
        "Set MAX_LENGTH_SEMANTICS_CONFIRMED = True in the config cell after "
        "confirming the section-3 max_length findings."
    )

if books:
    pid = BOOKS[A_BOOK_INDEX]["id"]
    bk = books[pid]
    page_indices = pdf.sample_pages_evenly(bk["n_pages"], A_NUM_PAGES)
    a_configs = {"gundam": GUNDAM_KWARGS, "base": BASE_SINGLE_KWARGS}

    for cfg_name, kwargs in a_configs.items():
        for idx in page_indices:
            image = bk["images"][idx]
            out_dir = OUTPUT_DIR / pid / "A" / cfg_name / f"p{idx:04d}"
            in_tokens = infer.estimate_input_tokens(
                tokenizer, PROMPT_SINGLE, n_pages=1,
                image_tokens_per_page=IMAGE_TOKENS_PER_PAGE,
            )
            infer.execute_run(
                run_key=dict(book_id=pid, experiment="A_single", config_name=cfg_name,
                             page_range=str(idx), input_pages=1, max_length=kwargs["max_length"]),
                results_path=RESULTS_PATH, runtime=runtime, revision=REVISION, config=kwargs,
                runner=(lambda image=image, kwargs=kwargs, out_dir=out_dir:
                        infer.run_inference(model, tokenizer, mode="single", image_files=[image],
                                            prompt=PROMPT_SINGLE, output_dir=out_dir,
                                            infer_kwargs=kwargs,
                                            max_length_semantics=MAX_LENGTH_SEMANTICS,
                                            max_length_slack=MAX_LENGTH_SLACK)),
                max_length=kwargs["max_length"], max_length_slack=MAX_LENGTH_SLACK,
                max_length_semantics=MAX_LENGTH_SEMANTICS, input_tokens=in_tokens,
                prompt_tokens=infer.count_tokens(tokenizer, PROMPT_SINGLE),
                evaluator=make_evaluator(bk["ground_truth"], slice_to_span=True),
                force=FORCE,
            )
    log.info("Experiment A complete.")
else:
    log.warning("No books configured; skipping Experiment A.")
""")

# --------------------------------------------------------------------------- #
md(r"""
## 6. Experiment B — multi-page scaling (core experiment)

`infer_multi` on the first N pages for N = 1, 5, 10, 25, 50, 100, 200, then the full book.
When a run truncates (`hit_max_length`), it is retried at raised `max_length` caps
(`MAX_LENGTH_ESCALATION`) until it stops truncating or OOMs — every attempt is logged. N stops
increasing after the first OOM, or when output is still truncated at the highest cap.
`last_page_covered` uses 3 distinctive lines from the final page (>=2 must match at ratio >= 90).
""")

code(r"""
if not MAX_LENGTH_SEMANTICS_CONFIRMED:
    raise RuntimeError(
        "Set MAX_LENGTH_SEMANTICS_CONFIRMED = True in the config cell after "
        "confirming the section-3 max_length findings."
    )


def run_multi_n(pid, n, max_length):
    bk = books[pid]
    images = bk["images"][:n]
    kwargs = dict(MULTI_KWARGS, max_length=max_length)
    out_dir = OUTPUT_DIR / pid / "B" / f"N{n:04d}_ml{max_length}"
    in_tokens = infer.estimate_input_tokens(
        tokenizer, PROMPT_MULTI, n_pages=n, image_tokens_per_page=IMAGE_TOKENS_PER_PAGE
    )
    last_image = images[-1]
    lp_cache = OUTPUT_DIR / pid / "lastpage" / f"p{n - 1:04d}"

    def last_page_check(outcome):
        return infer.last_page_covered(
            model, tokenizer, last_image=last_image,
            multi_output_cleaned=outcome.cleaned_text, cache_dir=lp_cache,
            single_infer_kwargs=BASE_SINGLE_KWARGS, prompt=PROMPT_SINGLE,
        )

    return infer.execute_run(
        run_key=dict(book_id=pid, experiment="B_multi", config_name="base_multi",
                     page_range=f"0-{n - 1}", input_pages=n, max_length=max_length),
        results_path=RESULTS_PATH, runtime=runtime, revision=REVISION, config=kwargs,
        runner=(lambda: infer.run_inference(model, tokenizer, mode="multi", image_files=images,
                                            prompt=PROMPT_MULTI, output_dir=out_dir,
                                            infer_kwargs=kwargs,
                                            max_length_semantics=MAX_LENGTH_SEMANTICS,
                                            max_length_slack=MAX_LENGTH_SLACK)),
        max_length=max_length, max_length_slack=MAX_LENGTH_SLACK,
        max_length_semantics=MAX_LENGTH_SEMANTICS, input_tokens=in_tokens,
        prompt_tokens=infer.count_tokens(tokenizer, PROMPT_MULTI),
        evaluator=make_evaluator(bk["ground_truth"], slice_to_span=(n < bk["n_pages"])),
        last_page_check=last_page_check, force=FORCE,
    )


if books:
    pid = BOOKS[A_BOOK_INDEX]["id"]
    n_pages = books[pid]["n_pages"]
    n_values = sorted({n for n in B_N_VALUES if n <= n_pages} | {n_pages})

    for n in n_values:
        row = run_multi_n(pid, n, BASE_MAX_LENGTH)
        if row.get("error"):
            log.warning("Stopping B at N=%d after error: %s", n, row["error"])
            break
        if row.get("hit_max_length"):
            oomed = False
            last_attempt = row
            for cap in MAX_LENGTH_ESCALATION:
                log.info("N=%d truncated; retrying at max_length=%d", n, cap)
                last_attempt = run_multi_n(pid, n, cap)
                if last_attempt.get("error"):
                    log.warning("Escalation stopped at N=%d cap=%d: %s",
                                n, cap, last_attempt["error"])
                    oomed = True
                    break
                if not last_attempt.get("hit_max_length"):
                    break
            if oomed:
                break
            if last_attempt.get("hit_max_length"):
                log.warning("N=%d still truncated at the highest cap; stopping B.", n)
                break
    log.info("Experiment B complete.")
else:
    log.warning("No books configured; skipping Experiment B.")
""")

# --------------------------------------------------------------------------- #
md(r"""
## 7. Experiment C — chunked fallback

The full book in chunks of K pages (K = 10, 25, 50) with a 1-page overlap. Each chunk is scored
against its matching GT span, then chunks are stitched (fuzzy seam de-dup) and the stitched book
is scored against the full ground truth. A synthetic `C_stitched` summary row aggregates wall /
GPU / VRAM across the chunks.
""")

code(r"""
if not MAX_LENGTH_SEMANTICS_CONFIRMED:
    raise RuntimeError(
        "Set MAX_LENGTH_SEMANTICS_CONFIRMED = True in the config cell after "
        "confirming the section-3 max_length findings."
    )


def run_chunked(pid, k, overlap):
    bk = books[pid]
    n_pages = bk["n_pages"]
    ranges = metrics.iter_chunks(n_pages, chunk_size=k, overlap=overlap)
    cleaned_chunks = []
    total_wall = total_gpu = 0.0
    peak_vram = 0.0
    error = None

    for ci, (start, end) in enumerate(ranges):
        images = bk["images"][start:end]
        kwargs = dict(MULTI_KWARGS)
        out_dir = OUTPUT_DIR / pid / "C" / f"K{k}" / f"chunk{ci:03d}_{start}-{end - 1}"
        in_tokens = infer.estimate_input_tokens(
            tokenizer, PROMPT_MULTI, n_pages=end - start,
            image_tokens_per_page=IMAGE_TOKENS_PER_PAGE,
        )
        row = infer.execute_run(
            run_key=dict(book_id=pid, experiment="C_chunk", config_name=f"K{k}",
                         page_range=f"{start}-{end - 1}", input_pages=end - start,
                         max_length=kwargs["max_length"], chunk_index=ci),
            results_path=RESULTS_PATH, runtime=runtime, revision=REVISION, config=kwargs,
            runner=(lambda images=images, kwargs=kwargs, out_dir=out_dir:
                    infer.run_inference(model, tokenizer, mode="multi", image_files=images,
                                        prompt=PROMPT_MULTI, output_dir=out_dir,
                                        infer_kwargs=kwargs,
                                        max_length_semantics=MAX_LENGTH_SEMANTICS,
                                        max_length_slack=MAX_LENGTH_SLACK)),
            max_length=kwargs["max_length"], max_length_slack=MAX_LENGTH_SLACK,
            max_length_semantics=MAX_LENGTH_SEMANTICS, input_tokens=in_tokens,
            prompt_tokens=infer.count_tokens(tokenizer, PROMPT_MULTI),
            evaluator=make_evaluator(bk["ground_truth"], slice_to_span=True), force=FORCE,
        )
        if row.get("error"):
            error = row["error"]
            log.warning("Chunk %d (K=%d) failed: %s", ci, k, error)
            break
        total_wall += row.get("wall_seconds") or 0.0
        total_gpu += row.get("gpu_seconds") or 0.0
        peak_vram = max(peak_vram, row.get("peak_vram_gb") or 0.0)
        cleaned = ""
        if row.get("cleaned_path") and Path(row["cleaned_path"]).exists():
            cleaned = Path(row["cleaned_path"]).read_text(encoding="utf-8", errors="replace")
        cleaned_chunks.append(cleaned)

    stitched = metrics.stitch_chunks(cleaned_chunks)
    ev = metrics.evaluate(stitched, bk["ground_truth"], slice_to_span=False,
                          ngram_n=NGRAM_N, n_deciles=N_DECILES)
    summary = dict(
        book_id=pid, experiment="C_stitched", config_name=f"K{k}",
        page_range=f"0-{n_pages - 1}", input_pages=n_pages,
        dtype=runtime.dtype_str, gpu_name=runtime.gpu_name, revision=REVISION,
        torch_version=runtime.torch_version, cuda_version=runtime.cuda_version,
        wall_seconds=round(total_wall, 3), gpu_seconds=round(total_gpu, 3),
        peak_vram_gb=round(peak_vram, 3), error=error, **ev.as_row(),
    )
    infer.append_result(RESULTS_PATH, summary)
    log.info("C stitched K=%d: cer=%.4f coverage=%.3f", k, ev.cer_lower, ev.coverage)
    return summary


if books:
    pid = BOOKS[A_BOOK_INDEX]["id"]
    for k in C_K_VALUES:
        run_chunked(pid, k, C_OVERLAP)
    log.info("Experiment C complete.")
else:
    log.warning("No books configured; skipping Experiment C.")
""")

# --------------------------------------------------------------------------- #
md(r"""
## 8. Metrics & analysis

Load every row from `results.jsonl` and build the summary table (cost per 300-page book is
computed from wall-clock seconds at `GPU_PRICE_PER_HOUR`).
""")

code(r"""
import pandas as pd

rows = infer.load_results(RESULTS_PATH)
summary = report.build_summary(
    rows, price_per_hour=GPU_PRICE_PER_HOUR, pages=BOOK_PAGES_FOR_COST
)
pd.set_option("display.max_rows", 300)
pd.set_option("display.width", 200)
summary
""")

# --------------------------------------------------------------------------- #
md(r"""
## 9. Report — charts, samples, verdict
""")

code(r"""
# Scaling: wall time, peak VRAM, output tokens vs pages (with the max_length line).
report.plot_scaling(rows, REPORT_DIR, experiment="B_multi", max_length=BASE_MAX_LENGTH)
""")

code(r"""
# CER by decile for the largest successful B run and each stitched C run (the key drift chart).
series = {}
b_ok = [r for r in rows if r.get("experiment") == "B_multi" and not r.get("error") and r.get("decile_cer")]
if b_ok:
    best_b = max(b_ok, key=lambda r: r.get("input_pages", 0))
    series[f"B: N={best_b['input_pages']}"] = best_b["decile_cer"]
for r in rows:
    if r.get("experiment") == "C_stitched" and r.get("decile_cer") and not r.get("error"):
        series[f"C: {r['config_name']}"] = r["decile_cer"]
report.plot_decile_cer(series, REPORT_DIR)
""")

code(r"""
# 3 side-by-side samples: page image + cleaned OCR excerpt from early / middle / late.
if books:
    pid = BOOKS[A_BOOK_INDEX]["id"]
    bk = books[pid]
    samples = []
    for label, idx in report.pick_sample_indices(bk["n_pages"]).items():
        out_dir = OUTPUT_DIR / pid / "samples" / f"{label}_p{idx:04d}"
        outcome = infer.run_inference(
            model, tokenizer, mode="single", image_files=[bk["images"][idx]],
            prompt=PROMPT_SINGLE, output_dir=out_dir, infer_kwargs=GUNDAM_KWARGS,
            capture_generate=False,
        )
        samples.append(report.Sample(
            label=f"{label} (page {idx})", image_path=bk["images"][idx],
            excerpt=outcome.cleaned_text,
        ))
    report.show_samples(samples)
""")

code(r"""
# Automatic GO / NO-GO verdict against the configurable thresholds.
thresholds = report.Thresholds(
    max_cer=THRESH_MAX_CER, max_decile_ratio=THRESH_MAX_DECILE_RATIO,
    max_repetition_ratio=THRESH_MAX_REP_RATIO, max_cost_per_book=MAX_COST_PER_BOOK,
)


def _cost(row):
    wall, pages = row.get("wall_seconds"), row.get("input_pages")
    if not wall or not pages:
        return float("inf")
    return metrics.cost_per_book(wall / pages, GPU_PRICE_PER_HOUR, pages=BOOK_PAGES_FOR_COST)


candidates = []
if books:
    pid = BOOKS[A_BOOK_INDEX]["id"]
    n_pages = books[pid]["n_pages"]
    for r in rows:
        if r.get("error") or not r.get("decile_cer"):
            continue
        if r.get("experiment") == "B_multi" and r.get("input_pages") == n_pages:
            candidates.append(report.Candidate(
                name=f"single-pass (N={n_pages})", cer=r["cer_lower"],
                decile_cer=r["decile_cer"], repeated_ngram_share=r["repeated_ngram_share"],
                gt_repeated_ngram_share=r["gt_repeated_ngram_share"], cost_per_book=_cost(r),
                single_pass_ok=(not r.get("hit_max_length") and r.get("last_page_covered") is True),
            ))
        if r.get("experiment") == "C_stitched":
            candidates.append(report.Candidate(
                name=f"chunk {r['config_name']}", cer=r["cer_lower"],
                decile_cer=r["decile_cer"], repeated_ngram_share=r["repeated_ngram_share"],
                gt_repeated_ngram_share=r["gt_repeated_ngram_share"], cost_per_book=_cost(r),
            ))

verdict = report.decide(candidates, thresholds)
report.print_verdict(verdict)
""")


# --------------------------------------------------------------------------- #
def to_source(text: str) -> list[str]:
    """nbformat wants a list of lines, each ending in \n except the last."""
    lines = text.splitlines()
    return [line + "\n" for line in lines[:-1]] + [lines[-1]] if lines else []


notebook = {
    "cells": [
        (
            {"cell_type": "markdown", "metadata": {}, "source": to_source(src)}
            if kind == "markdown"
            else {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": to_source(src),
            }
        )
        for kind, src in cells
    ],
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "L4"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out_path = Path(__file__).with_name("spike.ipynb")
out_path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
# Validate it is loadable JSON.
json.loads(out_path.read_text(encoding="utf-8"))
print(f"wrote {out_path} with {len(cells)} cells")
