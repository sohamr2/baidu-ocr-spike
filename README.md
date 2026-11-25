# Baidu Unlimited-OCR — Phase 0 spike

A throwaway-quality **spike** (not the product) to decide whether
[`baidu/Unlimited-OCR`](https://huggingface.co/baidu/Unlimited-OCR) (MIT, ~3B MoE on
DeepSeek-OCR) is good enough to turn 200–600 page scanned books into clean text. There is **no**
web app, FastAPI, or EPUB generation here — only measurements that answer four questions:

1. Does multi-page "whole book in one pass" work, or does it truncate / OOM / drift?
2. How accurate is it, and does accuracy drop later in long documents (the model's R-SWA
   sliding-window attention makes late-document drift the main worry)?
3. How much GPU time and VRAM per page, and what does a 300-page book cost?
4. If single-pass fails, does chunking (N pages at a time with overlap) work as a fallback?

## Layout

```
baidu-ocr-spike/
├── spike.ipynb            # run top-to-bottom on a Colab GPU runtime
├── build_notebook.py      # regenerates spike.ipynb from source cells (dev convenience)
├── requirements-spike.txt # pinned dependencies
├── ocr_spike/             # all reusable logic (notebook cells stay thin)
│   ├── pdf.py             # PDF -> cached 300-DPI PNGs, even page sampling
│   ├── infer.py           # runtime/dtype probe, model load, API introspection,
│   │                      #   inference wrappers, generate-capture, experiment runner
│   ├── metrics.py         # normalization, CER/WER, decile drift, coverage, repetition,
│   │                      #   GT-span infix location, chunk ranging + seam de-dup, cost
│   └── report.py          # summary table, charts, samples, GO/NO-GO verdict
└── tests/
    └── test_metrics.py    # CPU-only unit tests (metrics + det-marker parser)
```

## Run it in Colab

1. **GPU runtime**: `Runtime > Change runtime type > GPU` (L4 24 GB or A100). The model card uses
   `torch.bfloat16`; a free-tier **T4 does not support bf16** and the notebook will loudly fall
   back to `float16` (recorded in every result row). Results are only comparable to the model
   card on an L4/A100.
2. Get this repo onto Colab, e.g. `!git clone <your-repo-url>` then `%cd baidu-ocr-spike`, so
   `requirements-spike.txt` and the `ocr_spike` package are in the working directory.
3. Open `spike.ipynb` and run cells **in order**:
   - **Cell 0 (Install first)** runs before anything else. It checks the NVIDIA driver against the
     CUDA 12.9 torch wheels (falling back to a matching `cu12x` wheel if needed), installs the
     pinned deps, and prints **RESTART REQUIRED** if Colab shipped a different torch. If so,
     `Runtime > Restart session` and re-run from the **next** cell (skip cell 0).
   - **Config cell**: set `BOOKS` (your 3 books), `GPU_PRICE_PER_HOUR`, and thresholds.
   - The rest mount Drive, probe the GPU, load the model at a pinned commit, prep data, and run
     the experiments.
4. Everything (inputs, cached page images, raw/cleaned outputs, `results.jsonl`, report charts)
   is written under `MyDrive/ocr_spike/`, so a disconnect does not lose work. Re-running skips
   runs whose results already exist unless you set `FORCE = True`.

### Test data

Use public-domain books (pre-1929) that exist both as an Internet Archive scan (the PDF) and as a
Project Gutenberg plain-text file (the ground truth), ideally the same edition. Pick variety:
clean prose, one with footnotes, one with tables / two columns. Point `BOOKS` at the files:

```python
BOOKS = [
    dict(id="clean_prose", pdf_path=".../book1.pdf",
         ground_truth_path=".../book1_gutenberg.txt", notes="clean single-column prose"),
    # ... two more
]
```

Gutenberg header/license boilerplate is stripped automatically before comparison.

## What each experiment measures

- **A — single-page baseline.** 10 pages sampled evenly from one book, in both `gundam` and
  `base` single-page configs. Per-page accuracy and speed with no long-context effects.
- **B — multi-page scaling (the core experiment).** `infer_multi` on the first N pages for
  N = 1, 5, 10, 25, 50, 100, 200, then the full book. When a run **truncates**, it is retried at
  raised `max_length` caps (65536, 131072, configurable) until it stops truncating or OOMs — every
  attempt is logged. N stops growing after the first OOM, or when output is still truncated at the
  highest cap. This directly tests whether "whole book in one pass" holds.
- **C — chunked fallback.** The full book in chunks of K pages (K = 10, 25, 50) with a 1-page
  overlap. Each chunk is scored against its matching ground-truth span; chunks are then stitched
  with fuzzy seam de-duplication (rapidfuzz) and the stitched book is scored against the full
  ground truth.

### `max_length` and truncation

The model card passes `max_length=32768`. A 300-page book is ~150k+ output tokens, so at that cap
single-pass is expected to truncate. Section 3 of the notebook reads the remote modeling code to
determine whether `max_length` reaches `generate()` as **`max_length`** (prompt + image + output)
or **`max_new_tokens`** (output only); set `MAX_LENGTH_SEMANTICS` accordingly. Each run records
input tokens, output tokens, `hit_max_length`, and a `stop_reason`. When the remote code routes
through `model.generate`, the true total token count and EOS-vs-length stop reason are captured
directly rather than approximated.

## Metrics

Both texts are normalized identically (NFKC, markers stripped, hyphenated line-breaks rejoined,
whitespace collapsed, lowercased for the main metric; a case-sensitive CER is also reported).

- **CER / WER** via `jiwer`. For any partial-book output (Exp A pages, Exp B first-N runs, each
  Exp C chunk), the matching ground-truth span is located by **edlib infix alignment**
  (`mode="HW"`) and metrics are computed against that span only; the span offsets are recorded.
- **Drift by position**: a global edlib alignment split into 10 deciles, error rate per decile —
  the most important chart, since R-SWA models tend to degrade late in long documents.
- **Repetition / hallucination**: share of repeated 8-grams vs ground truth, and the longest
  repeated run.
- **Coverage**: share of ground-truth characters aligned to something (catches skipped pages).
- **Cost**: wall-clock GPU seconds per page × `GPU_PRICE_PER_HOUR`, projected to a 300-page book.

## How to read the verdict

The last cell prints an automatic **GO / NO-GO** decision against configurable thresholds:

- overall CER ≤ 3% on clean prose,
- no decile's CER above 2× the median decile,
- repeated-8-gram share no more than 1.5× ground truth,
- cost per 300-page book ≤ `MAX_COST_PER_BOOK` (default $1.00),
- and **either** full single-pass works (no truncation, last page covered) **or** a chunk size K
  meets every threshold.

A passing single-pass wins outright (simplest pipeline); otherwise the cheapest passing chunk
size wins. Each candidate's per-check PASS/FAIL and numbers are printed so you can see *why*.

## Tests

`tests/test_metrics.py` covers normalization, the det-marker parser, CER/WER, decile drift (with
errors injected only in the last 10%), coverage, GT-span location, repetition, chunk ranging, and
seam de-duplication. They are CPU-only (no torch):

```bash
pip install pytest jiwer rapidfuzz edlib
pytest tests -q
```

On Windows, `edlib` has no prebuilt wheel and needs a C toolchain; those alignment tests skip
automatically if `edlib` is missing. On the Colab/Linux target all tests run.

## Notes on the model

- Loaded from the **official** `baidu/Unlimited-OCR` repo only, at a resolved commit SHA (written
  to every result row). It requires `trust_remote_code=True`, which executes repo code — do not
  point this at random re-uploads.
- Decoding uses the remote code's greedy defaults plus the `no_repeat_ngram_*` settings from the
  model card; the full inference kwargs are recorded in each row's `config`.
