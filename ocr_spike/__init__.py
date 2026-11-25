"""Helpers for the Baidu Unlimited-OCR Phase 0 spike.

This package keeps the notebook cells thin: all reusable logic (PDF rasterizing,
model loading / inference, metrics, reporting) lives here so it can be lifted
into the real pipeline later.

Modules
-------
pdf
    Rasterize PDFs to cached PNGs and sample page indices.
infer
    Runtime/dtype detection, model loading at a pinned revision, API
    introspection, inference wrappers, output parsing, and the result-logging
    experiment runner.
metrics
    Text normalization, CER/WER, positional (decile) drift, repetition, and
    coverage, plus cost estimation.
report
    Summary tables, charts, side-by-side samples, and the GO / NO-GO verdict.
"""

from __future__ import annotations

__all__ = ["pdf", "infer", "metrics", "report"]
__version__ = "0.1.0"
