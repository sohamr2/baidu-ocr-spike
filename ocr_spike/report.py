"""Reporting: summary tables, charts, side-by-side samples, and the verdict.

Consumes the per-run rows (results.jsonl, optionally augmented with metrics
from :mod:`ocr_spike.metrics`) and produces the artifacts in the final notebook
section. Charts are saved to disk and returned so they also render inline.
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)

_SUMMARY_COLUMNS = [
    "book_id",
    "experiment",
    "config_name",
    "page_range",
    "input_pages",
    "dtype",
    "gpu_name",
    "torch_version",
    "revision",
    "max_length",
    "max_length_semantics",
    "input_tokens",
    "input_tokens_actual",
    "image_tokens_per_page_measured",
    "capture_fired",
    "output_tokens",
    "captured_total_tokens",
    "wall_seconds",
    "gpu_seconds",
    "peak_vram_gb",
    "hit_max_length",
    "stop_reason",
    "last_page_covered",
    "gt_span_start",
    "gt_span_end",
    "cer_lower",
    "cer_cased",
    "wer_lower",
    "coverage",
    "repeated_ngram_share",
    "gt_repeated_ngram_share",
    "cost_per_300",
    "error",
]


def build_summary(
    rows: Sequence[dict[str, Any]],
    *,
    price_per_hour: Optional[float] = None,
    pages: int = 300,
) -> pd.DataFrame:
    """Build a tidy summary table from result rows.

    Missing columns are filled with ``NA`` so partial runs still tabulate. When
    ``price_per_hour`` is given, ``cost_per_300`` is computed from
    ``wall_seconds`` (the billable clock) scaled to ``pages`` pages.
    """
    frame = pd.DataFrame(list(rows))
    if "wall_seconds" in frame.columns and "input_pages" in frame.columns and price_per_hour:
        per_page = frame["wall_seconds"] / frame["input_pages"]
        frame["cost_per_300"] = per_page * pages / 3600.0 * price_per_hour
    for column in _SUMMARY_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame[_SUMMARY_COLUMNS]


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
def _scaling_frame(rows: Sequence[dict[str, Any]], experiment: str) -> pd.DataFrame:
    frame = pd.DataFrame(list(rows))
    if frame.empty or "experiment" not in frame:
        return pd.DataFrame()
    frame = frame[(frame["experiment"] == experiment) & frame["error"].isna()]
    if frame.empty:
        return frame
    return frame.sort_values("input_pages")


def plot_scaling(
    rows: Sequence[dict[str, Any]],
    out_dir: Path,
    *,
    experiment: str = "B_multi",
    max_length: Optional[int] = None,
) -> Optional[Path]:
    """Plot wall time, peak VRAM, and output tokens vs input page count.

    The ``max_length`` line is drawn on the output-tokens panel so truncation is
    obvious. Returns the saved figure path (``None`` if there is nothing to plot).
    """
    frame = _scaling_frame(rows, experiment)
    if frame.empty:
        logger.warning("No successful '%s' rows to plot scaling.", experiment)
        return None

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    axes[0].plot(frame["input_pages"], frame["wall_seconds"], marker="o")
    axes[0].set_title("Wall time vs pages")
    axes[0].set_xlabel("input pages")
    axes[0].set_ylabel("wall seconds")

    axes[1].plot(frame["input_pages"], frame["peak_vram_gb"], marker="o", color="tab:red")
    axes[1].set_title("Peak VRAM vs pages")
    axes[1].set_xlabel("input pages")
    axes[1].set_ylabel("peak VRAM (GB)")

    axes[2].plot(
        frame["input_pages"], frame["output_tokens"], marker="o", color="tab:green"
    )
    if max_length is not None:
        axes[2].axhline(
            max_length, color="black", linestyle="--", label=f"max_length={max_length}"
        )
        axes[2].legend()
    axes[2].set_title("Output tokens vs pages")
    axes[2].set_xlabel("input pages")
    axes[2].set_ylabel("output tokens")

    for axis in axes:
        axis.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out_dir / "scaling.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    return path


def plot_decile_cer(
    series: dict[str, Sequence[float]],
    out_dir: Path,
) -> Optional[Path]:
    """Plot CER per decile for each labelled approach (the key drift chart).

    ``series`` maps a label (e.g. ``"B: 100 pages"``, ``"C: K=25"``) to its
    list of 10 per-decile CER values.
    """
    series = {k: v for k, v in series.items() if v}
    if not series:
        logger.warning("No decile CER series to plot.")
        return None

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(9, 5))
    deciles = list(range(1, 11))
    for label, values in series.items():
        axis.plot(deciles[: len(values)], values, marker="o", label=label)
    axis.set_title("CER by document decile (late-document drift)")
    axis.set_xlabel("document decile (1 = start, 10 = end)")
    axis.set_ylabel("CER")
    axis.set_xticks(deciles)
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.tight_layout()
    path = out_dir / "decile_cer.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    return path


# --------------------------------------------------------------------------- #
# Side-by-side samples
# --------------------------------------------------------------------------- #
@dataclass
class Sample:
    """A page image paired with its cleaned OCR excerpt."""

    label: str
    image_path: Path
    excerpt: str


def pick_sample_indices(n_pages: int) -> dict[str, int]:
    """Return early/middle/late page indices for qualitative review."""
    if n_pages <= 0:
        return {}
    return {
        "early": 0,
        "middle": n_pages // 2,
        "late": n_pages - 1,
    }


def show_samples(samples: Sequence[Sample], *, excerpt_chars: int = 600) -> None:
    """Render page image next to its cleaned OCR excerpt for each sample.

    Intended for an interactive notebook; uses matplotlib for the image and
    prints the excerpt beneath it.
    """
    import matplotlib.image as mpimg

    for sample in samples:
        fig, axis = plt.subplots(figsize=(6, 8))
        try:
            axis.imshow(mpimg.imread(str(sample.image_path)))
        except (OSError, ValueError) as exc:
            logger.warning("Could not show image %s: %r", sample.image_path, exc)
        axis.axis("off")
        axis.set_title(f"{sample.label} — {Path(sample.image_path).name}")
        plt.show()
        print(f"\n[{sample.label}] cleaned OCR excerpt:\n")
        print(sample.excerpt[:excerpt_chars])
        print("\n" + "-" * 72 + "\n")


# --------------------------------------------------------------------------- #
# Verdict
# --------------------------------------------------------------------------- #
@dataclass
class Thresholds:
    """Configurable GO / NO-GO thresholds."""

    max_cer: float = 0.03
    max_decile_ratio: float = 2.0
    max_repetition_ratio: float = 1.5
    max_cost_per_book: float = 1.00


@dataclass
class Candidate:
    """One approach to evaluate against thresholds.

    ``single_pass_ok`` is only meaningful for the single-pass candidate: it
    records whether the whole book fit in one successful, non-truncated pass.
    """

    name: str
    cer: float
    decile_cer: list[float]
    repeated_ngram_share: float
    gt_repeated_ngram_share: float
    cost_per_book: float
    single_pass_ok: bool = False


@dataclass
class CandidateResult:
    """Per-candidate threshold evaluation."""

    name: str
    passed: bool
    checks: dict[str, bool]
    detail: dict[str, float]


@dataclass
class Verdict:
    """Overall GO / NO-GO decision."""

    go: bool
    winner: Optional[str]
    reason: str
    candidates: list[CandidateResult] = field(default_factory=list)


def _evaluate_candidate(candidate: Candidate, thresholds: Thresholds) -> CandidateResult:
    median_decile = statistics.median(candidate.decile_cer) if candidate.decile_cer else 0.0
    worst_decile = max(candidate.decile_cer) if candidate.decile_cer else 0.0
    decile_ratio = (worst_decile / median_decile) if median_decile > 0 else float("inf")
    gt_share = candidate.gt_repeated_ngram_share
    rep_ratio = (
        candidate.repeated_ngram_share / gt_share
        if gt_share > 0
        else (1.0 if candidate.repeated_ngram_share == 0 else float("inf"))
    )

    checks = {
        "cer": candidate.cer <= thresholds.max_cer,
        "decile_ratio": decile_ratio <= thresholds.max_decile_ratio,
        "repetition_ratio": rep_ratio <= thresholds.max_repetition_ratio,
        "cost": candidate.cost_per_book <= thresholds.max_cost_per_book,
    }
    detail = {
        "cer": candidate.cer,
        "worst_decile_cer": worst_decile,
        "median_decile_cer": median_decile,
        "decile_ratio": decile_ratio,
        "repetition_ratio": rep_ratio,
        "cost_per_book": candidate.cost_per_book,
    }
    return CandidateResult(
        name=candidate.name,
        passed=all(checks.values()),
        checks=checks,
        detail=detail,
    )


def decide(candidates: Sequence[Candidate], thresholds: Thresholds) -> Verdict:
    """Compute the GO / NO-GO verdict and pick a winning approach.

    GO requires at least one candidate to pass every threshold. A passing
    single-pass candidate wins outright (simplest pipeline); otherwise the
    cheapest passing chunk approach wins.
    """
    results = [_evaluate_candidate(c, thresholds) for c in candidates]
    by_name = {c.name: c for c in candidates}
    passing = [r for r in results if r.passed]

    if not passing:
        return Verdict(
            go=False,
            winner=None,
            reason="No approach met all thresholds (CER, decile drift, repetition, cost).",
            candidates=results,
        )

    single = next(
        (r for r in passing if by_name[r.name].single_pass_ok),
        None,
    )
    if single is not None:
        return Verdict(
            go=True,
            winner=single.name,
            reason="Single-pass handled the whole book within all thresholds.",
            candidates=results,
        )

    cheapest = min(passing, key=lambda r: r.detail["cost_per_book"])
    return Verdict(
        go=True,
        winner=cheapest.name,
        reason=(
            "Single-pass did not qualify; "
            f"'{cheapest.name}' met every threshold at the lowest cost "
            f"(${cheapest.detail['cost_per_book']:.3f}/book)."
        ),
        candidates=results,
    )


def print_verdict(verdict: Verdict) -> None:
    """Print a human-readable GO / NO-GO summary."""
    banner = "GO" if verdict.go else "NO-GO"
    print("=" * 72)
    print(f"VERDICT: {banner}")
    if verdict.winner:
        print(f"Winning approach: {verdict.winner}")
    print(f"Reason: {verdict.reason}")
    print("=" * 72)
    for result in verdict.candidates:
        status = "PASS" if result.passed else "FAIL"
        print(f"\n[{status}] {result.name}")
        for check, ok in result.checks.items():
            value = result.detail.get(check)
            shown = f" ({value:.4f})" if isinstance(value, (int, float)) else ""
            print(f"    - {check:16s}: {'ok' if ok else 'FAIL'}{shown}")
