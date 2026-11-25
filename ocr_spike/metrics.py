"""Text normalization and accuracy metrics for the OCR spike (CPU-only).

This module deliberately avoids importing torch so the unit tests can run on a
plain CPU machine. It covers:

* marker parsing and normalization (NFKC, marker strip, hyphen rejoin,
  whitespace collapse, optional lowercase);
* ground-truth span location via edlib **infix** alignment (``mode="HW"``) so
  partial-book outputs are scored only against the matching span;
* CER / WER via :mod:`jiwer`;
* positional (decile) drift and coverage via an edlib global alignment;
* repetition / hallucination statistics;
* chunk ranging and fuzzy seam de-duplication for the chunked fallback; and
* a GPU-cost estimate per book.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Sequence

import jiwer
from rapidfuzz import fuzz

# edlib is imported lazily inside the functions that need it: it has no
# Windows source-build-free fallback, but the alignment-based metrics are the
# only consumers, so the rest of the module (parsing, normalization, seam
# de-duplication, repetition) stays importable without a C toolchain.

logger = logging.getLogger(__name__)

# Layout/control markers emitted by the model, e.g. ``<|det|>`` ... ``<|/det|>``.
_DET_RE = re.compile(r"<\|det\|>(.*?)<\|/det\|>", re.DOTALL)
_TAG_RE = re.compile(r"<\|/?[A-Za-z0-9_]+\|>")
# Project Gutenberg boilerplate delimiters.
_GUTENBERG_START = re.compile(
    r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
    re.IGNORECASE | re.DOTALL,
)
_GUTENBERG_END = re.compile(
    r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
    re.IGNORECASE | re.DOTALL,
)


# --------------------------------------------------------------------------- #
# Parsing / normalization
# --------------------------------------------------------------------------- #
@dataclass
class ParsedOutput:
    """Parsed model output.

    Attributes
    ----------
    raw:
        The unmodified model text.
    cleaned:
        ``raw`` with layout markers removed (still pre-normalization).
    block_types:
        Ordered list of block type labels found inside ``<|det|>`` markers
        (e.g. ``text``, ``table``, ``formula``, ``image``).
    """

    raw: str
    cleaned: str
    block_types: list[str] = field(default_factory=list)


def parse_markers(raw: str) -> ParsedOutput:
    """Split raw model output into raw text, marker-free text, and block types.

    The exact marker grammar comes from the model's remote code; this parser is
    deliberately defensive. The notebook prints real output so the grammar can
    be confirmed before the numbers are trusted.
    """
    block_types: list[str] = []
    for match in _DET_RE.finditer(raw):
        label = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)", match.group(1))
        if label:
            block_types.append(label.group(1).lower())
    # Remove whole det blocks (type + bbox metadata), then any remaining tags.
    cleaned = _DET_RE.sub(" ", raw)
    cleaned = _TAG_RE.sub(" ", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return ParsedOutput(raw=raw, cleaned=cleaned, block_types=block_types)


def strip_gutenberg_boilerplate(text: str) -> str:
    """Return the body of a Project Gutenberg text with header/license removed."""
    start_match = _GUTENBERG_START.search(text)
    end_match = _GUTENBERG_END.search(text)
    start = start_match.end() if start_match else 0
    end = end_match.start() if end_match else len(text)
    if end <= start:
        logger.warning("Gutenberg markers out of order or missing; using full text.")
        return text.strip()
    return text[start:end].strip()


def normalize_text(
    text: str,
    *,
    lowercase: bool = True,
    rejoin_hyphens: bool = True,
    strip_markers: bool = True,
) -> str:
    """Normalize ``text`` for comparison.

    Steps: NFKC → optional marker strip → optional hyphenated line-break
    rejoin → whitespace collapse → optional lowercase.
    """
    text = unicodedata.normalize("NFKC", text)
    if strip_markers:
        text = _DET_RE.sub(" ", text)
        text = _TAG_RE.sub(" ", text)
    if rejoin_hyphens:
        # "exam-\nple" -> "example"
        text = re.sub(r"-\s*\n\s*", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    if lowercase:
        text = text.lower()
    return text


# --------------------------------------------------------------------------- #
# Ground-truth span location (infix alignment)
# --------------------------------------------------------------------------- #
@dataclass
class SpanMatch:
    """Where a (normalized) hypothesis best matches inside a reference."""

    start: int
    end: int  # exclusive
    text: str


def locate_gt_span(hypothesis_norm: str, reference_norm: str) -> SpanMatch:
    """Locate the reference span that best matches ``hypothesis_norm``.

    Uses edlib infix alignment (``mode="HW"``): the hypothesis is aligned as a
    substring of the reference, so a partial-book OCR output is scored only
    against the matching ground-truth span. Offsets are in normalized-reference
    character coordinates.
    """
    import edlib

    if not hypothesis_norm or not reference_norm:
        return SpanMatch(0, len(reference_norm), reference_norm)
    result = edlib.align(hypothesis_norm, reference_norm, task="locations", mode="HW")
    locations = result.get("locations") or []
    if not locations:
        return SpanMatch(0, len(reference_norm), reference_norm)
    start, end = locations[0]
    start = 0 if start is None else int(start)
    end = len(reference_norm) - 1 if end is None else int(end)
    end_exclusive = min(end + 1, len(reference_norm))
    return SpanMatch(start, end_exclusive, reference_norm[start:end_exclusive])


# --------------------------------------------------------------------------- #
# Error rates
# --------------------------------------------------------------------------- #
def cer(reference: str, hypothesis: str) -> float:
    """Character error rate via jiwer. Returns 0.0 for an empty reference."""
    if not reference:
        return 0.0
    return float(jiwer.cer(reference, hypothesis))


def wer(reference: str, hypothesis: str) -> float:
    """Word error rate via jiwer. Returns 0.0 for an empty reference."""
    if not reference:
        return 0.0
    return float(jiwer.wer(reference, hypothesis))


def _iter_cigar(cigar: str):
    """Yield ``(count, op)`` tuples from an edlib CIGAR string."""
    for count, op in re.findall(r"(\d+)([=XID])", cigar):
        yield int(count), op


@dataclass
class AlignmentReport:
    """Positional alignment statistics from a global edlib alignment.

    ``decile_cer`` is the per-decile error rate over the reference; ``coverage``
    is the share of reference characters aligned to a match or substitution
    (i.e. not deleted), which catches skipped pages.
    """

    overall_cer: float
    decile_cer: list[float]
    coverage: float


def align_report(reference: str, hypothesis: str, *, n_deciles: int = 10) -> AlignmentReport:
    """Align ``hypothesis`` to ``reference`` and compute positional error rates.

    Uses an edlib global (NW) path alignment. Errors are attributed to the
    reference decile they occur in; insertions (extra hypothesis characters)
    are charged to the current reference position's decile.
    """
    import edlib

    if not reference:
        return AlignmentReport(0.0, [0.0] * n_deciles, 0.0)

    result = edlib.align(hypothesis, reference, task="path", mode="NW")
    ref_len = len(reference)
    bucket = ref_len / n_deciles
    errors = [0] * n_deciles
    ref_counts = [0] * n_deciles
    total_errors = 0
    aligned = 0  # matches + substitutions (reference chars covered)
    ref_pos = 0

    for count, op in _iter_cigar(result["cigar"]):
        for _ in range(count):
            decile = min(int(ref_pos / bucket), n_deciles - 1) if bucket else n_deciles - 1
            if op == "=":
                ref_counts[decile] += 1
                aligned += 1
                ref_pos += 1
            elif op == "X":
                ref_counts[decile] += 1
                errors[decile] += 1
                total_errors += 1
                aligned += 1
                ref_pos += 1
            elif op == "D":  # char in reference missing from hypothesis
                ref_counts[decile] += 1
                errors[decile] += 1
                total_errors += 1
                ref_pos += 1
            elif op == "I":  # extra char in hypothesis
                errors[decile] += 1
                total_errors += 1

    decile_cer = [
        (errors[i] / ref_counts[i]) if ref_counts[i] else 0.0 for i in range(n_deciles)
    ]
    overall = total_errors / ref_len
    coverage = aligned / ref_len
    return AlignmentReport(overall_cer=overall, decile_cer=decile_cer, coverage=coverage)


# --------------------------------------------------------------------------- #
# Repetition / hallucination
# --------------------------------------------------------------------------- #
@dataclass
class RepetitionStats:
    """Repetition statistics over whitespace-split tokens."""

    repeated_ngram_share: float
    longest_repeated_run: int


def repetition_stats(text: str, *, n: int = 8) -> RepetitionStats:
    """Share of ``n``-grams that occur more than once, and longest looping run.

    ``repeated_ngram_share`` is the fraction of ``n``-gram occurrences whose
    ``n``-gram appears at least twice. ``longest_repeated_run`` is the longest
    run of consecutive identical tokens (a common sliding-window loop failure).
    """
    tokens = text.split()
    if len(tokens) < n:
        return RepetitionStats(0.0, _longest_consecutive_run(tokens))
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    counts = Counter(ngrams)
    repeated = sum(c for c in counts.values() if c > 1)
    share = repeated / len(ngrams)
    return RepetitionStats(share, _longest_consecutive_run(tokens))


def _longest_consecutive_run(tokens: list[str]) -> int:
    """Longest run of consecutive identical tokens."""
    longest = 0
    current = 0
    previous: str | None = None
    for token in tokens:
        if token == previous:
            current += 1
        else:
            current = 1
            previous = token
        longest = max(longest, current)
    return longest


# --------------------------------------------------------------------------- #
# Chunk ranging and seam de-duplication (Experiment C)
# --------------------------------------------------------------------------- #
def iter_chunks(n_pages: int, chunk_size: int, overlap: int = 1) -> list[tuple[int, int]]:
    """Return ``(start, end)`` page ranges (end exclusive) covering a book.

    Consecutive chunks share ``overlap`` pages so the seam can be
    de-duplicated after OCR.
    """
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")
    if n_pages <= 0:
        return []
    ranges: list[tuple[int, int]] = []
    start = 0
    while start < n_pages:
        end = min(start + chunk_size, n_pages)
        ranges.append((start, end))
        if end >= n_pages:
            break
        start = end - overlap
    return ranges


def _overlap_cut(tail: str, head: str, *, min_score: float) -> int:
    """Return how many leading chars of ``head`` duplicate the end of ``tail``.

    Compares equal-length windows: the first ``length`` chars of ``head`` against
    the last ``length`` chars of ``tail``. The real overlap is a sharp
    similarity peak (prefix-of-head exactly equals suffix-of-tail), so the
    length that *maximizes* ``fuzz.ratio`` is chosen and only cut when that peak
    clears ``min_score``; otherwise 0 (no confident overlap).
    """
    max_len = min(len(tail), len(head))
    best_length = 0
    best_score = 0.0
    length = max_len
    while length > 20:
        score = fuzz.ratio(head[:length], tail[-length:])
        if score > best_score:
            best_score = score
            best_length = length
        length -= 5
    return best_length if best_score >= min_score else 0


def stitch_chunks(
    chunk_texts: Sequence[str],
    *,
    seam_window: int = 400,
    min_score: float = 70.0,
) -> str:
    """Concatenate chunk outputs, removing fuzzily-duplicated overlap text.

    For each seam, the tail of the accumulated text is compared against the head
    of the next chunk using rapidfuzz ``partial_ratio``; when the overlap scores
    above ``min_score`` the duplicated prefix is dropped from the next chunk.
    """
    stitched = ""
    for text in chunk_texts:
        text = text.strip()
        if not text:
            continue
        if not stitched:
            stitched = text
            continue
        tail = stitched[-seam_window:]
        head = text[:seam_window]
        cut = _overlap_cut(tail, head, min_score=min_score)
        stitched = stitched + "\n" + text[cut:]
    return stitched


# --------------------------------------------------------------------------- #
# Cost
# --------------------------------------------------------------------------- #
def cost_per_book(
    wall_seconds_per_page: float,
    price_per_hour: float,
    *,
    pages: int = 300,
) -> float:
    """Estimated GPU cost to process a ``pages``-page book, in dollars.

    Cost is based on wall-clock seconds (what you actually pay for), not pure
    GPU kernel time.
    """
    return wall_seconds_per_page / 3600.0 * price_per_hour * pages


# --------------------------------------------------------------------------- #
# One-shot evaluation
# --------------------------------------------------------------------------- #
@dataclass
class Evaluation:
    """All metrics for a single OCR output against its ground-truth slice."""

    cer_lower: float
    cer_cased: float
    wer_lower: float
    decile_cer: list[float]
    coverage: float
    repeated_ngram_share: float
    longest_repeated_run: int
    gt_repeated_ngram_share: float
    gt_span_start: int
    gt_span_end: int
    gt_chars: int
    ocr_chars: int

    def as_row(self) -> dict:
        """Flatten to primitive fields for a results.jsonl row."""
        return {
            "cer_lower": self.cer_lower,
            "cer_cased": self.cer_cased,
            "wer_lower": self.wer_lower,
            "decile_cer": self.decile_cer,
            "coverage": self.coverage,
            "repeated_ngram_share": self.repeated_ngram_share,
            "longest_repeated_run": self.longest_repeated_run,
            "gt_repeated_ngram_share": self.gt_repeated_ngram_share,
            "gt_span_start": self.gt_span_start,
            "gt_span_end": self.gt_span_end,
            "gt_chars": self.gt_chars,
            "ocr_chars": self.ocr_chars,
        }


def evaluate(
    ocr_text: str,
    ground_truth: str,
    *,
    slice_to_span: bool = False,
    ngram_n: int = 8,
    n_deciles: int = 10,
) -> Evaluation:
    """Compute the full metric set for one OCR output vs ground truth.

    ``ocr_text`` should be the cleaned (marker-free) model output; it is
    normalized here the same way as ``ground_truth``.

    When ``slice_to_span`` is True (any partial-book output: Exp A pages, Exp B
    first-N runs, each Exp C chunk before stitching), the matching reference
    span is located by infix alignment and all metrics are computed against that
    span only. The span offsets (normalized-reference coordinates) are recorded.
    """
    ref_lower_full = normalize_text(ground_truth, lowercase=True)
    hyp_lower = normalize_text(ocr_text, lowercase=True)
    ref_cased_full = normalize_text(ground_truth, lowercase=False)
    hyp_cased = normalize_text(ocr_text, lowercase=False)

    if slice_to_span:
        span_lower = locate_gt_span(hyp_lower, ref_lower_full)
        span_cased = locate_gt_span(hyp_cased, ref_cased_full)
        ref_lower = span_lower.text
        ref_cased = span_cased.text
        start, end = span_lower.start, span_lower.end
    else:
        ref_lower = ref_lower_full
        ref_cased = ref_cased_full
        start, end = 0, len(ref_lower_full)

    alignment = align_report(ref_lower, hyp_lower, n_deciles=n_deciles)
    hyp_rep = repetition_stats(hyp_lower, n=ngram_n)
    ref_rep = repetition_stats(ref_lower, n=ngram_n)

    return Evaluation(
        cer_lower=cer(ref_lower, hyp_lower),
        cer_cased=cer(ref_cased, hyp_cased),
        wer_lower=wer(ref_lower, hyp_lower),
        decile_cer=alignment.decile_cer,
        coverage=alignment.coverage,
        repeated_ngram_share=hyp_rep.repeated_ngram_share,
        longest_repeated_run=hyp_rep.longest_repeated_run,
        gt_repeated_ngram_share=ref_rep.repeated_ngram_share,
        gt_span_start=start,
        gt_span_end=end,
        gt_chars=len(ref_lower),
        ocr_chars=len(hyp_lower),
    )
