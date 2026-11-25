"""CPU-only unit tests for ocr_spike.metrics and the det-marker parser.

These import only :mod:`ocr_spike.metrics` (edlib / jiwer / rapidfuzz) and never
torch, so they run anywhere ``pytest`` does.
"""

from __future__ import annotations

import importlib.util
import string

import pytest

from ocr_spike import metrics

# edlib builds from source on Windows (no wheel); the alignment-based tests are
# skipped where it is unavailable. On the Colab/Linux target edlib installs as a
# wheel and every test runs.
_HAS_EDLIB = importlib.util.find_spec("edlib") is not None
requires_edlib = pytest.mark.skipif(not _HAS_EDLIB, reason="edlib not installed")


# --------------------------------------------------------------------------- #
# Normalization
# --------------------------------------------------------------------------- #
def test_normalize_collapses_whitespace_and_lowercases() -> None:
    raw = "Hello   WORLD\t\n  Foo"
    assert metrics.normalize_text(raw) == "hello world foo"


def test_normalize_preserves_case_when_disabled() -> None:
    assert metrics.normalize_text("Hello World", lowercase=False) == "Hello World"


def test_normalize_rejoins_hyphenated_linebreaks() -> None:
    raw = "exam-\nple of hyphen-\n  ation"
    assert metrics.normalize_text(raw) == "example of hyphenation"


def test_normalize_nfkc_folds_ligatures_and_fullwidth() -> None:
    # ﬁ ligature (U+FB01) -> "fi"; fullwidth A (U+FF21) -> "a"
    assert metrics.normalize_text("\ufb01le \uff21") == "file a"


def test_normalize_strips_markers() -> None:
    raw = "before <|det|>text [[1,2,3,4]]<|/det|> after <|ref|>x<|/ref|>"
    out = metrics.normalize_text(raw)
    assert "<|" not in out
    assert "before" in out and "after" in out


# --------------------------------------------------------------------------- #
# Det-marker parser
# --------------------------------------------------------------------------- #
def test_parse_markers_extracts_types_and_cleans() -> None:
    raw = (
        "intro text\n"
        "<|det|>text [[10, 20, 30, 40]]<|/det|>the body paragraph\n"
        "<|det|>table [[0,0,5,5]]<|/det|>\n"
        "<|det|>formula [[1,1,2,2]]<|/det|>"
    )
    parsed = metrics.parse_markers(raw)
    assert parsed.block_types == ["text", "table", "formula"]
    assert "<|det|>" not in parsed.cleaned
    assert "[[10, 20, 30, 40]]" not in parsed.cleaned
    assert "the body paragraph" in parsed.cleaned
    assert parsed.raw == raw  # raw is preserved verbatim


def test_parse_markers_no_markers_is_passthrough() -> None:
    parsed = metrics.parse_markers("plain text with no markers")
    assert parsed.block_types == []
    assert parsed.cleaned == "plain text with no markers"


# --------------------------------------------------------------------------- #
# CER / WER
# --------------------------------------------------------------------------- #
def test_cer_zero_for_identical() -> None:
    assert metrics.cer("the quick brown fox", "the quick brown fox") == 0.0


def test_cer_counts_single_substitution() -> None:
    # one substitution over 10 chars = 0.1
    assert abs(metrics.cer("abcdefghij", "abcdefghiX".lower()) - 0.1) < 1e-9


def test_cer_empty_reference_is_zero() -> None:
    assert metrics.cer("", "anything") == 0.0


def test_wer_counts_one_word_error() -> None:
    assert abs(metrics.wer("one two three four", "one two THREE four".lower())) < 1e-9
    assert abs(metrics.wer("one two three four", "one two tree four") - 0.25) < 1e-9


# --------------------------------------------------------------------------- #
# Decile drift: inject errors only in the last 10%
# --------------------------------------------------------------------------- #
@requires_edlib
def test_decile_cer_flags_late_document_drift() -> None:
    base = (string.ascii_lowercase + " ") * 200  # ~5400 chars, clean
    ref = metrics.normalize_text(base)
    hyp_chars = list(ref)
    # Corrupt roughly the last 10% of characters.
    start = int(len(hyp_chars) * 0.9)
    for i in range(start, len(hyp_chars)):
        if hyp_chars[i] != " ":
            hyp_chars[i] = "0"
    hyp = "".join(hyp_chars)

    report = metrics.align_report(ref, hyp, n_deciles=10)
    assert len(report.decile_cer) == 10
    # Early deciles clean, last decile clearly degraded.
    assert max(report.decile_cer[:8]) < 0.05
    assert report.decile_cer[-1] > 0.5
    assert report.decile_cer[-1] > 2 * (
        sorted(report.decile_cer)[len(report.decile_cer) // 2] + 1e-9
    )


@requires_edlib
def test_coverage_detects_skipped_tail() -> None:
    ref = metrics.normalize_text("alpha bravo charlie delta echo foxtrot golf hotel")
    hyp = metrics.normalize_text("alpha bravo charlie delta")  # dropped second half
    report = metrics.align_report(ref, hyp, n_deciles=10)
    assert report.coverage < 0.7


# --------------------------------------------------------------------------- #
# GT span location (infix / HW)
# --------------------------------------------------------------------------- #
@requires_edlib
def test_locate_gt_span_finds_infix() -> None:
    ref = metrics.normalize_text("zero one two three four five six seven eight nine")
    hyp = metrics.normalize_text("three four five six")
    span = metrics.locate_gt_span(hyp, ref)
    assert "three four five six" in span.text
    assert span.start > 0
    assert ref[span.start : span.end].strip().startswith("three")


@requires_edlib
def test_evaluate_slice_to_span_scores_only_matching_span() -> None:
    full = "a b c d e f g h i j k l m n o p q r s t u v w x y z " * 3
    partial = "g h i j k l m n o p"  # a small middle slice, OCR'd perfectly
    ev = metrics.evaluate(partial, full, slice_to_span=True)
    assert ev.cer_lower < 0.05
    assert ev.gt_span_end > ev.gt_span_start
    # The located span should be far shorter than the whole reference.
    assert (ev.gt_span_end - ev.gt_span_start) < len(metrics.normalize_text(full))


# --------------------------------------------------------------------------- #
# Repetition
# --------------------------------------------------------------------------- #
def test_repetition_share_high_for_looping_text() -> None:
    loop = "alpha beta gamma delta epsilon zeta eta theta "
    looping = loop * 6
    clean = (
        "the committee reviewed the annual report and approved the budget for "
        "the following fiscal year without any further amendments or delay today"
    )
    rep_loop = metrics.repetition_stats(looping, n=8)
    rep_clean = metrics.repetition_stats(clean, n=8)
    assert rep_loop.repeated_ngram_share > 0.5
    assert rep_clean.repeated_ngram_share == 0.0
    assert rep_loop.repeated_ngram_share > 1.5 * (rep_clean.repeated_ngram_share + 1e-9)


def test_longest_consecutive_run() -> None:
    rep = metrics.repetition_stats("a a a a b c c", n=8)
    assert rep.longest_repeated_run == 4


# --------------------------------------------------------------------------- #
# Chunk ranging + seam de-duplication
# --------------------------------------------------------------------------- #
def test_iter_chunks_overlaps_by_one() -> None:
    ranges = metrics.iter_chunks(12, chunk_size=5, overlap=1)
    assert ranges[0] == (0, 5)
    assert ranges[1][0] == 4  # one-page overlap with previous end
    assert ranges[-1][1] == 12  # covers to the end
    # every page is covered
    covered = set()
    for start, end in ranges:
        covered.update(range(start, end))
    assert covered == set(range(12))


def test_iter_chunks_rejects_bad_overlap() -> None:
    with pytest.raises(ValueError):
        metrics.iter_chunks(10, chunk_size=1, overlap=1)


def test_stitch_chunks_removes_duplicated_overlap() -> None:
    overlap = (
        "The quick brown fox jumps over the lazy dog near the quiet river bank "
        "in the early morning light before anyone else had woken up that day."
    )
    chunk_a = "Opening material that belongs only to the first chunk. " + overlap
    chunk_b = overlap + " Closing material that belongs only to the second chunk."

    stitched = metrics.stitch_chunks([chunk_a, chunk_b], seam_window=400, min_score=70.0)

    assert "Opening material" in stitched
    assert "Closing material" in stitched
    # The overlap sentence should not be duplicated after stitching.
    assert stitched.count("quiet river bank") == 1
    # And stitching must be shorter than naive concatenation.
    assert len(stitched) < len(chunk_a) + len(chunk_b)


def test_stitch_chunks_handles_no_overlap() -> None:
    a = "completely distinct first section with its own unique wording here now"
    b = "an entirely separate second section sharing nothing with the first one"
    stitched = metrics.stitch_chunks([a, b])
    assert a in stitched and b in stitched


# --------------------------------------------------------------------------- #
# Gutenberg boilerplate + cost
# --------------------------------------------------------------------------- #
def test_strip_gutenberg_boilerplate() -> None:
    text = (
        "Header junk and license noise.\n"
        "*** START OF THE PROJECT GUTENBERG EBOOK SAMPLE ***\n"
        "The real body of the book.\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK SAMPLE ***\n"
        "License trailer and legalese."
    )
    body = metrics.strip_gutenberg_boilerplate(text)
    assert body == "The real body of the book."


def test_cost_per_book_uses_wall_seconds() -> None:
    # 2 s/page, $1/hr, 300 pages = 600 s = 1/6 hr = $0.1667
    cost = metrics.cost_per_book(2.0, 1.0, pages=300)
    assert abs(cost - (600.0 / 3600.0)) < 1e-9
