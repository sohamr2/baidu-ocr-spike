"""PDF rasterization and page sampling for the OCR spike.

PDFs are converted to PNGs at a fixed DPI with PyMuPDF (matching the model
card) and cached on disk (Google Drive) so that a Colab disconnect does not
force a re-render. Pages already converted are skipped.
"""

from __future__ import annotations

import logging
from pathlib import Path

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

_PNG_STEM = "page_{index:04d}"


def count_pages(pdf_path: Path | str) -> int:
    """Return the number of pages in a PDF."""
    with fitz.open(str(pdf_path)) as doc:
        return doc.page_count


def page_image_path(out_dir: Path | str, index: int) -> Path:
    """Return the cache path for the rendered image of page ``index`` (0-based)."""
    return Path(out_dir) / f"{_PNG_STEM.format(index=index)}.png"


def pdf_to_images(
    pdf_path: Path | str,
    out_dir: Path | str,
    *,
    dpi: int = 300,
    force: bool = False,
) -> list[Path]:
    """Render every page of ``pdf_path`` to a PNG at ``dpi`` and cache in ``out_dir``.

    Parameters
    ----------
    pdf_path:
        Source PDF.
    out_dir:
        Directory for cached PNGs (created if needed). One file per page named
        ``page_0000.png`` (0-based).
    dpi:
        Render resolution. 300 DPI matches the model card.
    force:
        Re-render even if a cached PNG already exists.

    Returns
    -------
    list[Path]
        Paths to the per-page PNGs in page order.
    """
    pdf_path = Path(pdf_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    paths: list[Path] = []
    rendered = 0
    with fitz.open(str(pdf_path)) as doc:
        for index in range(doc.page_count):
            target = page_image_path(out_dir, index)
            paths.append(target)
            if target.exists() and not force:
                continue
            page = doc.load_page(index)
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            pixmap.save(str(target))
            rendered += 1
    logger.info(
        "Rasterized %s: %d pages (%d newly rendered at %d DPI) -> %s",
        pdf_path.name,
        len(paths),
        rendered,
        dpi,
        out_dir,
    )
    return paths


def sample_pages_evenly(n_pages: int, k: int) -> list[int]:
    """Return ``k`` page indices spread evenly across ``[0, n_pages)``.

    Used by Experiment A to pick representative pages without long-context
    effects. If ``k >= n_pages`` every page index is returned.
    """
    if n_pages <= 0:
        return []
    if k >= n_pages:
        return list(range(n_pages))
    if k <= 1:
        return [0]
    step = (n_pages - 1) / (k - 1)
    seen: list[int] = []
    for i in range(k):
        idx = round(i * step)
        if not seen or seen[-1] != idx:
            seen.append(idx)
    return seen
