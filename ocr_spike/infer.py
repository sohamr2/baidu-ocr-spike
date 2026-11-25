"""Model loading, API introspection, inference wrappers, and the experiment runner.

The model's ``infer`` / ``infer_multi`` methods come from remote code
(``trust_remote_code=True``), so nothing about their signatures or on-disk
outputs is assumed. :func:`introspect_api` prints the live signatures and the
cached modeling file; the inference wrappers capture both the return value and
any files written to ``output_path`` and reconcile them.

Where possible the true generation stop reason and token totals are recovered
by transiently wrapping ``model.generate`` (:class:`GenerateCapture`); otherwise
``hit_max_length`` is approximated from token counts and the documented
``max_length`` semantics.

Every run produces one dictionary suitable for appending to ``results.jsonl``.
OOMs are captured as rows rather than crashing the run.

Chunk ranging and seam de-duplication live in :mod:`ocr_spike.metrics` so they
stay importable (and unit-testable) without torch.
"""

from __future__ import annotations

import datetime as _dt
import inspect
import json
import logging
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import torch
from rapidfuzz import fuzz

from . import metrics

logger = logging.getLogger(__name__)

# Text-bearing files the model might write to output_path.
_TEXT_SUFFIXES = {".txt", ".md", ".mmd", ".json"}

# max_length semantics (confirmed by reading the remote modeling code).
SEMANTICS_TOTAL = "max_length"  # prompt + image + output tokens
SEMANTICS_NEW = "max_new_tokens"  # output tokens only


# --------------------------------------------------------------------------- #
# Runtime / dtype detection
# --------------------------------------------------------------------------- #
@dataclass
class RuntimeInfo:
    """Resolved GPU runtime and the dtype the spike will actually use."""

    gpu_name: str
    total_vram_gb: float
    cuda_version: str
    torch_version: str
    driver_version: str
    bf16_supported: bool
    dtype: torch.dtype
    dtype_str: str


def _nvidia_driver_version() -> str:
    """Return the NVIDIA driver version via ``nvidia-smi`` (``unknown`` on failure)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
            timeout=20,
        )
        return out.stdout.strip().splitlines()[0].strip()
    except (OSError, subprocess.SubprocessError, IndexError) as exc:
        logger.warning("Could not read NVIDIA driver version: %r", exc)
        return "unknown"


def nvidia_smi_max_cuda() -> str:
    """Return the max CUDA version the installed driver supports (``nvidia-smi``)."""
    try:
        out = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, check=True, timeout=20
        )
        match = re.search(r"CUDA Version:\s*([0-9.]+)", out.stdout)
        return match.group(1) if match else "unknown"
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("Could not read nvidia-smi CUDA version: %r", exc)
        return "unknown"


def check_driver_cuda_compat(wheel_cuda: str = "12.9") -> dict[str, Any]:
    """Compare the driver's max CUDA against the torch wheel's CUDA build.

    Returns a dict with ``driver_max_cuda``, ``wheel_cuda``, ``compatible`` and
    (when not compatible) a ``suggested_index_url`` for a matching cu12x wheel.
    PyTorch cu12x wheels bundle their own CUDA runtime and rely on CUDA 12
    minor-version forward compatibility, so any 12.x driver is normally fine.
    """
    driver_max = nvidia_smi_max_cuda()
    result: dict[str, Any] = {
        "driver_max_cuda": driver_max,
        "wheel_cuda": wheel_cuda,
        "compatible": True,
        "suggested_index_url": None,
    }
    try:
        driver_tuple = tuple(int(x) for x in driver_max.split(".")[:2])
        wheel_tuple = tuple(int(x) for x in wheel_cuda.split(".")[:2])
    except ValueError:
        return result
    if driver_tuple[0] != wheel_tuple[0] or driver_tuple < wheel_tuple:
        result["compatible"] = False
        major = wheel_tuple[0]
        minor_floor = min(driver_tuple[1], wheel_tuple[1]) if driver_tuple[0] == major else 1
        result["suggested_index_url"] = (
            f"https://download.pytorch.org/whl/cu{major}{minor_floor}"
        )
    return result


def detect_runtime() -> RuntimeInfo:
    """Probe the GPU and choose a dtype.

    Prefers ``bfloat16`` (as the model card uses). If the GPU does not support
    bf16 (e.g. free-tier T4), emits a loud warning and falls back to
    ``float16``; the chosen dtype is recorded so it lands in every result row.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU available. Select a Colab GPU runtime.")

    index = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(index)
    bf16 = torch.cuda.is_bf16_supported()
    if bf16:
        dtype, dtype_str = torch.bfloat16, "bfloat16"
    else:
        dtype, dtype_str = torch.float16, "float16"

    info = RuntimeInfo(
        gpu_name=props.name,
        total_vram_gb=props.total_memory / 1024**3,
        cuda_version=torch.version.cuda or "unknown",
        torch_version=torch.__version__,
        driver_version=_nvidia_driver_version(),
        bf16_supported=bf16,
        dtype=dtype,
        dtype_str=dtype_str,
    )

    logger.info("GPU name          : %s", info.gpu_name)
    logger.info("Total VRAM        : %.1f GB", info.total_vram_gb)
    logger.info("torch build       : %s (CUDA %s)", info.torch_version, info.cuda_version)
    logger.info("NVIDIA driver     : %s (max CUDA %s)", info.driver_version, nvidia_smi_max_cuda())
    logger.info("bf16 supported    : %s", info.bf16_supported)
    logger.info("dtype selected    : %s", info.dtype_str)
    if not bf16:
        bar = "=" * 72
        logger.warning(bar)
        logger.warning(
            "bf16 is NOT supported on %s -- falling back to float16.", info.gpu_name
        )
        logger.warning(
            "The model card uses bfloat16; float16 may change accuracy/stability."
        )
        logger.warning("Results are only comparable to the model card on an L4/A100.")
        logger.warning(bar)
    return info


# --------------------------------------------------------------------------- #
# Model loading (pinned revision)
# --------------------------------------------------------------------------- #
def resolve_revision(repo_id: str) -> str:
    """Resolve the current commit SHA of ``repo_id`` via the HF Hub API.

    The explicit SHA is used for loading and written to every result row, so
    runs are reproducible without any manual paste-back.
    """
    from huggingface_hub import HfApi

    sha = HfApi().model_info(repo_id).sha
    logger.info("Model repo        : %s", repo_id)
    logger.info("Resolved revision : %s", sha)
    return sha


def load_model(
    repo_id: str,
    dtype: torch.dtype,
    *,
    revision: Optional[str] = None,
    trust_remote_code: bool = True,
):
    """Load the pinned Unlimited-OCR model and tokenizer onto the GPU.

    If ``revision`` is None the current SHA is resolved first. Returns
    ``(model, tokenizer, resolved_revision)``. ``trust_remote_code`` executes
    repo code, which is why only the official ``baidu/Unlimited-OCR`` repo at a
    pinned commit should ever be passed here.
    """
    from transformers import AutoModel, AutoTokenizer

    resolved = revision or resolve_revision(repo_id)
    tokenizer = AutoTokenizer.from_pretrained(
        repo_id, revision=resolved, trust_remote_code=trust_remote_code
    )
    model = (
        AutoModel.from_pretrained(
            repo_id,
            revision=resolved,
            trust_remote_code=trust_remote_code,
            use_safetensors=True,
            torch_dtype=dtype,
        )
        .eval()
        .cuda()
    )
    logger.info("Loaded %s @ %s (dtype=%s)", repo_id, resolved, dtype)
    return model, tokenizer, resolved


def introspect_api(model: Any) -> dict[str, str]:
    """Return the live signatures of ``infer``/``infer_multi`` and the modeling file.

    This is the "don't guess the API" step: the returned dict is printed in the
    notebook and the modeling file is skimmed before anything is built on it.
    """
    report: dict[str, str] = {}
    for name in ("infer", "infer_multi"):
        func = getattr(model, name, None)
        if func is None:
            report[name] = "MISSING"
            continue
        try:
            report[name] = f"{name}{inspect.signature(func)}"
        except (TypeError, ValueError) as exc:
            report[name] = f"signature unavailable: {exc!r}"

    try:
        source_file = inspect.getsourcefile(type(model)) or "unknown"
    except TypeError:
        source_file = "unknown"
    report["model_class"] = type(model).__name__
    report["modeling_file"] = source_file
    report["module"] = type(model).__module__
    return report


# --------------------------------------------------------------------------- #
# Token accounting
# --------------------------------------------------------------------------- #
def count_tokens(tokenizer: Any, text: str) -> int:
    """Count tokens of ``text`` using the model tokenizer (0 for empty text)."""
    if not text:
        return 0
    try:
        return len(tokenizer.encode(text))
    except Exception as exc:  # tokenizer internals vary; never fail a run on this
        logger.warning("Token counting failed; falling back to whitespace: %r", exc)
        return len(text.split())


def estimate_input_tokens(
    tokenizer: Any,
    prompt: str,
    *,
    n_pages: int,
    image_tokens_per_page: int,
) -> int:
    """Estimate prompt + image input tokens.

    ``image_tokens_per_page`` is read from the remote modeling code (vision
    tokens per tile x tiles) and set in the config cell. Used for the
    ``max_length`` (total) semantics when a true generate-capture total is
    unavailable.
    """
    return count_tokens(tokenizer, prompt) + image_tokens_per_page * max(n_pages, 1)


# --------------------------------------------------------------------------- #
# Generate capture (true stop reason / token totals)
# --------------------------------------------------------------------------- #
@dataclass
class GenerateSummary:
    """Best-effort summary of captured ``generate`` calls.

    ``input_tokens`` is the measured input length (text + expanded image tokens)
    taken from the ids/embeds passed to ``generate`` on the call that fired.
    ``fired`` is False when no patched ``generate`` was invoked, in which case
    callers must fall back to the token-count estimate (and should warn).
    """

    generate_calls: int
    last_total_tokens: Optional[int]
    input_tokens: Optional[int]
    ended_with_eos: Optional[bool]
    fired: bool
    target: Optional[str]


def _infer_input_len(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Optional[int]:
    """Measure the sequence length passed into ``generate``.

    Handles both ``input_ids`` (``[batch, seq]``) and ``inputs_embeds``
    (``[batch, seq, hidden]``) by reading the batch-first sequence dim, so the
    count includes the model's expanded image tokens.
    """
    candidate = None
    for key in ("input_ids", "inputs_embeds", "inputs", "input"):
        value = kwargs.get(key)
        if isinstance(value, torch.Tensor):
            candidate = value
            break
    if candidate is None:
        for value in args:
            if isinstance(value, torch.Tensor):
                candidate = value
                break
    if candidate is None:
        return None
    try:
        return int(candidate.shape[1]) if candidate.ndim >= 2 else int(candidate.shape[0])
    except (IndexError, RuntimeError):
        return None


class GenerateCapture:
    """Transiently wrap ``generate`` on the model and every submodule that owns it.

    The remote ``infer``/``infer_multi`` code may call ``self.generate(...)`` or
    an inner module's ``generate`` (e.g. a language-model submodule). Rather than
    silently falling back to approximation, we patch the model plus every
    submodule exposing a callable ``generate`` and record which one fired, its
    returned ids (true total length, EOS vs length stop), and the measured input
    length. ``fired`` reports whether any wrapper was actually invoked.
    """

    def __init__(self, model: Any) -> None:
        self._model = model
        self._patched: list[tuple[Any, Callable[..., Any]]] = []
        self.records: list[dict[str, Any]] = []

    def _targets(self) -> list[tuple[str, Any]]:
        targets: list[tuple[str, Any]] = [("<model>", self._model)]
        for name, module in self._model.named_modules():
            if module is self._model:
                continue
            if callable(getattr(module, "generate", None)):
                targets.append((name, module))
        return targets

    def __enter__(self) -> "GenerateCapture":
        for name, module in self._targets():
            original = getattr(module, "generate", None)
            if not callable(original):
                continue

            def make_wrapper(orig: Callable[..., Any], target: str) -> Callable[..., Any]:
                def wrapper(*args: Any, **kwargs: Any) -> Any:
                    out = orig(*args, **kwargs)
                    self.records.append(
                        {"target": target, "out": out,
                         "input_len": _infer_input_len(args, kwargs)}
                    )
                    return out

                return wrapper

            try:
                module.generate = make_wrapper(original, name)  # type: ignore[assignment]
                self._patched.append((module, original))
            except (AttributeError, TypeError):
                pass
        return self

    def __exit__(self, *exc: object) -> None:
        for module, original in self._patched:
            try:
                module.generate = original  # type: ignore[assignment]
            except (AttributeError, TypeError):
                pass
        self._patched.clear()

    def summarize(self, eos_token_id: Optional[int]) -> GenerateSummary:
        """Summarize captured calls (uses the last call for totals/stop reason)."""
        if not self.records:
            return GenerateSummary(0, None, None, None, False, None)
        last = self.records[-1]
        out = last["out"]
        seq = getattr(out, "sequences", None)
        if seq is None and isinstance(out, torch.Tensor):
            seq = out
        total: Optional[int] = None
        ended: Optional[bool] = None
        if seq is not None:
            try:
                row = seq[0] if seq.ndim >= 2 else seq
                total = int(row.shape[-1])
                last_token = int(row[-1])
                ended = None if eos_token_id is None else (last_token == eos_token_id)
            except (IndexError, RuntimeError, ValueError):
                total = None
        return GenerateSummary(
            generate_calls=len(self.records),
            last_total_tokens=total,
            input_tokens=last.get("input_len"),
            ended_with_eos=ended,
            fired=True,
            target=last.get("target"),
        )


# --------------------------------------------------------------------------- #
# Output collection
# --------------------------------------------------------------------------- #
def collect_output(return_value: Any, output_dir: Path) -> tuple[str, dict[str, str]]:
    """Reconcile the method return value with files written to ``output_dir``.

    Prefers a non-empty string return value as the raw text; otherwise
    concatenates text-bearing files found under ``output_dir``. Always returns
    the map of discovered files so the notebook can confirm what the model
    actually wrote.
    """
    files: dict[str, str] = {}
    file_text_parts: list[str] = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file():
            continue
        files[str(path.relative_to(output_dir))] = str(path)
        if path.suffix.lower() in _TEXT_SUFFIXES:
            try:
                content = path.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                logger.warning("Could not read output file %s: %r", path, exc)
                continue
            if path.suffix.lower() == ".json":
                content = _extract_json_text(content)
            file_text_parts.append(content)

    if isinstance(return_value, str) and return_value.strip():
        raw = return_value
    else:
        raw = "\n".join(part for part in file_text_parts if part.strip())
    return raw, files


def _extract_json_text(content: str) -> str:
    """Best-effort extraction of text fields from a JSON results file."""
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return content
    keys = ("text", "result", "content", "markdown", "mmd", "ocr")
    if isinstance(data, dict):
        for key in keys:
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value
    return json.dumps(data, ensure_ascii=False)


# --------------------------------------------------------------------------- #
# Inference wrappers
# --------------------------------------------------------------------------- #
@dataclass
class InferenceOutcome:
    """Raw result of one inference call (before accuracy metrics)."""

    raw_text: str
    cleaned_text: str
    block_types: list[str]
    output_tokens: int
    wall_seconds: float
    gpu_seconds: float
    peak_vram_gb: float
    output_files: dict[str, str]
    raw_path: str
    cleaned_path: str
    generate_calls: int = 0
    captured_total_tokens: Optional[int] = None
    captured_input_tokens: Optional[int] = None
    captured_ended_eos: Optional[bool] = None
    capture_fired: bool = False
    capture_target: Optional[str] = None


def _run_timed(call: Callable[[], Any]) -> tuple[Any, float, float, float]:
    """Run ``call`` with VRAM reset and CUDA-event timing.

    Returns ``(return_value, wall_seconds, gpu_seconds, peak_vram_gb)``.
    """
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    wall_start = time.perf_counter()
    start_event.record()
    with torch.inference_mode():
        value = call()
    end_event.record()
    torch.cuda.synchronize()
    wall = time.perf_counter() - wall_start
    gpu_seconds = start_event.elapsed_time(end_event) / 1000.0
    peak_vram = torch.cuda.max_memory_allocated() / 1024**3
    return value, wall, gpu_seconds, peak_vram


def run_inference(
    model: Any,
    tokenizer: Any,
    *,
    mode: str,
    image_files: Sequence[Path],
    prompt: str,
    output_dir: Path,
    infer_kwargs: dict[str, Any],
    capture_generate: bool = True,
) -> InferenceOutcome:
    """Run a single ``infer`` or ``infer_multi`` call and collect its output.

    ``mode`` is ``"single"`` or ``"multi"``. Raw and cleaned text are also saved
    to ``output_dir`` (``raw.txt`` / ``cleaned.txt``) so results survive a
    disconnect regardless of what the model itself writes. When
    ``capture_generate`` is True the true stop reason / token total are recorded
    if the remote code routes through ``model.generate``.
    """
    if mode not in {"single", "multi"}:
        raise ValueError(f"mode must be 'single' or 'multi', got {mode!r}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = [str(Path(p)) for p in image_files]

    if mode == "single":
        def call() -> Any:
            return model.infer(
                tokenizer,
                prompt=prompt,
                image_file=files[0],
                output_path=str(output_dir),
                **infer_kwargs,
            )
    else:
        def call() -> Any:
            return model.infer_multi(
                tokenizer,
                prompt=prompt,
                image_files=files,
                output_path=str(output_dir),
                **infer_kwargs,
            )

    capture = GenerateCapture(model) if capture_generate else None

    def guarded() -> Any:
        if capture is None:
            return call()
        with capture:
            return call()

    return_value, wall, gpu_seconds, peak_vram = _run_timed(guarded)
    summary = (
        capture.summarize(getattr(tokenizer, "eos_token_id", None))
        if capture is not None
        else GenerateSummary(0, None, None, None, False, None)
    )
    if capture is not None and not summary.fired:
        logger.warning(
            "GenerateCapture did not fire (no patched generate was called); "
            "input tokens and stop reason are approximate for this run."
        )

    raw_text, output_files = collect_output(return_value, output_dir)
    parsed = metrics.parse_markers(raw_text)
    output_tokens = count_tokens(tokenizer, raw_text)

    raw_path = output_dir / "raw.txt"
    cleaned_path = output_dir / "cleaned.txt"
    raw_path.write_text(parsed.raw, encoding="utf-8")
    cleaned_path.write_text(parsed.cleaned, encoding="utf-8")

    return InferenceOutcome(
        raw_text=parsed.raw,
        cleaned_text=parsed.cleaned,
        block_types=parsed.block_types,
        output_tokens=output_tokens,
        wall_seconds=wall,
        gpu_seconds=gpu_seconds,
        peak_vram_gb=peak_vram,
        output_files=output_files,
        raw_path=str(raw_path),
        cleaned_path=str(cleaned_path),
        generate_calls=summary.generate_calls,
        captured_total_tokens=summary.last_total_tokens,
        captured_input_tokens=summary.input_tokens,
        captured_ended_eos=summary.ended_with_eos,
        capture_fired=summary.fired,
        capture_target=summary.target,
    )


def last_page_covered(
    model: Any,
    tokenizer: Any,
    *,
    last_image: Path,
    multi_output_cleaned: str,
    cache_dir: Path,
    single_infer_kwargs: dict[str, Any],
    prompt: str,
    min_score: float = 90.0,
    n_lines: int = 3,
    min_matches: int = 2,
) -> bool:
    """Check whether the final input page's content appears in the output.

    OCRs the last page once (cached under ``cache_dir``), takes its 3 longest
    distinctive lines as fingerprints, and requires at least ``min_matches`` of
    them to fuzzily match (rapidfuzz ``partial_ratio`` >= ``min_score``) inside
    the multi-page output.
    """
    cache_dir = Path(cache_dir)
    cached = cache_dir / "last_page_cleaned.txt"
    if cached.exists():
        reference = cached.read_text(encoding="utf-8", errors="replace")
    else:
        outcome = run_inference(
            model,
            tokenizer,
            mode="single",
            image_files=[last_image],
            prompt=prompt,
            output_dir=cache_dir,
            infer_kwargs=single_infer_kwargs,
            capture_generate=False,
        )
        reference = outcome.cleaned_text
        cached.write_text(reference, encoding="utf-8")

    lines = sorted(
        (line.strip() for line in reference.splitlines() if len(line.strip()) >= 20),
        key=len,
        reverse=True,
    )[:n_lines]
    if not lines:
        logger.warning("No distinctive lines on last page; cannot judge coverage.")
        return False
    required = min(min_matches, len(lines))
    matches = sum(
        1 for line in lines if fuzz.partial_ratio(line, multi_output_cleaned) >= min_score
    )
    return matches >= required


# --------------------------------------------------------------------------- #
# Result logging / experiment runner
# --------------------------------------------------------------------------- #
def load_results(results_path: Path) -> list[dict[str, Any]]:
    """Read all rows from a ``results.jsonl`` file (empty list if missing)."""
    results_path = Path(results_path)
    if not results_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in results_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def append_result(results_path: Path, row: dict[str, Any]) -> None:
    """Append one JSON row to ``results.jsonl``."""
    results_path = Path(results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def find_existing(
    results: Sequence[dict[str, Any]], run_key: dict[str, Any]
) -> Optional[dict[str, Any]]:
    """Return the most recent successful row matching ``run_key``, if any."""
    match: Optional[dict[str, Any]] = None
    for row in results:
        if all(row.get(k) == v for k, v in run_key.items()) and not row.get("error"):
            match = row
    return match


def is_oom(exc: BaseException) -> bool:
    """True if ``exc`` is a CUDA out-of-memory error."""
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def _compute_hit_max_length(
    *,
    output_tokens: int,
    input_tokens: int,
    captured_total_tokens: Optional[int],
    max_length: int,
    semantics: str,
    slack: int,
) -> bool:
    """Decide whether generation hit the length cap given the documented semantics."""
    threshold = max_length - slack
    if semantics == SEMANTICS_NEW:
        return output_tokens >= threshold
    # total semantics
    if captured_total_tokens is not None:
        return captured_total_tokens >= threshold
    return (input_tokens + output_tokens) >= threshold


def _measured_image_tokens(
    *,
    captured_input_tokens: Optional[int],
    prompt_tokens: Optional[int],
    n_pages: Optional[int],
) -> Optional[float]:
    """Measured image tokens per page = (actual input - prompt text) / pages.

    Lets us compare the real per-page image-token cost against the configured
    ``IMAGE_TOKENS_PER_PAGE`` estimate. Returns None when the capture or inputs
    needed to compute it are missing.
    """
    if captured_input_tokens is None or prompt_tokens is None or not n_pages:
        return None
    return round((captured_input_tokens - prompt_tokens) / n_pages, 1)


def _stop_reason(hit_max_length: bool, captured_ended_eos: Optional[bool]) -> str:
    """Return the stop reason, preferring the captured EOS signal when present."""
    if captured_ended_eos is not None:
        return "eos" if captured_ended_eos else "length"
    return "length(approx)" if hit_max_length else "eos_or_natural(approx)"


def execute_run(
    *,
    run_key: dict[str, Any],
    results_path: Path,
    runtime: RuntimeInfo,
    revision: str,
    config: dict[str, Any],
    runner: Callable[[], InferenceOutcome],
    max_length: int,
    max_length_slack: int,
    max_length_semantics: str,
    input_tokens: Optional[int] = None,
    prompt_tokens: Optional[int] = None,
    evaluator: Optional[Callable[[InferenceOutcome], dict[str, Any]]] = None,
    last_page_check: Optional[Callable[[InferenceOutcome], bool]] = None,
    force: bool = False,
) -> dict[str, Any]:
    """Run one experiment, logging exactly one row to ``results.jsonl``.

    Skips (and returns the cached row) when a matching successful result already
    exists and ``force`` is False. OOMs and other errors are captured in the
    ``error`` field instead of propagating. ``torch.cuda.empty_cache()`` is
    always called afterwards.

    ``run_key`` must uniquely identify the run (book id, experiment, page range,
    config name, max_length) and is merged into the logged row. ``evaluator``,
    when given, returns accuracy fields (including GT span offsets) merged into
    the row so metrics are persisted on Drive alongside the run.
    """
    existing = find_existing(load_results(results_path), run_key)
    if existing is not None and not force:
        logger.info("Skipping cached run: %s", run_key)
        return existing

    row: dict[str, Any] = {
        **run_key,
        "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "config": config,
        "dtype": runtime.dtype_str,
        "gpu_name": runtime.gpu_name,
        "torch_version": runtime.torch_version,
        "cuda_version": runtime.cuda_version,
        "driver_version": runtime.driver_version,
        "revision": revision,
        "max_length": max_length,
        "max_length_semantics": max_length_semantics,
        "input_tokens": input_tokens,  # estimate from IMAGE_TOKENS_PER_PAGE
        "input_tokens_actual": None,  # measured from GenerateCapture
        "image_tokens_per_page_measured": None,
        "capture_fired": None,
        "capture_target": None,
        "wall_seconds": None,
        "gpu_seconds": None,
        "peak_vram_gb": None,
        "output_tokens": None,
        "generate_calls": None,
        "captured_total_tokens": None,
        "hit_max_length": None,
        "stop_reason": None,
        "last_page_covered": None,
        "block_types": None,
        "raw_path": None,
        "cleaned_path": None,
        "error": None,
    }

    try:
        outcome = runner()
        hit_max = _compute_hit_max_length(
            output_tokens=outcome.output_tokens,
            input_tokens=input_tokens or 0,
            captured_total_tokens=outcome.captured_total_tokens,
            max_length=max_length,
            semantics=max_length_semantics,
            slack=max_length_slack,
        )
        covered = last_page_check(outcome) if last_page_check is not None else None
        measured_img = _measured_image_tokens(
            captured_input_tokens=outcome.captured_input_tokens,
            prompt_tokens=prompt_tokens,
            n_pages=run_key.get("input_pages"),
        )
        row.update(
            wall_seconds=round(outcome.wall_seconds, 3),
            gpu_seconds=round(outcome.gpu_seconds, 3),
            peak_vram_gb=round(outcome.peak_vram_gb, 3),
            output_tokens=outcome.output_tokens,
            generate_calls=outcome.generate_calls,
            captured_total_tokens=outcome.captured_total_tokens,
            input_tokens_actual=outcome.captured_input_tokens,
            image_tokens_per_page_measured=measured_img,
            capture_fired=outcome.capture_fired,
            capture_target=outcome.capture_target,
            hit_max_length=bool(hit_max),
            stop_reason=_stop_reason(hit_max, outcome.captured_ended_eos),
            last_page_covered=covered,
            block_types=outcome.block_types,
            raw_path=outcome.raw_path,
            cleaned_path=outcome.cleaned_path,
        )
        if not outcome.capture_fired:
            logger.warning(
                "Run %s: GenerateCapture did not fire; input_tokens_actual is "
                "unavailable and hit_max_length/stop_reason are approximate.",
                run_key,
            )
        if evaluator is not None:
            try:
                row.update(evaluator(outcome))
            except Exception as exc:  # noqa: BLE001 - metric failure must not drop the row
                logger.error("Evaluator failed for %s: %r", run_key, exc)
                row["metric_error"] = repr(exc)
        logger.info(
            "Run %s: out_tokens=%d wall=%.1fs gpu=%.1fs vram=%.1fGB hit_max=%s stop=%s",
            run_key,
            outcome.output_tokens,
            outcome.wall_seconds,
            outcome.gpu_seconds,
            outcome.peak_vram_gb,
            row["hit_max_length"],
            row["stop_reason"],
        )
    except Exception as exc:  # noqa: BLE001 - we log every failure as a row
        kind = "OOM" if is_oom(exc) else type(exc).__name__
        row["error"] = f"{kind}: {exc}"
        logger.error("Run %s failed: %s", run_key, row["error"])
    finally:
        torch.cuda.empty_cache()

    append_result(results_path, row)
    return row
