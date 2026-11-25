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
    # torch.cuda.is_bf16_supported() over-reports on pre-Ampere GPUs (e.g. T4, sm_75),
    # where bf16 is only emulated -- that ruins the timing/VRAM numbers this spike measures.
    # Require NATIVE bf16: compute capability >= 8.0 (Ampere+) AND torch agreeing.
    bf16 = props.major >= 8 and torch.cuda.is_bf16_supported()
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
class CallRecord:
    """One captured (outermost) ``generate`` call."""

    target: Optional[str]
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    stop_reason: str  # "eos" | "length" | "unknown"


@dataclass
class GenerateSummary:
    """Every outermost ``generate`` call captured during one inference.

    ``fired`` is False when no patched ``generate`` was invoked, in which case
    callers fall back to the token-count estimate (and should warn). Derived
    quantities are exposed as properties so result rows stay simple.
    """

    calls: list[CallRecord]
    fired: bool

    @property
    def generate_calls(self) -> int:
        return len(self.calls)

    @property
    def output_tokens_total(self) -> Optional[int]:
        if not self.calls:
            return None
        return sum(c.output_tokens or 0 for c in self.calls)

    @property
    def input_tokens_first(self) -> Optional[int]:
        return self.calls[0].input_tokens if self.calls else None

    @property
    def any_hit_max_length(self) -> Optional[bool]:
        if not self.calls:
            return None
        return any(c.stop_reason == "length" for c in self.calls)

    @property
    def overall_stop_reason(self) -> Optional[str]:
        if not self.calls:
            return None
        if self.any_hit_max_length:
            return "length"
        if any(c.stop_reason == "eos" for c in self.calls):
            return "eos"
        return "other"

    @property
    def first_target(self) -> Optional[str]:
        return self.calls[0].target if self.calls else None


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


def _seq_total_and_eos(out: Any, eos_token_id: Optional[int]) -> tuple[Optional[int], Optional[bool]]:
    """Return ``(total_tokens, ended_with_eos)`` for a ``generate`` return value."""
    seq = getattr(out, "sequences", None)
    if seq is None and isinstance(out, torch.Tensor):
        seq = out
    if seq is None:
        return None, None
    try:
        row = seq[0] if seq.ndim >= 2 else seq
        total = int(row.shape[-1])
        last_token = int(row[-1])
    except (IndexError, RuntimeError, ValueError):
        return None, None
    ended = None if eos_token_id is None else (last_token == eos_token_id)
    return total, ended


def _limit_reached(
    total: Optional[int],
    output_tokens: Optional[int],
    max_length: Optional[int],
    semantics: str,
    slack: int,
) -> bool:
    """Whether a call's token count reached the effective cap (per semantics)."""
    if max_length is None:
        return False
    threshold = max_length - slack
    if semantics == SEMANTICS_NEW:
        return output_tokens is not None and output_tokens >= threshold
    return total is not None and total >= threshold


class GenerateCapture:
    """Transiently wrap ``generate`` on the model and every submodule that owns it.

    The remote ``infer``/``infer_multi`` code may call ``self.generate(...)`` or
    an inner module's ``generate`` (e.g. a language-model submodule), and may call
    it once or once per page. We patch the model plus every submodule exposing a
    callable ``generate`` and record **every outermost** call — a depth counter
    skips nested patched calls so an outer ``generate`` that delegates to an inner
    ``generate`` is counted once. ``fired`` reports whether any wrapper ran.
    """

    def __init__(self, model: Any) -> None:
        self._model = model
        self._patched: list[tuple[Any, Callable[..., Any]]] = []
        self._depth = 0
        self.records: list[dict[str, Any]] = []

    def _targets(self) -> list[tuple[str, Any]]:
        targets: list[tuple[str, Any]] = [("<model>", self._model)]
        named = getattr(self._model, "named_modules", None)
        if callable(named):
            for name, module in named():
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
                    outermost = self._depth == 0
                    input_len = _infer_input_len(args, kwargs) if outermost else None
                    self._depth += 1
                    try:
                        out = orig(*args, **kwargs)
                    finally:
                        self._depth -= 1
                    if outermost and self._depth == 0:
                        self.records.append(
                            {"target": target, "input_len": input_len, "out": out}
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
        self._depth = 0

    def summarize(
        self,
        eos_token_id: Optional[int],
        *,
        max_length: Optional[int] = None,
        semantics: str = SEMANTICS_TOTAL,
        slack: int = 0,
    ) -> GenerateSummary:
        """Convert every recorded outermost call into a :class:`CallRecord`.

        ``stop_reason`` is ``"eos"`` when the final token is EOS; ``"length"``
        only when the call's token count reached the effective limit given
        ``max_length`` and ``semantics``; otherwise ``"other"``.
        """
        calls: list[CallRecord] = []
        for rec in self.records:
            total, ended = _seq_total_and_eos(rec["out"], eos_token_id)
            input_tokens = rec.get("input_len")
            if total is None:
                output_tokens = None
            elif input_tokens is not None and total >= input_tokens:
                output_tokens = total - input_tokens
            else:
                output_tokens = total
            if ended is True:
                stop_reason = "eos"
            elif _limit_reached(total, output_tokens, max_length, semantics, slack):
                stop_reason = "length"
            else:
                stop_reason = "other"
            calls.append(CallRecord(rec.get("target"), input_tokens, output_tokens, stop_reason))
        return GenerateSummary(calls=calls, fired=bool(self.records))


def derive_capture_fields(
    summary: GenerateSummary,
    *,
    n_pages: Optional[int],
    prompt_tokens: Optional[int],
) -> dict[str, Any]:
    """Flatten a :class:`GenerateSummary` into result-row fields."""
    fields: dict[str, Any] = {
        "capture_fired": summary.fired,
        "capture_target": summary.first_target if summary.fired else None,
        "generate_calls": summary.generate_calls if summary.fired else None,
        "output_tokens_total": summary.output_tokens_total if summary.fired else None,
        "input_tokens_first": summary.input_tokens_first if summary.fired else None,
        "any_hit_max_length": summary.any_hit_max_length if summary.fired else None,
        "per_page_generation": None,
        "image_tokens_per_page_measured": None,
    }
    if not summary.fired:
        return fields
    if n_pages:
        fields["per_page_generation"] = summary.generate_calls == n_pages
    fields["image_tokens_per_page_measured"] = _measure_image_tokens(
        summary, prompt_tokens=prompt_tokens, n_pages=n_pages
    )
    return fields


def _measure_image_tokens(
    summary: GenerateSummary,
    *,
    prompt_tokens: Optional[int],
    n_pages: Optional[int],
) -> Optional[float]:
    """Measured image tokens per page from the captured input lengths.

    One call: ``(input - prompt) / pages``. Per-page calls
    (``generate_calls == n_pages``): the mean of ``(input - prompt)`` across
    calls. Otherwise None (can't cleanly attribute).
    """
    if prompt_tokens is None:
        return None
    inputs = [c.input_tokens for c in summary.calls if c.input_tokens is not None]
    if not inputs:
        return None
    if summary.generate_calls == 1:
        if not n_pages:
            return None
        return round((inputs[0] - prompt_tokens) / n_pages, 1)
    if n_pages and summary.generate_calls == n_pages:
        values = [value - prompt_tokens for value in inputs]
        return round(sum(values) / len(values), 1)
    return None


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
    capture_summary: Optional[GenerateSummary] = None


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
    max_length_semantics: str = SEMANTICS_TOTAL,
    max_length_slack: int = 0,
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
        capture.summarize(
            getattr(tokenizer, "eos_token_id", None),
            max_length=infer_kwargs.get("max_length"),
            semantics=max_length_semantics,
            slack=max_length_slack,
        )
        if capture is not None
        else GenerateSummary(calls=[], fired=False)
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
        capture_summary=summary,
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
        "input_tokens_first": None,  # measured (text + image) from GenerateCapture
        "image_tokens_per_page_measured": None,
        "capture_fired": None,
        "capture_target": None,
        "generate_calls": None,
        "per_page_generation": None,
        "any_hit_max_length": None,
        "output_tokens_total": None,
        "wall_seconds": None,
        "gpu_seconds": None,
        "peak_vram_gb": None,
        "output_tokens": None,
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
        summary = outcome.capture_summary or GenerateSummary(calls=[], fired=False)
        capture_fields = derive_capture_fields(
            summary, n_pages=run_key.get("input_pages"), prompt_tokens=prompt_tokens
        )

        # Prefer the captured truth (any call stopped on length) over the
        # token-count approximation when the capture actually fired.
        if summary.fired and summary.any_hit_max_length is not None:
            hit_max = summary.any_hit_max_length
            stop_reason = summary.overall_stop_reason
        else:
            hit_max = _compute_hit_max_length(
                output_tokens=outcome.output_tokens,
                input_tokens=input_tokens or 0,
                captured_total_tokens=None,
                max_length=max_length,
                semantics=max_length_semantics,
                slack=max_length_slack,
            )
            stop_reason = _stop_reason(hit_max, None)

        covered = last_page_check(outcome) if last_page_check is not None else None
        row.update(capture_fields)
        row.update(
            wall_seconds=round(outcome.wall_seconds, 3),
            gpu_seconds=round(outcome.gpu_seconds, 3),
            peak_vram_gb=round(outcome.peak_vram_gb, 3),
            output_tokens=outcome.output_tokens,
            hit_max_length=bool(hit_max),
            stop_reason=stop_reason,
            last_page_covered=covered,
            block_types=outcome.block_types,
            raw_path=outcome.raw_path,
            cleaned_path=outcome.cleaned_path,
        )
        if not summary.fired:
            logger.warning(
                "Run %s: GenerateCapture did not fire; measured input tokens are "
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
