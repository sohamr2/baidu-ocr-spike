"""Tests for GenerateCapture using fake models (no GPU needed).

These need torch (for ``nn.Module`` fakes and tensor outputs) but run on CPU.
They are skipped where torch is unavailable; on the Colab/Linux target torch is
installed and they run.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from ocr_spike import infer  # noqa: E402

EOS = 2


def _seq(total: int, last_token: int) -> "torch.Tensor":
    """A [1, total] id tensor whose final token is ``last_token``."""
    row = torch.zeros((1, total), dtype=torch.long)
    row[0, -1] = last_token
    return row


class _Inner(nn.Module):
    """Inner language-model whose generate adds 7 tokens and ends on EOS."""

    def generate(self, input_ids=None, **kwargs):
        n = input_ids.shape[1]
        return _seq(n + 7, EOS)


class _OuterNested(nn.Module):
    """Outer model whose generate delegates to an inner module's generate."""

    def __init__(self) -> None:
        super().__init__()
        self.inner = _Inner()

    def generate(self, input_ids=None, **kwargs):
        return self.inner.generate(input_ids=input_ids)


class _Scripted(nn.Module):
    """Model whose generate replays a script of ``(new_tokens, last_token)``.

    ``infer_multi`` calls ``generate`` once per item in ``image_files``.
    """

    def __init__(self, script):
        super().__init__()
        self._script = list(script)
        self._i = 0

    def generate(self, input_ids=None, inputs_embeds=None, **kwargs):
        new_tokens, last_token = self._script[self._i]
        self._i += 1
        n = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]
        return _seq(n + new_tokens, last_token)

    def infer_multi(self, image_files, input_len=12, **kwargs):
        for _ in image_files:
            self.generate(input_ids=_seq(input_len, 0))


# --------------------------------------------------------------------------- #
# Nested generate -> counted once (outermost only)
# --------------------------------------------------------------------------- #
def test_nested_generate_counts_outermost_only() -> None:
    model = _OuterNested()
    with infer.GenerateCapture(model) as cap:
        model.generate(input_ids=_seq(10, 0))
    summary = cap.summarize(EOS)

    assert summary.fired is True
    assert summary.generate_calls == 1  # inner call not double-counted
    assert summary.first_target == "<model>"  # outermost wins
    assert summary.input_tokens_first == 10
    assert summary.calls[0].output_tokens == 7  # total 17 - input 10
    assert summary.output_tokens_total == 7
    assert summary.any_hit_max_length is False  # ended on EOS
    assert summary.overall_stop_reason == "eos"


# --------------------------------------------------------------------------- #
# Per-page generation -> one call per page
# --------------------------------------------------------------------------- #
def test_per_page_generation_counts_each_call() -> None:
    n_pages = 4
    model = _Scripted([(5, EOS)] * n_pages)
    with infer.GenerateCapture(model) as cap:
        model.infer_multi(image_files=list(range(n_pages)), input_len=12)
    summary = cap.summarize(EOS)

    assert summary.generate_calls == n_pages
    assert summary.input_tokens_first == 12
    assert summary.output_tokens_total == n_pages * 5
    assert summary.any_hit_max_length is False

    fields = infer.derive_capture_fields(summary, n_pages=n_pages, prompt_tokens=4)
    assert fields["capture_fired"] is True
    assert fields["generate_calls"] == n_pages
    assert fields["per_page_generation"] is True
    # mean of (input - prompt) across calls = 12 - 4 = 8
    assert fields["image_tokens_per_page_measured"] == 8.0


def test_any_hit_max_length_when_one_call_stops_on_length() -> None:
    # third call ends on a non-EOS token => stopped on length
    model = _Scripted([(5, EOS), (5, EOS), (5, 0)])
    with infer.GenerateCapture(model) as cap:
        model.infer_multi(image_files=[0, 1, 2], input_len=12)
    summary = cap.summarize(EOS)

    assert summary.generate_calls == 3
    assert summary.any_hit_max_length is True
    assert summary.overall_stop_reason == "length"
    assert [c.stop_reason for c in summary.calls] == ["eos", "eos", "length"]


# --------------------------------------------------------------------------- #
# Single call -> image tokens per page from one input length
# --------------------------------------------------------------------------- #
def test_single_call_image_tokens_measured() -> None:
    model = _Scripted([(7, EOS)])
    with infer.GenerateCapture(model) as cap:
        model.generate(input_ids=_seq(260, 0))  # 4 prompt + 256 image
    summary = cap.summarize(EOS)

    fields = infer.derive_capture_fields(summary, n_pages=1, prompt_tokens=4)
    assert summary.generate_calls == 1
    assert fields["input_tokens_first"] == 260
    assert fields["per_page_generation"] is True  # 1 call == 1 page
    assert fields["image_tokens_per_page_measured"] == 256.0


def test_inputs_embeds_sequence_length_is_measured() -> None:
    # inputs_embeds is [batch, seq, hidden]; seq must be read, not hidden.
    model = _Scripted([(7, EOS)])
    with infer.GenerateCapture(model) as cap:
        model.generate(inputs_embeds=torch.zeros((1, 300, 8)))
    summary = cap.summarize(EOS)
    assert summary.input_tokens_first == 300


# --------------------------------------------------------------------------- #
# Capture did not fire
# --------------------------------------------------------------------------- #
def test_capture_not_fired_reports_none_fields() -> None:
    model = _Scripted([(5, EOS)])
    with infer.GenerateCapture(model) as cap:
        pass  # never called generate
    summary = cap.summarize(EOS)

    assert summary.fired is False
    fields = infer.derive_capture_fields(summary, n_pages=3, prompt_tokens=4)
    assert fields["capture_fired"] is False
    assert fields["generate_calls"] is None
    assert fields["image_tokens_per_page_measured"] is None
    assert fields["per_page_generation"] is None
