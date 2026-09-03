# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import os

import pytest
import torch

from vllm.model_executor.kernels.linear.scaled_mm import triton as w8
from vllm.model_executor.layers.quantization import gfx1151_w4a4_archive as w4


def _force_short_writes(monkeypatch: pytest.MonkeyPatch, module) -> list[int]:
    original = os.write
    sizes: list[int] = []

    def short_write(fd: int, data) -> int:
        payload = bytes(data)
        count = max(1, len(payload) // 3)
        sizes.append(count)
        return original(fd, payload[:count])

    monkeypatch.setattr(module.os, "write", short_write)
    return sizes


def test_w4_trace_survives_short_writes(tmp_path, monkeypatch):
    trace = tmp_path / "w4.jsonl"
    monkeypatch.setenv("VLLM_GFX1151_W4A4_TRACE_PATH", str(trace))
    writes = _force_short_writes(monkeypatch, w4)
    w4.Gfx1151W4A4Runtime("incumbent").trace(
        family="w4a4", selected_route="fallback_wmma_baseline_padded"
    )
    rows = [json.loads(line) for line in trace.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["family"] == "w4a4"
    assert len(writes) > 1


def test_w8_trace_survives_short_writes(tmp_path, monkeypatch):
    trace = tmp_path / "w8.jsonl"
    monkeypatch.setenv("VLLM_GFX1151_W8A8_TRACE_PATH", str(trace))
    writes = _force_short_writes(monkeypatch, w8)
    w8._trace_gfx1151_w8a8(
        x=torch.empty((2, 4), dtype=torch.bfloat16),
        w_q=torch.empty((4, 8), dtype=torch.int8),
    )
    rows = [json.loads(line) for line in trace.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["route"] == "triton_int8_scaled_mm"
    assert len(writes) > 1