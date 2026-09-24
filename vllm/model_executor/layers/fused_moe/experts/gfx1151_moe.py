# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unqualified, explicitly enabled gfx1151 packed MoE experiment."""

import contextlib
import contextvars
import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

SOURCE_FILES = (
    "csrc/rocm/moe_q_gemm_rdna3.cu",
    "csrc/rocm/qdq_4_rdna3.cuh",
    "csrc/rocm/moe_q_gemm_gfx1151.cu",
    "vllm/model_executor/layers/fused_moe/experts/gfx1151_moe.py",
    "vllm/model_executor/layers/fused_moe/experts/rdna3_moe.py",
    "vllm/model_executor/layers/fused_moe/oracle/int_wna16.py",
)
_receipt = contextvars.ContextVar("gfx1151_moe_dispatch", default=None)


def enabled() -> bool:
    return os.environ.get("VLLM_EXPERIMENTAL_GFX1151_MOE") == "1"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_manifest(path: Path, source_root: Path) -> dict:
    """Bind the experimental library to its exact source; no qualification claim."""
    if not path.is_absolute():
        raise ValueError("gfx1151 MoE manifest must be an absolute path")
    data = json.loads(path.read_text())
    if (
        data.get("schema") != 1
        or data.get("architecture") != "gfx1151"
        or data.get("qualification") != "unqualified"
        or set(data.get("sources", {})) != set(SOURCE_FILES)
    ):
        raise ValueError("invalid gfx1151 MoE manifest")
    for name in SOURCE_FILES:
        if sha256(source_root / name) != data["sources"][name]:
            raise ValueError(f"gfx1151 MoE source hash mismatch: {name}")
    library = Path(data["library"])
    if not library.is_absolute() or sha256(library) != data["library_sha256"]:
        raise ValueError("gfx1151 MoE library hash mismatch")
    return data


@lru_cache(maxsize=1)
def _load(path: str):
    import torch

    data = verify_manifest(Path(path), Path(__file__).resolve().parents[5])
    if data.get("torch") != torch.__version__ or data.get("hip") != torch.version.hip:
        raise RuntimeError("gfx1151 MoE binary requires its original Torch/HIP ABI")
    torch.ops.load_library(data["library"])
    op = torch.ops._rocm_C.moe_gptq_gemm_gfx1151
    return op, data["library_sha256"]


def operator():
    if not enabled():
        raise RuntimeError("gfx1151 packed MoE experiment is disabled")
    path = os.environ.get("VLLM_GFX1151_MOE_MANIFEST", "")
    if not path:
        raise RuntimeError("VLLM_GFX1151_MOE_MANIFEST is required; no fallback")
    return _load(path)


def validate_checkpoint(w1, w2, s1, s2, group_size):
    """Validate the packed checkpoint before the in-place native shuffle."""
    import torch

    if group_size != 128 or w1.ndim != 3:
        raise ValueError("gfx1151 MoE requires rank-3 weights and group size 128")
    experts = w1.shape[0]
    if not 1 <= experts <= 512:
        raise ValueError("gfx1151 MoE requires 1..512 experts")
    for w, s, k, n in ((w1, s1, 2560, 1280), (w2, s2, 640, 2560)):
        if (
            tuple(w.shape) != (experts, k // 8, n)
            or tuple(s.shape) != (experts, k // 128, n)
            or w.dtype != torch.int32
            or s.dtype != torch.bfloat16
            or not w.is_contiguous()
            or not s.is_contiguous()
            or w.device != w1.device
            or s.device != w1.device
        ):
            raise ValueError("invalid gfx1151 packed weight or group-scale layout")
        if not bool((torch.isfinite(s) & (s > 0)).all()):
            raise ValueError("gfx1151 MoE scales must be positive and finite")


def gemm(*args, output_topk=0):
    op, digest = operator()
    op(*args, output_topk)
    receipt = _receipt.get()
    if receipt is not None:
        receipt.append(
            {
                "operator": "_rocm_C::moe_gptq_gemm_gfx1151",
                "library_sha256": digest,
                "input_shape": list(args[0].shape),
                "packed_shape": list(args[2].shape),
                "output_topk": output_topk,
                "status": "enqueued",
            }
        )


@contextlib.contextmanager
def capture_dispatch():
    """Capture this request's enqueues; the caller must synchronize and validate."""
    calls = []
    token = _receipt.set(calls)
    try:
        yield calls
    finally:
        _receipt.reset(token)
