# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
QUANT = ROOT / "vllm/model_executor/layers/quantization"
KERNELS = ROOT / "vllm/model_executor/kernels/linear"
W4 = QUANT / "gfx1151_w4a4_archive.py"
W8 = KERNELS / "scaled_mm/triton.py"
REGISTRY = QUANT / "__init__.py"
DONOR_ROOT = Path(
    os.environ.get("VLLM_GFX1151_DONOR_ROOT", "/__missing_gfx1151_donor__")
)
CHECKPOINT_CONFIG = Path(
    os.environ.get(
        "VLLM_GFX1151_CHECKPOINT_CONFIG", "/__missing_gfx1151_checkpoint_config__"
    )
)
requires_donor = pytest.mark.skipif(
    not DONOR_ROOT.is_dir(),
    reason="set VLLM_GFX1151_DONOR_ROOT for external archive-provenance tests",
)
requires_checkpoint = pytest.mark.skipif(
    not CHECKPOINT_CONFIG.is_file(),
    reason="set VLLM_GFX1151_CHECKPOINT_CONFIG for real-checkpoint contract test",
)
DONOR_W4 = DONOR_ROOT / "promotion-v1/payload/source/gfx1151_w4a4.py"
DONOR_W8 = DONOR_ROOT / "promotion-v1/payload/source/w8a8_scaled_mm_triton.py"
NATIVE_ROOT = ROOT / "csrc/quantization/gfx1151_archive_w4a4"
DONOR_NATIVE = {
    "gfx1151_w4a4_perf_op.hip": DONOR_ROOT / "w4a4-native/v2/gfx1151_w4a4_perf_op.hip",
    "gfx1151_w4a4_m128_v3_op.hip": DONOR_ROOT
    / "w4a4-native/v3/gfx1151_w4a4_m128_v3_op.hip",
    "gfx1151_w4a4_m128_qfloat_op.hip": DONOR_ROOT
    / "w4a4-native/v4q/gfx1151_w4a4_m128_qfloat_op.hip",
    "gfx1151_w4a4_m1_dot_pair2_signed_op.hip": DONOR_ROOT
    / "w4a4-native/v6f/gfx1151_w4a4_m1_dot_pair2_signed_op.hip",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@requires_donor
def test_w4_port_preserves_archived_python_source_semantics():
    assert W4.is_file()
    source = W4.read_text()
    donor = DONOR_W4.read_text()
    # The adaptation may change imports and asset-root wiring only. Core wire,
    # profile names, route labels and entry-point calls must remain verbatim.
    for token in (
        '"format": "gfx1151-w4a4-g32-v1"',
        '"weight_encoding": "symmetric_uint4b8"',
        '"activation": "dynamic_s4_group32"',
        '"gfx1151-w4a4-v4q-m1-pair2"',
        '"m128_v4q_quantizer_exact_v3"',
        '"m1_pair2_signed_dot8"',
        "torch.ops.gfx1151_w4a4_m128_v3.linear_quantized",
        "torch.ops.gfx1151_w4a4_m128_v4q.quantize",
        "torch.ops.gfx1151_w4a4_m1_dot_pair2_signed.run_quantized",
    ):
        assert token in donor
        assert token in source
    assert "promotion_authority" in source
    assert "False" in source


@requires_checkpoint
def test_w4_real_checkpoint_wire_remains_admissible_without_promotion_authority():
    assert _sha256(CHECKPOINT_CONFIG) == (
        "ac0c43b58d9a4a8d7cbe852e11d419b32f5c8c92459144c4c93122a392da808d"
    )
    quant = json.loads(CHECKPOINT_CONFIG.read_text())["quantization_config"]
    assert quant["route_scope"] == "selected_first_class_profile"
    source = W4.read_text()
    assert (
        'route_scopes = {"forced_server_route", "selected_first_class_profile"}'
        in source
    )
    assert "promotion_authority = False" in source


def test_w4_exact_method_is_registered_without_neutral_aliases():
    source = REGISTRY.read_text()
    assert '"gfx1151-w4a4-v4q-m1-pair2"' in source
    assert "gfx1151_w4a4_archive" in source
    assert '"gfx1151-w4a4"' not in source
    assert '"gfx1151-w8a8"' not in source


def _without_trailing_horizontal_whitespace(path: Path) -> str:
    text = path.read_text()
    normalized = "\n".join(line.rstrip(" \t") for line in text.splitlines())
    return normalized + ("\n" if text.endswith("\n") else "")


@requires_donor
def test_w4_native_sources_are_donor_semantics_with_whitespace_only_normalization():
    assert set(p.name for p in NATIVE_ROOT.glob("*.hip")) == set(DONOR_NATIVE)
    for name, donor in DONOR_NATIVE.items():
        assert (
            NATIVE_ROOT / name
        ).read_text() == _without_trailing_horizontal_whitespace(donor)


def test_w4_native_sources_use_current_stream_and_never_default_stream():
    for name in DONOR_NATIVE:
        source = (NATIVE_ROOT / name).read_text()
        assert "getCurrentCUDAStream" in source
        assert "getDefaultCUDAStream" not in source
        assert "HSA_OVERRIDE_GFX_VERSION" not in source


@requires_donor
def test_w8_port_retains_archived_trace_contract():
    source = W8.read_text()
    donor = DONOR_W8.read_text()
    for token in (
        "def _trace_gfx1151_w8a8",
        '"kernel": "TritonInt8ScaledMMLinearKernel"',
        '"route": "triton_int8_scaled_mm"',
        "_trace_gfx1151_w8a8(x=x, w_q=w_q)",
    ):
        assert token in donor
        assert token in source


def test_w8_rocm_registry_keeps_triton_candidate_and_explicit_backend_filter():
    source = (KERNELS / "__init__.py").read_text()
    assert (
        "PlatformEnum.ROCM: "
        "[AiterInt8ScaledMMLinearKernel, TritonInt8ScaledMMLinearKernel]" in source
    )
    assert '"triton": {' in source
    assert "TritonInt8ScaledMMLinearKernel" in source
    assert "_filter_kernels_by_backend(linear_backend, platform_kernels)" in source


def test_port_contract_explicitly_blocks_historical_promotion_policy():
    contract = ROOT / "gfx1151_archive_port_contract.json"
    value = __import__("json").loads(contract.read_text())
    assert value["schema"] == "gfx1151.archive-dual-family.rocm10-port.v2"
    assert value["target"] == {"rocm": "10.0", "arch": "gfx1151", "vllm": "0.27.0"}
    assert value["families"]["w4a4"]["wire"] == "gfx1151-w4a4-g32-v1"
    assert value["families"]["w8a8"]["wire"] == "compressed-tensors-int8-dynamic-token"
    assert value["historical_binaries_reused"] is False
    assert value["historical_promotion_reused"] is False
    assert value["gpu_qualified"] is False
    assert value["promotion_authority"] is False
    assert value["families"]["w4a4"]["build_manifest_sha256"] == (
        "34cd1fba7da7df504ed1d882a75d2d21494884e9f45f014c3716c2d8a638d217"
    )
    assert len(value["hardening_adaptations"]) == 4
