# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
BUILDER = ROOT / "tools/build_gfx1151_archive_w4a4_rocm10.py"
W4 = ROOT / "vllm/model_executor/layers/quantization/gfx1151_w4a4_archive.py"
NATIVE_ROOT = ROOT / "csrc/quantization/gfx1151_archive_w4a4"
EXPECTED = {
    "perf": ("gfx1151_w4a4_perf_op.hip", "95e787ec86fb2d0d787844517b08520966d711bbf7a5d33429fe499bffe849e7"),
    "m128_v3": ("gfx1151_w4a4_m128_v3_op.hip", "da90bff3782fd46b43138f1a9b793e3090da67cf32d13c3f6a440fe8a125d0e0"),
    "m128_v4q": ("gfx1151_w4a4_m128_qfloat_op.hip", "994af1d33ad9c8abe54c8475328b4f9c5f6de04790219cc3ce98ecb446e33abe"),
    "m1_pair2_signed": ("gfx1151_w4a4_m1_dot_pair2_signed_op.hip", "3f01ca4e79cd86601247faa58a9dfdd01c1150acbe99ced5c1d48b306b8c802c"),
}


def _load_builder():
    spec = importlib.util.spec_from_file_location("archive_builder", BUILDER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_builder_declares_all_four_exact_sources_and_safe_flags():
    module = _load_builder()
    actual = {
        name: (row["source"], row["source_sha256"])
        for name, row in module.MODULES.items()
    }
    assert actual == EXPECTED
    assert module.HOST_FLAGS == ["-O2", "-std=c++17"]
    assert module.HIP_FLAGS == [
        "-O2",
        "-std=c++17",
        "--offload-arch=gfx1151",
        "-Rpass-analysis=kernel-resource-usage",
    ]
    joined = " ".join(module.HOST_FLAGS + module.HIP_FLAGS)
    for forbidden in ("-O3", "-march=", "-mtune=", "-mwavefrontsize", "HSA_OVERRIDE_GFX_VERSION"):
        assert forbidden not in joined


def test_builder_environment_validation_fails_closed():
    module = _load_builder()
    with pytest.raises(RuntimeError, match="HSA_OVERRIDE_GFX_VERSION"):
        module.validate_environment(
            {"HSA_OVERRIDE_GFX_VERSION": "11.0.0"}, torch_hip="7.15.0", rocm_home="/opt/rocm"
        )
    with pytest.raises(RuntimeError, match="HIP-enabled"):
        module.validate_environment({}, torch_hip=None, rocm_home="/opt/rocm")
    with pytest.raises(RuntimeError, match="ROCm 10"):
        module.validate_environment({}, torch_hip="7.2.1", rocm_home="/opt/rocm")
    module.validate_environment({}, torch_hip="7.15.0", rocm_home="/opt/rocm")


def test_w4_runtime_uses_only_new_port_manifest_not_old_binary_authority():
    source = W4.read_text()
    assert "VLLM_GFX1151_W4A4_PORT_MANIFEST" in source
    for legacy in (
        "VLLM_GFX1151_W4A4_PERF_MANIFEST",
        "VLLM_GFX1151_W4A4_V3_MANIFEST",
        "VLLM_GFX1151_W4A4_V6F_MANIFEST",
        "VLLM_GFX1151_W4A4_V4Q_MANIFEST",
    ):
        assert legacy not in source
    assert '"CPU_BUILT_GPU_UNQUALIFIED"' in source
    assert '"historical_binaries_reused"' in source
    assert '"promotion_authority"' in source
    assert "EXPECTED_SOURCE_SHA256" in source


def test_bundle_manifest_is_exact_shape_and_route_closure_is_bounded():
    source = W4.read_text()
    for module in EXPECTED:
        assert f'"{module}"' in source
    assert 'set(modules) != set(EXPECTED_SOURCE_SHA256)' in source
    assert 'optional_extension_closure(self.route)' in source
    assert 'torch.ops.load_library' in source
    assert 'f"/proc/self/fd/{library_fd}"' in source


def test_native_sources_have_no_trailing_whitespace():
    for path in sorted(NATIVE_ROOT.glob("*.hip")):
        offenders = [
            number
            for number, line in enumerate(path.read_text().splitlines(), start=1)
            if line.endswith((" ", "\t"))
        ]
        assert offenders == [], f"{path.name}: trailing whitespace at {offenders}"
