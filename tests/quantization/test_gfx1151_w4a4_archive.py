# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm.model_executor.layers.quantization import get_quantization_config
from vllm.model_executor.layers.quantization import (
    gfx1151_w4a4_archive as gfx1151_w4a4,
)
from vllm.model_executor.layers.quantization.gfx1151_w4a4_archive import (
    Gfx1151W4A4IncumbentConfig,
    Gfx1151W4A4LinearMethod,
    Gfx1151W4A4M1Pair2Config,
    prepare_groupmajor_weights,
    projection_from_prefix,
)


EXACT_CONFIG = {
    "quant_method": "gfx1151-w4a4-incumbent",
    "bits": 4,
    "group_size": 32,
    "symmetric": True,
    "weight_encoding": "symmetric_uint4b8",
    "weight_packed_dtype": "int32",
    "weight_scale_dtype": "bfloat16",
    "activation": "dynamic_s4_group32",
    "route_scope": "forced_server_route",
    "format": "gfx1151-w4a4-g32-v1",
}


def test_both_forced_quantization_routes_are_registered() -> None:
    assert (
        get_quantization_config("gfx1151-w4a4-incumbent")
        is Gfx1151W4A4IncumbentConfig
    )
    assert (
        get_quantization_config("gfx1151-w4a4-m1-pair2")
        is Gfx1151W4A4M1Pair2Config
    )


@pytest.mark.parametrize(
    "config_class,route",
    [
        (Gfx1151W4A4IncumbentConfig, "incumbent"),
        (Gfx1151W4A4M1Pair2Config, "candidate"),
    ],
)
def test_config_accepts_only_the_exact_artifact_wire(config_class, route) -> None:
    config = config_class.from_config(dict(EXACT_CONFIG))
    assert config.route == route
    assert config.group_size == 32
    for key, wrong in (
        ("bits", 3),
        ("group_size", 128),
        ("symmetric", False),
        ("weight_encoding", "signed-nibbles"),
        ("format", "other"),
    ):
        mutated = dict(EXACT_CONFIG)
        mutated[key] = wrong
        with pytest.raises(ValueError, match="exact gfx1151 W4A4"):
            config_class.from_config(mutated)


def test_candidate_cli_override_is_explicit_and_incumbent_is_default() -> None:
    assert (
        Gfx1151W4A4M1Pair2Config.override_quantization_method(
            EXACT_CONFIG, "gfx1151-w4a4-m1-pair2"
        )
        == "gfx1151-w4a4-m1-pair2"
    )
    assert (
        Gfx1151W4A4IncumbentConfig.override_quantization_method(
            EXACT_CONFIG, None
        )
        is None
    )
    assert (
        Gfx1151W4A4M1Pair2Config.override_quantization_method(
            EXACT_CONFIG, "awq"
        )
        is None
    )


@pytest.mark.parametrize(
    "prefix,projection",
    [
        ("model.layers.0.self_attn.qkv_proj", "qkv_proj"),
        ("model.layers.1.self_attn.o_proj", "o_proj"),
        ("model.layers.2.mlp.gate_up_proj", "gate_up_proj"),
        ("model.layers.3.mlp.down_proj", "down_proj"),
    ],
)
def test_projection_identity_is_derived_from_exact_vllm_prefix(
    prefix: str, projection: str
) -> None:
    assert projection_from_prefix(prefix) == projection


def test_projection_identity_rejects_unquantized_or_ambiguous_layers() -> None:
    for prefix in ("lm_head", "model.embed_tokens", "model.layers.0.foo"):
        with pytest.raises(ValueError, match="unsupported W4A4 projection"):
            projection_from_prefix(prefix)


def test_create_weights_matches_materialized_checkpoint_shapes() -> None:
    config = Gfx1151W4A4IncumbentConfig.from_config(EXACT_CONFIG)
    method = Gfx1151W4A4LinearMethod(config, "model.layers.0.self_attn.qkv_proj")
    layer = torch.nn.Module()
    with (
        patch(
            "vllm.model_executor.parameter.get_tensor_model_parallel_rank",
            return_value=0,
        ),
        patch(
            "vllm.model_executor.parameter.get_tensor_model_parallel_world_size",
            return_value=1,
        ),
    ):
        method.create_weights(
            layer,
            input_size_per_partition=4096,
            output_partition_sizes=[4096, 1024, 1024],
            input_size=4096,
            output_size=6144,
            params_dtype=torch.bfloat16,
            weight_loader=lambda *args, **kwargs: None,
        )
    assert tuple(layer.weight_packed.shape) == (6144, 512)
    assert tuple(layer.weight_scale.shape) == (6144, 128)
    assert layer.weight_packed.dtype == torch.int32
    assert layer.weight_scale.dtype == torch.bfloat16


def test_create_weights_rejects_tensor_parallel_partitioning() -> None:
    config = Gfx1151W4A4IncumbentConfig.from_config(EXACT_CONFIG)
    method = Gfx1151W4A4LinearMethod(config, "model.layers.0.self_attn.o_proj")
    with pytest.raises(ValueError, match="tensor parallel size 1"):
        method.create_weights(
            torch.nn.Module(),
            input_size_per_partition=2048,
            output_partition_sizes=[4096],
            input_size=4096,
            output_size=4096,
            params_dtype=torch.bfloat16,
            weight_loader=lambda *args, **kwargs: None,
        )


def test_groupmajor_layout_is_exact_transpose_of_checkpoint_wire() -> None:
    packed = torch.arange(16 * 8, dtype=torch.int32).reshape(16, 8)
    scale = torch.arange(16 * 2, dtype=torch.bfloat16).reshape(16, 2)
    group_packed, group_scale = prepare_groupmajor_weights(packed, scale)
    assert tuple(group_packed.shape) == (2, 4, 16)
    assert tuple(group_scale.shape) == (2, 16)
    assert torch.equal(group_packed, packed.reshape(16, 2, 4).permute(1, 2, 0))
    assert torch.equal(group_scale, scale.transpose(0, 1))


class FakeRuntime:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def quantize_incumbent(self, x):
        self.calls.append(("quantize_incumbent", tuple(x.shape)))
        return "a-inc", "s-inc"

    def quantize_candidate(self, x):
        self.calls.append(("quantize_candidate", tuple(x.shape)))
        return "a-can", "s-can"

    def exact_m128(self, a, s, weight, scale):
        self.calls.append(("exact_m128", a, s, tuple(weight.shape), tuple(scale.shape)))
        return torch.full((128, weight.shape[2]), 4.0, dtype=torch.bfloat16)

    def fallback(self, x, packed, scale, route_code):
        self.calls.append(("fallback", tuple(x.shape), route_code))
        return torch.full((x.shape[0], packed.shape[0]), 3.0, dtype=torch.bfloat16)

    def m1_pair2(self, x, packed, scale):
        self.calls.append(("m1_pair2", tuple(x.shape)))
        return torch.full((x.shape[0], packed.shape[0]), 5.0, dtype=torch.bfloat16)

    def trace(self, **row):
        self.calls.append(("trace", row))


@pytest.mark.parametrize(
    "route",
    ["incumbent", "candidate"],
)
def test_m128_forced_routes_share_incumbent_quantizer_and_matrix(route: str) -> None:
    runtime = FakeRuntime()
    method = object.__new__(Gfx1151W4A4LinearMethod)
    method.route = route
    method.projection = "o_proj"
    method.runtime = runtime
    layer = SimpleNamespace(
        weight_packed=torch.zeros((4096, 512), dtype=torch.int32),
        weight_scale=torch.ones((4096, 128), dtype=torch.bfloat16),
        weight_groupmajor_packed=torch.zeros((128, 4, 4096), dtype=torch.int32),
        weight_groupmajor_scale=torch.ones((128, 4096), dtype=torch.bfloat16),
    )
    out = method.apply(layer, torch.ones((128, 4096), dtype=torch.bfloat16))
    names = [call[0] for call in runtime.calls]
    assert names[0] == "quantize_incumbent"
    assert "exact_m128" in names
    assert "quantize_candidate" not in names
    assert tuple(out.shape) == (128, 4096)


@pytest.mark.parametrize("route", ["incumbent", "candidate"])
def test_m1_uses_route_owner_and_other_shapes_retain_fallback(route: str) -> None:
    runtime = FakeRuntime()
    method = object.__new__(Gfx1151W4A4LinearMethod)
    method.route = route
    method.projection = "down_proj"
    method.runtime = runtime
    layer = SimpleNamespace(
        weight_packed=torch.zeros((4096, 1536), dtype=torch.int32),
        weight_scale=torch.ones((4096, 384), dtype=torch.bfloat16),
    )
    out = method.apply(layer, torch.ones((1, 12288), dtype=torch.bfloat16))
    expected = (
        ("fallback", (1, 12288), 1)
        if route == "incumbent"
        else ("m1_pair2", (1, 12288))
    )
    assert runtime.calls[0] == expected
    assert tuple(out.shape) == (1, 4096)


def test_m1_pair2_route_is_registered_under_an_explicit_method() -> None:
    config_class = getattr(gfx1151_w4a4, "Gfx1151W4A4M1Pair2Config", None)
    assert config_class is not None
    assert get_quantization_config("gfx1151-w4a4-m1-pair2") is config_class
    config = config_class.from_config(EXACT_CONFIG)
    assert config.route == "candidate"


def test_m1_pair2_route_holds_m128_quantizer_and_matrix_constant() -> None:
    runtime = FakeRuntime()
    method = object.__new__(Gfx1151W4A4LinearMethod)
    method.route = "candidate"
    method.projection = "o_proj"
    method.runtime = runtime
    layer = SimpleNamespace(
        weight_packed=torch.zeros((4096, 512), dtype=torch.int32),
        weight_scale=torch.ones((4096, 128), dtype=torch.bfloat16),
        weight_groupmajor_packed=torch.zeros((128, 4, 4096), dtype=torch.int32),
        weight_groupmajor_scale=torch.ones((128, 4096), dtype=torch.bfloat16),
    )
    method.apply(layer, torch.ones((128, 4096), dtype=torch.bfloat16))
    names = [call[0] for call in runtime.calls]
    assert names[0] == "quantize_incumbent"
    assert "exact_m128" in names
    assert "quantize_candidate" not in names
    assert "m1_pair2" not in names


def test_m1_pair2_candidate_is_selected_only_for_exactly_one_row() -> None:
    runtime = FakeRuntime()
    method = object.__new__(Gfx1151W4A4LinearMethod)
    method.route = "candidate"
    method.projection = "down_proj"
    method.runtime = runtime
    layer = SimpleNamespace(
        weight_packed=torch.zeros((4096, 1536), dtype=torch.int32),
        weight_scale=torch.ones((4096, 384), dtype=torch.bfloat16),
    )
    method.apply(layer, torch.ones((1, 12288), dtype=torch.bfloat16))
    assert runtime.calls[0] == ("m1_pair2", (1, 12288))
    runtime.calls.clear()
    method.apply(layer, torch.ones((2, 12288), dtype=torch.bfloat16))
    assert runtime.calls[0] == ("fallback", (2, 12288), 1)


def test_m1_incumbent_retains_the_existing_dot_fallback() -> None:
    runtime = FakeRuntime()
    method = object.__new__(Gfx1151W4A4LinearMethod)
    method.route = "incumbent"
    method.projection = "down_proj"
    method.runtime = runtime
    layer = SimpleNamespace(
        weight_packed=torch.zeros((4096, 1536), dtype=torch.int32),
        weight_scale=torch.ones((4096, 384), dtype=torch.bfloat16),
    )
    method.apply(layer, torch.ones((1, 12288), dtype=torch.bfloat16))
    assert runtime.calls[0] == ("fallback", (1, 12288), 1)


def test_apply_rejects_wrong_activation_dtype() -> None:
    runtime = FakeRuntime()
    method = object.__new__(Gfx1151W4A4LinearMethod)
    method.route = "candidate"
    method.projection = "o_proj"
    method.runtime = runtime
    layer = SimpleNamespace(weight_packed=torch.zeros((4096, 512), dtype=torch.int32))
    with pytest.raises(ValueError, match="BF16"):
        method.apply(layer, torch.ones((128, 4096), dtype=torch.float16))


def test_v4q_pair2_route_is_registered_under_an_explicit_method() -> None:
    config_class = getattr(gfx1151_w4a4, "Gfx1151W4A4V4QM1Pair2Config", None)
    assert config_class is not None
    assert (
        get_quantization_config("gfx1151-w4a4-v4q-m1-pair2") is config_class
    )
    config = config_class.from_config(EXACT_CONFIG)
    assert config.route == "v4q_pair2"


def test_final_pair2_routes_load_identical_optional_extension_closure() -> None:
    closure = getattr(gfx1151_w4a4, "optional_extension_closure", None)
    assert closure is not None
    assert closure("candidate") == ("v6f", "v4q")
    assert closure("v4q_pair2") == ("v6f", "v4q")
    assert closure("incumbent") == ()


def test_v4q_pair2_changes_only_the_exact_m128_quantizer() -> None:
    runtime = FakeRuntime()
    method = object.__new__(Gfx1151W4A4LinearMethod)
    method.route = "v4q_pair2"
    method.projection = "o_proj"
    method.runtime = runtime
    layer = SimpleNamespace(
        weight_packed=torch.zeros((4096, 512), dtype=torch.int32),
        weight_scale=torch.ones((4096, 128), dtype=torch.bfloat16),
        weight_groupmajor_packed=torch.zeros((128, 4, 4096), dtype=torch.int32),
        weight_groupmajor_scale=torch.ones((128, 4096), dtype=torch.bfloat16),
    )
    method.apply(layer, torch.ones((128, 4096), dtype=torch.bfloat16))
    names = [call[0] for call in runtime.calls]
    assert names[0] == "quantize_candidate"
    assert "exact_m128" in names
    assert "quantize_incumbent" not in names
    assert "m1_pair2" not in names


def test_v4q_pair2_shares_v6f_m1_and_incumbent_fallbacks() -> None:
    runtime = FakeRuntime()
    method = object.__new__(Gfx1151W4A4LinearMethod)
    method.route = "v4q_pair2"
    method.projection = "down_proj"
    method.runtime = runtime
    layer = SimpleNamespace(
        weight_packed=torch.zeros((4096, 1536), dtype=torch.int32),
        weight_scale=torch.ones((4096, 384), dtype=torch.bfloat16),
    )
    method.apply(layer, torch.ones((1, 12288), dtype=torch.bfloat16))
    assert runtime.calls[0] == ("m1_pair2", (1, 12288))
    runtime.calls.clear()
    method.apply(layer, torch.ones((2, 12288), dtype=torch.bfloat16))
    assert runtime.calls[0] == ("fallback", (2, 12288), 1)
