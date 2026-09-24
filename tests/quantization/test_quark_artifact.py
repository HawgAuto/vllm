# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU artifact contracts; install the runtime-support wheel before this suite."""

import copy
import inspect
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.forward_context import ForwardContext, override_forward_context
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonWNA16Experts
from vllm.model_executor.layers.linear import QKVParallelLinear, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.quark.quark import QuarkConfig

qm = pytest.importorskip("quark_mixed_moe")
md = pytest.importorskip("mtp_dense")
wire = pytest.importorskip("native_wire_binding")


def native_int4(codes):
    n, k = codes.shape
    padded = torch.zeros((k, (n + 7) // 8 * 8), dtype=torch.int64)
    padded[:, :n] = codes.T.to(torch.int64) & 15
    return (padded.reshape(k, -1, 8) << (4 * torch.arange(8))).sum(-1).to(torch.int32)


@pytest.mark.parametrize("bits,n,k,group", [(4, 9, 128, 128), (8, 7, 64, 64)])
def test_native_conversion_preserves_all_codes_and_fp32_scales(bits, n, k, group):
    offset = 1 << (bits - 1)
    codes = (torch.arange(n * k).reshape(n, k) % (2 * offset) - offset).to(torch.int8)
    source = native_int4(codes) if bits == 4 else codes.T.contiguous()
    scales = (torch.arange(n * (k // group)).reshape(n, -1).float() + 1) / 7
    packed, actual_scales = wire.convert_native(
        source,
        scales,
        n=n,
        k=k,
        bits=bits,
        group=group,
    )
    torch.testing.assert_close(packed, md.pack_signed(codes, bits), rtol=0, atol=0)
    torch.testing.assert_close(actual_scales, scales, rtol=0, atol=0)
    assert actual_scales.dtype == torch.float32
    torch.testing.assert_close(
        md.dequant_oracle(packed, actual_scales, bits, group),
        codes.float() * scales.repeat_interleave(group, 1),
    )


def test_native_conversion_rejects_nonzero_output_padding():
    source = native_int4(torch.zeros((1, 128), dtype=torch.int8))
    source[0, 0] = 16
    with pytest.raises(ValueError, match="output padding"):
        wire.convert_native(source, torch.ones(1, 1), n=1, k=128, bits=4, group=128)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("inf"), float("nan")])
def test_native_conversion_rejects_invalid_scales(bad):
    with pytest.raises(ValueError, match="invalid scales"):
        wire.convert_native(
            torch.zeros(64, 1, dtype=torch.int8),
            torch.tensor([[bad]]),
            n=1,
            k=64,
            bits=8,
            group=64,
        )


def base_config():
    return QuarkConfig(
        {
            "version": "0.12.post1",
            "export": {
                "file2file_pack_reorder": False,
                "weight_format": "real_quantized",
                "pack_method": "reorder",
                "kv_cache_group": [],
            },
            "global_quant_config": {
                "weight": {
                    "dtype": "int4",
                    "qscheme": "per_group",
                    "group_size": 128,
                    "ch_axis": -1,
                    "symmetric": True,
                    "is_dynamic": False,
                },
                "input_tensors": None,
                "output_tensors": None,
                "bias": None,
            },
            "exclude": [
                "model.visual.*",
                "*.mlp.gate",
                "*hyper_connection*",
                "*embed*",
            ],
            "layer_quant_config": {},
            "layer_type_quant_config": {},
        }
    )


def mixed_layer():
    parallel = FusedMoEParallelConfig(
        tp_size=1,
        pcp_size=1,
        dp_size=1,
        ep_size=1,
        sp_size=1,
        tp_rank=0,
        pcp_rank=0,
        dp_rank=0,
        ep_rank=0,
        use_ep=False,
        all2all_backend="allgather_reducescatter",
        enable_eplb=False,
    )
    config = FusedMoEConfig(
        num_experts=2,
        experts_per_token=2,
        hidden_dim=2560,
        intermediate_size=640,
        num_local_experts=2,
        num_logical_experts=2,
        activation=MoEActivation.SILU,
        device="cpu",
        routing_method=RoutingMethodType.Default,
        moe_parallel_config=parallel,
        in_dtype=torch.bfloat16,
    )
    method = qm.QuarkMixedMoEMethod(
        config, qm.PoolPlan((0,), (1,), 2), int8_wire=qm.INT8_WIRE
    )
    layer = nn.Module()
    method.create_weights(layer, 2, 2560, 640, torch.bfloat16)
    layer.quant_method = method
    return layer, method


@pytest.fixture
def loaded_mixed():
    layer, method = mixed_layer()
    for expert in range(2):
        for shard in ("w1", "w3", "w2"):
            n, k = (2560, 640) if shard == "w2" else (640, 2560)
            name = "w2" if shard == "w2" else "w13"
            codes = torch.full((n, k), -7 if expert == 0 else -127, dtype=torch.int8)
            source = native_int4(codes) if expert == 0 else codes.T.contiguous()
            scales = torch.full((n, k // 128), 0.1234567)
            for suffix, payload in (("_weight", source), ("_weight_scale", scales)):
                parameter = getattr(layer, name + suffix)
                assert parameter.weight_loader(
                    parameter,
                    payload,
                    name + suffix,
                    shard,
                    expert,
                    return_success=True,
                )
    method.process_weights_after_loading(layer)
    return layer, method


def test_mixed_loader_finalizes_real_upstream_experts_losslessly(loaded_mixed):
    layer, method = loaded_mixed
    assert method.loaded_piece_count == 12
    assert all(type(native) is TritonWNA16Experts for native in method.native.values())
    assert torch.all(layer.mixed_pools["4"].w13_weight == 0x11)
    assert torch.all(layer.mixed_pools["8"].w2_weight == 1)
    assert torch.all(layer.mixed_pools["8"].w2_scale == torch.tensor(0.1234567))
    assert layer.mixed_lookup_4.tolist() == [0, -1]
    assert layer.mixed_lookup_8.tolist() == [-1, 0]
    with pytest.raises(ValueError, match="late weight load"):
        layer.w2_weight.weight_loader(
            layer.w2_weight, torch.empty(0), "w2_weight", "w2", 0
        )


def test_mixed_loader_rejects_missing_duplicate_and_wrong_wire():
    layer, method = mixed_layer()
    with pytest.raises(ValueError, match="incomplete"):
        method.process_weights_after_loading(layer)
    parameter = layer.w13_weight_scale
    scale = torch.ones(640, 20)
    with pytest.raises(ValueError, match="shape/dtype"):
        parameter.weight_loader(parameter, scale.bfloat16(), "scale", "w1", 0)
    parameter.weight_loader(parameter, scale, "scale", "w1", 0)
    with pytest.raises(ValueError, match="duplicate"):
        parameter.weight_loader(parameter, scale, "scale", "w1", 0)


def test_padding_sentinels_require_real_mask_and_discard_nan_slots(loaded_mixed):
    layer, method = loaded_mixed
    x = torch.ones(2, 2560, dtype=torch.bfloat16)
    ids = torch.tensor([[0, 1], [-1, -1]])
    weights = torch.tensor([[0.25, 0.75], [0.5, 0.5]])
    calls = []

    def record(bits, native, kwargs):
        inspect.signature(native.apply).bind(**kwargs)
        assert torch.all(kwargs["topk_weights"] == 1)
        local = kwargs["expert_map"][kwargs["topk_ids"].flatten()]
        kwargs["output"].fill_(0)
        kwargs["output"][local >= 0] = bits
        kwargs["output"][2:] = float("nan")
        calls.append(bits)

    context = ForwardContext({}, {}, {}, is_padding=torch.tensor([False, True]))
    with override_forward_context(context):
        result = method.apply_cpu_test(layer, x, weights, ids, record)
    assert calls == [4, 8]
    assert torch.all(result[0] == 7)
    assert torch.all(result[1] == 0)
    with override_forward_context(None), pytest.raises(RuntimeError, match="expert ID"):
        method.apply_cpu_test(layer, x, weights, ids, record)
    context.is_padding[:] = False
    with (
        override_forward_context(context),
        pytest.raises(RuntimeError, match="expert ID"),
    ):
        method.apply_cpu_test(layer, x, weights, ids, record)


def policy():
    names = [f"mtp.layers.0.self_attn.{part}_proj.weight" for part in ("q", "k", "v")]
    specs = [
        dict(
            name=name,
            source_shape=[n, 64],
            weight_bits=8,
            group_size=64,
            algorithm="RTN",
            activations="BF16",
            symmetric=True,
        )
        for name, n in zip(names, (4, 2, 2))
    ]
    specs += [dict(specs[0], name=f"mtp.dense_{i}.weight") for i in range(16)]
    specs += [
        dict(name=f"mtp.norm_{i}.weight", source_shape=[64], weight_bits=16)
        for i in range(10)
    ]
    specs += [
        dict(
            name=f"mtp.experts_{i}.weight",
            source_shape=[512, 640, 2560],
            weight_bits=4,
            group_size=128,
            algorithm="RTN",
        )
        for i in range(2)
    ]
    return md.Policy(specs, 40)


def test_dense_policy_overrides_only_draft_quantization_and_keeps_bf16():
    base = base_config()
    p = policy()
    manifest = {
        "recipe_sha256": md.RECIPE_SHA,
        "tensors": {s["name"]: md.metadata(s) for s in p.dense},
    }
    draft = md.bind_draft(base, p, manifest)
    assert not hasattr(base, "_mtp_dense_policy")
    assert (
        draft.get_quant_method_target(
            "mtp.layers.40.self_attn.qkv_proj", QKVParallelLinear
        )[2]
        is md.MTPDenseMethod
    )
    assert (
        draft.get_quant_method_target("model.visual.proj", QKVParallelLinear)[2]
        is UnquantizedLinearMethod
    )
    assert (
        draft.get_quant_method_target("mtp.norm_0", QKVParallelLinear)[2]
        is UnquantizedLinearMethod
    )
    assert (
        p.resolve("mtp.layers.41.self_attn.qkv_proj", namespace="draft_prefix") is None
    )
    assert (
        p.resolve("model.layers.40.self_attn.qkv_proj", namespace="draft_prefix")
        == p.dense[:3]
    )


def test_dense_loader_uses_native_qkv_partitions_and_rejects_incomplete():
    p = policy()
    layer = QKVParallelLinear(
        64,
        2,
        2,
        1,
        bias=False,
        params_dtype=torch.bfloat16,
        quant_config=md.BoundConfig(p),
        prefix="mtp.layers.40.self_attn.qkv_proj",
        disable_tp=True,
    )
    method = layer.quant_method
    for shard, n in zip(("q", "k", "v"), (4, 2, 2)):
        layer.weight.weight_loader(
            layer.weight, torch.full((n, 64), 128, dtype=torch.uint8), shard
        )
    with pytest.raises(ValueError, match="missing"):
        method.process_weights_after_loading(layer)
    for shard, n in zip(("q", "k", "v"), (4, 2, 2)):
        layer.weight_scale.weight_loader(
            layer.weight_scale, torch.full((n, 1), 1 / 7), shard
        )
    method.process_weights_after_loading(layer)
    assert method.ready
    assert method.row_map == [
        (s["name"], start, n) for s, start, n in zip(p.dense, (0, 4, 6), (4, 2, 2))
    ]
    assert layer.weight_scale.dtype == torch.float32
    with pytest.raises(RuntimeError, match="GPU dispatch only"):
        method.apply(layer, torch.zeros(1, 64, dtype=torch.bfloat16))


def test_mixed_selection_fails_closed_without_locked_recipe(tmp_path):
    config = copy.deepcopy(base_config().quant_config)
    recipe = tmp_path / "recipe.json"
    recipe.write_text("{}")
    config["mixed_moe"] = dict(
        route=qm.ROUTE, recipe_path=str(recipe), int8_wire=qm.INT8_WIRE
    )
    with pytest.raises(ValueError, match="recipe hash"):
        QuarkConfig.from_config(config)


def test_native_mtp_loader_intercepts_dense_and_keeps_bf16_payloads():
    from vllm.models.qwen4_exp.amd.mtp import Qwen4ExpMTP

    p = policy()
    qkv = QKVParallelLinear(
        64,
        2,
        2,
        1,
        bias=False,
        params_dtype=torch.bfloat16,
        quant_config=md.BoundConfig(p),
        prefix="mtp.layers.40.self_attn.qkv_proj",
        disable_tp=True,
    )
    attention = nn.Module()
    attention.qkv_proj = qkv
    decoder = nn.Module()
    decoder.self_attn = attention
    owner = Qwen4ExpMTP.__new__(Qwen4ExpMTP)
    nn.Module.__init__(owner)
    owner.model = nn.Module()
    owner.model.layers = nn.ModuleList([decoder])
    owner.model.fc_embedding = nn.Linear(64, 64, bias=False, dtype=torch.bfloat16)
    owner.model._mtp_dense_policy = p
    payloads = []
    for spec in p.dense[:3]:
        n, k = spec["source_shape"]
        payloads += [
            (
                "model.language_model." + spec["name"],
                torch.full((n, k), 129, dtype=torch.uint8),
            ),
            (spec["name"] + "_scale", torch.full((n, 1), 0.1234567)),
        ]
    bf16 = torch.full((64, 64), 0.75, dtype=torch.bfloat16)
    payloads.append(("mtp.fc_embedding.weight", bf16))
    loaded = owner.load_weights(iter(payloads))
    assert loaded == {
        "model.layers.0.self_attn.qkv_proj.weight",
        "model.layers.0.self_attn.qkv_proj.weight_scale",
        "model.fc_embedding.weight",
    }
    qkv.quant_method.process_weights_after_loading(qkv)
    torch.testing.assert_close(owner.model.fc_embedding.weight, bf16, rtol=0, atol=0)
    assert torch.all(qkv.weight == 129)
    assert torch.all(qkv.weight_scale == torch.tensor(0.1234567))


def test_hyperconnection_policy_preserves_separate_int8_matrices_and_bf16_norm():
    from vllm.models.qwen4_exp.amd.hyperconnection import (
        GatedResidual,
        HyperConnectionConfig,
    )

    roster = copy.deepcopy(policy().roster)
    prefix = "mtp.layers.0.attn_hyper_connection"
    for i, (leaf, shape) in enumerate(
        [
            ("input_mix_weight_down", [64, 256]),
            ("input_mix_weight_up", [256, 64]),
            ("block_inject_weight", [4, 256]),
        ]
    ):
        roster[i]["name"] = f"{prefix}.{leaf}.weight"
        roster[i]["source_shape"] = shape
    p = md.Policy(roster, 40)
    quant = base_config()
    quant._mtp_dense_policy = p
    config = HyperConnectionConfig(
        hc_count=4, hidden_size=64, hc_lowrank=64, params_dtype=torch.bfloat16
    )
    layer = GatedResidual(
        config, prefix="mtp.layers.40.attn_hyper_connection", quant_config=quant
    )
    assert layer.pad_size == 0
    assert not hasattr(layer, "input_mix_weight_down_block_inject")
    for name in ("input_mix_weight_down", "input_mix_weight_up", "block_inject_weight"):
        projection = getattr(layer, name)
        assert isinstance(projection.quant_method, md.MTPDenseMethod)
        assert projection.weight.dtype == torch.uint8
        assert projection.weight_scale.dtype == torch.float32
    assert layer.hc_norm.weight.dtype == torch.bfloat16


def test_shared_gate_adapter_retains_native_owner_and_int4_wire(monkeypatch):
    import vllm.model_executor.parameter as parameter_module
    from vllm.model_executor.layers.linear import ReplicatedLinear

    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        parameter_module, "get_tensor_model_parallel_world_size", lambda: 1
    )
    roster = copy.deepcopy(policy().roster)
    roster[0].update(
        name="mtp.layers.0.mlp.shared_expert_gate.weight",
        source_shape=[1, 2560],
        weight_bits=4,
        group_size=128,
    )
    quant = base_config()
    quant._mtp_dense_policy = md.Policy(roster, 40)
    gate = ReplicatedLinear(
        2560, 1, bias=False, params_dtype=torch.bfloat16, disable_tp=True
    )
    block = SimpleNamespace(
        is_fused_shared_expert_enabled=False,
        shared_expert=nn.Module(),
        shared_expert_gate=gate,
    )
    md.wire_shared_gate(block, quant, "mtp.layers.40.mlp")
    assert block.shared_expert_gate is gate
    assert gate.weight.shape == (1, 1280)
    assert gate.weight.dtype == torch.uint8
    assert gate.weight_scale.dtype == torch.float32


def test_checkpoint_binding_rejects_changed_tensor_and_bad_expert_name():
    import mtp_checkpoint as checkpoint

    tensor = torch.tensor([[1, -2]], dtype=torch.int8)
    metadata = checkpoint.describe_tensor(tensor)
    checkpoint.validate_tensor(tensor.clone(), metadata)
    tensor[0, 0] = 2
    with pytest.raises(ValueError, match="binding mismatch"):
        checkpoint.validate_tensor(tensor, metadata)
    with pytest.raises(ValueError, match="checkpoint name"):
        checkpoint.expert_piece(
            "mtp.layers.0.mlp.experts.-1.gate_proj.weight", policy()
        )


def test_upstream_mapper_preserves_explicit_mixed_pools_and_bf16_exclusions():
    from vllm.model_executor.layers.fused_moe import RoutedExperts
    from vllm.model_executor.model_loader.utils import configure_quant_config
    from vllm.models.qwen4_exp.amd.model import Qwen4ExpForCausalLM

    plan = qm.PoolPlan((0,), (1,), 2)
    config = qm.QuarkMixedConfig(
        base_config(),
        {"model.language_model.layers.0.mlp.experts": plan},
        int8_wire=qm.INT8_WIRE,
    )
    configure_quant_config(config, Qwen4ExpForCausalLM)
    assert config.mixed_plans == {"model.layers.0.mlp.experts": plan}
    assert config.get_quant_method_target(
        "model.layers.0.mlp.experts", RoutedExperts
    ) == (
        None,
        None,
        qm.QuarkMixedMoEMethod,
    )
    assert (
        config.get_quant_method_target(
            "model.visual.blocks.0.attn.qkv", QKVParallelLinear
        )[2]
        is UnquantizedLinearMethod
    )
    assert (
        config.get_quant_method_target("model.layers.0.mlp.gate", QKVParallelLinear)[2]
        is UnquantizedLinearMethod
    )


@pytest.mark.parametrize("bits,group,n", [(4, 128, 1), (8, 64, 7)])
def test_dense_dispatch_preserves_stock_kernel_abi_with_padded_rows(bits, group, n):
    """Check launch arguments only; the recorder performs no GPU computation."""
    spec = dict(
        name="mtp.dense.weight",
        source_shape=[n, 128],
        weight_bits=bits,
        group_size=group,
        algorithm="RTN",
        activations="BF16",
        symmetric=True,
    )
    recorder = md.TestCallRecorder()
    method = md.MTPDenseMethod([spec], test_recorder=recorder)
    layer = nn.Module()
    method.create_weights(layer, 128, [n], 128, n, torch.bfloat16)
    layer.weight.weight_loader(layer.weight, torch.zeros_like(layer.weight))
    layer.weight_scale.weight_loader(
        layer.weight_scale, torch.ones_like(layer.weight_scale)
    )
    method.process_weights_after_loading(layer)
    result = method.apply(layer, torch.ones(17, 128, dtype=torch.bfloat16))
    assert result.shape == (17, n)
    call = recorder.calls[0]
    assert call["sorted_ids"] == list(range(32))
    assert call["experts"] == [0, 0]
    assert call["scales_dtype"] == torch.float32
    assert call["kwargs"]["MUL_ROUTED_WEIGHT"] is False
    assert call["kwargs"]["top_k"] == 1
    assert call["kwargs"]["BLOCK_SIZE_K"] == 64
    assert call["kwargs"]["use_int4_w4a16"] is (bits == 4)
    assert call["kwargs"]["use_int8_w8a16"] is (bits == 8)
    with pytest.raises(RuntimeError, match="activations must be finite"):
        method.apply(layer, torch.full((1, 128), float("nan"), dtype=torch.bfloat16))
    assert len(recorder.calls) == 1
