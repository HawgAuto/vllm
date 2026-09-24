# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contracts for bounded PLE I/O; no GPU or model qualification."""

import json
import os
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from vllm.config import LoadConfig
from vllm.config.compilation import CompilationMode, CUDAGraphMode
from vllm.model_executor.model_loader import weight_utils
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.models.qwen4_exp.amd.ple_layer import (
    Qwen4ExpPLESSDEmbedding,
    _ple_checkpoint_prefix,
    validate_qwen4_ple_ssd_config,
)
from vllm.models.qwen4_exp.common.ple_ssd import PLESafetensorsReader

PREFIX = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"


@pytest.fixture
def checkpoint(tmp_path):
    rows = torch.arange(24, dtype=torch.float32).reshape(8, 3).to(torch.bfloat16)
    ordinary = torch.tensor([[-128, 127], [3, -7]], dtype=torch.int8)
    tensors = [
        {f"{PREFIX}.shard_0.weight": rows[:4]},
        {f"{PREFIX}.shard_1.weight": rows[4:], "mtp.dense.weight": ordinary},
        {"model.norm.weight": torch.ones(3, dtype=torch.bfloat16)},
    ]
    files = []
    mapping = {}
    for i, values in enumerate(tensors):
        path = tmp_path / f"model-{i}.safetensors"
        save_file(values, str(path))
        files.append(str(path))
        mapping.update(dict.fromkeys(values, path.name))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": mapping})
    )
    return tmp_path, files, rows, ordinary


def reader_for(root, **kwargs):
    return PLESafetensorsReader(
        root,
        num_embeddings=8,
        embedding_dim=3,
        split_ngram_parts=2,
        tensor_prefix=PREFIX,
        **kwargs,
    )


def test_gather_reads_only_requested_rows_and_preserves_order(checkpoint, monkeypatch):
    root, _, rows, _ = checkpoint
    with reader_for(root, max_batch_rows=4) as reader:
        reads = []
        pread = os.pread

        def record(fd, count, offset):
            reads.append(count)
            return pread(fd, count, offset)

        monkeypatch.setattr(os, "pread", record)
        actual = torch.frombuffer(
            bytearray(reader.gather_bytes([7, 0, 7, 1])), dtype=torch.bfloat16
        ).reshape(4, 3)
        torch.testing.assert_close(actual, rows[[7, 0, 7, 1]], rtol=0, atol=0)
        assert sum(reads) == 3 * 3 * 2
        assert max(reads) <= 2 * 3 * 2
        assert reader.stats()["requested_rows"] == 4
        assert reader.stats()["unique_rows"] == 3
        assert reader.gather_bytes([]) == b""


@pytest.mark.parametrize(
    "ids,error",
    [
        ([-1], ValueError),
        ([8], ValueError),
        ([True], TypeError),
        ([1.0], TypeError),
        (range(5), ValueError),
    ],
)
def test_gather_rejects_invalid_or_unbounded_ids(checkpoint, ids, error):
    with reader_for(checkpoint[0], max_batch_rows=4) as reader:
        with pytest.raises(error):
            reader.gather_bytes(ids)
        assert reader.stats()["read_calls"] == 0


def test_embedding_chunks_and_closes_reader(checkpoint):
    reader = reader_for(checkpoint[0], max_batch_rows=2)
    embedding = Qwen4ExpPLESSDEmbedding(
        reader=reader,
        num_embeddings=8,
        embedding_dim=3,
        allow_mixed_shard_pread=True,
    )
    assert not list(embedding.parameters())
    ids = torch.tensor([[7, 0, 1], [2, 4, 5]])
    torch.testing.assert_close(embedding(ids), checkpoint[2][ids], rtol=0, atol=0)
    assert reader.stats()["requested_rows"] == 6
    with pytest.raises(ValueError, match="int32 or int64"):
        embedding(ids.float())
    embedding.close()
    with pytest.raises(ValueError, match="closed"):
        embedding(ids)


def test_draft_loader_never_maps_ple_even_without_ple_modules(checkpoint, monkeypatch):
    root, files, _, ordinary = checkpoint
    monkeypatch.setenv("VLLM_QWEN4_PLE_SSD", "1")
    loader = DefaultModelLoader(LoadConfig(safetensors_load_strategy="lazy"))
    monkeypatch.setattr(
        loader, "_prepare_weights", lambda *args: (str(root), files, True)
    )
    safe_open = weight_utils.safe_open
    opened = []

    def guarded_open(path, **kwargs):
        assert path == files[2], "PLE-containing file reached mmap loader"
        opened.append(path)
        return safe_open(path, **kwargs)

    monkeypatch.setattr(weight_utils, "safe_open", guarded_open)
    source = DefaultModelLoader.Source(str(root), revision=None, prefix="draft.")
    result = dict(loader._get_weights_iterator(source))
    assert set(result) == {"draft.mtp.dense.weight", "draft.model.norm.weight"}
    assert opened == [files[2]]
    torch.testing.assert_close(result["draft.mtp.dense.weight"], ordinary)


@pytest.mark.parametrize("strategy", ["eager", "prefetch"])
def test_ssd_loader_rejects_bulk_strategies_before_read(
    checkpoint, monkeypatch, strategy
):
    monkeypatch.setenv("VLLM_QWEN4_PLE_SSD", "1")
    loader = DefaultModelLoader(LoadConfig(safetensors_load_strategy=strategy))
    root, files, _, _ = checkpoint
    monkeypatch.setattr(
        loader, "_prepare_weights", lambda *args: (str(root), files, True)
    )
    with pytest.raises(ValueError, match="lazy safetensors"):
        loader._get_weights_iterator(
            DefaultModelLoader.Source(str(root), revision=None)
        )


def test_reader_rejects_index_mismatch_and_truncated_payload(checkpoint):
    root, files, _, _ = checkpoint
    index = root / "model.safetensors.index.json"
    data = json.loads(index.read_text())
    data["weight_map"][f"{PREFIX}.shard_0.weight"] = "model-1.safetensors"
    index.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="exactly match"):
        reader_for(root)
    index.unlink()
    with open(files[0], "r+b") as file:
        file.truncate(os.path.getsize(files[0]) - 2)
    with pytest.raises(ValueError, match="file size"):
        reader_for(root)


def test_reader_rejects_checkpoint_change_after_open(checkpoint):
    with reader_for(checkpoint[0]) as reader:
        with open(checkpoint[1][0], "r+b") as file:
            file.truncate(os.path.getsize(checkpoint[1][0]) - 2)
        with pytest.raises(OSError, match="identity changed"):
            reader.gather_bytes([0])


def test_ssd_allows_native_multistep_mtp_with_explicit_graph_boundary(monkeypatch):
    monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", "0")
    config = SimpleNamespace(
        model_config=SimpleNamespace(dtype=torch.bfloat16),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1, pipeline_parallel_size=1
        ),
        speculative_config=SimpleNamespace(num_speculative_tokens=3),
        load_config=LoadConfig(safetensors_load_strategy="lazy"),
        compilation_config=SimpleNamespace(
            mode=CompilationMode.VLLM_COMPILE,
            cudagraph_mode=CUDAGraphMode.PIECEWISE,
            splitting_ops=["vllm::qwen4_exp_amd_ple_ssd_embedding"],
        ),
    )
    validate_qwen4_ple_ssd_config(config)
    config.compilation_config.splitting_ops = []
    with pytest.raises(ValueError, match="boundary|splitting op"):
        validate_qwen4_ple_ssd_config(config)
    config.compilation_config.cudagraph_mode = CUDAGraphMode.FULL
    with pytest.raises(ValueError, match="cannot be captured"):
        validate_qwen4_ple_ssd_config(config)


@pytest.mark.parametrize(
    "runtime",
    [
        "model.layers.1.ple.ple_embedding.ngram_embedding",
        "language_model.model.layers.1.ple.ple_embedding.ngram_embedding",
        PREFIX,
    ],
)
def test_checkpoint_prefix_matches_native_wrappers(runtime):
    assert _ple_checkpoint_prefix(runtime) == PREFIX


def test_draft_loader_rejects_malformed_ple_before_mmap(checkpoint, monkeypatch):
    from vllm.models.qwen4_exp.common.ple_ssd import ple_ssd_weights_iterator

    _, files, rows, _ = checkpoint
    save_file({f"{PREFIX}.shard_bad.weight": rows[:4]}, files[0])

    def forbidden(*args, **kwargs):
        pytest.fail("malformed PLE must not reach mmap")

    monkeypatch.setattr(weight_utils, "safe_open", forbidden)
    with pytest.raises(ValueError, match="unsupported SSD PLE tensor"):
        list(ple_ssd_weights_iterator([files[0]], False))


def test_mixed_reader_never_stages_another_layers_ple(tmp_path, monkeypatch):
    """A shared physical shard may contain several large PLE tables."""
    from vllm.models.qwen4_exp.common.ple_ssd import ple_ssd_weights_iterator

    rows = torch.arange(24).reshape(8, 3).to(torch.bfloat16)
    other_prefix = PREFIX.replace("layers.1.", "layers.2.")
    tensors = {
        f"{prefix}.shard_{part}.weight": rows[part * 4 : (part + 1) * 4].clone()
        for prefix in (PREFIX, other_prefix)
        for part in range(2)
    }
    tensors["model.norm.weight"] = torch.ones(3, dtype=torch.bfloat16)
    path = tmp_path / "model.safetensors"
    save_file(tensors, str(path))

    def forbidden(*args, **kwargs):
        pytest.fail("mixed PLE file must not reach mmap")

    monkeypatch.setattr(weight_utils, "safe_open", forbidden)
    with reader_for(tmp_path) as reader:
        actual = dict(reader.iter_mixed_shard_weights())
    assert set(actual) == {"model.norm.weight"}
    torch.testing.assert_close(
        actual["model.norm.weight"], tensors["model.norm.weight"]
    )
    assert set(dict(ple_ssd_weights_iterator([str(path)], False))) == set(actual)


def test_ssd_custom_op_refreshes_static_output_and_skips_profile_io(checkpoint):
    """Exercise the eager op contract; this does not simulate graph capture."""
    from vllm.forward_context import ForwardContext, override_forward_context
    from vllm.models.qwen4_exp.amd.ple_layer import qwen4_exp_amd_ple_ssd_embedding

    embedding = Qwen4ExpPLESSDEmbedding(
        reader=reader_for(checkpoint[0]),
        num_embeddings=8,
        embedding_dim=3,
        allow_mixed_shard_pread=True,
    )
    owner = SimpleNamespace(ple_embedding=SimpleNamespace(ngram_embedding=embedding))
    context = ForwardContext({"ple": owner}, {}, {})
    output = torch.empty(1, 6, dtype=torch.bfloat16)
    address = output.data_ptr()
    with override_forward_context(context):
        for ids in (torch.tensor([[0, 1]]), torch.tensor([[7, 4]])):
            qwen4_exp_amd_ple_ssd_embedding(ids, output, "ple")
            torch.testing.assert_close(output, checkpoint[2][ids].flatten(-2))
            assert output.data_ptr() == address
        reads = embedding.stats()["read_calls"]
        context.attn_metadata = None
        qwen4_exp_amd_ple_ssd_embedding(ids, output, "ple")
        assert torch.count_nonzero(output) == 0
        assert embedding.stats()["read_calls"] == reads
    embedding.close()


def test_ssd_graph_capture_defers_uninitialized_ids_without_weakening_replay(checkpoint, monkeypatch):
    """Graph construction cannot dereference an unexecuted producer's IDs."""
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture
    from vllm.forward_context import ForwardContext, override_forward_context
    from vllm.models.qwen4_exp.amd.ple_layer import _ssd_ple_lookup_or_profile_zeros

    embedding = Qwen4ExpPLESSDEmbedding(
        reader=reader_for(checkpoint[0]),
        num_embeddings=8,
        embedding_dim=3,
        allow_mixed_shard_pread=True,
    )
    ids = torch.tensor([[9223231297218904063]])
    context = ForwardContext({}, {}, {})
    context.attn_metadata = {}
    monkeypatch.setattr(BreakableCUDAGraphCapture, "is_active", staticmethod(lambda: True))
    with override_forward_context(context):
        reads = embedding.stats()["read_calls"]
        output = _ssd_ple_lookup_or_profile_zeros(embedding, ids)
        assert output.shape == (1, 1, 3)
        assert torch.count_nonzero(output) == 0
        assert embedding.stats()["read_calls"] == reads
        monkeypatch.setattr(BreakableCUDAGraphCapture, "is_active", staticmethod(lambda: False))
        with pytest.raises(ValueError, match="row ID out of range"):
            _ssd_ple_lookup_or_profile_zeros(embedding, ids)
    embedding.close()
