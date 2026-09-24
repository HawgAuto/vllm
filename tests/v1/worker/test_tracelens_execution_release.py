# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from unittest.mock import Mock

import pytest
import torch

from vllm.config import (
    ProfilerConfig,
)
from vllm.profiler.wrapper import (
    TorchProfilerWrapper,
)
from vllm.v1.worker.gpu_worker import Worker


def test_tracelens_capture_records_observed_geometry_and_manifest(tmp_path):
    """V2 capture descriptors and real tensor metadata need no name inference."""
    import gzip
    import json

    from vllm.profiler.tracelens import capture_subsystem, graph_capture_step
    from vllm.profiler.wrapper import create_graph_capture_profiler

    config = ProfilerConfig(
        profiler="torch",
        torch_profiler_dir=str(tmp_path),
        torch_profiler_activities=["CPU"],
        capture_torch_profiler=True,
    )
    wrapper = create_graph_capture_profiler(config, global_rank=0)
    model = torch.nn.Linear(3, 2, bias=False)
    value = torch.arange(12, dtype=torch.float32).reshape(3, 4).t()
    expected = model(value)
    with (
        wrapper.capture_cuda_graphs(),
        capture_subsystem("decoder", model),
        graph_capture_step(4, "FULL"),
    ):
        actual = model(value)
    torch.testing.assert_close(actual, expected)
    directory = tmp_path / "capture_traces"
    manifest = json.loads((directory / "execution_details.json").read_text())
    assert len(manifest) == 1
    assert manifest[0]["batch_size"] == 4
    assert manifest[0]["mode"] == "FULL"
    with gzip.open(directory / manifest[0]["file"], "rt") as stream:
        trace = json.load(stream)
    assert any(e["name"] == "capture_4_FULL" for e in trace["traceEvents"])
    sidecar = json.loads(
        next((directory / "execution_metadata").glob("*.json")).read_text()
    )
    assert sidecar["capture_complete"]
    assert not sidecar["errors"]
    assert not sidecar["exact_kernel_replay_ready"]
    mm = next(row for row in sidecar["operators"] if row["operator"] == "aten::mm")
    assert mm["inputs"][0]["shape"] == [4, 3]
    assert mm["inputs"][0]["stride"] == [1, 4]
    assert mm["inputs"][0]["dtype"] == "torch.float32"
    assert mm["inputs"][0]["layout"] == "torch.strided"
    assert not mm["inputs"][0]["values_captured"]
    assert "Linear" in mm["module_stack"]
    assert mm["python_stack"]
    assert not mm["exact_replay_admitted"]


def test_tracelens_capture_default_off_and_context_restored(tmp_path):
    from vllm.profiler.tracelens import graph_capture_step
    from vllm.profiler.wrapper import create_graph_capture_profiler

    config = ProfilerConfig(profiler="torch", torch_profiler_dir=str(tmp_path))
    assert create_graph_capture_profiler(config, global_rank=0) is None
    with graph_capture_step(4, "FULL"):
        torch.ones(1).add_(1)
    assert not list(tmp_path.iterdir())


def test_tracelens_failed_capture_is_not_classified_as_ready(tmp_path):
    import json

    from vllm.profiler.tracelens import graph_capture_step
    from vllm.profiler.wrapper import create_graph_capture_profiler

    config = ProfilerConfig(
        profiler="torch",
        torch_profiler_dir=str(tmp_path),
        torch_profiler_activities=["CPU"],
        capture_torch_profiler=True,
    )
    wrapper = create_graph_capture_profiler(config, global_rank=0)
    with (
        pytest.raises(RuntimeError, match="capture failed"),
        wrapper.capture_cuda_graphs(),
        graph_capture_step(1, "FULL"),
    ):
        torch.ones(1).add_(1)
        raise RuntimeError("capture failed")
    directory = tmp_path / "capture_traces"
    assert not (directory / "execution_details.json").exists()
    sidecar = json.loads(
        next((directory / "execution_metadata").glob("*.json")).read_text()
    )
    assert not sidecar["capture_complete"]
    with graph_capture_step(1, "FULL"):
        torch.ones(1)
    assert len(list(directory.glob("*.gz"))) == 1


def test_tracelens_steady_profile_preserves_trace_and_observed_sidecar(tmp_path):
    import gzip
    import json

    config = ProfilerConfig(
        profiler="torch",
        torch_profiler_dir=str(tmp_path),
        torch_profiler_dump_cuda_time_total=False,
        detailed_trace_annotation=True,
        torch_profiler_with_stack=False,
        torch_profiler_record_shapes=False,
    )
    wrapper = TorchProfilerWrapper(
        config, worker_name="observed", local_rank=1, activities=["CPU"]
    )
    wrapper.start()
    with wrapper.annotate_context_manager(
        "execute_2_context_1(sq2sk2sqsq4sqsk4)_generation_0(sq0sk0sqsq0sqsk0)"
    ):
        torch.ones(2, 3).sin()
    wrapper.stop()
    with gzip.open(next(tmp_path.glob("*.pt.trace.json.gz")), "rt") as stream:
        trace = json.load(stream)
    sidecar = json.loads(
        next((tmp_path / "execution_metadata").glob("*.json")).read_text()
    )
    assert sidecar["trace_id"] == trace["trace_id"]
    assert sidecar["operator_metadata_observed"]
    assert not sidecar["exact_kernel_replay_ready"]


def test_tracelens_does_not_merge_ambiguous_capture_descriptors(tmp_path):
    import json
    from dataclasses import dataclass

    from vllm.profiler.tracelens import capture_subsystem, graph_capture_step
    from vllm.profiler.wrapper import create_graph_capture_profiler

    @dataclass
    class Descriptor:
        num_reqs: int

    wrapper = create_graph_capture_profiler(
        ProfilerConfig(
            profiler="torch",
            torch_profiler_dir=str(tmp_path),
            torch_profiler_activities=["CPU"],
            capture_torch_profiler=True,
        ),
        global_rank=0,
    )
    with (
        wrapper.capture_cuda_graphs(),
        capture_subsystem("decoder", torch.nn.Identity()),
    ):
        for requests in [1, 2]:
            with graph_capture_step(4, "FULL", descriptor=Descriptor(requests)):
                torch.ones(4).sin()
    entries = json.loads(
        (tmp_path / "capture_traces/execution_details.json").read_text()
    )
    assert [row["descriptor"]["num_reqs"] for row in entries] == [1, 2]
    assert all(row["mode"] == "ambiguous_FULL" for row in entries)
    assert all(not row["exact_replay_match"] for row in entries)


@pytest.mark.parametrize("rank", [0, 1])
def test_tracelens_capture_does_not_replace_serving_profiler(tmp_path, rank):
    from vllm.profiler.tracelens import graph_capture_step

    worker = object.__new__(Worker)
    worker.rank = rank
    worker.profiler = None
    worker.use_v2_model_runner = True
    worker.profiler_config = ProfilerConfig(
        profiler="torch",
        torch_profiler_dir=str(tmp_path),
        torch_profiler_activities=["CPU"],
        capture_torch_profiler=True,
    )
    worker.model_runner = Mock()
    worker.model_runner.needs_cudagraph_capture.return_value = True
    with worker._get_cudagraph_capture_context(), graph_capture_step(1, "FULL"):
        torch.ones(1).sin()
    assert worker.profiler is None
    assert (tmp_path / "capture_traces").exists() == (rank == 0)
