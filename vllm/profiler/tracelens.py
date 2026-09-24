# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Execution-derived TraceLens metadata, without replay-input fabrication."""

import contextvars
import hashlib
import json
import math
import os
import struct
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import torch
from torch._C._profiler import _EventType, _TensorMetadata

if TYPE_CHECKING:
    from vllm.config import ProfilerConfig

SCHEMA = "vllm.tracelens.execution.v1"
_binding: contextvars.ContextVar[
    tuple["ProfilerConfig", Sequence[torch.profiler.ProfilerActivity]] | None
] = contextvars.ContextVar("tracelens_capture", default=None)
_subsystem: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "tracelens_subsystem", default=None
)


def _input_metadata(value):
    if isinstance(value, _TensorMetadata):
        return {
            "kind": "tensor",
            "shape": list(value.sizes),
            "stride": list(value.strides),
            "dtype": str(value.dtype),
            "layout": str(value.layout),
            "device": str(value.device),
            "values_captured": False,
        }
    if isinstance(value, (list, tuple)):
        return {"kind": "list", "items": [_input_metadata(v) for v in value]}
    # None may represent an argument the profiler could not encode.
    if value is None:
        return {"kind": "unavailable"}
    if isinstance(value, float) and not math.isfinite(value):
        return {
            "kind": "scalar",
            "value": {
                "encoding": "ieee754-binary64",
                "bits": struct.pack(">d", value).hex(),
            },
        }
    if isinstance(value, (bool, int, float, str)):
        return {"kind": "scalar", "value": value}
    return {"kind": "unavailable"}


def execution_metadata(profiler) -> dict:
    """Read observed operator arguments and Python/module ancestry from Kineto."""
    rows: list[dict[str, Any]] = []
    python_calls: list[dict[str, Any]] = []
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "trace_id": profiler.get_trace_id(),
        "operators": rows,
        "python_calls": python_calls,
        "errors": [],
    }
    try:
        roots = profiler.profiler.kineto_results.experimental_event_tree()
        pending = [(event, [], []) for event in reversed(roots)]
        while pending:
            event, frames, modules = pending.pop()
            tag, fields = event.typed
            if tag == _EventType.PyCall:
                frame = fields.callsite
                callsite = {
                    "file": frame.file_name,
                    "line": frame.line_number,
                    "function": frame.function_name,
                }
                python_calls.append(
                    {
                        "event_id": event.id,
                        "parent_id": event.parent.id if event.parent else None,
                        "thread_id": event.start_tid,
                        "start_time_ns": event.start_time_ns,
                        "end_time_ns": event.end_time_ns,
                        "callsite": callsite,
                        "attribution_method": "observed_python_call",
                    }
                )
                frames = frames + [callsite]
                if fields.module is not None:
                    modules = modules + [fields.module.cls_name]
            if tag == _EventType.TorchOp:
                rows.append(
                    {
                        "event_id": event.id,
                        "correlation_id": event.correlation_id,
                        "parent_id": event.parent.id if event.parent else None,
                        "thread_id": event.start_tid,
                        "operator": event.name,
                        "inputs": [_input_metadata(v) for v in fields.inputs],
                        "python_stack": frames,
                        "module_stack": modules,
                        "attribution_method": "kineto_execution_tree",
                        "input_provenance": "observed_operator_arguments",
                        "exact_replay_admitted": False,
                        "replay_rejection": (
                            "kernel_launcher_arguments_and_values_missing"
                        ),
                    }
                )
            pending.extend(
                (child, frames, modules) for child in reversed(event.children)
            )
    except Exception as exc:
        result["errors"].append(f"{type(exc).__name__}: {exc}")
    result["operator_metadata_observed"] = bool(rows) and not result["errors"]
    result["all_kernel_inputs_observed"] = False
    # Observed operator inputs are not necessarily a fused kernel's inputs.
    result["exact_kernel_replay_ready"] = False
    return result


def _write_json(path: Path, value) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, allow_nan=False), encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def export_execution_metadata(profiler, directory: str, name: str) -> Path:
    """Write a sidecar independently of the unmodified raw profiler trace."""
    path = Path(directory) / "execution_metadata" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(path, execution_metadata(profiler))
    return path


@contextmanager
def capture_session(config, activities) -> Iterator[None]:
    """Bind capture settings; warmup remains outside each profiling window."""
    token = _binding.set((config, activities))
    try:
        yield
    finally:
        _binding.reset(token)


@contextmanager
def capture_subsystem(role: str, model) -> Iterator[None]:
    """Label the actual runner call site, never a role inferred from names."""
    if _binding.get() is None:
        yield
        return
    model_type = type(model)
    token = _subsystem.set(
        {
            "role": role,
            "model_class": (
                f"{model_type.__module__}.{model_type.__qualname__}"
                if model is not None
                else None
            ),
            "role_provenance": "runner_capture_callsite",
        }
    )
    try:
        yield
    finally:
        _subsystem.reset(token)


@contextmanager
def graph_capture_step(
    num_tokens: int, mode: str, *, descriptor=None
) -> Iterator[None]:
    """Record one real capture with TraceLens's existing capture annotation."""
    binding = _binding.get()
    if binding is None:
        yield
        return
    config, activities = binding
    capture_descriptor = asdict(descriptor) if descriptor is not None else {}
    capture_descriptor.update(num_tokens=num_tokens, cg_mode=mode)
    subsystem = _subsystem.get()
    role = subsystem["role"] if subsystem else "unknown"
    label_mode = mode if role == "decoder" else f"{role}_{mode}"
    label = f"capture_{num_tokens}_{label_mode}"
    directory = Path(config.torch_profiler_dir) / "capture_traces"
    directory.mkdir(parents=True, exist_ok=True)
    name = f"graph_capture_rank_0.{os.getpid()}.{uuid4().hex}"
    trace = directory / f"{name}.pt.trace.json.gz"
    complete = False
    profiler = torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        with_stack=True,
        with_modules=True,
        profile_memory=config.torch_profiler_with_memory,
    )
    try:
        with profiler:
            profiler.add_metadata_json("vllm_tracelens_schema", json.dumps(SCHEMA))
            profiler.add_metadata_json(
                "vllm_capture_descriptor", json.dumps(capture_descriptor)
            )
            profiler.add_metadata_json("vllm_capture_subsystem", json.dumps(subsystem))
            with torch.profiler.record_function(label):
                yield
        complete = True
    finally:
        profiler.export_chrome_trace(str(trace))
        metadata = execution_metadata(profiler)
        with trace.open("rb") as trace_file:
            trace_digest = hashlib.file_digest(trace_file, "sha256").hexdigest()
        metadata.update(
            {
                "raw_trace": trace.name,
                "raw_trace_sha256": trace_digest,
                "capture_complete": complete,
                "capture": capture_descriptor,
                "subsystem": subsystem,
                "timing_scope": "graph_capture_not_serving_phase",
            }
        )
        sidecar_dir = directory / "execution_metadata"
        sidecar_dir.mkdir(exist_ok=True)
        _write_json(sidecar_dir / f"{name}.json", metadata)
        if complete:
            # Rank zero is the only writer. Keep sidecars in a subdirectory so
            # TraceLens does not mistake them for Chrome trace files.
            manifest = directory / "execution_details.json"
            entries = json.loads(manifest.read_text()) if manifest.exists() else []
            entry = {
                "file": trace.name,
                "batch_size": num_tokens,
                "mode": label_mode,
                "capture_mode": label_mode,
                "descriptor": capture_descriptor,
                "subsystem": subsystem,
                "exact_replay_match": False,
            }
            # TraceLens indexes only (batch_size, mode). Do not let different
            # request/LoRA/ubatch descriptors silently overwrite one another.
            duplicates = [
                old
                for old in entries
                if old["batch_size"] == num_tokens
                and old.get("capture_mode", old["mode"]) == label_mode
            ]
            if duplicates:
                for duplicate in [*duplicates, entry]:
                    duplicate["mode"] = f"ambiguous_{label_mode}"
            entries.append(entry)
            _write_json(manifest, entries)
