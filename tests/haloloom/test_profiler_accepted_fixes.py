# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
from pathlib import Path


ROOT = Path(__file__).parents[2]


def _method(tree: ast.Module, class_name: str, method_name: str) -> ast.FunctionDef:
    owner = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    return next(
        node
        for node in owner.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )


def test_torch_profiler_keeps_explicit_stack_disable_independent():
    tree = ast.parse((ROOT / "vllm/profiler/wrapper.py").read_text())
    init = _method(tree, "TorchProfilerWrapper", "__init__")
    kwargs_call = next(
        node
        for node in ast.walk(init)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "dict"
        and any(keyword.arg == "with_stack" for keyword in node.keywords)
    )
    value = next(
        keyword.value for keyword in kwargs_call.keywords if keyword.arg == "with_stack"
    )
    assert ast.unparse(value) == "profiler_config.torch_profiler_with_stack"


def test_torch_profiler_releases_state_even_when_stop_raises():
    tree = ast.parse((ROOT / "vllm/profiler/wrapper.py").read_text())
    stop = _method(tree, "TorchProfilerWrapper", "_call_stop")
    try_node = next(node for node in stop.body if isinstance(node, ast.Try))
    assert try_node.finalbody
    final_source = "\n".join(ast.unparse(node) for node in try_node.finalbody)
    assert "del self.profiler" in final_source
    assert "gc.collect()" in final_source
