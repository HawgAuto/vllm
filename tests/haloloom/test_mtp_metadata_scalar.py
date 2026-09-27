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


def test_compacted_spec_decode_token_count_is_python_int():
    tree = ast.parse(
        (ROOT / "vllm/v1/spec_decode/llm_base_proposer.py").read_text()
    )
    prepare = _method(tree, "SpecDecodeBaseProposer", "prepare_inputs")
    assignment = next(
        node
        for node in ast.walk(prepare)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "total_num_tokens"
            for target in node.targets
        )
    )
    assert ast.unparse(assignment.value) == "int(new_query_start_loc_np[-1])"
