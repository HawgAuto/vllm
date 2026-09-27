# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
from pathlib import Path


ROOT = Path(__file__).parents[2]


def test_excluded_quark_embedding_uses_unquantized_method():
    tree = ast.parse(
        (
            ROOT
            / "vllm/model_executor/layers/quantization/quark/quark.py"
        ).read_text()
    )
    owner = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "QuarkConfig"
    )
    target = next(
        node
        for node in owner.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "get_quant_method_target"
    )
    source = ast.unparse(target)
    assert "issubclass(layer_type, VocabParallelEmbedding)" in source
    assert "return (None, None, UnquantizedEmbeddingMethod)" in source
