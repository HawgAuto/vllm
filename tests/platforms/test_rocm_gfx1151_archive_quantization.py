# SPDX-License-Identifier: Apache-2.0

import ast
from pathlib import Path


EXPECTED = [
    "gfx1151-w4a4-incumbent",
    "gfx1151-w4a4-m1-pair2",
    "gfx1151-w4a4-v4q-m1-pair2",
]


def test_registered_archive_w4_methods_are_rocm_admissible_in_stable_order():
    source = Path(__file__).resolve().parents[2] / "vllm/platforms/rocm.py"
    tree = ast.parse(source.read_text())
    values = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "RocmPlatform":
            for statement in node.body:
                if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
                    if statement.target.id == "supported_quantization" and statement.value is not None:
                        values = ast.literal_eval(statement.value)
    assert values is not None
    assert len(values) == len(set(values))
    assert values[-3:] == EXPECTED


if __name__ == "__main__":
    test_registered_archive_w4_methods_are_rocm_admissible_in_stable_order()
    print("PASS")
