# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from vllm.model_executor.layers.quantization.gfx1151_w4a4_archive import (
    Gfx1151W4A4Runtime,
)

ASSETS = Path(
    "/opt/venv/lib/python3.14/site-packages/vllm/model_executor/"
    "layers/quantization/gfx1151_archive_assets"
)


def _copy_assets(tmp_path: Path) -> Path:
    target = tmp_path / "assets"
    shutil.copytree(ASSETS, target)
    for member in target.iterdir():
        member.chmod(0o444)
    return target / "port-manifest.json"


def _rewrite(path: Path, payload: dict) -> None:
    path.chmod(0o644)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")
    path.chmod(0o444)


def test_real_port_manifest_passes():
    row = Gfx1151W4A4Runtime._load_port_manifest(ASSETS / "port-manifest.json")
    assert row["status"] == "CPU_BUILT_GPU_UNQUALIFIED"
    assert set(row["modules"]) == {
        "perf",
        "m128_v3",
        "m128_v4q",
        "m1_pair2_signed",
    }


@pytest.mark.parametrize(
    "claim",
    [
        "historical_binaries_reused",
        "historical_promotion_reused",
        "gpu_qualified",
        "quality_qualified",
        "performance_qualified",
        "production_ready",
        "promotion_authority",
    ],
)
def test_true_claim_fails_closed(tmp_path: Path, claim: str):
    path = _copy_assets(tmp_path)
    row = json.loads(path.read_text())
    row[claim] = True
    _rewrite(path, row)
    with pytest.raises(RuntimeError, match="claim must remain false"):
        Gfx1151W4A4Runtime._load_port_manifest(path)


def test_unknown_top_level_field_fails_closed(tmp_path: Path):
    path = _copy_assets(tmp_path)
    row = json.loads(path.read_text())
    row["unexpected"] = 1
    _rewrite(path, row)
    with pytest.raises(RuntimeError, match="shape mismatch"):
        Gfx1151W4A4Runtime._load_port_manifest(path)


def test_duplicate_json_key_fails_closed(tmp_path: Path):
    path = _copy_assets(tmp_path)
    text = path.read_text().replace(
        '"schema_version": 1,', '"schema_version": 1, "schema_version": 1,'
    )
    path.chmod(0o644)
    path.write_text(text)
    path.chmod(0o444)
    with pytest.raises(RuntimeError, match="duplicate"):
        Gfx1151W4A4Runtime._load_port_manifest(path)


def test_mutable_member_fails_closed(tmp_path: Path):
    path = _copy_assets(tmp_path)
    row = json.loads(path.read_text())
    member = path.parent / row["modules"]["perf"]["library"]
    member.chmod(0o644)
    with pytest.raises(RuntimeError, match="identity drift"):
        Gfx1151W4A4Runtime._load_port_manifest(path)


def test_symlink_member_fails_closed(tmp_path: Path):
    path = _copy_assets(tmp_path)
    row = json.loads(path.read_text())
    member = path.parent / row["modules"]["perf"]["library"]
    replacement = path.parent / "real-perf.so"
    member.rename(replacement)
    os.symlink(replacement.name, member)
    with pytest.raises(RuntimeError, match="symlink"):
        Gfx1151W4A4Runtime._load_port_manifest(path)


def test_absolute_and_escape_member_paths_fail_closed(tmp_path: Path):
    path = _copy_assets(tmp_path)
    row = json.loads(path.read_text())
    row["modules"]["perf"]["library"] = str(
        (path.parent / row["modules"]["perf"]["library"]).resolve()
    )
    _rewrite(path, row)
    with pytest.raises(RuntimeError, match="relative"):
        Gfx1151W4A4Runtime._load_port_manifest(path)

    path = _copy_assets(tmp_path / "second")
    row = json.loads(path.read_text())
    row["modules"]["perf"]["library"] = "../escape.so"
    _rewrite(path, row)
    with pytest.raises(RuntimeError, match="escapes"):
        Gfx1151W4A4Runtime._load_port_manifest(path)


def test_module_and_source_identity_drift_fail_closed(tmp_path: Path):
    path = _copy_assets(tmp_path)
    row = json.loads(path.read_text())
    del row["modules"]["m128_v4q"]
    _rewrite(path, row)
    with pytest.raises(RuntimeError, match="module closure"):
        Gfx1151W4A4Runtime._load_port_manifest(path)

    path = _copy_assets(tmp_path / "second")
    row = json.loads(path.read_text())
    row["modules"]["m128_v4q"]["source_sha256"] = "0" * 64
    _rewrite(path, row)
    with pytest.raises(RuntimeError, match="archived source identity"):
        Gfx1151W4A4Runtime._load_port_manifest(path)


@pytest.mark.parametrize("field", ["module", "source", "library"])
def test_canonical_module_artifact_identity_drift_fails_closed(
    tmp_path: Path, field: str
):
    path = _copy_assets(tmp_path)
    row = json.loads(path.read_text())
    row["modules"]["perf"][field] = "renamed-artifact"
    _rewrite(path, row)
    with pytest.raises(RuntimeError, match="artifact identity"):
        Gfx1151W4A4Runtime._load_port_manifest(path)


def test_validated_library_inode_remains_bound_after_path_replacement(tmp_path: Path):
    path = _copy_assets(tmp_path)
    manifest = Gfx1151W4A4Runtime._load_port_manifest(
        path, bind_libraries=True
    )
    try:
        row = manifest["modules"]["perf"]
        library = path.parent / row["library"]
        replacement = path.parent / "replacement.so"
        replacement.write_bytes(b"replacement")
        replacement.chmod(0o444)
        library.unlink()
        replacement.rename(library)
        fd_path = Path(f"/proc/self/fd/{row['_library_fd']}")
        assert fd_path.read_bytes() != library.read_bytes()
        assert __import__("hashlib").sha256(fd_path.read_bytes()).hexdigest() == (
            row["library_sha256"]
        )
    finally:
        Gfx1151W4A4Runtime._close_bound_libraries(manifest)
