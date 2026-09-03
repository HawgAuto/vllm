#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build the archived W4A4 source family for ROCm 10/gfx1151.

This is a CPU/build-only operation. It never opens a GPU device and it does not
carry forward historical shared objects or historical promotion authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Mapping

HOST_FLAGS = ["-O2", "-std=c++17"]
HIP_FLAGS = [
    "-O2",
    "-std=c++17",
    "--offload-arch=gfx1151",
    "-Rpass-analysis=kernel-resource-usage",
]
MODULES: dict[str, dict[str, str]] = {
    "perf": {
        "module": "gfx1151_w4a4_perf_ext_rocm10",
        "source": "gfx1151_w4a4_perf_op.hip",
        "source_sha256": "95e787ec86fb2d0d787844517b08520966d711bbf7a5d33429fe499bffe849e7",
    },
    "m128_v3": {
        "module": "gfx1151_w4a4_m128_v3_ext_rocm10",
        "source": "gfx1151_w4a4_m128_v3_op.hip",
        "source_sha256": "da90bff3782fd46b43138f1a9b793e3090da67cf32d13c3f6a440fe8a125d0e0",
    },
    "m128_v4q": {
        "module": "gfx1151_w4a4_m128_v4q_ext_rocm10",
        "source": "gfx1151_w4a4_m128_qfloat_op.hip",
        "source_sha256": "994af1d33ad9c8abe54c8475328b4f9c5f6de04790219cc3ce98ecb446e33abe",
    },
    "m1_pair2_signed": {
        "module": "gfx1151_w4a4_m1_dot_pair2_signed_ext_rocm10",
        "source": "gfx1151_w4a4_m1_dot_pair2_signed_op.hip",
        "source_sha256": "3f01ca4e79cd86601247faa58a9dfdd01c1150acbe99ced5c1d48b306b8c802c",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_environment(
    env: Mapping[str, str], *, torch_hip: str | None, rocm_home: str | None
) -> None:
    if env.get("HSA_OVERRIDE_GFX_VERSION"):
        raise RuntimeError("HSA_OVERRIDE_GFX_VERSION must be unset")
    if not torch_hip:
        raise RuntimeError("a HIP-enabled Torch build is required")
    if not torch_hip.startswith("7.15."):
        raise RuntimeError(f"ROCm 10 Torch HIP 7.15 is required, got {torch_hip}")
    if Path(rocm_home or "").as_posix() != "/opt/rocm":
        raise RuntimeError(f"ROCm 10 toolchain root must be /opt/rocm, got {rocm_home}")


def _fresh_directory(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise RuntimeError(f"refusing non-empty output directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def build(source_root: Path, build_root: Path, asset_root: Path) -> dict[str, Any]:
    import torch
    from torch.utils.cpp_extension import ROCM_HOME, load

    validate_environment(
        os.environ, torch_hip=torch.version.hip, rocm_home=str(ROCM_HOME)
    )
    source_root = source_root.resolve()
    build_root = build_root.resolve()
    asset_root = asset_root.resolve()
    _fresh_directory(build_root)
    _fresh_directory(asset_root)
    os.environ["MAX_JOBS"] = "1"
    os.environ["PYTORCH_ROCM_ARCH"] = "gfx1151"

    built_modules: dict[str, dict[str, Any]] = {}
    for key, spec in MODULES.items():
        source = source_root / spec["source"]
        if not source.is_file() or sha256(source) != spec["source_sha256"]:
            raise RuntimeError(f"archived source identity drift: {key}: {source}")
        module_build = build_root / key
        module_build.mkdir()
        staged = module_build / f"{source.stem}.cu"
        shutil.copyfile(source, staged)
        result = Path(
            load(
                name=spec["module"],
                sources=[str(staged)],
                extra_cflags=HOST_FLAGS,
                extra_cuda_cflags=HIP_FLAGS,
                build_directory=str(module_build),
                with_cuda=True,
                is_python_module=False,
                verbose=True,
            )
        ).resolve()
        if not result.is_file():
            raise RuntimeError(f"extension output missing: {key}: {result}")
        source_out = asset_root / spec["source"]
        library_out = asset_root / f"{spec['module']}.so"
        shutil.copyfile(source, source_out)
        shutil.copyfile(result, library_out)
        source_out.chmod(0o444)
        library_out.chmod(0o444)
        built_modules[key] = {
            "module": spec["module"],
            "source": source_out.name,
            "source_bytes": source_out.stat().st_size,
            "source_sha256": sha256(source_out),
            "library": library_out.name,
            "library_bytes": library_out.stat().st_size,
            "library_sha256": sha256(library_out),
        }

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "CPU_BUILT_GPU_UNQUALIFIED",
        "target": {"rocm": "10.0", "torch_hip": torch.version.hip, "arch": "gfx1151"},
        "torch": torch.__version__,
        "rocm_home": str(ROCM_HOME),
        "host_flags": HOST_FLAGS,
        "hip_flags": HIP_FLAGS,
        "modules": built_modules,
        "historical_binaries_reused": False,
        "historical_promotion_reused": False,
        "gpu_qualified": False,
        "quality_qualified": False,
        "performance_qualified": False,
        "production_ready": False,
        "promotion_authority": False,
    }
    manifest_path = asset_root / "port-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    manifest_path.chmod(0o444)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--build-root", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    args = parser.parse_args()
    manifest = build(args.source_root, args.build_root, args.asset_root)
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
