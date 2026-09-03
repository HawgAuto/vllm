# SPDX-License-Identifier: Apache-2.0
"""Experimental gfx1151 symmetric W4 x dynamic S4/G32 routes."""

from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import os
import stat
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import torch
from torch import nn

from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.utils import set_weight_attrs
from vllm.model_executor.parameter import GroupQuantScaleParameter, ModelWeightParameter

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization import QuantizationMethods
else:
    QuantizationMethods = str

GFX1151_W4A4_INCUMBENT = "gfx1151-w4a4-incumbent"
GFX1151_W4A4_M1_PAIR2 = "gfx1151-w4a4-m1-pair2"
GFX1151_W4A4_V4Q_M1_PAIR2 = "gfx1151-w4a4-v4q-m1-pair2"
_FORCED_METHODS = {
    GFX1151_W4A4_INCUMBENT,
    GFX1151_W4A4_M1_PAIR2,
    GFX1151_W4A4_V4Q_M1_PAIR2,
}
_ASSET_ROOT = Path(__file__).with_name("gfx1151_archive_assets")
promotion_authority = False
_PACKAGED_ASSETS = {
    "VLLM_GFX1151_W4A4_PORT_MANIFEST": "port-manifest.json",
}
EXPECTED_SOURCE_SHA256 = {
    "perf": "95e787ec86fb2d0d787844517b08520966d711bbf7a5d33429fe499bffe849e7",
    "m128_v3": "da90bff3782fd46b43138f1a9b793e3090da67cf32d13c3f6a440fe8a125d0e0",
    "m128_v4q": "994af1d33ad9c8abe54c8475328b4f9c5f6de04790219cc3ce98ecb446e33abe",
    "m1_pair2_signed": "3f01ca4e79cd86601247faa58a9dfdd01c1150acbe99ced5c1d48b306b8c802c",
}
EXPECTED_MODULE_ARTIFACTS = {
    "perf": {
        "module": "gfx1151_w4a4_perf_ext_rocm10",
        "source": "gfx1151_w4a4_perf_op.hip",
        "library": "gfx1151_w4a4_perf_ext_rocm10.so",
    },
    "m128_v3": {
        "module": "gfx1151_w4a4_m128_v3_ext_rocm10",
        "source": "gfx1151_w4a4_m128_v3_op.hip",
        "library": "gfx1151_w4a4_m128_v3_ext_rocm10.so",
    },
    "m128_v4q": {
        "module": "gfx1151_w4a4_m128_v4q_ext_rocm10",
        "source": "gfx1151_w4a4_m128_qfloat_op.hip",
        "library": "gfx1151_w4a4_m128_v4q_ext_rocm10.so",
    },
    "m1_pair2_signed": {
        "module": "gfx1151_w4a4_m1_dot_pair2_signed_ext_rocm10",
        "source": "gfx1151_w4a4_m1_dot_pair2_signed_op.hip",
        "library": "gfx1151_w4a4_m1_dot_pair2_signed_ext_rocm10.so",
    },
}
_EXACT_WIRE = {
    "bits": 4,
    "group_size": 32,
    "symmetric": True,
    "weight_encoding": "symmetric_uint4b8",
    "weight_packed_dtype": "int32",
    "weight_scale_dtype": "bfloat16",
    "activation": "dynamic_s4_group32",
    "format": "gfx1151-w4a4-g32-v1",
}
_PROJECTION_SHAPES = {
    "qkv_proj": (128, 6144, 4096),
    "o_proj": (128, 4096, 4096),
    "gate_up_proj": (128, 24576, 4096),
    "down_proj": (128, 4096, 12288),
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_fd(fd: int) -> str:
    digest = hashlib.sha256()
    os.lseek(fd, 0, os.SEEK_SET)
    while chunk := os.read(fd, 8 * 1024 * 1024):
        digest.update(chunk)
    os.lseek(fd, 0, os.SEEK_SET)
    return digest.hexdigest()


def _open_immutable_at(directory_fd: int, name: str, *, label: str) -> int:
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        fd = os.open(name, flags, dir_fd=directory_fd)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise RuntimeError(f"W4A4 {label} must not be a symlink: {name}") from exc
        raise RuntimeError(f"W4A4 {label} cannot be opened immutably: {name}") from exc
    metadata = os.fstat(fd)
    if not stat.S_ISREG(metadata.st_mode) or (metadata.st_mode & 0o777) != 0o444:
        os.close(fd)
        raise RuntimeError(
            f"W4A4 {label} identity drift: expected regular immutable mode 0444 file"
        )
    return fd


def _strict_json_fd(fd: int) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise RuntimeError(f"duplicate JSON key in W4A4 manifest: {key}")
            result[key] = value
        return result

    os.lseek(fd, 0, os.SEEK_SET)
    chunks = []
    while chunk := os.read(fd, 1024 * 1024):
        chunks.append(chunk)
    os.lseek(fd, 0, os.SEEK_SET)
    parsed = json.loads(
        b"".join(chunks).decode("utf-8"), object_pairs_hook=reject_duplicates
    )
    if not isinstance(parsed, dict):
        raise RuntimeError("W4A4 manifest root must be an object")
    return parsed


def _validate_relative_member(value: str) -> None:
    member = Path(value)
    if member.is_absolute():
        raise RuntimeError("W4A4 port member path must be relative")
    if ".." in member.parts:
        raise RuntimeError(
            f"W4A4 relative manifest member escapes manifest directory: {value}"
        )
    if len(member.parts) != 1 or member.name in {"", "."}:
        raise RuntimeError(f"W4A4 port member path must be a flat filename: {value}")


def _append_json_line(path: str, line: bytes) -> None:
    fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        remaining = memoryview(line)
        while remaining:
            written = os.write(fd, remaining)
            if written <= 0:
                raise OSError("W4A4 trace write made no progress")
            remaining = remaining[written:]
    finally:
        os.close(fd)


def _exact_wire(config: dict[str, Any], route_scopes: set[str]) -> bool:
    return all(config.get(key) == value for key, value in _EXACT_WIRE.items()) and (
        config.get("route_scope") in route_scopes
    )


def projection_from_prefix(prefix: str) -> str:
    for projection in _PROJECTION_SHAPES:
        if prefix.endswith(f".{projection}"):
            return projection
    raise ValueError(f"unsupported W4A4 projection prefix: {prefix}")


def prepare_groupmajor_weights(
    weight_packed: torch.Tensor, weight_scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if weight_packed.ndim != 2 or weight_scale.ndim != 2:
        raise ValueError("checkpoint W4A4 weights and scales must be rank two")
    n = int(weight_packed.shape[0])
    groups = int(weight_scale.shape[1])
    if tuple(weight_packed.shape) != (n, groups * 4):
        raise ValueError("checkpoint W4A4 packed shape mismatch")
    if tuple(weight_scale.shape) != (n, groups):
        raise ValueError("checkpoint W4A4 scale shape mismatch")
    return (
        weight_packed.reshape(n, groups, 4).permute(1, 2, 0).contiguous(),
        weight_scale.transpose(0, 1).contiguous(),
    )


class _Gfx1151W4A4Config(QuantizationConfig):
    route: Literal["incumbent", "candidate", "v4q_pair2"]
    method_name: str
    route_scopes = {"forced_server_route"}

    def __init__(
        self, route: Literal["incumbent", "candidate", "v4q_pair2"]
    ) -> None:
        super().__init__()
        self.route = route
        self.group_size = 32

    def get_name(self) -> QuantizationMethods:
        return cast(QuantizationMethods, self.method_name)

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 0

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> _Gfx1151W4A4Config:
        if not _exact_wire(config, cls.route_scopes):
            raise ValueError(
                "checkpoint does not declare the exact gfx1151 W4A4/G32 wire"
            )
        return cls()  # type: ignore[call-arg]

    @classmethod
    def override_quantization_method(
        cls,
        hf_quant_cfg: dict[str, Any],
        user_quant: str | None,
        hf_config: Any = None,
    ) -> QuantizationMethods | None:
        if _exact_wire(hf_quant_cfg, cls.route_scopes) and user_quant in _FORCED_METHODS:
            return cast(QuantizationMethods, user_quant)
        return None

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> LinearMethodBase | None:
        if not isinstance(layer, LinearBase):
            return None
        try:
            projection_from_prefix(prefix)
        except ValueError:
            return UnquantizedLinearMethod()
        return Gfx1151W4A4LinearMethod(self, prefix)


@register_quantization_config(GFX1151_W4A4_INCUMBENT)
class Gfx1151W4A4IncumbentConfig(_Gfx1151W4A4Config):
    method_name = GFX1151_W4A4_INCUMBENT

    def __init__(self) -> None:
        super().__init__("incumbent")


@register_quantization_config(GFX1151_W4A4_M1_PAIR2)
class Gfx1151W4A4M1Pair2Config(_Gfx1151W4A4Config):
    method_name = GFX1151_W4A4_M1_PAIR2

    def __init__(self) -> None:
        super().__init__("candidate")


@register_quantization_config(GFX1151_W4A4_V4Q_M1_PAIR2)
class Gfx1151W4A4V4QM1Pair2Config(_Gfx1151W4A4Config):
    method_name = GFX1151_W4A4_V4Q_M1_PAIR2
    route_scopes = {"forced_server_route", "selected_first_class_profile"}

    def __init__(self) -> None:
        super().__init__("v4q_pair2")


def optional_extension_closure(
    route: Literal["incumbent", "candidate", "v4q_pair2"],
) -> tuple[str, ...]:
    if route in ("candidate", "v4q_pair2"):
        return ("v6f", "v4q")
    return ()


class Gfx1151W4A4Runtime:
    """Hash-bound lazy loader for CPU-built, GPU-unqualified port operators."""

    def __init__(
        self, route: Literal["incumbent", "candidate", "v4q_pair2"]
    ) -> None:
        self.route: Literal["incumbent", "candidate", "v4q_pair2"] = route
        self._loaded = False
        self._lock = threading.Lock()

    @staticmethod
    def _required_path(name: str) -> Path:
        value = os.environ.get(name)
        if not value:
            filename = _PACKAGED_ASSETS.get(name)
            if filename is None:
                raise RuntimeError(f"unknown W4A4 runtime input: {name}")
            candidate = _ASSET_ROOT / filename
        else:
            candidate = Path(value).absolute()
        if candidate.is_symlink():
            raise RuntimeError(f"required W4A4 runtime file is a symlink: {candidate}")
        path = candidate.resolve()
        if not path.is_file():
            raise RuntimeError(f"required W4A4 runtime file is absent: {path}")
        return path

    @staticmethod
    def _load_port_manifest(
        path: Path, *, bind_libraries: bool = False
    ) -> dict[str, Any]:
        directory_fd = os.open(
            path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
        )
        bound_library_fds: list[int] = []
        try:
            manifest_fd = _open_immutable_at(
                directory_fd, path.name, label="port manifest"
            )
            try:
                manifest = _strict_json_fd(manifest_fd)
            finally:
                os.close(manifest_fd)
            expected_top = {
                "schema_version",
                "status",
                "target",
                "torch",
                "rocm_home",
                "host_flags",
                "hip_flags",
                "modules",
                "historical_binaries_reused",
                "historical_promotion_reused",
                "gpu_qualified",
                "quality_qualified",
                "performance_qualified",
                "production_ready",
                "promotion_authority",
            }
            if set(manifest) != expected_top:
                raise RuntimeError("W4A4 port manifest shape mismatch")
            target = manifest.get("target")
            if (
                manifest.get("schema_version") != 1
                or manifest.get("status") != "CPU_BUILT_GPU_UNQUALIFIED"
                or target
                != {"rocm": "10.0", "torch_hip": torch.version.hip, "arch": "gfx1151"}
                or manifest.get("rocm_home") != "/opt/rocm"
            ):
                raise RuntimeError("W4A4 ROCm 10 port identity mismatch")
            for claim in (
                "historical_binaries_reused",
                "historical_promotion_reused",
                "gpu_qualified",
                "quality_qualified",
                "performance_qualified",
                "production_ready",
                "promotion_authority",
            ):
                if manifest.get(claim) is not False:
                    raise RuntimeError(f"W4A4 port claim must remain false: {claim}")
            modules = manifest.get("modules")
            if not isinstance(modules, dict) or set(modules) != set(EXPECTED_SOURCE_SHA256):
                raise RuntimeError("W4A4 port module closure mismatch")
            expected_row = {
                "module",
                "source",
                "source_bytes",
                "source_sha256",
                "library",
                "library_bytes",
                "library_sha256",
            }
            for name, expected_source_sha in EXPECTED_SOURCE_SHA256.items():
                row = modules[name]
                if not isinstance(row, dict) or set(row) != expected_row:
                    raise RuntimeError(f"W4A4 port module shape mismatch: {name}")
                for kind in ("source", "library"):
                    _validate_relative_member(row[kind])
                expected_artifacts = EXPECTED_MODULE_ARTIFACTS[name]
                if any(
                    row.get(field) != expected_artifacts[field]
                    for field in ("module", "source", "library")
                ):
                    raise RuntimeError(f"W4A4 port artifact identity mismatch: {name}")
                if row.get("source_sha256") != expected_source_sha:
                    raise RuntimeError(f"W4A4 archived source identity drift: {name}")
                for kind in ("source", "library"):
                    value = row[kind]
                    member_fd = _open_immutable_at(
                        directory_fd, value, label=f"port {kind} {name}"
                    )
                    metadata = os.fstat(member_fd)
                    if (
                        metadata.st_size != row.get(f"{kind}_bytes")
                        or _sha256_fd(member_fd) != row.get(f"{kind}_sha256")
                    ):
                        os.close(member_fd)
                        raise RuntimeError(f"W4A4 port {kind} identity drift: {name}")
                    if kind == "library" and bind_libraries:
                        row["_library_fd"] = member_fd
                        bound_library_fds.append(member_fd)
                    else:
                        os.close(member_fd)
            return manifest
        except BaseException:
            for fd in bound_library_fds:
                try:
                    os.close(fd)
                except OSError:
                    pass
            raise
        finally:
            os.close(directory_fd)

    @staticmethod
    def _close_bound_libraries(manifest: dict[str, Any]) -> None:
        for row in manifest.get("modules", {}).values():
            library_fd = row.pop("_library_fd", None)
            if library_fd is not None:
                try:
                    os.close(library_fd)
                except OSError:
                    pass

    def ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
                raise RuntimeError("gfx1151 W4A4 requires exactly one visible HIP device")
            properties = torch.cuda.get_device_properties(0)
            if "gfx1151" not in str(getattr(properties, "gcnArchName", "")):
                raise RuntimeError("gfx1151 W4A4 loaded on a non-gfx1151 device")
            manifest_path = self._required_path(
                "VLLM_GFX1151_W4A4_PORT_MANIFEST"
            )
            manifest = self._load_port_manifest(
                manifest_path, bind_libraries=True
            )
            try:
                modules = manifest["modules"]
                closure = ["perf", "m128_v3"]
                optional = optional_extension_closure(self.route)
                if "v6f" in optional:
                    closure.append("m1_pair2_signed")
                if "v4q" in optional:
                    closure.append("m128_v4q")
                for name in closure:
                    library_fd = modules[name]["_library_fd"]
                    torch.ops.load_library(f"/proc/self/fd/{library_fd}")
            finally:
                self._close_bound_libraries(manifest)
            self._loaded = True

    def quantize_incumbent(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.ensure_loaded()
        return torch.ops.gfx1151_w4a4_perf.quantize_s4_groupmajor(x)

    def quantize_candidate(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.route != "v4q_pair2":
            raise RuntimeError("v4q quantizer requested outside its M128 route")
        self.ensure_loaded()
        return torch.ops.gfx1151_w4a4_m128_v4q.quantize(x, 2)

    def exact_m128(
        self,
        activation_packed: torch.Tensor,
        activation_scale: torch.Tensor,
        weight_packed: torch.Tensor,
        weight_scale: torch.Tensor,
    ) -> torch.Tensor:
        self.ensure_loaded()
        return torch.ops.gfx1151_w4a4_m128_v3.linear_quantized(
            activation_packed, activation_scale, weight_packed, weight_scale, 5
        )

    def fallback(
        self,
        x: torch.Tensor,
        weight_packed: torch.Tensor,
        weight_scale: torch.Tensor,
        route_code: int,
    ) -> torch.Tensor:
        self.ensure_loaded()
        return torch.ops.gfx1151_w4a4_perf.linear_rowmajor(
            x, weight_packed, weight_scale, route_code
        )

    def m1_pair2(
        self,
        x: torch.Tensor,
        weight_packed: torch.Tensor,
        weight_scale: torch.Tensor,
    ) -> torch.Tensor:
        if (
            self.route not in ("candidate", "v4q_pair2")
            or int(x.shape[0]) != 1
        ):
            raise RuntimeError("signed pair2 is restricted to the two M=1 routes")
        self.ensure_loaded()
        activation_packed, activation_scale = (
            torch.ops.gfx1151_w4a4_perf.quantize_s4_rowmajor(x)
        )
        return torch.ops.gfx1151_w4a4_m1_dot_pair2_signed.run_quantized(
            activation_packed,
            activation_scale,
            weight_packed,
            weight_scale,
            2,
        )

    def trace(self, **row: Any) -> None:
        trace_path = os.environ.get("VLLM_GFX1151_W4A4_TRACE_PATH")
        if not trace_path:
            return
        payload = {
            "schema_version": 1,
            "pid": os.getpid(),
            "monotonic_ns": time.monotonic_ns(),
            "forced_server_route": self.route,
            **row,
        }
        line = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
        _append_json_line(trace_path, line.encode())


_RUNTIMES: dict[str, Gfx1151W4A4Runtime] = {}
_RUNTIME_LOCK = threading.Lock()


def _get_runtime(
    route: Literal["incumbent", "candidate", "v4q_pair2"],
) -> Gfx1151W4A4Runtime:
    with _RUNTIME_LOCK:
        runtime = _RUNTIMES.get(route)
        if runtime is None:
            runtime = Gfx1151W4A4Runtime(route)
            _RUNTIMES[route] = runtime
        return runtime


class Gfx1151W4A4LinearMethod(LinearMethodBase):
    def __init__(self, config: _Gfx1151W4A4Config, prefix: str) -> None:
        self.route = config.route
        self.projection = projection_from_prefix(prefix)
        self.runtime = _get_runtime(config.route)

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs: Any,
    ) -> None:
        if (
            input_size_per_partition != input_size
            or sum(output_partition_sizes) != output_size
        ):
            raise ValueError("gfx1151 W4A4 currently requires tensor parallel size 1")
        if params_dtype != torch.bfloat16:
            raise ValueError("gfx1151 W4A4 requires BF16 model activations")
        _, expected_n, expected_k = _PROJECTION_SHAPES[self.projection]
        if (output_size, input_size) != (expected_n, expected_k):
            raise ValueError(
                f"projection shape mismatch for {self.projection}: "
                f"{(output_size, input_size)} != {(expected_n, expected_k)}"
            )
        weight_loader = extra_weight_attrs.pop("weight_loader")
        weight_packed = ModelWeightParameter(
            data=torch.empty(output_size, input_size // 8, dtype=torch.int32),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        weight_scale = GroupQuantScaleParameter(
            data=torch.empty(
                output_size, input_size // 32, dtype=torch.bfloat16
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_packed", weight_packed)
        layer.register_parameter("weight_scale", weight_scale)
        set_weight_attrs(weight_packed, extra_weight_attrs)
        set_weight_attrs(weight_scale, extra_weight_attrs)

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        if layer.weight_packed.dtype != torch.int32:
            raise RuntimeError("materialized W4A4 packed weights are not int32")
        if layer.weight_scale.dtype != torch.bfloat16:
            raise RuntimeError("materialized W4A4 scales are not BF16")
        group_packed, group_scale = prepare_groupmajor_weights(
            layer.weight_packed, layer.weight_scale
        )
        layer.register_buffer(
            "weight_groupmajor_packed", group_packed, persistent=False
        )
        layer.register_buffer(
            "weight_groupmajor_scale", group_scale, persistent=False
        )
        self.runtime.ensure_loaded()

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.dtype != torch.bfloat16:
            raise ValueError("gfx1151 W4A4 requires BF16 activations")
        k = int(x.shape[-1])
        n = int(layer.weight_packed.shape[0])
        if k != int(layer.weight_packed.shape[1]) * 8:
            raise ValueError("activation K does not match packed W4A4 weight")
        x_2d = x.reshape(-1, k).contiguous()
        m = int(x_2d.shape[0])
        exact_shape = _PROJECTION_SHAPES[self.projection]
        if (m, n, k) == exact_shape:
            if not hasattr(layer, "weight_groupmajor_packed"):
                raise RuntimeError("group-major W4A4 weights were not prepared")
            if self.route == "v4q_pair2":
                activation_packed, activation_scale = (
                    self.runtime.quantize_candidate(x_2d)
                )
                route_label = "m128_v4q_quantizer_exact_v3"
            else:
                activation_packed, activation_scale = (
                    self.runtime.quantize_incumbent(x_2d)
                )
                route_label = "m128_incumbent_quantizer_exact_v3"
            output = self.runtime.exact_m128(
                activation_packed,
                activation_scale,
                layer.weight_groupmajor_packed,
                layer.weight_groupmajor_scale,
            )
        elif self.route in ("candidate", "v4q_pair2") and m == 1:
            output = self.runtime.m1_pair2(
                x_2d, layer.weight_packed, layer.weight_scale
            )
            route_label = "m1_pair2_signed_dot8"
        else:
            route_code = 1 if m <= 8 else 2
            if m == 1:
                route_label = "m1_incumbent_dot8"
            elif route_code == 1:
                route_label = "fallback_dot_baseline"
            else:
                route_label = "fallback_wmma_baseline_padded"
            output = self.runtime.fallback(
                x_2d, layer.weight_packed, layer.weight_scale, route_code
            )
        if bias is not None:
            output = output + bias
        self.runtime.trace(
            projection=self.projection,
            selected_route=route_label,
            M=m,
            N=n,
            K=k,
        )
        return output.reshape(*x.shape[:-1], n)


__all__ = [
    "GFX1151_W4A4_INCUMBENT",
    "GFX1151_W4A4_M1_PAIR2",
    "GFX1151_W4A4_V4Q_M1_PAIR2",
    "Gfx1151W4A4IncumbentConfig",
    "Gfx1151W4A4M1Pair2Config",
    "Gfx1151W4A4V4QM1Pair2Config",
    "Gfx1151W4A4LinearMethod",
    "optional_extension_closure",
    "prepare_groupmajor_weights",
    "projection_from_prefix",
]
