# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicit-I/O safetensors row reader for Qwen4Exp PLE tables."""

import json
import os
import stat as stat_module
import struct
import threading
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from itertools import islice
from pathlib import Path
from typing import Any

import regex as re

_HEADER_LIMIT = 16 * 1024 * 1024
_INDEX_LIMIT = 64 * 1024 * 1024
_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "F8_E8M0": 1,
    "U16": 2,
    "I16": 2,
    "F16": 2,
    "BF16": 2,
    "U32": 4,
    "I32": 4,
    "F32": 4,
    "U64": 8,
    "I64": 8,
    "F64": 8,
}


class _TensorLocation:
    __slots__ = ("fd", "offset", "rows", "path", "file_size", "device", "inode")

    def __init__(
        self,
        fd: int,
        offset: int,
        rows: int,
        path: Path,
        file_size: int,
        device: int,
        inode: int,
    ) -> None:
        self.fd = fd
        self.offset = offset
        self.rows = rows
        self.path = path
        self.file_size = file_size
        self.device = device
        self.inode = inode


class _ReadRange:
    __slots__ = ("location", "offset", "length", "rows")

    def __init__(self, location: _TensorLocation, offset: int, row_index: int) -> None:
        self.location = location
        self.offset = offset
        self.length = 0
        self.rows: list[tuple[int, int]] = [(row_index, 0)]


class PLESafetensorsReader:
    """Read exact BF16 PLE rows with bounded, cache-free ``pread`` calls."""

    def __init__(
        self,
        model_dir: str | os.PathLike[str],
        *,
        num_embeddings: int,
        embedding_dim: int,
        split_ngram_parts: int,
        tensor_prefix: str | None = None,
        max_batch_rows: int = 65536,
        workers: int = 8,
    ) -> None:
        self._validate_positive_int("num_embeddings", num_embeddings)
        self._validate_positive_int("embedding_dim", embedding_dim)
        self._validate_positive_int("split_ngram_parts", split_ngram_parts)
        self._validate_positive_int("max_batch_rows", max_batch_rows)
        self._validate_positive_int("workers", workers)
        if tensor_prefix is not None and (
            not isinstance(tensor_prefix, str) or not tensor_prefix
        ):
            raise ValueError("tensor_prefix must be a non-empty string or None")

        root = Path(model_dir).resolve()
        if not root.is_dir():
            raise ValueError(f"model_dir is not a directory: {root}")
        self._num_embeddings = num_embeddings
        self._embedding_dim = embedding_dim
        self._parts = split_ngram_parts
        self._row_bytes = embedding_dim * 2
        self._max_batch_rows = max_batch_rows
        self._worker_count = workers
        self._executor: ThreadPoolExecutor | None = None
        self._fds: list[int] = []
        self._locations: list[_TensorLocation] = []
        self._closed = False
        self._lifecycle_lock = threading.RLock()
        self._stats_lock = threading.Lock()
        self._read_calls = 0
        self._read_bytes = 0
        self._requested_rows = 0
        self._unique_rows = 0
        self._gather_wall_seconds = 0.0

        try:
            entries = self._discover(root, tensor_prefix)
            prefix = self._select_prefix(entries, tensor_prefix)
            names = [f"{prefix}.shard_{index}.weight" for index in range(self._parts)]
            expected = set(names)
            matching = set()
            for name in entries:
                split = self._split_name(name)
                if split is not None and split[0] == prefix:
                    matching.add(name)
            if matching != expected:
                raise ValueError("PLE shard names are incomplete or out of range")

            shard_size = (num_embeddings + split_ngram_parts - 1) // split_ngram_parts
            for index, name in enumerate(names):
                metadata, fd, path, file_size, data_start, stat = entries[name]
                expected_rows = max(
                    0, min(shard_size, num_embeddings - index * shard_size)
                )
                self._validate_ple_tensor(name, metadata, expected_rows, embedding_dim)
                data_offset = metadata["data_offsets"][0]
                self._locations.append(
                    _TensorLocation(
                        fd,
                        data_start + data_offset,
                        expected_rows,
                        path,
                        file_size,
                        stat.st_dev,
                        stat.st_ino,
                    )
                )
            self.tensor_names = frozenset(names)
            selected_paths = {entries[name][2] for name in names}
            mixed_paths = {
                entry[2]
                for name, entry in entries.items()
                if entry[2] in selected_paths and name not in self.tensor_names
            }
            self.ple_only_shard_paths = frozenset(
                str(path) for path in selected_paths - mixed_paths
            )
            self.mixed_shard_paths = frozenset(str(path) for path in mixed_paths)
            self._mixed_weights = {
                name: entry
                for name, entry in entries.items()
                if entry[2] in mixed_paths
                and ".ple_embedding.ngram_embedding." not in name
            }
        except BaseException:
            self.close()
            raise

    def iter_mixed_shard_weights(self):
        """Yield ordinary tensors from mixed files without mapping PLE bytes.

        One CPU tensor is staged at a time; each explicit read is at most 64 MiB.
        PLE lookups retain their separate bounded row-reader and cache policy.
        """
        for name, entry in self._mixed_weights.items():
            with self._lifecycle_lock:
                if self._closed:
                    raise RuntimeError("PLE reader is closed")
                tensor = _pread_tensor(entry)
            yield name, tensor
            del tensor

    @staticmethod
    def _validate_positive_int(name: str, value: object) -> None:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value <= 0:
            raise ValueError(f"{name} must be positive")

    @staticmethod
    def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    @classmethod
    def _decode_json(cls, data: bytes, source: Path) -> Any:
        try:
            return json.loads(
                data.decode("utf-8"), object_pairs_hook=cls._unique_object
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid JSON in {source}") from error

    def _discover(
        self, root: Path, tensor_prefix: str | None
    ) -> dict[str, tuple[dict[str, Any], int, Path, int, int, os.stat_result]]:
        indexes = sorted(root.glob("*.safetensors.index.json"))
        if len(indexes) > 1:
            raise ValueError("multiple safetensors index files are ambiguous")
        index_map: dict[str, str] | None = None
        if indexes:
            raw = self._read_bounded_regular_file(indexes[0], _INDEX_LIMIT)
            index = self._decode_json(raw, indexes[0])
            if not isinstance(index, dict) or not isinstance(
                index.get("weight_map"), dict
            ):
                raise ValueError("safetensors index has no weight_map object")
            index_map = index["weight_map"]
            if not index_map or any(
                not isinstance(name, str) or not isinstance(path, str)
                for name, path in index_map.items()
            ):
                raise ValueError("safetensors weight_map is empty or malformed")
            paths = sorted(set(index_map.values()))
            for relative in paths:
                candidate = Path(relative)
                if (
                    candidate.is_absolute()
                    or candidate.name != relative
                    or candidate.suffix != ".safetensors"
                ):
                    raise ValueError(f"unsafe safetensors shard path: {relative}")
            prefix = self._select_prefix(index_map, tensor_prefix)
            selected_paths = {
                relative
                for name, relative in index_map.items()
                if (split := self._split_name(name)) is not None and split[0] == prefix
            }
            paths = sorted(selected_paths)
            # Validate every tensor in selected files, including native tensors.
            index_map = {
                name: relative
                for name, relative in index_map.items()
                if relative in selected_paths
            }
        else:
            paths = sorted(path.name for path in root.glob("*.safetensors"))
            if not paths:
                raise ValueError("no local safetensors files found")

        entries: dict[
            str, tuple[dict[str, Any], int, Path, int, int, os.stat_result]
        ] = {}
        actual_map: dict[str, str] = {}
        for relative in paths:
            path = root / relative
            flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)
            try:
                fd = os.open(path, flags)
            except OSError as error:
                raise OSError(f"cannot open safetensors shard {path}") from error
            self._fds.append(fd)
            stat = os.fstat(fd)
            if not stat_module.S_ISREG(stat.st_mode):
                raise ValueError(f"safetensors shard is not a regular file: {path}")
            header, data_start = self._read_header(fd, path, stat.st_size)
            for name, metadata in header.items():
                if name == "__metadata__":
                    if not isinstance(metadata, dict) or any(
                        not isinstance(key, str) or not isinstance(value, str)
                        for key, value in metadata.items()
                    ):
                        raise ValueError(f"malformed __metadata__ in {path}")
                    continue
                if name in entries:
                    raise ValueError(f"duplicate tensor across shards: {name}")
                entries[name] = (
                    metadata,
                    fd,
                    path,
                    stat.st_size,
                    data_start,
                    stat,
                )
                actual_map[name] = relative

        if index_map is not None and index_map != actual_map:
            raise ValueError("safetensors index does not exactly match shard headers")
        return entries

    @staticmethod
    def _read_bounded_regular_file(path: Path, limit: int) -> bytes:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            fd = os.open(path, flags)
        except OSError as error:
            raise OSError(f"cannot open metadata file {path}") from error
        try:
            file_stat = os.fstat(fd)
            if not stat_module.S_ISREG(file_stat.st_mode):
                raise ValueError(f"metadata is not a regular file: {path}")
            if file_stat.st_size > limit:
                raise ValueError(f"metadata file exceeds size limit: {path}")
            data = os.pread(fd, file_stat.st_size, 0)
            if len(data) != file_stat.st_size:
                raise OSError(f"short metadata read from {path}")
            return data
        finally:
            os.close(fd)

    @classmethod
    def _read_header(
        cls, fd: int, path: Path, file_size: int
    ) -> tuple[dict[str, Any], int]:
        prefix = os.pread(fd, 8, 0)
        if len(prefix) != 8:
            raise ValueError(f"truncated safetensors length in {path}")
        header_size = struct.unpack("<Q", prefix)[0]
        if header_size < 2 or header_size > _HEADER_LIMIT:
            raise ValueError(f"invalid safetensors header size in {path}")
        data_start = 8 + header_size
        if data_start > file_size:
            raise ValueError(f"truncated safetensors header in {path}")
        encoded = os.pread(fd, header_size, 8)
        if len(encoded) != header_size:
            raise ValueError(f"short safetensors header read in {path}")
        header = cls._decode_json(encoded, path)
        if not isinstance(header, dict):
            raise ValueError(f"safetensors header is not an object in {path}")

        ranges: list[tuple[int, int]] = []
        for name, metadata in header.items():
            if name == "__metadata__":
                continue
            if not isinstance(name, str) or not isinstance(metadata, dict):
                raise ValueError(f"malformed tensor metadata in {path}")
            dtype = metadata.get("dtype")
            shape = metadata.get("shape")
            offsets = metadata.get("data_offsets")
            if (
                dtype not in _DTYPE_BYTES
                or not isinstance(shape, list)
                or any(
                    isinstance(value, bool) or not isinstance(value, int) or value < 0
                    for value in shape
                )
                or not isinstance(offsets, list)
                or len(offsets) != 2
                or any(
                    isinstance(value, bool) or not isinstance(value, int)
                    for value in offsets
                )
            ):
                raise ValueError(f"malformed tensor metadata for {name}")
            start, end = offsets
            elements = 1
            for dimension in shape:
                elements *= dimension
            if (
                start < 0
                or end < start
                or end - start != elements * _DTYPE_BYTES[dtype]
            ):
                raise ValueError(f"invalid data range for {name}")
            ranges.append((start, end))

        cursor = 0
        for start, end in sorted(ranges):
            if start != cursor:
                raise ValueError(f"gapped or overlapping tensor ranges in {path}")
            cursor = end
        if data_start + cursor != file_size:
            raise ValueError(f"safetensors file size does not match header in {path}")
        return header, data_start

    @staticmethod
    def _split_name(name: str) -> tuple[str, int] | None:
        match = re.fullmatch(r"(.+)\.shard_([0-9]+)\.weight", name)
        if match is None:
            return None
        return match.group(1), int(match.group(2))

    def _select_prefix(
        self,
        entries: Iterable[str],
        requested: str | None,
    ) -> str:
        if requested is not None:
            return requested
        candidates: dict[str, set[int]] = {}
        for name in entries:
            split = self._split_name(name)
            if split is not None:
                candidates.setdefault(split[0], set()).add(split[1])
        complete = [
            prefix
            for prefix, indices in candidates.items()
            if indices == set(range(self._parts))
        ]
        if len(complete) != 1:
            raise ValueError(
                "tensor_prefix is required for ambiguous or missing PLE data"
            )
        return complete[0]

    @staticmethod
    def _validate_ple_tensor(
        name: str, metadata: dict[str, Any], rows: int, embedding_dim: int
    ) -> None:
        if metadata.get("dtype") != "BF16":
            raise ValueError(f"{name} must use BF16")
        if metadata.get("shape") != [rows, embedding_dim]:
            raise ValueError(
                f"{name} has shape {metadata.get('shape')}, expected "
                f"[{rows}, {embedding_dim}]"
            )

    def _row_location(self, row_id: int) -> tuple[_TensorLocation, int]:
        shard_size = (self._num_embeddings + self._parts - 1) // self._parts
        shard_index = row_id // shard_size
        row_in_shard = row_id - shard_index * shard_size
        return self._locations[shard_index], row_in_shard

    def gather_bytes(self, row_ids: Iterable[int]) -> bytes:
        """Return raw little-endian BF16 rows in the requested order."""
        with self._lifecycle_lock:
            return self._gather_locked(row_ids)

    @property
    def max_batch_rows(self) -> int:
        """Maximum rows accepted by one gather, for bounded runtime staging."""
        return self._max_batch_rows

    def _gather_locked(self, row_ids: Iterable[int]) -> bytes:
        if self._closed:
            raise ValueError("reader is closed")
        try:
            ids = list(islice(iter(row_ids), self._max_batch_rows + 1))
        except TypeError as error:
            raise TypeError("row_ids must be iterable") from error
        if len(ids) > self._max_batch_rows:
            raise ValueError("row_ids exceeds max_batch_rows")
        for row_id in ids:
            if isinstance(row_id, bool) or not isinstance(row_id, int):
                raise TypeError("row IDs must be integers, not bool")
            if row_id < 0 or row_id >= self._num_embeddings:
                raise ValueError(f"row ID out of range: {row_id}")
        if not ids:
            return b""

        unique_ids = sorted(set(ids))
        ranges: list[_ReadRange] = []
        previous_location: _TensorLocation | None = None
        previous_offset = -1
        for row_id in unique_ids:
            location, row_in_shard = self._row_location(row_id)
            offset = location.offset + row_in_shard * self._row_bytes
            if (
                ranges
                and location is previous_location
                and offset == previous_offset + self._row_bytes
            ):
                ranges[-1].rows.append((row_id, ranges[-1].length))
            else:
                ranges.append(_ReadRange(location, offset, row_id))
            ranges[-1].length += self._row_bytes
            previous_location = location
            previous_offset = offset

        with self._stats_lock:
            self._requested_rows += len(ids)
            self._unique_rows += len(unique_ids)
        started = time.perf_counter()
        try:
            if len(ranges) == 1 or self._worker_count == 1:
                chunks = [self._read_range(read_range) for read_range in ranges]
            else:
                if self._executor is None:
                    self._executor = ThreadPoolExecutor(
                        max_workers=self._worker_count,
                        thread_name_prefix="ple-pread",
                    )
                futures = [
                    self._executor.submit(self._read_range, read_range)
                    for read_range in ranges
                ]
                chunks = []
                first_error: BaseException | None = None
                for future in futures:
                    try:
                        chunks.append(future.result())
                    except BaseException as error:
                        chunks.append(b"")
                        if first_error is None:
                            first_error = error
                if first_error is not None:
                    raise first_error
            rows: dict[int, bytes] = {}
            for read_range, chunk in zip(ranges, chunks):
                for row_id, relative in read_range.rows:
                    rows[row_id] = chunk[relative : relative + self._row_bytes]
            return b"".join(rows[row_id] for row_id in ids)
        finally:
            elapsed = time.perf_counter() - started
            with self._stats_lock:
                self._gather_wall_seconds += elapsed

    def _read_range(self, read_range: _ReadRange) -> bytes:
        location = read_range.location
        stat = os.fstat(location.fd)
        if (
            stat.st_dev != location.device
            or stat.st_ino != location.inode
            or stat.st_size != location.file_size
        ):
            raise OSError(f"safetensors shard identity changed: {location.path}")
        with self._stats_lock:
            self._read_calls += 1
        data = os.pread(location.fd, read_range.length, read_range.offset)
        with self._stats_lock:
            self._read_bytes += len(data)
        if len(data) != read_range.length:
            raise OSError(
                f"short PLE row read from {location.path}: "
                f"got {len(data)}, expected {read_range.length}"
            )
        return data

    def stats(self) -> dict[str, int | float]:
        """Return a snapshot of cumulative actual gather and I/O counters."""
        with self._stats_lock:
            return {
                "read_calls": self._read_calls,
                "read_bytes": self._read_bytes,
                "requested_rows": self._requested_rows,
                "unique_rows": self._unique_rows,
                "gather_wall_seconds": self._gather_wall_seconds,
            }

    def close(self) -> None:
        """Close the reader's executor and all owned shard descriptors."""
        with self._lifecycle_lock:
            if self._closed:
                return
            self._closed = True
            if self._executor is not None:
                self._executor.shutdown(wait=True, cancel_futures=True)
                self._executor = None
            for fd in self._fds:
                with suppress(OSError):
                    os.close(fd)
            self._fds.clear()

    def __enter__(self) -> "PLESafetensorsReader":
        if self._closed:
            raise ValueError("reader is closed")
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()


__all__ = ["PLESafetensorsReader"]


def _pread_tensor(entry):
    import torch

    metadata, fd, path, file_size, data_start, original_stat = entry
    identity = (original_stat.st_dev, original_stat.st_ino, file_size)
    if any(
        (stat.st_dev, stat.st_ino, stat.st_size) != identity
        for stat in (os.fstat(fd), path.stat())
    ):
        raise RuntimeError("mixed PLE checkpoint identity/size changed")
    dtypes = {
        "BOOL": torch.bool,
        "U8": torch.uint8,
        "I8": torch.int8,
        "I16": torch.int16,
        "I32": torch.int32,
        "I64": torch.int64,
        "F16": torch.float16,
        "BF16": torch.bfloat16,
        "F32": torch.float32,
        "F64": torch.float64,
    }
    dtype = dtypes.get(metadata["dtype"])
    if dtype is None:
        raise ValueError("unsupported mixed PLE ordinary tensor dtype")
    start, end = metadata["data_offsets"]
    length = end - start
    if not length:
        return torch.empty(metadata["shape"], dtype=dtype, device="cpu")
    payload = bytearray(length)
    offset = 0
    while offset < length:
        count = min(length - offset, 64 * 1024 * 1024)
        chunk = os.pread(fd, count, data_start + start + offset)
        if not chunk:
            raise EOFError("short mixed PLE ordinary tensor payload")
        payload[offset : offset + len(chunk)] = chunk
        offset += len(chunk)
    return torch.frombuffer(payload, dtype=dtype).reshape(metadata["shape"])


def ple_ssd_weights_iterator(
    hf_weights_files: list[str],
    use_tqdm_on_load: bool,
    *,
    local_expert_ids: set[int] | None = None,
):
    """Exclude PLE before mmap, including when the draft has no PLE modules.

    LOADFAST v2 (nb2-loadfast-v2), byte-identical stream to the image original:
    (1) consecutive ordinary files go through ONE upstream lazy iterator call (v1);
    (2) ordinary/mixed classification from model.safetensors.index.json; mixed files and files
        absent from the index keep the bounded-header path; every tensor yielded from an ordinary
        run is re-checked for PLE names (index/header disagreement raises instead of diverging);
    (3) bounded posix_fadvise(WILLNEED) prefetch of index-ordinary files ahead of the consumer
        (page-cache warming only; the bytes read are unchanged).
    """
    from vllm.model_executor.model_loader.weight_utils import (
        _natural_sort_key,
        safetensors_weights_iterator,
        should_skip_weight,
    )

    order = sorted(hf_weights_files, key=_natural_sort_key)
    paths = [Path(filename).resolve() for filename in order]
    index = _lf2_load_index(paths)
    known, ple_files, weight_map = index if index is not None else (set(), set(), {})
    pf_files = [p for p in paths if p.name in known and p.name not in ple_files]
    ordinal = {p.name: i for i, p in enumerate(pf_files)}
    prefetch = None
    window = int(os.environ.get("DIRECT_LF2_PREFETCH_WINDOW_BYTES", str(6 << 30)))
    if pf_files and os.environ.get("DIRECT_LF2_PREFETCH", "1") == "1" and window > 0:
        prefetch = _LF2Prefetch(pf_files, window)
        prefetch.start()
    stats = {"index": index is not None, "index_ordinary": 0, "header_path": 0, "runs": 0}
    pending: list[str] = []

    def _flush():
        if pending:
            run = list(pending)
            pending.clear()
            stats["runs"] += 1
            for name, tensor in safetensors_weights_iterator(
                run,
                use_tqdm_on_load,
                "lazy",
                local_expert_ids=local_expert_ids,
            ):
                if ".ple_embedding.ngram_embedding." in name:
                    raise ValueError(f"PLE tensor in a shard classified as ordinary: {name}")
                if prefetch is not None:
                    o = ordinal.get(weight_map.get(name))
                    if o is not None:
                        prefetch.consumer = o
                yield name, tensor

    try:
        for path in paths:
            if path.name in known and path.name not in ple_files:
                st = os.stat(path, follow_symlinks=False)
                if not stat_module.S_ISREG(st.st_mode):
                    raise ValueError(f"safetensors shard is not a regular file: {path}")
                stats["index_ordinary"] += 1
                pending.append(str(path))
                continue
            stats["header_path"] += 1
            flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)
            fd = os.open(path, flags)
            try:
                stat = os.fstat(fd)
                if not stat_module.S_ISREG(stat.st_mode):
                    raise ValueError(f"safetensors shard is not a regular file: {path}")
                header, data_start = PLESafetensorsReader._read_header(
                    fd, path, stat.st_size
                )
                ple_names = {
                    name for name in header if ".ple_embedding.ngram_embedding." in name
                }
                for name in ple_names:
                    metadata = header[name]
                    if (
                        not re.search(
                            r"\.ple_embedding\.ngram_embedding\.shard_[0-9]+\.weight$",
                            name,
                        )
                        or metadata["dtype"] != "BF16"
                        or len(metadata["shape"]) != 2
                    ):
                        raise ValueError(f"unsupported SSD PLE tensor: {name}")
                if not ple_names:
                    pending.append(str(path))
                else:
                    yield from _flush()
                    for name, metadata in header.items():
                        if (
                            name == "__metadata__"
                            or name in ple_names
                            or should_skip_weight(name, local_expert_ids)
                        ):
                            continue
                        tensor = _pread_tensor(
                            (metadata, fd, path, stat.st_size, data_start, stat)
                        )
                        yield name, tensor
                        del tensor
            finally:
                os.close(fd)
        yield from _flush()
    finally:
        if prefetch is not None:
            prefetch.stop.set()
            prefetch.join(timeout=5)
            stats["prefetch"] = prefetch.summary()
        print("DIRECT_LF2 " + json.dumps(stats, sort_keys=True), flush=True)


def _lf2_load_index(paths):
    parents = {p.parent for p in paths}
    if len(parents) != 1:
        return None
    index_path = next(iter(parents)) / "model.safetensors.index.json"
    try:
        weight_map = json.loads(index_path.read_text())["weight_map"]
    except (OSError, ValueError, KeyError):
        return None
    if not isinstance(weight_map, dict):
        return None
    known = set(weight_map.values())
    ple_files = {f for n, f in weight_map.items() if ".ple_embedding.ngram_embedding." in n}
    return known, ple_files, weight_map


class _LF2Prefetch(threading.Thread):
    """Issue posix_fadvise(WILLNEED) for ordinary shards, at most `window` bytes ahead."""

    def __init__(self, files, window):
        super().__init__(name="direct-lf2-prefetch", daemon=True)
        self.files, self.window = files, window
        self.consumer = 0
        self.stop = threading.Event()
        self.cum = [0]
        self.issued = self.skipped = self.waits = 0
        self.issued_bytes = 0
        self.fadvise_s = 0.0
        self.error = None
        self.t0 = time.time()
        self.t_done = None

    def _size(self, k):
        while len(self.cum) <= k + 1:
            j = len(self.cum) - 1
            self.cum.append(self.cum[-1] + os.stat(self.files[j]).st_size)
        return self.cum[k + 1] - self.cum[k]

    def run(self):
        try:
            k = 0
            n = len(self.files)
            while k < n and not self.stop.is_set():
                c = min(self.consumer, n - 1)
                if k < c:
                    self.skipped += c - k
                    k = c
                    continue
                self._size(k)
                if self.cum[k] - self.cum[c] > self.window:
                    self.waits += 1
                    self.stop.wait(0.002)
                    continue
                t = time.perf_counter()
                fd = os.open(self.files[k], os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
                try:
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_WILLNEED)
                finally:
                    os.close(fd)
                self.fadvise_s += time.perf_counter() - t
                self.issued += 1
                self.issued_bytes += self.cum[k + 1] - self.cum[k]
                k += 1
            self.t_done = time.time()
        except Exception as exc:  # prefetch is advisory; never affects loading
            self.error = repr(exc)

    def summary(self):
        return {"files": len(self.files), "issued": self.issued, "issued_bytes": self.issued_bytes,
                "skipped_overtaken": self.skipped, "window_waits": self.waits,
                "fadvise_s": round(self.fadvise_s, 3), "window": self.window, "error": self.error,
                "done_after_s": None if self.t_done is None else round(self.t_done - self.t0, 3)}
