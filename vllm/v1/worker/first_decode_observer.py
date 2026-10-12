# SPDX-License-Identifier: Apache-2.0
"""Opt-in, worker-local first-decode breadcrumb observer.

Importing this module is inert.  The environment is inspected only when
``get_first_decode_observer`` is first called from a worker execution seam.
"""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any

ENABLE_ENV = "VLLM_FIRST_DECODE_OBSERVER"
PATH_ENV = "VLLM_FIRST_DECODE_OBSERVER_PATH"
MAX_RECORD_BYTES = 512


class ObserverError(RuntimeError):
    """Base class for fail-closed diagnostic observer failures."""


class ObserverOpenError(ObserverError):
    pass


class ObserverWriteError(ObserverError):
    pass


class FirstDecodeObserver:
    """Write bounded JSONL breadcrumbs to one pre-opened append-only FD."""

    def __init__(
        self,
        fd: int | None,
        write: Callable[[int, bytes | memoryview], int] = os.write,
    ) -> None:
        self._fd = fd
        self._write = write
        self._iteration = 0
        self._kind = "unknown"
        self._request_count = 0
        self._closed = False

    @classmethod
    def from_environ(cls, environ: Mapping[str, str]) -> FirstDecodeObserver:
        enabled = environ.get(ENABLE_ENV, "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if not enabled:
            return cls(None)
        path = environ.get(PATH_ENV, "")
        if not path:
            raise ObserverOpenError(f"{PATH_ENV} is required when {ENABLE_ENV}=1")
        flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT
        flags |= getattr(os, "O_CLOEXEC", 0)
        try:
            fd = os.open(path, flags, 0o600)
        except OSError as exc:
            raise ObserverOpenError(f"cannot open breadcrumb path {path!r}: {exc}") from exc
        return cls(fd)

    @property
    def enabled(self) -> bool:
        return self._fd is not None

    def begin_iteration(self, kind: str, request_ids: Sequence[Any]) -> int:
        if self._fd is None:
            return 0
        if kind not in {"prompt", "decode", "mixed", "no_work", "unknown"}:
            raise ObserverError(f"invalid iteration kind: {kind!r}")
        self._iteration += 1
        self._kind = kind
        self._request_count = len(request_ids)
        return self._iteration

    def emit(self, phase: str, edge: str, *, route: str | None = None) -> None:
        fd = self._fd
        if fd is None:
            return
        if self._closed:
            raise ObserverWriteError("observer FD is closed")
        if edge not in {"enter", "exit"}:
            raise ObserverError(f"invalid edge: {edge!r}")
        record: dict[str, Any] = {
            "monotonic_ns": time.monotonic_ns(),
            "pid": os.getpid(),
            "tid": threading.get_native_id(),
            "iteration": self._iteration,
            "request_count": self._request_count,
            "kind": self._kind,
            "phase": phase,
            "edge": edge,
        }
        if route is not None:
            record["route"] = route
        payload = json.dumps(
            record, separators=(",", ":"), ensure_ascii=True, sort_keys=True
        ).encode("ascii") + b"\n"
        if len(payload) > MAX_RECORD_BYTES:
            raise ObserverWriteError(
                f"breadcrumb record exceeds {MAX_RECORD_BYTES} bytes"
            )
        view = memoryview(payload)
        while view:
            try:
                written = self._write(fd, view)
            except InterruptedError:
                continue
            except OSError as exc:
                raise ObserverWriteError(f"breadcrumb write failed: {exc}") from exc
            if written <= 0:
                raise ObserverWriteError("breadcrumb write returned a zero-byte short write")
            if written > len(view):
                raise ObserverWriteError("breadcrumb write returned an invalid byte count")
            view = view[written:]

    @contextmanager
    def phase(self, phase: str, *, route: str | None = None) -> Iterator[None]:
        self.emit(phase, "enter", route=route)
        try:
            yield
        except BaseException:
            # The unmatched enter is the durable localization oracle.
            raise
        else:
            self.emit(phase, "exit", route=route)

    def close(self) -> None:
        if self._fd is not None and not self._closed:
            os.close(self._fd)
            self._closed = True


_worker_local_observer: FirstDecodeObserver | None = None


def get_first_decode_observer() -> FirstDecodeObserver:
    """Initialize once in the calling worker; module import remains inert."""
    global _worker_local_observer
    if _worker_local_observer is None:
        _worker_local_observer = FirstDecodeObserver.from_environ(os.environ)
    return _worker_local_observer
