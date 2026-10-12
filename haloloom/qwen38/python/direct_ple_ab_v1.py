# SPDX-License-Identifier: Apache-2.0
"""Direct-session PLE SSD gather A/B. Private sequential benchmark interface.

The native path returns exactly the bytes PLESafetensorsReader.gather_bytes
would return for the same row IDs: identical shard/offset mapping, buffered
pread, no row cache. Arm selection comes from an atomically replaced phase
file written by the isolated bench between requests. Missing env or missing
phase file means the installed reader path (no behaviour change).
"""
import collections, ctypes, json, os, threading, time
from pathlib import Path

import numpy as np

ARMS = ('installed', 'native', 'shadow')
PHASE_ENV = 'DIRECT_PLE_AB_PHASE'
LIB_ENV = 'DIRECT_PLE_NATIVE_LIB'
THREADS_ENV = 'DIRECT_PLE_NATIVE_THREADS'
COUNTS = collections.Counter()
_LIB = None
_LOCK = threading.Lock()
_LAST_CASE = None


def read_arm():
    path = os.environ.get(PHASE_ENV)
    if not path:
        return 'installed', None
    try:
        data = json.loads(Path(path).read_text())
    except FileNotFoundError:
        return 'installed', None
    if not isinstance(data, dict) or set(data) != {'arm', 'case'}:
        raise ValueError('invalid PLE A/B phase schema')
    if data['arm'] not in ARMS or not isinstance(data['case'], str) or not 0 < len(data['case']) <= 128:
        raise ValueError('invalid PLE A/B phase value')
    return data['arm'], data['case']


def _lib():
    global _LIB
    with _LOCK:
        if _LIB is None:
            path = os.environ.get(LIB_ENV)
            if not path:
                raise RuntimeError(f'{LIB_ENV} is required for native PLE arm')
            lib = ctypes.CDLL(path)  # CDLL releases the GIL for the call
            lib.ple_gather.restype = ctypes.c_int
            lib.ple_gather.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int64, ctypes.c_int]
            _LIB = lib
        return _LIB


MIN_RUNS_PER_THREAD = 512  # v2: decode lookups (<=16 preads/token) stay single-threaded; no pthread churn


def _threads(runs=None):
    value = int(os.environ.get(THREADS_ENV, '16'))
    if not 1 <= value <= 256:
        raise ValueError('native PLE thread count out of range')
    if runs is not None:
        value = max(1, min(value, int(runs) // MIN_RUNS_PER_THREAD))
    return value


def gather_rows_native(reader, ids):
    """Return uint16 [len(ids), dim] rows, byte-identical to reader.gather_bytes."""
    ids = np.ascontiguousarray(ids, dtype=np.int64)
    dim = reader._embedding_dim
    rb = reader._row_bytes
    if ids.ndim != 1:
        raise ValueError('row IDs must be one-dimensional')
    if ids.size > reader._max_batch_rows:
        raise ValueError('row_ids exceeds max_batch_rows')
    if ids.size == 0:
        return np.empty((0, dim), np.uint16)
    if int(ids.min()) < 0 or int(ids.max()) >= reader._num_embeddings:
        raise ValueError('row ID out of range')
    with reader._lifecycle_lock:
        if reader._closed:
            raise ValueError('reader is closed')
        locs = reader._locations
        for loc in locs:  # same identity guard as _read_range, once per call
            st = os.fstat(loc.fd)
            if st.st_dev != loc.device or st.st_ino != loc.inode or st.st_size != loc.file_size:
                raise OSError(f'safetensors shard identity changed: {loc.path}')
        fds = np.array([loc.fd for loc in locs], np.int32)
        base = np.array([loc.offset for loc in locs], np.int64)
        rows = np.array([loc.rows for loc in locs], np.int64)
        shard = (reader._num_embeddings + reader._parts - 1) // reader._parts
        uniq, inv = np.unique(ids, return_inverse=True)
        sh = uniq // shard
        rin = uniq - sh * shard
        if np.any(rin >= rows[sh]):
            raise ValueError('row beyond shard rows')
        off = base[sh] + rin * rb
        brk = np.ones(uniq.size, bool)
        brk[1:] = (sh[1:] != sh[:-1]) | (off[1:] != off[:-1] + rb)
        starts = np.flatnonzero(brk)
        counts = np.diff(np.append(starts, uniq.size))
        rfd = np.ascontiguousarray(fds[sh[starts]])
        roff = np.ascontiguousarray(off[starts])
        rlen = np.ascontiguousarray(counts * rb)
        rout = np.ascontiguousarray(starts.astype(np.int64) * rb)
        buf = np.empty((uniq.size, dim), np.uint16)
        t0 = time.perf_counter()
        rc = _lib().ple_gather(rfd.ctypes.data, roff.ctypes.data, rlen.ctypes.data, rout.ctypes.data,
                               buf.ctypes.data, ctypes.c_int64(starts.size), ctypes.c_int(_threads(starts.size)))
        elapsed = time.perf_counter() - t0
        if rc != 0:
            raise OSError(f'native PLE gather failed rc={rc}')
        with reader._stats_lock:
            reader._requested_rows += int(ids.size)
            reader._unique_rows += int(uniq.size)
            reader._read_calls += int(starts.size)
            reader._read_bytes += int(rlen.sum())
            reader._gather_wall_seconds += elapsed
    return buf[inv]


def record(arm, case, rows, ms):
    global _LAST_CASE
    COUNTS[(case, arm, 'calls')] += 1
    COUNTS[(case, arm, 'rows')] += rows
    if rows >= 1024:
        print('DIRECT_PLE_AB_GATHER ' + json.dumps({'case': case, 'arm': arm, 'rows': rows, 'host_ms': ms,
                                                   'scope': 'host lookup wall incl. D2H ids/H2D rows; not GPU timing'},
                                                  sort_keys=True), flush=True)
    if case != _LAST_CASE and _LAST_CASE is not None:
        flush_counts(_LAST_CASE)
    _LAST_CASE = case


def flush_counts(case):
    summary = {f'{a}:{k}': v for (c, a, k), v in COUNTS.items() if c == case}
    print('DIRECT_PLE_AB_CASE_TOTALS ' + json.dumps({'case': case, 'totals': summary}, sort_keys=True), flush=True)
