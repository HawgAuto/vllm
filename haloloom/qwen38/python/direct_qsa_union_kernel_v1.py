"""Experimental single-request, native-unique-index QSA query reuse.
Not a model adapter. Only admit indices from the qualified unique QSA producer.
Repeated logical indices carry multiplicity in general attention and are NOT
supported by this private set-based prototype. No quantization/precision change.
All metadata is recomputed per invocation on the calling stream, graph-safe.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _scatter(I, R, B, SI: tl.constexpr, TOP: tl.constexpr,
             NB: tl.constexpr, LIMIT: tl.constexpr, QG: tl.constexpr,
             TILE: tl.constexpr):
    row = tl.program_id(0)
    j = tl.program_id(1) * TILE + tl.arange(0, TILE)
    idx = tl.load(I + row * SI + j, j < TOP, other=-1)
    req = tl.load(R + row)
    valid = (j < TOP) & (idx >= 0) & (idx < LIMIT) & (req == 0)
    block = tl.maximum(idx, 0) // 4
    bit = 1 << ((row % QG) * 4 + idx % 4)
    tl.atomic_or(B + (row // QG) * NB + block, bit, mask=valid,
                 sem="relaxed")


@triton.jit
def _scatter_lead(I, R, B, SI: tl.constexpr, TOP: tl.constexpr,
                  NB: tl.constexpr, LIMIT: tl.constexpr, QG: tl.constexpr,
                  TILE: tl.constexpr):
    # Same bitmap as _scatter (integer OR is associative/commutative/idempotent), with run-combined atomics:
    # entry i (block b) is a leader iff entry i-1 has a different block, or entries i-1..i-4 all have block b
    # (i is >= 4 into a run).  A leader ORs its own bit plus the bits of entries i+1..i+3 having block b, so leaders sit
    # at run offsets 0,4,8,.. and cover every valid entry; validity masks are exactly _scatter's.
    row = tl.program_id(0)
    j = tl.program_id(1) * TILE + tl.arange(0, TILE)
    req = tl.load(R + row)
    base = I + row * SI
    i0 = tl.load(base + j, j < TOP, other=-1)
    v0 = (j < TOP) & (i0 >= 0) & (i0 < LIMIT) & (req == 0)
    b0 = tl.where(v0, i0 // 4, -1)
    jm = j - 1
    im = tl.load(base + jm, (jm >= 0) & (jm < TOP), other=-1)
    b1 = tl.where((jm >= 0) & (jm < TOP) & (im >= 0) & (im < LIMIT) & (req == 0), im // 4, -1)
    jm = j - 2
    im = tl.load(base + jm, (jm >= 0) & (jm < TOP), other=-1)
    b2 = tl.where((jm >= 0) & (jm < TOP) & (im >= 0) & (im < LIMIT) & (req == 0), im // 4, -1)
    jm = j - 3
    im = tl.load(base + jm, (jm >= 0) & (jm < TOP), other=-1)
    b3 = tl.where((jm >= 0) & (jm < TOP) & (im >= 0) & (im < LIMIT) & (req == 0), im // 4, -1)
    jm = j - 4
    im = tl.load(base + jm, (jm >= 0) & (jm < TOP), other=-1)
    b4 = tl.where((jm >= 0) & (jm < TOP) & (im >= 0) & (im < LIMIT) & (req == 0), im // 4, -1)
    lead = v0 & ((b1 != b0) | ((b2 == b0) & (b3 == b0) & (b4 == b0)))
    sh = (row % QG) * 4
    bits = tl.where(v0, 1 << (sh + i0 % 4), 0)
    jp = j + 1
    ip = tl.load(base + jp, jp < TOP, other=-1)
    bits = bits | tl.where((jp < TOP) & (ip >= 0) & (ip < LIMIT) & (req == 0) & (ip // 4 == b0), 1 << (sh + ip % 4), 0)
    jp = j + 2
    ip = tl.load(base + jp, jp < TOP, other=-1)
    bits = bits | tl.where((jp < TOP) & (ip >= 0) & (ip < LIMIT) & (req == 0) & (ip // 4 == b0), 1 << (sh + ip % 4), 0)
    jp = j + 3
    ip = tl.load(base + jp, jp < TOP, other=-1)
    bits = bits | tl.where((jp < TOP) & (ip >= 0) & (ip < LIMIT) & (req == 0) & (ip // 4 == b0), 1 << (sh + ip % 4), 0)
    tl.atomic_or(B + (row // QG) * NB + tl.maximum(b0, 0), bits, mask=lead, sem="relaxed")


@triton.jit
def _count(B, C, NB: tl.constexpr, NT: tl.constexpr, TILE: tl.constexpr):
    group = tl.program_id(0)
    tile = tl.program_id(1)
    x = tile * TILE + tl.arange(0, TILE)
    v = tl.load(B + group * NB + x, x < NB, other=0)
    tl.store(C + group * NT + tile, tl.sum((v != 0).to(tl.int32)))


@triton.jit
def _compact(B, C, K, M, NB: tl.constexpr, NT: tl.constexpr,
             MAXU: tl.constexpr, CT: tl.constexpr, TILE: tl.constexpr):
    group = tl.program_id(0)
    tile = tl.program_id(1)
    x = tile * TILE + tl.arange(0, TILE)
    v = tl.load(B + group * NB + x, x < NB, other=0)
    t = tl.arange(0, CT)
    counts = tl.load(C + group * NT + t, t < NT, other=0)
    offset = tl.sum(tl.where(t < tile, counts, 0))
    rank = offset + tl.cumsum((v != 0).to(tl.int32)) - 1
    valid = (x < NB) & (v != 0) & (rank < MAXU)
    tl.store(K + group * MAXU + rank, x, valid)
    tl.store(M + group * MAXU + rank, v, valid)


@triton.jit
def _attend(Q, K, V, TAB, U, MASK, COUNTS, O,
            SQ0: tl.constexpr, SQ1: tl.constexpr,
            SK0: tl.constexpr, SK1: tl.constexpr, SK2: tl.constexpr,
            SV0: tl.constexpr, SV1: tl.constexpr, SV2: tl.constexpr,
            SO0: tl.constexpr, SO1: tl.constexpr,
            NR: tl.constexpr, NP: tl.constexpr, PAGE: tl.constexpr,
            TW: tl.constexpr, MAXU: tl.constexpr, NT: tl.constexpr,
            CT: tl.constexpr, QG: tl.constexpr, GH: tl.constexpr,
            PH: tl.constexpr, HD: tl.constexpr, BM: tl.constexpr,
            BN: tl.constexpr):
    group = tl.program_id(0)
    kvhead = tl.program_id(1)
    h = tl.arange(0, BM)
    d = tl.arange(0, HD)
    n = tl.arange(0, BN)
    qslot = h // PH
    row = group * QG + qslot
    head = kvhead * GH + h % PH
    active = (row < NR) & (h % PH < GH)
    query = tl.load(Q + row[:, None] * SQ0 + head[:, None] * SQ1 + d[None, :],
                    active[:, None], other=0)
    t = tl.arange(0, CT)
    counts = tl.load(COUNTS + group * NT + t, t < NT, other=0)
    count = tl.minimum(tl.sum(counts), MAXU)
    maximum = tl.full((BM,), -1.0e20, tl.float32)
    norm = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, HD), tl.float32)
    scale: tl.constexpr = (HD ** -0.5) * 1.4426950408889634
    for tile in range(tl.cdiv(count * 4, BN)):
        col = tile * BN + n
        ui = col // 4
        block = tl.load(U + group * MAXU + ui, ui < count, other=0)
        bits = tl.load(MASK + group * MAXU + ui, ui < count, other=0)
        logical = block * 4 + col % 4
        pg = logical // PAGE
        off = logical % PAGE
        valid = (ui < count) & (pg < TW)
        physical = tl.load(TAB + pg, valid, other=-1)
        valid &= (physical >= 0) & (physical < NP)
        phys = tl.maximum(physical, 0).to(tl.int64)
        keys = tl.load(K + phys[None, :] * SK0 + off[None, :] * SK1
                        + kvhead * SK2 + d[:, None], valid[None, :], other=0)
        values = tl.load(V + phys[:, None] * SV0 + off[:, None] * SV1
                          + kvhead * SV2 + d[None, :], valid[:, None], other=0)
        membership = ((bits[None, :] >> (qslot[:, None] * 4 + col[None, :] % 4)) & 1) != 0
        use = active[:, None] & valid[None, :] & membership
        scores = tl.dot(query, keys) * scale
        scores = tl.where(use, scores, -1.0e20)
        newmax = tl.maximum(maximum, tl.max(scores, 1))
        alpha = tl.exp2(maximum - newmax)
        p = tl.where(use, tl.exp2(scores - newmax[:, None]), 0.0)
        acc = tl.dot(p.to(values.dtype), values, acc=acc * alpha[:, None])
        norm = norm * alpha + tl.sum(p, 1)
        maximum = newmax
    value = tl.where(norm[:, None] > 0, acc / tl.maximum(norm[:, None], 1.0e-20), 0.0)
    tl.store(O + row[:, None] * SO0 + head[:, None] * SO1 + d[None, :],
             value, active[:, None])


def _contract(q, k_cache, v_cache, logical_indices, block_table, token_to_req, out, query_group):
    assert query_group in (2, 4)
    assert q.is_cuda and q.dtype == k_cache.dtype == v_cache.dtype == torch.bfloat16
    assert q.ndim == 3 and k_cache.ndim == v_cache.ndim == 4
    assert k_cache.shape == v_cache.shape and q.shape[2] == k_cache.shape[3] == 256
    assert q.shape[1] == k_cache.shape[2] * 12
    assert block_table.ndim == 2 and block_table.shape[0] == 1 and block_table.shape[1] > 0
    assert block_table.stride(1) == 1 and block_table.dtype == torch.int32
    assert logical_indices.ndim == 2 and logical_indices.shape[0] == q.shape[0]
    assert logical_indices.shape[1] > 0 and logical_indices.stride(1) == 1
    assert logical_indices.dtype in (torch.int32, torch.int64)
    assert token_to_req.shape == (q.shape[0],) and token_to_req.stride(0) == 1
    assert token_to_req.dtype == torch.int32 and out.shape == q.shape and out.dtype == q.dtype
    assert all(t.device == q.device for t in (k_cache,v_cache,logical_indices,block_table,token_to_req,out))
    assert all(t.stride(-1) == 1 for t in (q,k_cache,v_cache,out))
    assert k_cache.shape[0] > 0 and k_cache.shape[1] > 0


def build_union(logical_indices, token_to_req, capacity, query_group, lead=False):
    """Private metadata construction; caller has validated the tensor contract."""
    rows, top = logical_indices.shape
    groups = triton.cdiv(rows, query_group)
    nb = triton.cdiv(capacity, 4)
    nt = triton.cdiv(nb, 1024)
    maxu = min(nb, query_group * top)  # also safe for non-grouped unique indices
    bitmap = torch.zeros((groups, nb), device=logical_indices.device, dtype=torch.int32)
    counts = torch.empty((groups, nt), device=logical_indices.device, dtype=torch.int32)
    keys = torch.empty((groups, maxu), device=logical_indices.device, dtype=torch.int32)
    masks = torch.empty_like(keys)
    if rows:
        (_scatter_lead if lead else _scatter)[(rows, triton.cdiv(top, 256))](logical_indices, token_to_req, bitmap,
            logical_indices.stride(0), top, nb, capacity, query_group, 256)
        _count[(groups, nt)](bitmap, counts, nb, nt, 1024)
        _compact[(groups, nt)](bitmap, counts, keys, masks, nb, nt, maxu,
                              triton.next_power_of_2(nt), 1024)
    return keys, masks, counts


# ---- DIRECT QSA capacity bound (overlay nb2-qsacap-ab-v2: + power-of-two bucket + run-combined _scatter_lead) ----
# build_union sized its [groups, cdiv(capacity,4)] bitmap by the request's full KV capacity (page 1568 x 84 = 131712 tokens),
# zeroing and scanning 135 MB per layer even for a 4096-token context.  Every valid logical index is a causal context
# position < max_seq_len, so bounding capacity to the context sets exactly the same bits; _compact emits keys in ascending
# block order independent of the tile count, and the attend kernels use NT/MAXU only as a count bound and row stride, so
# the per-group key/mask lists (and therefore attention) are unchanged.  First bounded use per process builds BOTH unions
# and asserts identical totals and key/mask prefixes.  On when DIRECT_QSA_CAP=1, else when direct_nb1.nb_level() >= 5.
_QC_CHECKED = [False]
_QC_LOGGED = set()


def _qc_log(tag, **kw):
    if tag in _QC_LOGGED:
        return
    _QC_LOGGED.add(tag)
    print('DIRECT_QSA_CAP ' + _qn_json.dumps(dict(tag=tag, pid=_qn_os.getpid(), **kw), sort_keys=True, default=str), flush=True)


def _qc_on():
    f = _qn_os.environ.get('DIRECT_QSA_CAP')
    if f is not None:
        return f == '1'
    try:
        import direct_nb1 as _nb
        return _nb.nb_level() >= 5
    except Exception:
        return False


def _qc_selfcheck(logical_indices, token_to_req, full_cap, cap, query_group, bounded):
    ub, mb, cb = bounded
    uf, mf, cf = build_union(logical_indices, token_to_req, full_cap, query_group)  # original _scatter, full capacity
    tb = cb.sum(1); tf = cf.sum(1)
    ok = bool(torch.equal(tb, tf)) and int(tf.max()) <= ub.shape[1]
    if ok:
        live = torch.arange(ub.shape[1], device=ub.device).view(1, -1) < tb.view(-1, 1)
        ok = (bool(torch.equal(ub[live], uf[:, :ub.shape[1]][live]))
              and bool(torch.equal(mb[live], mf[:, :ub.shape[1]][live])))
    _qc_log('selfcheck', equal=ok, rows=int(logical_indices.shape[0]), full_cap=int(full_cap), cap=int(cap),
            nb_full=int(uf.shape[1]), maxu_bounded=int(ub.shape[1]), nt_full=int(cf.shape[1]), nt_bounded=int(cb.shape[1]),
            max_total=int(tf.max()))
    if not ok:
        raise RuntimeError('DIRECT_QSA_CAP selfcheck mismatch: bounded union != full-capacity union')
    _QC_CHECKED[0] = True


def native_unique_attention(q, k_cache, v_cache, logical_indices, block_table,
                            token_to_req, out, *, query_group=4, context_len=None):
    _contract(q,k_cache,v_cache,logical_indices,block_table,token_to_req,out,query_group)
    if q.shape[0] == 0:
        return out
    full_cap = k_cache.shape[1] * block_table.shape[1]
    cap = full_cap
    if isinstance(context_len, int) and 0 < context_len < full_cap and _qc_on():
        # v2: power-of-two capacity bucket (>= context, >= 1024): NB/LIMIT/NT/MAXU are tl.constexpr, so an exact
        # per-request capacity would JIT new kernel variants for every context length.
        cap = min(full_cap, max(1024, 1 << (int(context_len) - 1).bit_length()))
    if _un_on() and _un_ok(logical_indices, token_to_req, cap, query_group):  # [nb2-union1-ab-v1] exact native union
        u, mask, counts = _un_build(logical_indices, token_to_req, cap, query_group, lead=cap != full_cap)
    else:
        u, mask, counts = build_union(logical_indices, token_to_req, cap, query_group, lead=cap != full_cap)
    if cap != full_cap:
        _qc_log('first_bounded', cap=int(cap), full_cap=int(full_cap), rows=int(q.shape[0]))
        if not _QC_CHECKED[0]:
            _qc_selfcheck(logical_indices, token_to_req, full_cap, cap, query_group, (u, mask, counts))
    nt = counts.shape[1]
    if query_group == 4 and _qn_on(q.shape[0]) and _qn_contract(q, k_cache, v_cache, block_table):
        g = _gf_take(out, q.shape[0])
        _qn_run(q, k_cache, v_cache, block_table, u, mask, counts, out, nt, gate=g)
        return out
    _attend[(u.shape[0], k_cache.shape[2])](q,k_cache,v_cache,block_table,u,mask,counts,out,
        *q.stride()[:2], *k_cache.stride()[:3], *v_cache.stride()[:3], *out.stride()[:2],
        q.shape[0],k_cache.shape[0],k_cache.shape[1],block_table.shape[1],u.shape[1],nt,
        triton.next_power_of_2(nt),query_group,12,16,256,query_group*16,64,
        num_warps=4 if query_group==2 else 8, num_stages=1)
    return out


# ---- DIRECT QSA NATIVE A/B path (overlay moe-wmma-unified-v7): faithful gfx1151 WMMA port of _attend ----
# qsa-native-v1/qsa_native_v2.hip, gated bitwise vs this module's _attend (qsa-native-v2-gate-01: 4096/10976 tokens).
import os as _qn_os, json as _qn_json, ctypes as _qn_ct
_QN_HSACO = _qn_os.environ.get('DIRECT_QSA_NATIVE_HSACO', '/opt/venv/lib/python3.14/site-packages/direct_qsa_native_v2.hsaco')
_QN_FN = _qn_os.environ.get('DIRECT_QSA_NATIVE_FN', 'qsa_attend_v2')
_QN_MIN_T = int(_qn_os.environ.get('DIRECT_QSA_NATIVE_MIN_TOKENS', '256'))
_QN_MOD = None
_QN_CHECKED = False
_QN_LOGGED = set()
_QN_COUNT = [0]


def _qn_log(tag, **kw):
    if tag in _QN_LOGGED:
        return
    _QN_LOGGED.add(tag)
    print('DIRECT_QSA_NATIVE ' + _qn_json.dumps(dict(tag=tag, pid=_qn_os.getpid(), **kw), sort_keys=True, default=str), flush=True)


def _qn_on(t):
    force = _qn_os.environ.get('DIRECT_QSA_NATIVE')
    if force is not None:
        return force == '1' and t >= _QN_MIN_T
    if t < _QN_MIN_T:
        return False
    import direct_moe_ab_adapter_v1 as _ab
    ph = _ab.read_phase()
    return ph is not None and ph['arm'] == 'wmma5'


def _qn_contract(q, kc, vc, bt):
    ok = (q.shape[1] == kc.shape[2] * 12 and q.shape[2] == 256 and kc.shape[2] == 2
          and all(t.data_ptr() % 16 == 0 for t in (q, kc, vc))
          and q.stride(0) % 8 == 0 and q.stride(1) % 8 == 0
          and all(s % 8 == 0 for s in kc.stride()[:3]) and all(s % 8 == 0 for s in vc.stride()[:3])
          and bt.shape[0] == 1 and bt.stride(1) == 1)
    if not ok:
        _qn_log('contract_fallback', q=list(q.shape), qs=list(q.stride()), ks=list(kc.stride()), vs=list(vc.stride()))
    return ok


def _qn_launch(q, kc, vc, bt, u, mask, counts, out, nt, gate=None):
    global _QN_MOD, _GF_MOD
    import direct_hip_module_v1 as hm
    if gate is not None:
        if _GF_MOD is None:
            _GF_MOD = hm.Module(_GF_HSACO)
            _qn_log('gate_module', hsaco=_GF_HSACO, fn=_GF_FN, attrs=_GF_MOD.attrs(_GF_FN))
        I64 = _qn_ct.c_int64; I32 = _qn_ct.c_int32
        _GF_MOD.launch(_GF_FN, (u.shape[0], kc.shape[2], 1), (384, 1, 1),
            [hm.ptr(q), hm.ptr(kc), hm.ptr(vc), hm.ptr(bt), hm.ptr(u), hm.ptr(mask), hm.ptr(counts), hm.ptr(out),
             I64(q.stride(0)), I64(q.stride(1)), I64(kc.stride(0)), I64(kc.stride(1)), I64(kc.stride(2)),
             I64(vc.stride(0)), I64(vc.stride(1)), I64(vc.stride(2)), I64(out.stride(0)), I64(out.stride(1)),
             I32(q.shape[0]), I32(kc.shape[0]), I32(kc.shape[1]), I32(bt.shape[1]), I32(u.shape[1]), I32(nt),
             hm.ptr(gate), I64(gate.stride(0)), I64(gate.stride(1))],
            stream=torch.cuda.current_stream().cuda_stream)
        return
    if _QN_MOD is None:
        _QN_MOD = hm.Module(_QN_HSACO)
        _qn_log('module', hsaco=_QN_HSACO, fn=_QN_FN, attrs=_QN_MOD.attrs(_QN_FN))
    assert u.dtype == mask.dtype == counts.dtype == bt.dtype == torch.int32
    assert u.is_contiguous() and mask.is_contiguous() and counts.is_contiguous()
    I64 = _qn_ct.c_int64; I32 = _qn_ct.c_int32
    _QN_MOD.launch(_QN_FN, (u.shape[0], kc.shape[2], 1), (384, 1, 1),
        [hm.ptr(q), hm.ptr(kc), hm.ptr(vc), hm.ptr(bt), hm.ptr(u), hm.ptr(mask), hm.ptr(counts), hm.ptr(out),
         I64(q.stride(0)), I64(q.stride(1)), I64(kc.stride(0)), I64(kc.stride(1)), I64(kc.stride(2)),
         I64(vc.stride(0)), I64(vc.stride(1)), I64(vc.stride(2)), I64(out.stride(0)), I64(out.stride(1)),
         I32(q.shape[0]), I32(kc.shape[0]), I32(kc.shape[1]), I32(bt.shape[1]), I32(u.shape[1]), I32(nt)],
        stream=torch.cuda.current_stream().cuda_stream)


def _qn_run(q, kc, vc, bt, u, mask, counts, out, nt, gate=None):
    global _QN_CHECKED, _GF_CHECKED
    if gate is not None:
        if not _GF_CHECKED:
            ref = torch.empty_like(out)
            _qn_run(q, kc, vc, bt, u, mask, counts, ref, nt)          # ungated native (+ its own v3g-vs-_attend selfcheck)
            gf_apply_(ref, gate)                                       # verbatim inductor mul_sigmoid body
            out.fill_(float('nan'))
            _qn_launch(q, kc, vc, bt, u, mask, counts, out, nt, gate=gate)
            eq = bool(torch.equal(out, ref))
            _qn_log('gate_selfcheck', tokens=int(q.shape[0]), equal=eq, nan_left=int(torch.isnan(out).sum()))
            if not eq:
                raise RuntimeError('DIRECT_QSA_GATE_FUSE selfcheck mismatch (v3k gated vs native+ref gate)')
            _GF_CHECKED = True
            return
        _qn_launch(q, kc, vc, bt, u, mask, counts, out, nt, gate=gate)
        _GF_COUNT[0] += 1
        if _GF_COUNT[0] in (1, 100):
            _qn_log('gate_used_%d' % _GF_COUNT[0], tokens=int(q.shape[0]))
        return
    _QN_COUNT[0] += 1
    if not _QN_CHECKED:
        ref = torch.empty_like(out)
        _attend[(u.shape[0], kc.shape[2])](q, kc, vc, bt, u, mask, counts, ref,
            *q.stride()[:2], *kc.stride()[:3], *vc.stride()[:3], *ref.stride()[:2],
            q.shape[0], kc.shape[0], kc.shape[1], bt.shape[1], u.shape[1], nt,
            triton.next_power_of_2(nt), 4, 12, 16, 256, 64, 64, num_warps=8, num_stages=1)
        _qn_launch(q, kc, vc, bt, u, mask, counts, out, nt)
        eq = bool(torch.equal(out, ref))
        _qn_log('selfcheck', tokens=int(q.shape[0]), equal=eq, maxu=int(u.shape[1]), nt=int(nt))
        if not eq:
            raise RuntimeError('DIRECT_QSA_NATIVE selfcheck mismatch vs Triton _attend')
        _QN_CHECKED = True
        return
    _qn_launch(q, kc, vc, bt, u, mask, counts, out, nt)
    if _QN_COUNT[0] in (2, 100):
        _qn_log('used_%d' % _QN_COUNT[0], tokens=int(q.shape[0]))


# ---- nb2-gatefuse-v1: QSA output gate ownership (op function opens a ctx; native union path may consume rows [0:n]) ----
_GF_HSACO = _qn_os.environ.get('DIRECT_QSA_GATE_HSACO', '/opt/venv/lib/python3.14/site-packages/direct_qsa_native_v3k.hsaco')
_GF_FN = _qn_os.environ.get('DIRECT_QSA_GATE_FN', 'qsa_attend_v3k_gate')
_GF_MOD = None
_GF_CHECKED = False
_GF_COUNT = [0]
_GF_CTX = [None]


@triton.jit
def _gf_ref(O, G, N, SG0, SG1, XBLOCK: tl.constexpr):
    # body copied from the production inductor kernel triton_poi_fused_mul_sigmoid_slice_view_0 (in place)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < N
    x3 = xindex
    x0 = (xindex % 256)
    x1 = ((xindex // 256) % 24)
    x2 = xindex // 6144
    tmp0 = tl.load(O + (x3), xmask).to(tl.float32)
    tmp1 = tl.load(G + (x0 + SG1 * x1 + SG0 * x2), xmask).to(tl.float32)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tl.store(O + (x3), tmp3, xmask)


def gf_apply_(o, g):
    assert o.dim() == 3 and tuple(o.shape[1:]) == (24, 256) and o.is_contiguous() and o.dtype == torch.bfloat16
    assert g.shape == o.shape and g.stride(2) == 1 and g.dtype == torch.bfloat16
    n = o.numel()
    if n:
        _gf_ref[(triton.cdiv(n, 1024),)](o, g, n, g.stride(0), g.stride(1), XBLOCK=1024, num_warps=4)
    return o


def gf_begin(output, gate):
    if _GF_CTX[0] is not None:
        raise RuntimeError('DIRECT_QSA_GATE_FUSE nested gate ctx')
    ctx = {'out_ptr': output.data_ptr(), 'out_stride': tuple(output.stride()), 'gate': gate, 'consumed': 0, 'takes': 0}
    _GF_CTX[0] = ctx
    return ctx


def gf_end(ctx):
    assert _GF_CTX[0] is ctx
    _GF_CTX[0] = None
    return ctx['consumed']


def _gf_take(out, rows):
    ctx = _GF_CTX[0]
    if ctx is None:
        return None
    ctx['takes'] += 1
    g = ctx['gate']
    ok = (ctx['consumed'] == 0 and ctx['takes'] == 1 and out.data_ptr() == ctx['out_ptr']
          and tuple(out.stride()) == ctx['out_stride'] and out.dim() == 3 and rows <= g.shape[0]
          and tuple(out.shape[1:]) == tuple(g.shape[1:]) == (24, 256) and g.stride(2) == 1
          and g.data_ptr() % 16 == 0 and g.stride(0) % 8 == 0 and g.stride(1) % 8 == 0 and g.dtype == out.dtype)
    if not ok:
        _qn_log('gate_take_refused', rows=int(rows), takes=ctx['takes'], gs=list(g.stride()), os=list(out.stride()))
        return None
    ctx['consumed'] = int(rows)
    return g[:rows]


# ---- nb2-union1-ab-v1: exact native union build (qsa-native-v1/qsa_union_build_v1.hip) ----
# One workgroup per 4-row group: LDS 1-bit block-presence map + popcount prefix scan + per-rank LDS mask OR.  Same
# validity/bit semantics as _scatter/_scatter_lead, same per-1024-block tile counts as _count, same ascending ranks and
# MAXU truncation as _compact => keys/masks (ranks < count) and counts are bit-identical (operator gate union1-gate-01),
# with no [groups, cap/4] bitmap memset/scan.  First native use per process also builds the original union on the same
# inputs and raises on any difference.  On when DIRECT_QSA_UNION_NATIVE=1 (=0 off), else when direct_nb1.nb_level() >= 7.
_UN_HSACO = _qn_os.environ.get('DIRECT_QSA_UNION_HSACO', '/opt/venv/lib/python3.14/site-packages/direct_qsa_union_build_v1.hsaco')
_UN_MIN_CAP = int(_qn_os.environ.get('DIRECT_QSA_UNION_MIN_CAP', '8192'))  # gate union1-gate-02: c2 cap 4096 is 0.94x, keep original
_UN_MOD = None
_UN_CHECKED = [False]
_UN_COUNT = [0]


def _un_on():
    f = _qn_os.environ.get('DIRECT_QSA_UNION_NATIVE')
    if f is not None:
        return f == '1'
    try:
        import direct_nb1 as _nb
        return _nb.nb_level() >= 7
    except Exception:
        return False


def _un_ok(logical_indices, token_to_req, capacity, query_group):
    rows, top = logical_indices.shape
    nb = triton.cdiv(capacity, 4)
    maxu = min(nb, query_group * top)
    if capacity < _UN_MIN_CAP:
        return False
    ok = (query_group == 4 and rows > 0 and nb <= 33792 and maxu <= 12288
          and logical_indices.dtype in (torch.int32, torch.int64) and logical_indices.stride(1) == 1
          and token_to_req.dtype == torch.int32 and token_to_req.is_contiguous()
          and not torch.cuda.is_current_stream_capturing())
    if not ok:
        _qn_log('union_native_fallback', rows=int(rows), top=int(top), nb=int(nb), maxu=int(maxu), qg=int(query_group),
                dtype=str(logical_indices.dtype))
    return ok


def _un_launch(logical_indices, token_to_req, capacity, query_group):
    global _UN_MOD
    import direct_hip_module_v1 as hm
    if _UN_MOD is None:
        _UN_MOD = hm.Module(_UN_HSACO)
        _qn_log('union_module', hsaco=_UN_HSACO, attrs=_UN_MOD.attrs('qsa_union_build_v1_i32'))
    rows, top = logical_indices.shape
    groups = triton.cdiv(rows, query_group)
    nb = triton.cdiv(capacity, 4)
    nt = triton.cdiv(nb, 1024)
    maxu = min(nb, query_group * top)
    keys = torch.empty((groups, maxu), device=logical_indices.device, dtype=torch.int32)
    masks = torch.empty_like(keys)
    counts = torch.empty((groups, nt), device=logical_indices.device, dtype=torch.int32)
    fn = 'qsa_union_build_v1_i64' if logical_indices.dtype == torch.int64 else 'qsa_union_build_v1_i32'
    _UN_MOD.launch(fn, (groups, 1, 1), (256, 1, 1),
        [hm.ptr(logical_indices), _qn_ct.c_longlong(logical_indices.stride(0)), hm.ptr(token_to_req), _qn_ct.c_int(rows),
         _qn_ct.c_int(top), _qn_ct.c_longlong(capacity), _qn_ct.c_int(nb), _qn_ct.c_int(nt), _qn_ct.c_int(maxu),
         hm.ptr(keys), hm.ptr(masks), hm.ptr(counts)],
        stream=torch.cuda.current_stream().cuda_stream)
    return keys, masks, counts


def _un_build(logical_indices, token_to_req, capacity, query_group, lead=False):
    k, m, c = _un_launch(logical_indices, token_to_req, capacity, query_group)
    _UN_COUNT[0] += 1
    if not _UN_CHECKED[0]:
        uk, um, uc = build_union(logical_indices, token_to_req, capacity, query_group, lead=lead)
        ok = bool(torch.equal(uc, c))
        if ok:
            tot = torch.clamp(uc.sum(1), max=uk.shape[1])
            live = torch.arange(uk.shape[1], device=uk.device).view(1, -1) < tot.view(-1, 1)
            ok = bool(torch.equal(uk[live], k[live])) and bool(torch.equal(um[live], m[live]))
        _qn_log('union_selfcheck', equal=ok, rows=int(logical_indices.shape[0]), capacity=int(capacity), lead=bool(lead),
                maxu=int(k.shape[1]), nt=int(c.shape[1]), max_total=int(uc.sum(1).max()))
        if not ok:
            raise RuntimeError('DIRECT_QSA_UNION_NATIVE selfcheck mismatch vs build_union')
        _UN_CHECKED[0] = True
    elif _UN_COUNT[0] in (2, 100):
        _qn_log('union_used_%d' % _UN_COUNT[0], rows=int(logical_indices.shape[0]), capacity=int(capacity))
    return k, m, c
