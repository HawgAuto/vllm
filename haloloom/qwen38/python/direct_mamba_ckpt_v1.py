"""Direct campaign: internal Mamba prefill checkpoints for Qwen4Exp GDN + PLE short-conv (AMD).

Removes the align-mode tail split ("stop at the last block boundary so the recurrent state
there can be cached") by exporting that boundary state from inside the single prefill step.

Upstream contract (vllm.v1.kv_cache_interface / layers/mamba/checkpoint.py):
  * MambaSpec.num_prefill_checkpoint_blocks > 0 on EVERY Mamba spec -> scheduler stops splitting
    the last prefill chunk; MambaManager reserves block column cdiv(seq_len, B) - 2 and registers
    it as the cached state at position get_mamba_prefill_checkpoint_position(seq_len, B).
  * Backends must write exactly those blocks, computed with compute_mamba_prefill_checkpoints so
    they decline in lockstep with the scheduler and manager.

GDN: chunk-aligned restart (a = off - off % 64): [qs, qs+a) -> exact state at a; [qs+a, qs+off) from it
-> exact checkpoint state (written to the checkpoint block in the cache dtype, like the split path's
state write); [qs+a, qe) from it -> outputs + final state. Bitwise-equivalent to the one-call prefill. Conv state = last conv_state.shape[-1] raw
(pre-conv) inputs before the checkpoint. PLE: dilated short-conv history slice before the checkpoint.
Only the GDN/PLE layers change; attention/MoE/dense run once over the full chunk.

Gated by DIRECT_MAMBA_PREFILL_CHECKPOINT=1; v1 supports non-speculative (no MTP) serving only.
"""
from __future__ import annotations

import dataclasses
import json
import os
import sys
from dataclasses import dataclass

import torch

ENV = 'DIRECT_MAMBA_PREFILL_CHECKPOINT'
ALIGNMENT = 16
STATS = {'gdn_builds': 0, 'ple_builds': 0, 'gdn_layer_exports': 0, 'ple_layer_exports': 0,
         'gdn_rows': 0, 'ple_rows': 0}
_LOGGED = {'spec': 0}


def enabled() -> bool:
    value = os.environ.get(ENV, '0')
    if value not in ('0', '1'):
        raise ValueError(f'{ENV} must be 0 or 1')
    return value == '1'


def _log(tag: str, payload: dict) -> None:
    print(f'{tag} ' + json.dumps(payload, sort_keys=True), file=sys.stderr, flush=True)


def maybe_checkpoint_spec(spec, vllm_config):
    """Add one prefill checkpoint block to a Qwen4Exp GDN/PLE MambaSpec when enabled."""
    from vllm.v1.kv_cache_interface import MambaSpec
    if not enabled() or not isinstance(spec, MambaSpec):
        return spec
    if vllm_config.cache_config.mamba_cache_mode != 'align':
        raise ValueError(f'{ENV}=1 requires --mamba-cache-mode align')
    if vllm_config.num_speculative_tokens:
        raise ValueError(f'{ENV}=1 v1 does not support speculative decoding (MTP)')
    out = dataclasses.replace(spec, num_prefill_checkpoint_blocks=1, prefill_checkpoint_alignment=ALIGNMENT)
    if _LOGGED['spec'] < 2:
        _LOGGED['spec'] += 1
        _log('DIRECT_MAMBA_CKPT_SPEC', {'mamba_type': str(spec.mamba_type), 'block_size': spec.block_size,
                                        'alignment': ALIGNMENT, 'shapes': [list(s) for s in spec.shapes],
                                        'dtypes': [str(d) for d in spec.dtypes]})
    return out


def _row_checkpoints(builder, m):
    """Upstream-identical per-row checkpoint offsets/cols for all rows of this group."""
    from vllm.model_executor.layers.mamba.checkpoint import compute_mamba_prefill_checkpoints
    spec = builder.kv_cache_spec
    vcfg = builder.vllm_config
    assert m.seq_lens_cpu_upper_bound is not None
    query_lens = m.query_start_loc_cpu.diff().tolist()
    seq_lens = m.seq_lens_cpu_upper_bound.tolist()[: m.num_reqs]
    hash_block_size = vcfg.cache_config.prefix_match_unit or spec.block_size
    sc = vcfg.speculative_config
    drop = sc is not None and sc.use_eagle_block_drop()
    return compute_mamba_prefill_checkpoints(seq_lens, query_lens[: m.num_reqs], hash_block_size=hash_block_size,
                                             mamba_block_size=spec.block_size,
                                             checkpoint_alignment=spec.prefill_checkpoint_alignment,
                                             drop_eagle_block=drop)


def _prefill_rows(builder, m, num_prefills: int, has_spec: bool):
    """Return (rows, offsets, cols) for rows that need a checkpoint, validated against the prefill tail."""
    if builder.kv_cache_spec.num_prefill_checkpoint_blocks == 0 or num_prefills == 0:
        return None
    offsets, cols = _row_checkpoints(builder, m)
    rows = [r for r, o in enumerate(offsets) if o > 0]
    if not rows:
        return None
    if has_spec:
        raise RuntimeError('direct mamba checkpoint: spec-decode batch with checkpoint rows (unsupported in v1)')
    first_prefill = m.num_reqs - num_prefills
    if any(r < first_prefill for r in rows):
        raise RuntimeError(f'direct mamba checkpoint: checkpoint row outside prefill tail {rows} first={first_prefill}')
    return rows, [offsets[r] for r in rows], [cols[r] for r in rows], first_prefill


def _state_indices(m, rows, cols, device):
    from vllm.utils.torch_utils import async_tensor_h2d
    from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
    r = async_tensor_h2d(rows, dtype=torch.int64, device=device)
    c = async_tensor_h2d(cols, dtype=torch.int64, device=device)
    idx = m.block_table_tensor[r, c].to(torch.int64)
    return idx, NULL_BLOCK_ID


@dataclass
class GDNCall:
    cu: torch.Tensor               # int32 cu_seqlens (device)
    ci: torch.Tensor | None        # FLA chunk indices
    co: torch.Tensor | None        # FLA chunk offsets
    idx: torch.Tensor | None       # prefill-local token indices (None on the single-row slice path)
    lo: int = 0                    # single-row slice [lo, hi)
    hi: int = 0


@dataclass
class GDNCheckpoint:
    """Chunk-aligned restart plan, bitwise-equivalent to the one-call (no-cache) GDN prefill.

    a_j = off_j - off_j % CHUNK.
      call1: [qs, qs + a_j) for checkpoint rows, whole query for others  -> exact state at a_j
      call2: [qs + a_j, qs + off_j) from that state (only rows with a_j < off_j) -> exact checkpoint
      call3: [qs + a_j, qe) from that state -> outputs for [a_j, len) and the row's final state
    """
    single: bool
    num_prefill_tokens: int
    ck_rows: torch.Tensor          # prefill-local rows with a checkpoint (int64)
    state_idx: torch.Tensor        # checkpoint block ids (int64)
    conv_end_pos: torch.Tensor     # checkpoint token index in the non-spec token stream (int64)
    call1: GDNCall
    call3: GDNCall
    call2: GDNCall | None
    call2_pos: torch.Tensor | None  # positions within ck_rows covered by call2 (int64)
    summary: dict


def make_gdn_plan(pq, off_by_local, num_decode_tokens, device, chunk_md, h2d, state_idx):
    """Pure planner shared by the GDN builder and the GPU unit gate.

    pq: prefill-local query_start_loc (python list, len num_prefills + 1); off_by_local: {row: offset}.
    chunk_md(cu_dev, cu_cpu) -> (chunk_indices, chunk_offsets); h2d(list, dtype) -> device tensor.
    """
    from vllm.third_party.flash_linear_attention.ops.utils import FLA_CHUNK_SIZE as CHUNK
    n = len(pq) - 1
    seg1, seg2, seg3, idx1, idx2, idx3, ckl, pos2 = [], [], [], [], [], [], [], []
    for j in range(n):
        qs, qe = pq[j], pq[j + 1]
        o = off_by_local.get(j, 0)
        if not o:
            seg1.append(qe - qs); idx1.extend(range(qs, qe)); continue
        if not (CHUNK <= o < qe - qs):
            raise RuntimeError(f'direct gdn checkpoint: bad offset {o} for query {qe - qs}')
        a = o - o % CHUNK
        seg1.append(a); idx1.extend(range(qs, qs + a))
        seg3.append(qe - qs - a); idx3.extend(range(qs + a, qe))
        if a < o:
            pos2.append(len(ckl)); seg2.append(o - a); idx2.extend(range(qs + a, qs + o))
        ckl.append(j)

    def mk(segs, idx, lo=0, hi=0, single=False):
        cu = [0]
        for s in segs:
            cu.append(cu[-1] + s)
        cu_cpu = torch.tensor(cu, dtype=torch.int32)
        cu_dev = h2d(cu, torch.int32)
        ci, co = chunk_md(cu_dev, cu_cpu)
        return GDNCall(cu=cu_dev, ci=ci, co=co, idx=None if single else h2d(idx, torch.int64), lo=lo, hi=hi)

    single = n == 1
    if single:
        o = off_by_local[0]; a = o - o % CHUNK; T = pq[1]
        c1 = mk(seg1, None, 0, a, True); c3 = mk(seg3, None, a, T, True)
        c2 = mk(seg2, None, a, o, True) if seg2 else None
    else:
        c1 = mk(seg1, idx1); c3 = mk(seg3, idx3); c2 = mk(seg2, idx2) if seg2 else None
    summary = {'seg1': seg1, 'seg2': seg2, 'seg3': seg3, 'single': single, 'chunk': CHUNK}
    return GDNCheckpoint(single=single, num_prefill_tokens=pq[-1], ck_rows=h2d(ckl, torch.int64), state_idx=state_idx,
                         conv_end_pos=h2d([num_decode_tokens + pq[j] + off_by_local[j] for j in ckl], torch.int64),
                         call1=c1, call3=c3, call2=c2, call2_pos=h2d(pos2, torch.int64) if seg2 else None,
                         summary=summary)


@dataclass
class PLECheckpoint:
    ck_rows: torch.Tensor          # prefill-local rows (int64, device)
    offsets: torch.Tensor          # checkpoint offset within each row's query (int64, device)
    state_idx: torch.Tensor        # checkpoint block ids (int64, device)
    summary: dict


def build_gdn(builder, m, md):
    """Attach two-call checkpoint metadata to a GDNAttentionMetadata (non-spec path)."""
    from vllm.utils.torch_utils import async_tensor_h2d
    sel = _prefill_rows(builder, m, md.num_prefills, md.spec_sequence_masks is not None)
    if sel is None:
        return None
    rows, offs, cols, first_prefill = sel
    device = m.query_start_loc.device
    qsl = m.query_start_loc_cpu[: m.num_reqs + 1].tolist()
    base = qsl[first_prefill]
    pq = [x - base for x in qsl[first_prefill:]]
    num_prefills = md.num_prefills
    assert len(pq) == num_prefills + 1 and pq[-1] == md.num_prefill_tokens, (pq[-1], md.num_prefill_tokens)
    off_by_local = {r - first_prefill: o for r, o in zip(rows, offs)}
    state_idx, _ = _state_indices(m, rows, cols, device)
    ck = make_gdn_plan(pq, off_by_local, md.num_decode_tokens, device,
                       lambda cu_dev, cu_cpu: builder._build_chunk_metadata(cu_dev, cu_cpu, device),
                       lambda vals, dt: async_tensor_h2d(vals, dtype=dt, device=device), state_idx)
    ck.summary.update({'rows': rows, 'offsets': offs, 'cols': cols, 'num_prefills': num_prefills,
                       'num_prefill_tokens': md.num_prefill_tokens})
    STATS['gdn_builds'] += 1; STATS['gdn_rows'] += len(rows)
    _log('DIRECT_MAMBA_CKPT_BUILD', {'kind': 'gdn', **ck.summary, 'stats': dict(STATS)})
    return ck


def build_ple(builder, m, md):
    from vllm.utils.torch_utils import async_tensor_h2d
    sel = _prefill_rows(builder, m, md.num_prefills, getattr(md, 'spec_sequence_masks', None) is not None)
    if sel is None:
        return None
    rows, offs, cols, first_prefill = sel
    if md.num_decodes + md.num_prefills != m.num_reqs:
        raise RuntimeError('direct mamba checkpoint: PLE row partition mismatch')
    device = m.query_start_loc.device
    state_idx, _ = _state_indices(m, rows, cols, device)
    local = [r - first_prefill for r in rows]
    summary = {'rows': rows, 'offsets': offs, 'cols': cols, 'num_prefills': md.num_prefills}
    STATS['ple_builds'] += 1; STATS['ple_rows'] += len(rows)
    _log('DIRECT_MAMBA_CKPT_BUILD', {'kind': 'ple', **summary, 'stats': dict(STATS)})
    return PLECheckpoint(ck_rows=async_tensor_h2d(local, dtype=torch.int64, device=device),
                         offsets=async_tensor_h2d(offs, dtype=torch.int64, device=device),
                         state_idx=state_idx, summary=summary)


def gdn_export_conv(ck: GDNCheckpoint, x: torch.Tensor, conv_state: torch.Tensor) -> None:
    """conv_state: (blocks, dim, L) view; x: (non-spec tokens, dim) raw pre-conv inputs."""
    n = ck.ck_rows.numel()
    L = conv_state.shape[-1]
    if x.shape[-1] != conv_state.shape[-2]:
        raise RuntimeError(f'direct gdn conv export: dim mismatch {tuple(x.shape)} vs {tuple(conv_state.shape)}')
    pos = ck.conv_end_pos.view(n, 1) - L + torch.arange(L, device=x.device, dtype=torch.int64).view(1, L)
    vals = x.index_select(0, pos.reshape(-1)).view(n, L, x.shape[-1]).transpose(1, 2)
    conv_state[ck.state_idx] = vals.to(conv_state.dtype)


def _take(c: GDNCall, t):
    return t[:, c.lo:c.hi] if c.idx is None else t.index_select(1, c.idx)


def _run(chunk_fn, c: GDNCall, tensors, h0):
    q, k, v, g, b = (_take(c, t) for t in tensors)
    return chunk_fn(q=q, k=k, v=v, g=g, beta=b, initial_state=h0, output_final_state=True, cu_seqlens=c.cu,
                    chunk_indices=c.ci, chunk_offsets=c.co, use_qk_l2norm_in_kernel=False)


_GDN_DIRECT_OUT_LOGGED = set()


def _gdn_direct_out_on():
    v = os.environ.get('DIRECT_GDN_DIRECT_OUT')
    if v is not None:
        on = v == '1'
    else:
        try:
            from direct_moe_ab_adapter_v1 import read_phase as _rp
            ph = _rp()
            on = ph is not None and ph['arm'] == 'wmma5'
        except Exception:
            on = False
    if on not in _GDN_DIRECT_OUT_LOGGED:
        _GDN_DIRECT_OUT_LOGGED.add(on)
        print('DIRECT_GDN_DIRECT_OUT ' + json.dumps({'fast': on, 'pid': os.getpid()}), flush=True)
    return on


def _out_ok(out_buf, q, v, T):
    return (out_buf is not None and out_buf.is_contiguous() and out_buf.dim() == v.dim() and out_buf.shape[0] == 1
            and out_buf.shape[1] == T and tuple(out_buf.shape[2:]) == tuple(v.shape[2:]) and out_buf.dtype == q.dtype)


def _run_into(chunk_fn, c: GDNCall, tensors, h0, out_flat):
    q, k, v, g, b = (_take(c, t) for t in tensors)
    return chunk_fn(q=q, k=k, v=v, g=g, beta=b, initial_state=h0, output_final_state=True, cu_seqlens=c.cu,
                    chunk_indices=c.ci, chunk_offsets=c.co, use_qk_l2norm_in_kernel=False, core_attn_out=out_flat)


def gdn_two_call(chunk_fn, ck: GDNCheckpoint, q, k, v, g, beta, initial_state, ssm_state, prefill_state_indices, out_buf=None):
    """Chunk-aligned restart prefill (see GDNCheckpoint). Returns (output, final_state) like chunk_fn."""
    tensors = (q, k, v, g, beta)
    if ck.single and _out_ok(out_buf, q, v, ck.num_prefill_tokens) and _gdn_direct_out_on():
        # v11 exact data movement: call1/call3 write their chunk_fwd_o outputs straight into the two contiguous
        # token ranges of the caller's output buffer (same kernels, same inputs, only the destination differs),
        # removing the torch.cat copy. call2 (state only) is unchanged.
        per_tok = out_buf[0, 0].numel()
        flat = out_buf.view(-1)
        a = ck.call1.hi
        o1, f1 = _run_into(chunk_fn, ck.call1, tensors, initial_state, flat[: a * per_tok])
        state_a = f1.index_select(0, ck.ck_rows)
        o3, f3 = _run_into(chunk_fn, ck.call3, tensors, state_a, flat[a * per_tok: ck.num_prefill_tokens * per_tok])
        if o1.data_ptr() != flat.data_ptr() or o3.data_ptr() != flat[a * per_tok:].data_ptr():
            raise RuntimeError('direct gdn direct-out: chunk_fn did not write into the supplied buffer')
        ckpt_state = state_a
        if ck.call2 is not None:
            _, f2 = _run(chunk_fn, ck.call2, tensors, state_a.index_select(0, ck.call2_pos))
            ckpt_state = state_a.clone()
            ckpt_state.index_copy_(0, ck.call2_pos, f2)
        ssm_state[prefill_state_indices] = f3.to(ssm_state.dtype)
        ssm_state[ck.state_idx] = ckpt_state.to(ssm_state.dtype)
        STATS['gdn_layer_exports'] += 1
        STATS['gdn_direct_out'] = STATS.get('gdn_direct_out', 0) + 1
        return out_buf, f3
    o1, f1 = _run(chunk_fn, ck.call1, tensors, initial_state)
    state_a = f1.index_select(0, ck.ck_rows)
    o3, f3 = _run(chunk_fn, ck.call3, tensors, state_a)
    ckpt_state = state_a
    if ck.call2 is not None:
        _, f2 = _run(chunk_fn, ck.call2, tensors, state_a.index_select(0, ck.call2_pos))
        ckpt_state = state_a.clone()
        ckpt_state.index_copy_(0, ck.call2_pos, f2)
    if ck.single:
        if (out_buf is not None and out_buf.dtype == o1.dtype and out_buf.dim() == o1.dim()
                and out_buf.shape[0] == o1.shape[0] and out_buf.shape[1] == o1.shape[1] + o3.shape[1]
                and tuple(out_buf.shape[2:]) == tuple(o1.shape[2:]) and out_buf.is_contiguous()):
            torch.cat((o1, o3), dim=1, out=out_buf)  # exact: same bytes, one pass fewer (no later copy)
            o = out_buf
        else:
            o = torch.cat((o1, o3), dim=1)
        final = f3
    else:
        o = o1.new_empty((o1.shape[0], ck.num_prefill_tokens, *o1.shape[2:]))
        o.index_copy_(1, ck.call1.idx, o1)
        o.index_copy_(1, ck.call3.idx, o3)
        final = f1.clone()
        final.index_copy_(0, ck.ck_rows, f3)
    ssm_state[prefill_state_indices] = final.to(ssm_state.dtype)
    ssm_state[ck.state_idx] = ckpt_state.to(ssm_state.dtype)
    STATS['gdn_layer_exports'] += 1
    return o, final


def ple_export(ck: PLECheckpoint, history: torch.Tensor, conv_state: torch.Tensor, L: int) -> None:
    """history: (num_prefills, hidden, L + max_len) = [initial_state | tokens]; state before offset o = cols [o, o+L)."""
    n = ck.ck_rows.numel()
    hidden = history.shape[1]
    dev = history.device
    pos = ck.offsets.view(n, 1, 1) + torch.arange(L, device=dev, dtype=torch.int64).view(1, 1, L)
    vals = history[ck.ck_rows.view(n, 1, 1), torch.arange(hidden, device=dev).view(1, hidden, 1), pos]
    existing = conv_state.index_select(0, ck.state_idx)
    existing[..., :L] = vals.to(conv_state.dtype)
    conv_state.index_copy_(0, ck.state_idx, existing)
    STATS['ple_layer_exports'] += 1
