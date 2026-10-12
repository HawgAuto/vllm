"""[nb2-mqa4-ab-v1] Native QSA indexer scores: qsa_mqa_scores_v4 (HIP, gfx1151) as a drop-in for the Triton
ops/qsa.py::_qsa_mqa_paged_kernel launch inside qsa_mqa_paged().  BITWISE candidate: identical logits and visible_blocks
(operator gates attempts/mqa1-gate-01 .. mqa5-gate-01; v4 26/26 cases incl. c2..c5 geometries, odd row blocking,
non-contiguous q, invalid pages/requests, +-0 / denormal-product / huge / inf / NaN values, exact ties).

Same arithmetic as the reference (tree read from the reference LLIR): per (row, key, head) lane-chunk partials
T_l = ((p0+p1)+(p2+p3))+((p4+p5)+(p6+p7)) over dims 8l..8l+7 (p = q*k, exact in fp32), DPP-order cross-chunk tree,
score = (((0+relu(d0))+relu(d1))+relu(d2))+relu(d3), IEEE division by score_divisor, -inf where invalid.

On when DIRECT_QSA_MQA_NATIVE=1 (forced) / =0 (forced off), else when direct_nb1.nb_level() >= 6.
Eligibility (else the untouched reference launch runs): NUM_HEADS 4, HEAD_DIM 128, bf16 q/cache with unit last stride and
16-byte aligned rows, single-request page table (num_requests == 1, as the reference's specialised variant), contiguous
int32 token_to_req / int64 query_positions / int32 sequence_lengths, >= DIRECT_QSA_MQA_MIN_ROWS rows, not capturing.
"""
import ctypes
import json
import os
import sys

import torch

_HSACO = os.environ.get('DIRECT_QSA_MQA_HSACO', '/opt/venv/lib/python3.14/site-packages/direct_qsa_mqa_scores_v4.hsaco')
_FN = 'qsa_mqa_scores_v4'
_RPB = int(os.environ.get('DIRECT_QSA_MQA_RPB', '128'))
_MIN_ROWS = int(os.environ.get('DIRECT_QSA_MQA_MIN_ROWS', '65'))
_AUDIT = os.environ.get('DIRECT_QSA_MQA_AUDIT') == '1'
_I32 = 0x7FFFFFFF
_mod = [None]
_count = [0]


def _report(tag, **kw):
    print('DIRECT_QSA_MQA_' + tag + ' ' + json.dumps(kw, sort_keys=True), file=sys.stderr, flush=True)


def enabled() -> bool:
    forced = os.environ.get('DIRECT_QSA_MQA_NATIVE')
    if forced is not None:
        return forced == '1'
    import direct_nb1 as _nb
    return _nb.nb_level() >= 6


def eligible(q, k_cache, page_table, token_to_req, query_positions, sequence_lengths) -> bool:
    try:
        return bool(
            q.is_cuda and q.dtype == torch.bfloat16 and q.dim() == 3 and q.shape[1] == 4 and q.shape[2] == 128
            and q.stride(2) == 1 and q.shape[0] >= _MIN_ROWS and q.data_ptr() % 16 == 0
            and (q.stride(0) * 2) % 16 == 0 and (q.stride(1) * 2) % 16 == 0
            and k_cache.is_cuda and k_cache.dtype == torch.bfloat16 and k_cache.dim() == 4 and k_cache.shape[2] == 1
            and k_cache.shape[3] == 128 and k_cache.stride(3) == 1 and k_cache.data_ptr() % 16 == 0
            and (k_cache.stride(0) * 2) % 16 == 0 and (k_cache.stride(1) * 2) % 16 == 0
            and page_table.dtype == torch.int32 and page_table.dim() == 2 and page_table.shape[0] == 1
            and page_table.stride(1) == 1
            and token_to_req.dtype == torch.int32 and token_to_req.is_contiguous()
            and query_positions.dtype == torch.int64 and query_positions.is_contiguous()
            and sequence_lengths.dtype == torch.int32 and sequence_lengths.is_contiguous()
            and max(q.stride(0), q.stride(1), k_cache.stride(0), k_cache.stride(1)) <= _I32
            and not torch.cuda.is_current_stream_capturing())
    except Exception:
        return False


def run(q, k_cache, page_table, token_to_req, query_positions, sequence_lengths, visible_blocks, logits,
        columns, compress_ratio, score_divisor) -> None:
    import direct_hip_module_v1 as H
    if _mod[0] is None:
        _mod[0] = H.Module(_HSACO)
        _report('LOADED', hsaco=_HSACO, fn=_FN, rpb=_RPB, min_rows=_MIN_ROWS, attrs=_mod[0].attrs(_FN))
    rows = q.shape[0]
    assert logits.shape == (rows, columns) and logits.stride(1) == 1 and logits.stride(0) <= _I32
    grid = ((columns + 255) // 256, (rows + _RPB - 1) // _RPB, 1)
    _mod[0].launch(_FN, grid, (256, 1, 1), [
        H.ptr(q), H.ptr(k_cache), H.ptr(page_table), H.ptr(token_to_req), H.ptr(query_positions), H.ptr(sequence_lengths),
        H.ptr(visible_blocks), H.ptr(logits),
        ctypes.c_int(q.stride(0)), ctypes.c_int(q.stride(1)), ctypes.c_int(k_cache.stride(0)), ctypes.c_int(k_cache.stride(1)),
        ctypes.c_int(logits.stride(0)), ctypes.c_int(rows), ctypes.c_int(columns), ctypes.c_int(k_cache.shape[0]),
        ctypes.c_int(k_cache.shape[1]), ctypes.c_int(page_table.shape[1]), ctypes.c_int(int(compress_ratio)),
        ctypes.c_float(float(score_divisor)), ctypes.c_int(_RPB)])
    _count[0] += 1
    if _AUDIT:
        case = _case()
        if case != _cur[0]:
            if _cur[0] is not None:
                _report('CASE', case=_cur[0], dispatches=_cur[1], rows=_cur[2])
            _cur[0], _cur[1], _cur[2] = case, 0, 0
        _cur[1] += 1; _cur[2] += rows
        if _count[0] <= 4:
            _report('DISPATCH', n=_count[0], rows=rows, columns=columns, width=int(page_table.shape[1]),
                    page_size=int(k_cache.shape[1]), grid=list(grid), case=case)


_cur = [None, 0, 0]
_case_cache = [None, None]


def _case():
    # audit only: the nb-phase 'case' label of the request this dispatch belongs to (stat-cached, like direct_nb1)
    try:
        import direct_nb1 as _nb
        st = _nb.PHASE.stat(); key = (st.st_mtime_ns, st.st_size, st.st_ino)
        if _case_cache[0] != key:
            _case_cache[0], _case_cache[1] = key, json.loads(_nb.PHASE.read_text()).get('case')
        return _case_cache[1]
    except Exception:
        return None


# Appended to the hash-verified parent adapter; parent functions remain intact.
import hashlib as _hashlib
from pathlib import Path as _Path
_inc_run=run
_inc_eligible=eligible
_succ_mod=[None]
_checked_shapes=set()
_succ_count=[0]
_fallback_seen=set()

def _candidate_on():
    import direct_nb1 as nb
    return nb.nb_level()>=9

def eligible(q,k_cache,page_table,token_to_req,query_positions,sequence_lengths):
    ok=_inc_eligible(q,k_cache,page_table,token_to_req,query_positions,sequence_lengths)
    if not ok and _candidate_on() and not torch.cuda.is_current_stream_capturing():
        key=(_case(),tuple(q.shape),tuple(q.stride()))
        if key not in _fallback_seen:
            _fallback_seen.add(key)
            _report('SUCCESSOR_BASELINE_PATH',case=key[0],shape=list(q.shape),stride=list(q.stride()),
                    reason='outside_inherited_min_rows' if q.shape[0]<_MIN_ROWS else 'unexpected_eligibility_failure')
        if q.shape[0]>=_MIN_ROWS:
            raise RuntimeError('unexpected successor prefill fallback')
    return ok

def _launch_successor(q,k_cache,page_table,token_to_req,query_positions,sequence_lengths,visible_blocks,logits,
                      columns,compress_ratio,score_divisor):
    import direct_hip_module_v1 as H
    path=os.environ['DIRECT_QSA_SUCCESSOR_HSACO'];fn='qsa_mqa_scores_v7'
    if _succ_mod[0] is None:
        digest=_hashlib.sha256(_Path(path).read_bytes()).hexdigest()
        assert digest==os.environ['DIRECT_QSA_SUCCESSOR_SHA256']
        _succ_mod[0]=H.Module(path)
        _report('SUCCESSOR_LOADED',hsaco=path,sha256=digest,fn=fn,attrs=_succ_mod[0].attrs(fn))
    rows=q.shape[0];grid=((columns+255)//256,(rows+_RPB-1)//_RPB,1)
    _succ_mod[0].launch(fn,grid,(256,1,1),[
        H.ptr(q),H.ptr(k_cache),H.ptr(page_table),H.ptr(token_to_req),H.ptr(query_positions),H.ptr(sequence_lengths),
        H.ptr(visible_blocks),H.ptr(logits),
        ctypes.c_int(q.stride(0)),ctypes.c_int(q.stride(1)),ctypes.c_int(k_cache.stride(0)),ctypes.c_int(k_cache.stride(1)),
        ctypes.c_int(logits.stride(0)),ctypes.c_int(rows),ctypes.c_int(columns),ctypes.c_int(k_cache.shape[0]),
        ctypes.c_int(k_cache.shape[1]),ctypes.c_int(page_table.shape[1]),ctypes.c_int(int(compress_ratio)),
        ctypes.c_float(float(score_divisor)),ctypes.c_int(_RPB)])

def run(q,k_cache,page_table,token_to_req,query_positions,sequence_lengths,visible_blocks,logits,
        columns,compress_ratio,score_divisor):
    on=_candidate_on()
    args=(q,k_cache,page_table,token_to_req,query_positions,sequence_lengths)
    if not on:
        _inc_run(*args,visible_blocks,logits,columns,compress_ratio,score_divisor)
    else:
        key=(tuple(q.shape),tuple(q.stride()),tuple(k_cache.stride()),int(columns),int(page_table.shape[1]),int(compress_ratio),float(score_divisor),_RPB)
        if key not in _checked_shapes:
            ref_l=torch.empty_like(logits);ref_v=torch.empty_like(visible_blocks)
            _inc_run(*args,ref_v,ref_l,columns,compress_ratio,score_divisor)
            _launch_successor(*args,visible_blocks,logits,columns,compress_ratio,score_divisor)
            errors=int((ref_l.view(torch.int32)!=logits.view(torch.int32)).sum().item())
            vis_errors=int((ref_v!=visible_blocks).sum().item())
            _report('SUCCESSOR_SELFCHECK',case=_case(),shape=list(q.shape),columns=int(columns),mismatches=errors,visible_mismatches=vis_errors)
            if errors or vis_errors:raise RuntimeError('successor live-model bit mismatch')
            _checked_shapes.add(key)
        else:
            _launch_successor(*args,visible_blocks,logits,columns,compress_ratio,score_divisor)
    _succ_count[0]+=1
    _report('SUCCESSOR_DISPATCH',case=_case(),arm='i1' if on else 'qsa3',n=_succ_count[0],rows=int(q.shape[0]),columns=int(columns),rpb=_RPB)
