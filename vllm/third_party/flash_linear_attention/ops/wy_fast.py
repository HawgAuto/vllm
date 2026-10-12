# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

# ruff: noqa: E501

import torch

from vllm.triton_utils import tl, triton

from .index import prepare_chunk_indices


@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=["H", "K", "V", "BT", "BK", "BV", "IS_VARLEN"],
)
@triton.jit(do_not_specialize=["T"])
def recompute_w_u_fwd_kernel(
    k,
    v,
    beta,
    w,
    u,
    A,
    g,
    cu_seqlens,
    chunk_indices,
    T,
    stride_vt,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(1), tl.program_id(0)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
    p_beta = tl.make_block_ptr(
        beta + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,)
    )
    p_g = tl.make_block_ptr(g + (bos * H + i_h), (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_A = tl.make_block_ptr(
        A + (bos * H + i_h) * BT, (T, BT), (H * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0)
    )
    b_beta = tl.load(p_beta, boundary_check=(0,))
    b_A = tl.load(p_A, boundary_check=(0, 1))
    b_g = tl.exp(tl.load(p_g, boundary_check=(0,)))

    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(
            v + bos * stride_vt + i_h * V,
            (T, V),
            (stride_vt, 1),
            (i_t * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )
        p_u = tl.make_block_ptr(
            u + (bos * H + i_h) * V,
            (T, V),
            (H * V, 1),
            (i_t * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_vb = (b_v * b_beta[:, None]).to(b_v.dtype)
        b_u = tl.dot(b_A, b_vb, allow_tf32=False)
        tl.store(p_u, b_u.to(p_u.dtype.element_ty), boundary_check=(0, 1))

    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(
            k + (bos * Hg + i_h // (H // Hg)) * K,
            (T, K),
            (Hg * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        p_w = tl.make_block_ptr(
            w + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_kb = (b_k * b_beta[:, None] * b_g[:, None]).to(b_k.dtype)
        b_w = tl.dot(b_A, b_kb)
        tl.store(p_w, b_w.to(p_w.dtype.element_ty), boundary_check=(0, 1))


def recompute_w_u_fwd(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g_cumsum: torch.Tensor,
    A: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    chunk_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, Hg, K, V = *k.shape, v.shape[-1]
    H = v.shape[-2]
    BT = A.shape[-1]

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    BK = 64
    BV = 64
    # DIRECT v-view (overlay nb2-gdnvv-ab-v1): v may be a token-strided view (dense heads/features); anything else is
    # made contiguous.  stride_vt == H*V for contiguous v => identical addresses to the original kernel.
    if v.stride(-1) != 1 or v.stride(-2) != V or (B > 1 and v.stride(0) != T * v.stride(1)):
        v = v.contiguous()
    stride_vt = v.stride(1)
    u = v.new_empty(B, T, H, V)
    w = k.new_empty(B, T, H, K)
    if cu_seqlens is not None and _wy_on():
        _wy_run(k, v, beta, w, u, A, g_cumsum, cu_seqlens, chunk_indices, T, H, Hg, K, V, BT, B, NT, BK, BV, stride_vt)
        return w, u
    recompute_w_u_fwd_kernel[(B * H, NT)](
        k=k,
        v=v,
        beta=beta,
        w=w,
        u=u,
        A=A,
        g=g_cumsum,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        stride_vt=stride_vt,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
    )
    return w, u


# ---- DIRECT recompute_w_u pinned-config A/B path (overlay moe-wmma-unified-v10) ----
# Autotune key [H,K,V,BT,BK,BV,IS_VARLEN] excludes T; BK/BV are wrapper constants (64/64); production autotunes w8/s3 once.
# Exact screen wy-configs-01: all 80 configs bitwise-equal; BK32/BV32/w8/s1 is 1.72-1.77x faster at T=960/3136/4096/12288.
# wmma5 arm (varlen path only, as screened) launches the bare JIT with that config; first call self-checks vs production.
import os as _wy_os, json as _wy_json
_WY_LOGGED = set()
_WY_CHECKED = [False]
_WY_CFG = dict(BK=int(_wy_os.environ.get('DIRECT_WY_BK', '32')), BV=int(_wy_os.environ.get('DIRECT_WY_BV', '32')),
               num_warps=int(_wy_os.environ.get('DIRECT_WY_WARPS', '8')), num_stages=int(_wy_os.environ.get('DIRECT_WY_STAGES', '1')))


def _wy_log(tag, **kw):
    if tag in _WY_LOGGED:
        return
    _WY_LOGGED.add(tag)
    print('DIRECT_WY ' + _wy_json.dumps(dict(tag=tag, pid=_wy_os.getpid(), **kw), sort_keys=True, default=str), flush=True)


def _wy_on():
    force = _wy_os.environ.get('DIRECT_WY_PIN')
    if force is not None:
        return force == '1'
    try:
        import direct_moe_ab_adapter_v1 as _ab
    except Exception:
        return False
    ph = _ab.read_phase()
    return ph is not None and ph['arm'] == 'wmma5'


def _wy_launch(k, v, beta, w, u, A, g, cu_seqlens, chunk_indices, T, H, Hg, K, V, BT, B, NT, stride_vt):
    jit = recompute_w_u_fwd_kernel.fn.fn
    jit[(B * H, NT)](
        k=k, v=v, beta=beta, w=w, u=u, A=A, g=g, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
        T=T, stride_vt=stride_vt, H=H, Hg=Hg, K=K, V=V, BT=BT, BK=_WY_CFG['BK'], BV=_WY_CFG['BV'], IS_VARLEN=True,
        num_warps=_WY_CFG['num_warps'], num_stages=_WY_CFG['num_stages'])


def _wy_run(k, v, beta, w, u, A, g, cu_seqlens, chunk_indices, T, H, Hg, K, V, BT, B, NT, BK_prod, BV_prod, stride_vt):
    _wy_launch(k, v, beta, w, u, A, g, cu_seqlens, chunk_indices, T, H, Hg, K, V, BT, B, NT, stride_vt)
    _wy_log('pinned_first_use', stride_vt=int(stride_vt), v_contig=bool(v.is_contiguous()), T=int(T), B=int(B), H=int(H), Hg=int(Hg), K=int(K), V=int(V), BT=int(BT), cfg=_WY_CFG)
    if not _WY_CHECKED[0]:
        w2 = torch.empty_like(w)
        u2 = torch.empty_like(u)
        recompute_w_u_fwd_kernel[(B * H, NT)](
            k=k, v=v, beta=beta, w=w2, u=u2, A=A, g=g, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
            T=T, stride_vt=stride_vt, H=H, Hg=Hg, K=K, V=V, BT=BT, BK=BK_prod, BV=BV_prod)
        eq = bool(torch.equal(w, w2) and torch.equal(u, u2))
        _wy_log('selfcheck', T=int(T), equal=eq, cfg=_WY_CFG, prod_BK=BK_prod, prod_BV=BV_prod,
                prod_cfg=str(recompute_w_u_fwd_kernel.fn.best_config))
        if not eq:
            raise RuntimeError('DIRECT_WY selfcheck mismatch vs autotuned production kernel')
        _WY_CHECKED[0] = True
