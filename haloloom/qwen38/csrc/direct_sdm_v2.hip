// sdm_v2.hip -- scalar-decode (M=1, 10 slots) mixed W4/W8 MoE route for gfx1151, two launches per layer.
//
// Successor of sdm_v1 (bit-exact vs the incumbent scalar pool-pair route).  Same arithmetic contract:
//   b   = bf16_rne(fma(q, s, -ZP*s))  (exact for finite s > 0)
//   acc = v_wmma_f32_16x16x16_bf16 chain over K ascending in 16-chunks from +0, WMMA dual-row layout (A replicated)
//   act = bf16(bf16(silu(gate)) * up)  (variant siluv, exhaustively gated vs the native activation)
//   down = bf16_rne(acc), -0 -> +0
// New in v2 (no arithmetic change):
//   * raw buffer loads: per-WG SGPR descriptor base + loop-invariant 32-bit voffset + SGPR soffset (no 64-bit VALU address math)
//   * W8 down: coalesced 128 B row-segment tile loads staged through LDS (storage layout unchanged, [e][n][k])
//   * up: optional LDS-staged b128 weight loads (LDSU=1)
//   * route guards folded into up (block 0,0 wave 0): same predicates as scalar_route_guards + the three _assert_async
//     calls; violation -> flags written then s_trap (fail-closed process abort, like the device assert)
//   * slot reduction (+ optional shared-expert fold) folded into down via last-arriver per column tile: replicates
//     _slots_reduce(HAS_B=False) exactly: products x*w (no FMA), 4 accumulators by slot%4, ((a0+a1)+a2)+a3, then
//     HAS_S: bf16(value) + bf16(gate*shared) in fp32, bf16 RNE.
#include <hip/hip_runtime.h>
#include <stdint.h>

typedef __bf16 bf16x16 __attribute__((ext_vector_type(16)));
typedef float f32x8 __attribute__((ext_vector_type(8)));
typedef uint32_t u32x2 __attribute__((ext_vector_type(2)));
typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));
typedef uint32_t u32x8 __attribute__((ext_vector_type(8)));

__device__ __forceinline__ uint32_t rne_hi(float f) {
  uint32_t u = __float_as_uint(f);
  return u + 0x7fffu + ((u >> 16) & 1u);
}
__device__ __forceinline__ uint32_t pack_hi(uint32_t lo_elem, uint32_t hi_elem) {
  return __builtin_amdgcn_perm(hi_elem, lo_elem, 0x07060302u);
}
__device__ __forceinline__ uint32_t f32_to_bf16_rne_nan(float f) {
  uint32_t u = __float_as_uint(f);
  if ((u & 0x7fffffffu) > 0x7f800000u) return ((u >> 16) | 0x40u) & 0xffffu;
  u += 0x7fffu + ((u >> 16) & 1u);
  return u >> 16;
}
__device__ __forceinline__ float ubyte_f(uint32_t x, int c) { return (float)((x >> (8 * c)) & 0xffu); }
__device__ __forceinline__ float bf16_bits_to_f32(uint32_t b) { return __uint_as_float(b << 16); }

template <int SILUV>
__device__ __forceinline__ uint32_t silu_mul_v(uint32_t g, uint32_t u) {
  const float x = bf16_bits_to_f32(g);
  float e;
  if constexpr (SILUV == 0 || SILUV == 2) e = expf(-x); else e = __expf(-x);
  float s;
  if constexpr (SILUV == 0 || SILUV == 1) s = x / (1.0f + e); else s = __fdividef(x, 1.0f + e);
  const uint32_t sb = f32_to_bf16_rne_nan(s);
  return f32_to_bf16_rne_nan(bf16_bits_to_f32(sb) * bf16_bits_to_f32(u));
}
__device__ __forceinline__ uint32_t silu_mul(uint32_t g, uint32_t u, int v) {
  switch (v) {
    case 1: return silu_mul_v<1>(g, u);
    case 2: return silu_mul_v<2>(g, u);
    case 3: return silu_mul_v<3>(g, u);
    default: return silu_mul_v<0>(g, u);
  }
}
__device__ __forceinline__ f32x8 wmma(const u32x8& a, const u32x8& b, const f32x8& c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(__builtin_bit_cast(bf16x16, a), __builtin_bit_cast(bf16x16, b), c);
}

__device__ __forceinline__ __amdgpu_buffer_rsrc_t rsrc(const void* p) {
  return __builtin_amdgcn_make_buffer_rsrc((void*)p, (short)0, 0x7FFFFFFF, 0x31004000);
}
template <int CPL> __device__ __forceinline__ uint32_t bld(__amdgpu_buffer_rsrc_t r, int vo, int so) {
  if constexpr (CPL == 1) return (uint32_t)__builtin_amdgcn_raw_buffer_load_b8(r, vo, so, 0);
  else if constexpr (CPL == 2) return (uint32_t)__builtin_amdgcn_raw_buffer_load_b16(r, vo, so, 0);
  else return __builtin_amdgcn_raw_buffer_load_b32(r, vo, so, 0);
}
__device__ __forceinline__ float bldf(__amdgpu_buffer_rsrc_t r, int vo, int so) {
  return __builtin_bit_cast(float, __builtin_amdgcn_raw_buffer_load_b32(r, vo, so, 0));
}
template <int CPL> __device__ __forceinline__ void stc(__bf16* p, const uint32_t* v) {
  if constexpr (CPL == 1) *(uint16_t*)p = (uint16_t)v[0];
  else if constexpr (CPL == 2) *(uint32_t*)p = v[0] | (v[1] << 16);
  else *(u32x2*)p = u32x2{v[0] | (v[1] << 16), v[2] | (v[3] << 16)};
}
__device__ __forceinline__ long long slot_id(const void* ids, int ids64, int slot) {
  return ids64 ? ((const long long*)ids)[slot] : (long long)((const int*)ids)[slot];
}

// ---------------------------------------------------------------------------------------------------------------
// Route guards (scalar_route_guards + the three torch._assert_async calls).  One wave, lanes 0..9 live.
__device__ __forceinline__ void route_guard(const void* __restrict__ ids, int ids64, const float* __restrict__ wts,
                                         const uint8_t* __restrict__ pad, int has_padding, int* __restrict__ flags) {
  const int lane = threadIdx.x & 31;
  const bool live = lane < 10;
  long long id = -1;
  float w = 0.0f;
  if (live) { id = slot_id(ids, ids64, lane); w = wts[lane]; }
  bool v = (id >= 0) && (id < 512);
  if (has_padding) v = v || ((id == -1) && (pad[0] != 0));
  const bool bad_valid = live && !v;
  const bool bad_fin = live && ((w != w) || (__builtin_fabsf(w) == __builtin_inff()));
  const int lo = (int)(id & 0xffffffffll), hi = (int)(id >> 32);
  bool dup = false;
#pragma unroll
  for (int q = 0; q < 10; ++q) {
    const long long o = ((long long)__shfl(hi, q) << 32) | (unsigned int)__shfl(lo, q);
    if (live && q > lane && o == id && id != -1) dup = true;
  }
  const bool ok_valid = __ballot(bad_valid) == 0, ok_fin = __ballot(bad_fin) == 0, ok_uniq = __ballot(dup) == 0;
  if (lane == 0 && flags) { flags[0] = ok_valid; flags[1] = ok_fin; flags[2] = ok_uniq; }
  if (!(ok_valid && ok_fin && ok_uniq)) {
    __threadfence_system();
    __builtin_trap();
  }
}

// ---------------------------------------------------------------------------------------------------------------
// gate_up (+SiLU*up), MODE 2: WG = 2 waves, wave 0 = gate cols, wave 1 = matching up cols, 32*CPL intermediates/WG.
//   W4 [E][kb=1280][n=1280] bytes (N-contiguous), S4 [E][g=20][n=1280] f32
//   W8 [E][k=2560][n=1280] bytes (N-contiguous), S8 [E][n=1280][g=20] f32
// LDSU=1: per chunk, the wave's NR x (32*CPL) byte tile is fetched with b128 loads (16 B/lane) and redistributed
//         through a private per-wave LDS ring; LDSU=0: one CPL-byte buffer load per row per lane.
template <int Q, int CPL, int PF, int LDSU>
__device__ __forceinline__ void up_body(const uint32_t* __restrict__ xs, const uint8_t* __restrict__ W, const float* __restrict__ S,
                                        int local, int i0, int lane, int wave, uint32_t* __restrict__ bits_out,
                                        uint8_t* __restrict__ lds_w) {
  constexpr int NR = (Q == 4) ? 8 : 16;
  constexpr int KCH = 160;
  constexpr int RS = 1280;
  constexpr float ZP = (Q == 4) ? 8.0f : 128.0f;
  constexpr int TB = 32 * CPL;               // tile bytes per row (contiguous columns of this wave)
  constexpr int TILE = NR * TB;              // bytes per chunk tile
  constexpr int NV = (TILE + 511) / 512;     // b128 loads per lane per chunk (LDSU)
  static_assert(!LDSU || (TILE % 512) == 0, "LDS tile must be a multiple of 512 B");
  const int j = lane & 15, h = lane >> 4;
  const uint8_t* wb = W + (size_t)local * (size_t)(Q == 4 ? 1280u * 1280u : 2560u * 1280u);
  const float* sb = S + (size_t)local * 25600u;
  const __amdgpu_buffer_rsrc_t rs = rsrc(sb);
  // one descriptor per 4 rows: row offset (r&3)*RS <= 3840 rides in the 12-bit immediate, chunk offset in soffset
  __amdgpu_buffer_rsrc_t rwr[NR / 4];
#pragma unroll
  for (int q = 0; q < NR / 4; ++q) rwr[q] = rsrc(wb + q * 4 * RS);
  const int cbase = (wave ? 640 : 0) + i0;                  // first column of this wave's contiguous tile
  const int coff = h * 16 * CPL + CPL * j;                  // this lane's column offset inside the tile
  const int col = cbase + coff;
  auto sload = [&](int c, int g) -> float {
    return (Q == 4) ? bldf(rs, (col + c) * 4, g * 5120) : bldf(rs, (col + c) * 80, g * 4);
  };
  float sc[CPL], sn[CPL];
#pragma unroll
  for (int c = 0; c < CPL; ++c) sn[c] = sload(c, 0);
  // LDSU geometry: piece p = v*32 + lane -> row p / (TB/16), byte (p % (TB/16))*16
  constexpr int PPR = TB / 16;               // 16-byte pieces per row
  uint32_t raw[PF][LDSU ? 1 : NR];
  u32x4 vraw[PF][LDSU ? NV : 1];
  auto issue = [&](int p, int kc) {
    if constexpr (LDSU) {
#pragma unroll
      for (int v = 0; v < NV; ++v) {
        const int pc = v * 32 + lane, r = pc / PPR, b = (pc % PPR) * 16;
        vraw[p][v] = __builtin_amdgcn_raw_buffer_load_b128(rwr[0], cbase + b + r * RS, kc * NR * RS, 0);
      }
    } else {
#pragma unroll
      for (int r = 0; r < NR; ++r) raw[p][r] = bld<CPL>(rwr[r >> 2], col + (r & 3) * RS, kc * NR * RS);
    }
  };
#pragma unroll
  for (int p = 0; p < PF; ++p) issue(p, p);
  f32x8 acc[CPL];
#pragma unroll
  for (int i = 0; i < CPL; ++i) acc[i] = f32x8{0, 0, 0, 0, 0, 0, 0, 0};
  for (int kc0 = 0; kc0 < KCH; kc0 += PF) {
#pragma unroll
    for (int p = 0; p < PF; ++p) {
      const int kc = kc0 + p;
      if ((kc & 7) == 0) {
#pragma unroll
        for (int c = 0; c < CPL; ++c) sc[c] = sn[c];
        if (kc + 8 < KCH) {
#pragma unroll
          for (int c = 0; c < CPL; ++c) sn[c] = sload(c, (kc >> 3) + 1);
        }
      }
      uint32_t rr[NR];  // LDSU: filled from LDS; l0: aliases raw[p] (no copy), reloaded after this chunk's dequant
      if constexpr (LDSU) {
        // private ring slot (2 tiles) of this wave
        uint8_t* t = lds_w + (kc & 1) * TILE;
#pragma unroll
        for (int v = 0; v < NV; ++v) {
          const int pc = v * 32 + lane;
          *(u32x4*)(t + (pc / PPR) * TB + (pc % PPR) * 16) = vraw[p][v];
        }
        __builtin_amdgcn_wave_barrier();
        __builtin_amdgcn_s_waitcnt(0xFC07);  // gfx11 encoding: vmcnt(63) lgkmcnt(0) expcnt(7)
        __builtin_amdgcn_wave_barrier();
#pragma unroll
        for (int r = 0; r < NR; ++r) {
          if constexpr (CPL == 1) rr[r] = t[r * TB + coff];
          else if constexpr (CPL == 2) rr[r] = *(const uint16_t*)(t + r * TB + coff);
          else rr[r] = *(const uint32_t*)(t + r * TB + coff);
        }
      } else {
#pragma unroll
        for (int r = 0; r < NR; ++r) rr[r] = raw[p][r];
      }
      if constexpr (LDSU) {
        if (kc + PF < KCH) issue(p, kc + PF);
      }
      const u32x4* x4 = (const u32x4*)(xs + kc * 8);
      const u32x4 a0 = x4[0], a1 = x4[1];
      const u32x8 a = u32x8{a0[0], a0[1], a0[2], a0[3], a1[0], a1[1], a1[2], a1[3]};
      uint32_t lo4[NR], hi4[NR];
      if constexpr (Q == 4) {
#pragma unroll
        for (int r = 0; r < NR; ++r) { lo4[r] = rr[r] & 0x0F0F0F0Fu; hi4[r] = (rr[r] >> 4) & 0x0F0F0F0Fu; }
      }
#pragma unroll
      for (int c = 0; c < CPL; ++c) {
        const float s0 = sc[c], m0 = -ZP * s0;
        u32x8 b;
#pragma unroll
        for (int r2 = 0; r2 < 8; ++r2) {
          float q0, q1;
          if constexpr (Q == 4) { q0 = ubyte_f(lo4[r2], c); q1 = ubyte_f(hi4[r2], c); }
          else { q0 = ubyte_f(rr[2 * r2], c); q1 = ubyte_f(rr[2 * r2 + 1], c); }
          b[r2] = pack_hi(rne_hi(fmaf(q0, s0, m0)), rne_hi(fmaf(q1, s0, m0)));
        }
        acc[c] = wmma(a, b, acc[c]);
      }
      if constexpr (!LDSU) {
        if (kc + PF < KCH) issue(p, kc + PF);
      }
    }
  }
#pragma unroll
  for (int c = 0; c < CPL; ++c) bits_out[c] = f32_to_bf16_rne_nan(acc[c][0]);  // row 0 (lanes 0-15) / row 1 (16-31)
}

template <int CPL, int PF, int LDSU>
__device__ __forceinline__ void up_kernel(const __bf16* __restrict__ x, const void* __restrict__ ids, int ids64,
                                          const long long* __restrict__ L4, const long long* __restrict__ L8, int nexp,
                                          const uint8_t* __restrict__ W4, const float* __restrict__ S4,
                                          const uint8_t* __restrict__ W8, const float* __restrict__ S8,
                                          __bf16* __restrict__ act, int siluv, const float* __restrict__ wts,
                                          const uint8_t* __restrict__ pad, int has_padding, int* __restrict__ flags) {
  constexpr int TILE8 = 16 * 32 * CPL;  // largest per-chunk tile (W8)
  __shared__ __attribute__((aligned(16))) uint32_t xs[1280];
  __shared__ uint32_t ub[32 * CPL];
  __shared__ __attribute__((aligned(16))) uint8_t ldsw[LDSU ? 2 * 2 * TILE8 : 16];
  for (int i = threadIdx.x; i < 320; i += 64) ((u32x4*)xs)[i] = ((const u32x4*)x)[i];
  const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
  if (blockIdx.x == 0 && blockIdx.y == 0 && wave == 0) route_guard(ids, ids64, wts, pad, has_padding, flags);
  __syncthreads();
  const int slot = blockIdx.y;
  const long long eid = slot_id(ids, ids64, slot);
  long long l4 = -1, l8 = -1;
  if (eid >= 0 && eid < nexp) { l4 = L4[eid]; l8 = L8[eid]; }
  __bf16* arow = act + slot * 640;
  const int i0 = blockIdx.x * (32 * CPL);
  const int j = lane & 15, h = lane >> 4, idx = h * 16 * CPL + CPL * j;
  uint32_t bits[CPL];
  bool zero = false;
  uint8_t* myl = ldsw + (LDSU ? wave * 2 * TILE8 : 0);
  if (l4 >= 0) up_body<4, CPL, PF, LDSU>(xs, W4, S4, __builtin_amdgcn_readfirstlane((int)l4), i0, lane, wave, bits, myl);
  else if (l8 >= 0) up_body<8, CPL, PF, LDSU>(xs, W8, S8, __builtin_amdgcn_readfirstlane((int)l8), i0, lane, wave, bits, myl);
  else zero = true;
  if (wave == 1) {
#pragma unroll
    for (int c = 0; c < CPL; ++c) ub[idx + c] = zero ? 0u : bits[c];
  }
  __syncthreads();
  if (wave == 0) {
    uint32_t v[CPL];
#pragma unroll
    for (int c = 0; c < CPL; ++c) v[c] = zero ? 0u : silu_mul(bits[c], ub[idx + c], siluv);
    stc<CPL>(arow + i0 + idx, v);
  }
}

// ---------------------------------------------------------------------------------------------------------------
// down, MODE 0 (dual-row: lanes 0-15 / 16-31 carry two 16*CPL column blocks).  One wave: NW = 32*CPL columns of
// one slot, K=640 (40 chunks).
//   W4 [E][kb=320][n=2560] bytes (N-contiguous), S4 [E][g=5][n=2560] f32
//   W8 [E][n=2560][k=640] bytes (K-contiguous),  S8 [E][n=2560][g=5] f32
// W8L=1: per 8-chunk stage (one scale group) the wave's NW x 128 B tile is fetched with coalesced b128 loads
//        (8 lanes per 128 B row segment) into a single LDS tile (row stride 144 B), register-prefetched one stage
//        ahead; lanes then read their own column's 16 B per chunk.  W8L=0: v1 pattern (16 B per lane per row).
constexpr int W8RS = 144;
template <int Q, int CPL, int PF, int W8L>
__device__ __forceinline__ void dn_body(const uint32_t* __restrict__ as, const uint8_t* __restrict__ W, const float* __restrict__ S,
                                        int local, int n0, int lane, uint32_t* __restrict__ vout, uint8_t* __restrict__ tile) {
  constexpr int KCH = 40;
  constexpr float ZP = (Q == 4) ? 8.0f : 128.0f;
  constexpr int NW = 32 * CPL;
  constexpr int NRAW = (Q == 4) ? 8 : 4 * CPL;  // dwords per chunk per lane
  const int j = lane & 15, h = lane >> 4;
  const int coff = h * 16 * CPL + CPL * j;
  const int col = n0 + coff;
  const uint8_t* wb = W + (size_t)local * ((Q == 4) ? 819200u : 1638400u);
  const float* sb = S + (size_t)local * 12800u;
  const __amdgpu_buffer_rsrc_t rw = rsrc(wb), rs = rsrc(sb);
  __amdgpu_buffer_rsrc_t rwr[4];  // W4: one descriptor per 2 rows (row offset (r&1)*2560 in the immediate)
#pragma unroll
  for (int q = 0; q < 4; ++q) rwr[q] = rsrc(wb + q * 2 * 2560);
  auto sload = [&](int c, int g) -> float {
    return (Q == 4) ? bldf(rs, (col + c) * 4, g * 10240) : bldf(rs, (col + c) * 20, g * 4);
  };
  f32x8 acc[CPL];
#pragma unroll
  for (int i = 0; i < CPL; ++i) acc[i] = f32x8{0, 0, 0, 0, 0, 0, 0, 0};
  auto chunk = [&](const uint32_t* rr, int kc, const float* sc) {
    const u32x4* a4 = (const u32x4*)(as + kc * 8);
    const u32x4 a0 = a4[0], a1 = a4[1];
    const u32x8 a = u32x8{a0[0], a0[1], a0[2], a0[3], a1[0], a1[1], a1[2], a1[3]};
    uint32_t lo4[8], hi4[8];
    if constexpr (Q == 4) {
#pragma unroll
      for (int r = 0; r < 8; ++r) { lo4[r] = rr[r] & 0x0F0F0F0Fu; hi4[r] = (rr[r] >> 4) & 0x0F0F0F0Fu; }
    }
#pragma unroll
    for (int c = 0; c < CPL; ++c) {
      const float s0 = sc[c], m0 = -ZP * s0;
      u32x8 b;
#pragma unroll
      for (int r2 = 0; r2 < 8; ++r2) {
        float q0, q1;
        if constexpr (Q == 4) { q0 = ubyte_f(lo4[r2], c); q1 = ubyte_f(hi4[r2], c); }
        else {
          const uint32_t w = rr[4 * c + (r2 >> 1)];
          q0 = ubyte_f(w, (2 * r2) & 3); q1 = ubyte_f(w, ((2 * r2) & 3) + 1);
        }
        b[r2] = pack_hi(rne_hi(fmaf(q0, s0, m0)), rne_hi(fmaf(q1, s0, m0)));
      }
      acc[c] = wmma(a, b, acc[c]);
    }
  };
  if constexpr (Q == 8 && W8L) {
    // NW rows x 128 B per stage; pieces: p = i*32 + lane -> row p>>3, seg p&7
    constexpr int NI = NW / 4;
    u32x4 g[NI];
    auto gload = [&](int st) {
#pragma unroll
      for (int i = 0; i < NI; ++i) {
        const int p = i * 32 + lane, r = p >> 3, sg = p & 7;
        g[i] = __builtin_amdgcn_raw_buffer_load_b128(rw, (n0 + r) * 640 + sg * 16, st * 128, 0);
      }
    };
    float sc[CPL], sn[CPL];
#pragma unroll
    for (int c = 0; c < CPL; ++c) sn[c] = sload(c, 0);
    gload(0);
    for (int st = 0; st < 5; ++st) {
      __syncthreads();
#pragma unroll
      for (int i = 0; i < NI; ++i) {
        const int p = i * 32 + lane, r = p >> 3, sg = p & 7;
        *(u32x4*)(tile + r * W8RS + sg * 16) = g[i];
      }
#pragma unroll
      for (int c = 0; c < CPL; ++c) sc[c] = sn[c];
      if (st + 1 < 5) {
        gload(st + 1);
#pragma unroll
        for (int c = 0; c < CPL; ++c) sn[c] = sload(c, st + 1);
      }
      __syncthreads();
#pragma unroll
      for (int kk = 0; kk < 8; ++kk) {
        uint32_t rr[NRAW];
#pragma unroll
        for (int c = 0; c < CPL; ++c) {
          const u32x4 t = *(const u32x4*)(tile + (coff + c) * W8RS + kk * 16);
          rr[4 * c + 0] = t[0]; rr[4 * c + 1] = t[1]; rr[4 * c + 2] = t[2]; rr[4 * c + 3] = t[3];
        }
        chunk(rr, st * 8 + kk, sc);
      }
    }
  } else {
    auto load = [&](uint32_t* dst, int kc) {
      if constexpr (Q == 4) {
#pragma unroll
        for (int r = 0; r < 8; ++r) dst[r] = bld<CPL>(rwr[r >> 1], col + (r & 1) * 2560, kc * 8 * 2560);
      } else {
#pragma unroll
        for (int c = 0; c < CPL; ++c) {
          const u32x4 t = __builtin_amdgcn_raw_buffer_load_b128(rw, (col + c) * 640, kc * 16, 0);
          dst[4 * c + 0] = t[0]; dst[4 * c + 1] = t[1]; dst[4 * c + 2] = t[2]; dst[4 * c + 3] = t[3];
        }
      }
    };
    float sc[CPL], sn[CPL];
#pragma unroll
    for (int c = 0; c < CPL; ++c) sn[c] = sload(c, 0);
    uint32_t raw[PF][NRAW];
#pragma unroll
    for (int p = 0; p < PF; ++p) load(raw[p], p);
    for (int kc0 = 0; kc0 < KCH; kc0 += PF) {
#pragma unroll
      for (int p = 0; p < PF; ++p) {
        const int kc = kc0 + p;
        if ((kc & 7) == 0) {
#pragma unroll
          for (int c = 0; c < CPL; ++c) sc[c] = sn[c];
          if (kc + 8 < KCH) {
#pragma unroll
            for (int c = 0; c < CPL; ++c) sn[c] = sload(c, (kc >> 3) + 1);
          }
        }
        chunk(raw[p], kc, sc);
        if (kc + PF < KCH) load(raw[p], kc + PF);
      }
    }
  }
#pragma unroll
  for (int c = 0; c < CPL; ++c) {
    uint32_t b = f32_to_bf16_rne_nan(acc[c][0] * 1.0f);
    vout[c] = (b == 0x8000u) ? 0u : b;  // (x4 + x8) with the other pool's +0: -0 -> +0
  }
}

// One WG = 1 wave.  grid (2560/NW, 10).  After writing its slot's columns, every WG bumps cnt[blockIdx.x]; the 10th
// (last) WG of a column tile performs the exact _slots_reduce (+ optional shared fold) for those columns and resets
// the counter for the next launch / graph replay.
template <int CPL, int PF, int W8L>
__device__ __forceinline__ void dn_kernel(const __bf16* __restrict__ act, const void* __restrict__ ids, int ids64,
                                          const long long* __restrict__ L4, const long long* __restrict__ L8, int nexp,
                                          const uint8_t* __restrict__ W4, const float* __restrict__ S4,
                                          const uint8_t* __restrict__ W8, const float* __restrict__ S8,
                                          __bf16* __restrict__ out, const float* __restrict__ wts,
                                          const __bf16* __restrict__ shared, const __bf16* __restrict__ gate, int has_s,
                                          int mask_padding, __bf16* __restrict__ y, int* __restrict__ cnt) {
  constexpr int NW = 32 * CPL;
  __shared__ __attribute__((aligned(16))) uint32_t as[320];
  __shared__ __attribute__((aligned(16))) uint8_t tile[W8L ? NW * W8RS : 16];
  __shared__ int s_last;
  const int slot = blockIdx.y;
  for (int i = threadIdx.x; i < 80; i += 32) ((u32x4*)as)[i] = ((const u32x4*)(act + slot * 640))[i];
  __syncthreads();
  const int lane = threadIdx.x & 31;
  const int n0 = blockIdx.x * NW;
  const long long eid = slot_id(ids, ids64, slot);
  long long l4 = -1, l8 = -1;
  if (eid >= 0 && eid < nexp) { l4 = L4[eid]; l8 = L8[eid]; }
  __bf16* orow = out + slot * 2560;
  const int j = lane & 15, h = lane >> 4;
  const int col = n0 + h * 16 * CPL + CPL * j;
  uint32_t v[CPL];
  if (l4 >= 0) dn_body<4, CPL, PF, W8L>(as, W4, S4, __builtin_amdgcn_readfirstlane((int)l4), n0, lane, v, tile);
  else if (l8 >= 0) dn_body<8, CPL, PF, W8L>(as, W8, S8, __builtin_amdgcn_readfirstlane((int)l8), n0, lane, v, tile);
  else {
#pragma unroll
    for (int c = 0; c < CPL; ++c) v[c] = 0u;
  }
  stc<CPL>(orow + col, v);
  // ---- last-arriver exact slot reduction ----
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) s_last = (__hip_atomic_fetch_add(cnt + blockIdx.x, 1, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT) == 9);
  __syncthreads();
  if (!s_last) return;
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
  bool act_s[10];
  float wv[10];
#pragma unroll
  for (int s = 0; s < 10; ++s) {
    act_s[s] = slot_id(ids, ids64, s) >= 0;
    wv[s] = wts[s];
  }
  const float gv = has_s ? bf16_bits_to_f32(*(const uint16_t*)gate) : 0.0f;
  uint32_t yo[CPL];
#pragma unroll
  for (int c = 0; c < CPL; ++c) {
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
#pragma unroll
    for (int s = 0; s < 10; ++s) {
      const uint16_t xb = *(const volatile uint16_t*)(out + s * 2560 + col + c);
      float xv = bf16_bits_to_f32(xb);
      if (mask_padding) xv = act_s[s] ? xv : 0.0f;
      const float pr = __fmul_rn(xv, wv[s]);
      if (s % 4 == 0) a0 = __fadd_rn(a0, pr);
      else if (s % 4 == 1) a1 = __fadd_rn(a1, pr);
      else if (s % 4 == 2) a2 = __fadd_rn(a2, pr);
      else a3 = __fadd_rn(a3, pr);
    }
    float value = __fadd_rn(__fadd_rn(__fadd_rn(a0, a1), a2), a3);
    if (has_s) {
      const float r = bf16_bits_to_f32(f32_to_bf16_rne_nan(value));
      const float sv = bf16_bits_to_f32(*(const uint16_t*)(shared + col + c));
      const float scv = bf16_bits_to_f32(f32_to_bf16_rne_nan(__fmul_rn(gv, sv)));
      value = __fadd_rn(scv, r);
    }
    yo[c] = f32_to_bf16_rne_nan(value);
  }
  stc<CPL>(y + col, yo);
  if (threadIdx.x == 0) __hip_atomic_store(cnt + blockIdx.x, 0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

#define UPK(CPL, PF, LDSU)                                                                                             \
  extern "C" __global__ void __launch_bounds__(64) sdm2_up_m2_c##CPL##_p##PF##_l##LDSU(                               \
      const __bf16* __restrict__ x, const void* __restrict__ ids, int ids64, const long long* __restrict__ L4,      \
      const long long* __restrict__ L8, int nexp, const uint8_t* __restrict__ W4, const float* __restrict__ S4,     \
      const uint8_t* __restrict__ W8, const float* __restrict__ S8, __bf16* __restrict__ act, int siluv,            \
      const float* __restrict__ wts, const uint8_t* __restrict__ pad, int has_padding, int* __restrict__ flags) {   \
    up_kernel<CPL, PF, LDSU>(x, ids, ids64, L4, L8, nexp, W4, S4, W8, S8, act, siluv, wts, pad, has_padding, flags); \
  }
#define DNK(CPL, PF, W8L)                                                                                              \
  extern "C" __global__ void __launch_bounds__(32) sdm2_dn_m0_c##CPL##_p##PF##_l##W8L(                                \
      const __bf16* __restrict__ act, const void* __restrict__ ids, int ids64, const long long* __restrict__ L4,    \
      const long long* __restrict__ L8, int nexp, const uint8_t* __restrict__ W4, const float* __restrict__ S4,     \
      const uint8_t* __restrict__ W8, const float* __restrict__ S8, __bf16* __restrict__ out,                       \
      const float* __restrict__ wts, const __bf16* __restrict__ shared, const __bf16* __restrict__ gate, int has_s, \
      int mask_padding, __bf16* __restrict__ y, int* __restrict__ cnt) {                                            \
    dn_kernel<CPL, PF, W8L>(act, ids, ids64, L4, L8, nexp, W4, S4, W8, S8, out, wts, shared, gate, has_s,          \
                            mask_padding, y, cnt);                                                                  \
  }

UPK(2, 4, 0) UPK(2, 8, 0) UPK(4, 4, 0) UPK(2, 4, 1) UPK(2, 8, 1) UPK(4, 4, 1)
DNK(1, 8, 0) DNK(1, 4, 0) DNK(2, 4, 0) DNK(1, 8, 1) DNK(1, 4, 1) DNK(2, 4, 1)

// ---------------------------------------------------------------------------------------------------------------
// Kernel-boundary probe: a dependent chain of trivial kernels (one store each) for graph per-node cost.
extern "C" __global__ void __launch_bounds__(64) chain_probe(int* __restrict__ p) {
  if (threadIdx.x == 0 && blockIdx.x == 0) p[0] = p[0] + 1;
}

// Elementwise SiLU-mul for the exhaustive gate (verbatim from sdm_v1).
extern "C" __global__ void __launch_bounds__(256) silu_mul_exh(const uint16_t* __restrict__ g, const uint16_t* __restrict__ u,
                                                               uint16_t* __restrict__ out, int n, int siluv) {
  const int i = blockIdx.x * 256 + threadIdx.x;
  if (i < n) out[i] = (uint16_t)silu_mul(g[i], u[i], siluv);
}
