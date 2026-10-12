// qsa_mqa_scores_v4: bitwise-faithful native replacement for ops/qsa.py::_qsa_mqa_paged_kernel
// Same exact arithmetic as v1/v3 (tree documented in qsa_mqa_scores_v1.hip):
//   T_l = ((p0+p1)+(p2+p3)) + ((p4+p5)+(p6+p7)) over dims 8l..8l+7, p = q*k exact in fp32
//   dot = (((T0+T8)+(T4+T12)) + ((T2+T10)+(T6+T14))) + (((T1+T9)+(T5+T13)) + ((T3+T11)+(T7+T15)))
//   score = (((0 + max(d0,0)) + max(d1,0)) + max(d2,0)) + max(d3,0); score / divisor (IEEE); where(valid, s, -inf)
// Structure: one key column per lane (128 fp32 VGPRs, loaded once per workgroup); query values are wave-uniform and
// are read with uniform (scalar) global loads, converted bf16->fp32 on the scalar unit and consumed as SGPR operands.
// No LDS.  The 16 lane-chunk partials of one (row, head) are formed in two halves (dims 0..63, 64..127).
#include <hip/hip_runtime.h>
#include <stdint.h>

#define KT 256
#define HD 128
#define NH 4
#define NT 256

__device__ __forceinline__ void unpack8(const uint4 v, float* o) {
  o[0] = __uint_as_float(v.x << 16); o[1] = __uint_as_float(v.x & 0xffff0000u);
  o[2] = __uint_as_float(v.y << 16); o[3] = __uint_as_float(v.y & 0xffff0000u);
  o[4] = __uint_as_float(v.z << 16); o[5] = __uint_as_float(v.z & 0xffff0000u);
  o[6] = __uint_as_float(v.w << 16); o[7] = __uint_as_float(v.w & 0xffff0000u);
}

// chunk partial for dims 8L..8L+7 with q packed bf16 (uniform) and k fp32 (per lane)
#define CHUNKU(QV, L)                                                                                          \
  (((__uint_as_float((QV).x << 16) * kr[8 * (L) + 0] + __uint_as_float((QV).x & 0xffff0000u) * kr[8 * (L) + 1]) + \
    (__uint_as_float((QV).y << 16) * kr[8 * (L) + 2] + __uint_as_float((QV).y & 0xffff0000u) * kr[8 * (L) + 3])) + \
   ((__uint_as_float((QV).z << 16) * kr[8 * (L) + 4] + __uint_as_float((QV).z & 0xffff0000u) * kr[8 * (L) + 5]) + \
    (__uint_as_float((QV).w << 16) * kr[8 * (L) + 6] + __uint_as_float((QV).w & 0xffff0000u) * kr[8 * (L) + 7])))

extern "C" __global__ __launch_bounds__(NT) void qsa_mqa_scores_v4(
    const uint16_t* __restrict__ q, const uint16_t* __restrict__ kc, const int32_t* __restrict__ table,
    const int32_t* __restrict__ tok2req, const int64_t* __restrict__ qpos, const int32_t* __restrict__ seqlens,
    int32_t* __restrict__ visible_out, float* __restrict__ logits,
    int stride_q_row, int stride_q_head, int stride_cache_block, int stride_cache_token, int stride_logits_row,
    int num_rows, int num_columns, int num_pages, int page_size, int table_width, int compress_ratio,
    float score_divisor, int rows_per_block) {
  const int tid = threadIdx.x;
  const int col0 = blockIdx.x * KT;
  const int c = col0 + tid;
  const int wave_col0 = col0 + (tid & ~31);
  const int rowA = blockIdx.y * rows_per_block;
  const int rowB = min(num_rows, rowA + rows_per_block);
  bool colok = false;
  int64_t pp = 0; int off = 0;
  if (c < num_columns) {
    const int lp = c / page_size;
    off = c - lp * page_size;
    if (lp < table_width) {
      const int p = table[lp];
      colok = (p >= 0) && (p < num_pages);
      pp = p;
    }
  }
  float kr[HD];
#pragma unroll
  for (int v = 0; v < HD / 8; ++v) {
    uint4 raw = make_uint4(0u, 0u, 0u, 0u);
    if (colok) raw = *reinterpret_cast<const uint4*>(kc + pp * (int64_t)stride_cache_block + (int64_t)off * stride_cache_token + v * 8);
    unpack8(raw, &kr[v * 8]);
  }
  const int sl_raw = seqlens[0];
  for (int r = rowA; r < rowB; ++r) {
    const int req = tok2req[r];
    const int64_t pos = qpos[r];
    const bool req_ok = (req >= 0) && (req < 1);
    const int sl = req_ok ? sl_raw : 0;
    const int64_t a = (pos + 1) / (int64_t)compress_ratio;
    const int64_t b = (int64_t)(sl / compress_ratio);
    const int64_t v = a < b ? a : b;
    if (blockIdx.x == 0 && tid == 0) visible_out[r] = (int32_t)v;
    const int vis = req_ok ? (v <= 0 ? 0 : (v > 0x7fffffff ? 0x7fffffff : (int)v)) : 0;
    float out = -__builtin_inff();
    if (wave_col0 < vis) {
      float score = 0.0f;
      const uint4* qr = reinterpret_cast<const uint4*>(q + (int64_t)r * stride_q_row);
#pragma unroll 1
      for (int h = 0; h < NH; ++h) {
        const uint4* qh = reinterpret_cast<const uint4*>(reinterpret_cast<const uint16_t*>(qr) + (int64_t)h * stride_q_head);
        // first half: T0..T7
        float T0, T1, T2, T3, T4, T5, T6, T7;
        { const uint4 a0 = qh[0], a1 = qh[1], a2 = qh[2], a3 = qh[3];
          T0 = CHUNKU(a0, 0); T1 = CHUNKU(a1, 1); T2 = CHUNKU(a2, 2); T3 = CHUNKU(a3, 3); }
        { const uint4 a4 = qh[4], a5 = qh[5], a6 = qh[6], a7 = qh[7];
          T4 = CHUNKU(a4, 4); T5 = CHUNKU(a5, 5); T6 = CHUNKU(a6, 6); T7 = CHUNKU(a7, 7); }
        float A0, A1, A2, A3, A4, A5, A6, A7;
        { const uint4 b8 = qh[8], b9 = qh[9], b10 = qh[10], b11 = qh[11];
          A0 = T0 + CHUNKU(b8, 8); A1 = T1 + CHUNKU(b9, 9); A2 = T2 + CHUNKU(b10, 10); A3 = T3 + CHUNKU(b11, 11); }
        { const uint4 b12 = qh[12], b13 = qh[13], b14 = qh[14], b15 = qh[15];
          A4 = T4 + CHUNKU(b12, 12); A5 = T5 + CHUNKU(b13, 13); A6 = T6 + CHUNKU(b14, 14); A7 = T7 + CHUNKU(b15, 15); }
        const float dot = ((A0 + A4) + (A2 + A6)) + ((A1 + A5) + (A3 + A7));
        score = score + fmaxf(dot, 0.0f);
      }
      const float s = score / score_divisor;
      out = (colok && c < vis) ? s : -__builtin_inff();
    }
    if (c < num_columns) logits[(int64_t)r * stride_logits_row + c] = out;
  }
}
