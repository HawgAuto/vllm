// decode23 qsa-normrope (code object v3 = v2 + qng_r2): sum-of-squares order fixed to screen-identified V1;
// rsqrt implementation is a compile-time RS variant screened exhaustively against torch.rsqrt.
// (v1 code object: attempts/decode23-qsa-normrope-01 -> V1 var bits 6000/6000 but device rsqrtf != torch.rsqrt)
// fused QSA-indexer GemmaRMSNorm + interleaved-MRoPE (NeoX) for gfx1151.
//
// Replaces, per QSA decode layer, the eager sequence (q: 14 kernels, k: 14 kernels):
//   w1 = weight.float() + 1.0                       (bf16->f32 copy, add)
//   x.to(f32); var = x.pow(2).mean(-1); x*rsqrt(var+eps); x*w1; .to(bf16)   (8 kernels)
//   cos_sin = cache[positions]; cos,sin = chunk; .contiguous() x2           (gather + 2 copies)
//   _triton_mrope_forward (in place, bf16 arithmetic)                        (1 kernel)
// with ONE launch. Arithmetic is replicated op-for-op:
//   * all f32 ops are separate IEEE ops (built with -ffp-contract=off);
//   * the mean's sum-of-squares order is a compile-time VARIANT selected by an exact screen against the
//     installed torch reduce kernel (no variant is assumed);
//   * rsqrtf is the device-library rsqrt (screened exhaustively against torch.rsqrt);
//   * the rope is Triton's bf16 semantics: every bf16 op = f32 op then RNE to bf16 (no contraction);
//     cos_row = (t + h) + w over masked (other=0) bf16 loads, exactly like the Triton kernel.
// Scope: head_dim 128, rotary_dim 64 (NeoX halves of 32), mrope interleaved; one row of 128 per block.
#include <hip/hip_runtime.h>
#include <stdint.h>

static __device__ __forceinline__ float bf2f(uint16_t b) { return __uint_as_float(((uint32_t)b) << 16); }
static __device__ __forceinline__ uint16_t f2bf(float f) {          // torch/LLVM RNE (NaN -> quiet)
  uint32_t u = __float_as_uint(f);
  if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)0x7fc0u;   // c10::BFloat16 NaN
  u += 0x7fffu + ((u >> 16) & 1u);
  return (uint16_t)(u >> 16);
}
static __device__ __forceinline__ uint16_t bmul(uint16_t a, uint16_t b) { return f2bf(bf2f(a) * bf2f(b)); }
static __device__ __forceinline__ uint16_t badd(uint16_t a, uint16_t b) { return f2bf(bf2f(a) + bf2f(b)); }
static __device__ __forceinline__ uint16_t bsub(uint16_t a, uint16_t b) { return f2bf(bf2f(a) - bf2f(b)); }


template <int RS>
static __device__ __forceinline__ float rsq(float x) {
  if (RS == 0) return rsqrtf(x);                              // device-lib approx (rejected by screen 01)
  if (RS == 1) return 1.0f / sqrtf(x);                        // correctly rounded sqrt, IEEE div
  if (RS == 2) return (float)(1.0 / sqrt((double)x));         // double-precision reference
  return __frsqrt_rn(x);                                      // HIP round-to-nearest rsqrt
}

// Sum of 128 squares in lane 0 of wave 0 (all 32 lanes of wave 0 must call). sq[] in LDS.
template <int V>
static __device__ __forceinline__ float sumsq(const float* sq, int l) {
  float v;
  if (V == 0 || V == 3) {          // block_x 128, scalar per thread; LDS tree 64,32; then shfl
    v = (sq[l] + sq[l + 64]) + (sq[l + 32] + sq[l + 96]);
  } else if (V == 1 || V == 4) {   // vectorized x4 per thread, 32 threads: ((a0+a1)+a2)+a3
    v = ((sq[4 * l] + sq[4 * l + 1]) + sq[4 * l + 2]) + sq[4 * l + 3];
  } else {                         // scalar, 4 strided per thread (stride 32)
    v = ((sq[l] + sq[l + 32]) + sq[l + 64]) + sq[l + 96];
  }
  if (V < 3) {
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) v = v + __shfl_down(v, o, 32);
  } else {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = v + __shfl_down(v, o, 32);
  }
  return v;
}

// grid: (rows), block: 128. Row r belongs to token t = r / heads.
// pos: int64 or int32 (pos64) [3][T] with element strides (ps0, ps1). cache: bf16 [maxpos][64] = [cos(32) | sin(32)].
template <int V, int RS>
static __device__ void qnr_body(const uint16_t* __restrict__ x, long sx, const uint16_t* __restrict__ w, float eps,
                                const uint16_t* __restrict__ cache, long cs, const void* __restrict__ pos, int pos64,
                                long ps0, long ps1, int heads, int T, uint16_t* __restrict__ out, long so) {
  __shared__ float sq[128];
  __shared__ uint16_t nb[128];
  __shared__ float s_rs;
  const int r = blockIdx.x, t = threadIdx.x, tok = r / heads;
  const float f = bf2f(x[(long)r * sx + t]);
  sq[t] = f * f;
  __syncthreads();
  if (t < 32) {
    const float s = sumsq<V>(sq, t);
    if (t == 0) {
      const float var = s * (1.0f / 128.0f);   // MeanOps project: acc * factor (factor = 1/128, exact)
      s_rs = rsq<RS>(var + eps);
    }
  }
  __syncthreads();
  const float w1 = bf2f(w[t]) + 1.0f;
  const float y = (f * s_rs) * w1;
  nb[t] = f2bf(y);
  __syncthreads();
  uint16_t* o = out + (long)r * so;
  if (t < 32) {
    const int j = t;
    const bool hm = ((j % 3) == 1) && (j <= 3 * 11);
    const bool wm = ((j % 3) == 2) && (j <= 3 * 10);
    const bool tm = !(hm || wm);
    const long i0 = 0 * ps0 + (long)tok * ps1, i1 = 1 * ps0 + (long)tok * ps1, i2 = 2 * ps0 + (long)tok * ps1;
    const long p0 = pos64 ? ((const int64_t*)pos)[i0] : (long)((const int32_t*)pos)[i0];
    const long p1 = pos64 ? ((const int64_t*)pos)[i1] : (long)((const int32_t*)pos)[i1];
    const long p2 = pos64 ? ((const int64_t*)pos)[i2] : (long)((const int32_t*)pos)[i2];
    const uint16_t tc = tm ? cache[p0 * cs + j] : (uint16_t)0, hc = hm ? cache[p1 * cs + j] : (uint16_t)0,
                   wc = wm ? cache[p2 * cs + j] : (uint16_t)0;
    const uint16_t ts = tm ? cache[p0 * cs + 32 + j] : (uint16_t)0, hs = hm ? cache[p1 * cs + 32 + j] : (uint16_t)0,
                   ws = wm ? cache[p2 * cs + 32 + j] : (uint16_t)0;
    const uint16_t c = badd(badd(tc, hc), wc), sn = badd(badd(ts, hs), ws);
    const uint16_t q1 = nb[j], q2 = nb[j + 32];
    o[j] = bsub(bmul(q1, c), bmul(q2, sn));
    o[j + 32] = badd(bmul(q2, c), bmul(q1, sn));
  } else if (t >= 64) {
    o[t] = nb[t];
  }
}

#define QNR(RS)                                                                                             \
  extern "C" __global__ void __launch_bounds__(128) qnr_r##RS(                                              \
      const uint16_t* x, long sx, const uint16_t* w, float eps, const uint16_t* cache, long cs,             \
      const void* pos, int pos64, long ps0, long ps1, int heads, int T, uint16_t* out, long so) {           \
    qnr_body<1, RS>(x, sx, w, eps, cache, cs, pos, pos64, ps0, ps1, heads, T, out, so);                     \
  }
QNR(0) QNR(1) QNR(2) QNR(3)

#define QN(RS)                                                                                              \
  extern "C" __global__ void __launch_bounds__(128) qn_r##RS(const uint16_t* x, const uint16_t* w, float eps, \
                                                            uint16_t* out, float* var_out) {                \
    __shared__ float sq[128];                                                                               \
    __shared__ float s_rs;                                                                                  \
    const int r = blockIdx.x, t = threadIdx.x;                                                              \
    const float f = bf2f(x[(long)r * 128 + t]);                                                             \
    sq[t] = f * f;                                                                                          \
    __syncthreads();                                                                                        \
    if (t < 32) {                                                                                           \
      const float s = sumsq<1>(sq, t);                                                                      \
      if (t == 0) { const float var = s * (1.0f / 128.0f); var_out[r] = var; s_rs = rsq<RS>(var + eps); }   \
    }                                                                                                       \
    __syncthreads();                                                                                        \
    out[(long)r * 128 + t] = f2bf((f * s_rs) * (bf2f(w[t]) + 1.0f));                                        \
  }
QN(0) QN(1) QN(2) QN(3)


// ---- code object v3: exact norm (V1 order, RS2 rsqrt; screen 03: 6000/6000 both shapes, rsqrt 0/2139095039) fused with the
// cos/sin gather + .contiguous() copies of apply_qsa_rope. The rotation itself stays the installed Triton _triton_mrope_forward
// (it lowers bf16 products to v_dot2_bf16_bf16; not re-emulated). grid: rows, block 128.
// out: bf16 [rows][128] contiguous; cos_out/sin_out: bf16 [3][T][32] contiguous (== cache[positions].chunk(2,-1)[i].contiguous()).
extern "C" __global__ void __launch_bounds__(128) qng_r2(
    const uint16_t* __restrict__ x, long sx, const uint16_t* __restrict__ w, float eps, const uint16_t* __restrict__ cache,
    long cs, const void* __restrict__ pos, int pos64, long ps0, long ps1, int heads, int T, uint16_t* __restrict__ out,
    uint16_t* __restrict__ cos_out, uint16_t* __restrict__ sin_out) {
  __shared__ float sq[128];
  __shared__ float s_rs;
  const int r = blockIdx.x, t = threadIdx.x;
  const float f = bf2f(x[(long)r * sx + t]);
  sq[t] = f * f;
  __syncthreads();
  if (t < 32) {
    const float s = sumsq<1>(sq, t);
    if (t == 0) s_rs = rsq<2>(s * (1.0f / 128.0f) + eps);
  }
  __syncthreads();
  out[(long)r * 128 + t] = f2bf((f * s_rs) * (bf2f(w[t]) + 1.0f));
  if (r % heads == 0 && t < 96) {                 // one block per token gathers its 3 position rows
    const int tok = r / heads, d = t / 32, j = t % 32;
    const long i = (long)d * ps0 + (long)tok * ps1;
    const long p = pos64 ? ((const int64_t*)pos)[i] : (long)((const int32_t*)pos)[i];
    cos_out[((long)d * T + tok) * 32 + j] = cache[p * cs + j];
    sin_out[((long)d * T + tok) * 32 + j] = cache[p * cs + 32 + j];
  }
}

// Exhaustive rsqrt probes: out[i] = rsq<RS>(bits(base + i)).
#define PR(RS)                                                                                              \
  extern "C" __global__ void rsqrt_probe_r##RS(uint32_t base, uint32_t n, float* out) {                     \
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;                                               \
    if (i < n) out[i] = rsq<RS>(__uint_as_float(base + i));                                                 \
  }
PR(0) PR(1) PR(2) PR(3)
