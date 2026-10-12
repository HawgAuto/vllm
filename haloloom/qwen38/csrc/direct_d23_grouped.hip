#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <type_traits>
#if !defined(__gfx1151__)
#error gfx1151 required
#endif
#define __HIP__GFX1X__
#define LDS_SIZE 65536
template <typename T>
__device__ __forceinline__ float2 __s22float2(T v);

template <typename T>
__device__ __forceinline__ T __float2s(float v);

template <typename T>
__device__ __forceinline__ T __float22s2_rn(float2 v);

template <>
__device__ __forceinline__ half __float2s(float v) {
  return __float2half(v);
}

template <>
__device__ __forceinline__ float2 __s22float2(__half2 v) {
  return __half22float2(v);
}

template <>
__device__ __forceinline__ __half2 __float22s2_rn(float2 v) {
  return __float22half2_rn(v);
}
template <>
__device__ __forceinline__ __hip_bfloat16 __float2s(float v) {
  return __float2bfloat16(v);
}

template <>
__device__ __forceinline__ float2 __s22float2(__hip_bfloat162 v) {
  return __bfloat1622float2(v);
}

template <>
__device__ __forceinline__ __hip_bfloat162 __float22s2_rn(float2 v) {
  return __float22bfloat162_rn(v);
}

template <typename T>
__device__ __forceinline__ T loadnt(T* addr) {
  return __builtin_nontemporal_load(addr);
}

__device__ __forceinline__ float4 load_ntmprl(const float4* addr) {
  auto addr_alias = reinterpret_cast<const float*>(addr);
  auto dat0 = loadnt(addr_alias);
  auto dat1 = loadnt(addr_alias + 1);
  auto dat2 = loadnt(addr_alias + 2);
  auto dat3 = loadnt(addr_alias + 3);
  return make_float4(dat0, dat1, dat2, dat3);
}
#if defined(__HIP__GFX9__) && !defined(__HIP__GFX1X__)
  #define DOT2C(V0, V2, V3)                                          \
    if constexpr (std::is_same_v<scalar_t, half>) {                  \
      asm("v_dot2c_f32_f16 %0, %2, %3"                               \
          : "=v"(V0)                                                 \
          : "0"(V0), "v"(V2), "v"(V3));                              \
    } else if constexpr (std::is_same_v<scalar_t, __hip_bfloat16>) { \
      float2 s = __bfloat1622float2(*((__hip_bfloat162*)(&(V2)))) *  \
                 __bfloat1622float2(*((__hip_bfloat162*)(&(V3))));   \
      V0 += (s.x + s.y);                                             \
    }
#elif defined(__HIP__GFX1X__)
  // gfx1x: v_dot2_f32_f16 (VOP3-P, dot10-insts, available on gfx11+gfx12)
  #define DOT2C(V0, V2, V3)                                               \
    if constexpr (std::is_same_v<scalar_t, half>) {                       \
      asm("v_dot2_f32_f16 %0, %1, %2, %0" : "+v"(V0) : "v"(V2), "v"(V3)); \
    } else if constexpr (std::is_same_v<scalar_t, __hip_bfloat16>) {      \
      float2 s = __bfloat1622float2(*((__hip_bfloat162*)(&(V2)))) *       \
                 __bfloat1622float2(*((__hip_bfloat162*)(&(V3))));        \
      V0 += (s.x + s.y);                                                  \
    }
#endif

// To avoid LLVM silently upcasting to double
__device__ inline unsigned int min__(uint32_t a, uint32_t b) {
  return min(a, b);
}

template <typename scalar_t, int THRDS, int YTILE, int WvPrGrp, int A_CHUNK,
          int UNRL, int N>
__device__ __forceinline__ void d23_tile(const int K, const int Kbp, const int Kap, const int M,
                     const int Bx, const int By, const scalar_t* B,
                     const scalar_t* __restrict__ BIAS, scalar_t* C,
                     const scalar_t* s, uint32_t m) {
  constexpr int max_lds_len = LDS_SIZE / 2;
  #if defined(__HIP__MI3XX__)
  constexpr bool use_mfma = (std::is_same_v<scalar_t, __hip_bfloat16>);
  #else
  constexpr bool use_mfma = false;
  #endif
  using scalar8 =
      __attribute__((__vector_size__((A_CHUNK / 2) * sizeof(float)))) float;
  using half4 =
      __attribute__((__vector_size__((A_CHUNK / 2) * sizeof(__bf16)))) __bf16;
  union bigType {
    scalar_t h[A_CHUNK];
    float f[A_CHUNK / 2];
    float2 f2[A_CHUNK / 4];
    double d[A_CHUNK / 4];
    half4 h4[A_CHUNK / 4];
    scalar8 h8;
  };

  //----------------------------------------------------
  // Reserving 64/160 KB of LDS to have 1 WG / CU
  // Goal is to bring the activation matrix A to the LDS
  // and use it across the lifetime of the work group
  // TODO: When activation matrix is larger than 64 KB
  //	     then this is not going to work!
  //----------------------------------------------------
  {
    //----------------------------------------------------
    // 'sum' accumulates the matrix A x B computation
    // split across 64 lanes.
    //
    // YTILE represents how many column of weight matrix
    // are being worked on by each wave.
    //----------------------------------------------------
    float sum[N][YTILE] = {};
    scalar8 sum4[N][YTILE] = {};

    for (uint32_t k1 = 0; k1 < K; k1 += THRDS * A_CHUNK * UNRL) {
      bigType bigA[N][UNRL] = {};
      bigType bigB[YTILE][UNRL];
      // Fetch the weight matrix from memory!
  #pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + threadIdx.x * A_CHUNK;
        const scalar_t* B_ = &B[min__(k_, K - A_CHUNK)];
        for (int y = 0; y < YTILE; y++)
          bigB[y][k2].h8 = (loadnt((scalar8*)(&B_[min__(y + m, M - 1) * Kbp])));
      }

      // Fetch activation matrix from either just LDS or from both LDS / memory
  #pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + threadIdx.x * A_CHUNK;
        if (k_ >= K) break;
        for (int n = 0; n < N; n++) {
          bigA[n][k2] = *((const bigType*)(&(s[k_ + Kap * n])));
        }
      }

      // Do the matrix multiplication in interleaved manner
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        for (uint32_t n = 0; n < N; n++) {
          for (int y = 0; y < YTILE; y++) {
            if constexpr (!use_mfma)
              for (uint32_t b = 0; b < A_CHUNK / 2; b++) {
                DOT2C(sum[n][y], bigA[n][k2].f[b], bigB[y][k2].f[b])
              }
            else
              for (uint32_t b = 0; b < A_CHUNK / 4; b++)
                sum4[n][y] = __builtin_amdgcn_mfma_f32_4x4x4bf16_1k(
                    bigA[n][k2].h4[b], bigB[y][k2].h4[b], sum4[n][y], 0, 0, 0);
          }
        }
      }
    }
    __builtin_amdgcn_sched_barrier(0);
    //----------------------------------------------------
    // Final reduction step using shuffle
    //----------------------------------------------------
    if constexpr (!use_mfma) {
      for (int n = 0; n < N; n++) {
        for (int y = 0; y < YTILE; y++) {
          sum[n][y] += __builtin_amdgcn_mov_dpp(sum[n][y], 0x118, 0xf, 0xf,
                                                1);  // row_shr8
          sum[n][y] += __builtin_amdgcn_mov_dpp(sum[n][y], 0x114, 0xf, 0xf,
                                                1);  // row_shr4
          sum[n][y] += __builtin_amdgcn_mov_dpp(sum[n][y], 0x112, 0xf, 0xf,
                                                1);  // row_shr2
          sum[n][y] += __builtin_amdgcn_mov_dpp(sum[n][y], 0x111, 0xf, 0xf,
                                                1);  // row_shr1
  #if defined(__HIP__GFX9__)
          sum[n][y] += __builtin_amdgcn_mov_dpp(sum[n][y], 0x142, 0xf, 0xf,
                                                1);  // ROW_BCAST15
          sum[n][y] += __builtin_amdgcn_mov_dpp(sum[n][y], 0x143, 0xf, 0xf,
                                                1);  // ROW_BCAST31
  #else
          sum[n][y] += __shfl_xor(sum[n][y], 16);
  #endif
        }
      }

      if (threadIdx.x == (THRDS - 1)) {
        scalar_t biases[N][YTILE] = {};
        if (BIAS)
          for (int n = 0; n < N; n++) {
            for (int y = 0; y < YTILE; y++) {
              biases[n][y] = BIAS[(m + y) % Bx + (n % By) * Bx];
            }
          }
        for (int n = 0; n < N; n++) {
          for (int y = 0; y < YTILE; y++) {
            if constexpr (std::is_same_v<scalar_t, half>) {
              sum[n][y] += __half2float(biases[n][y]);
            } else if constexpr (std::is_same_v<scalar_t, __hip_bfloat16>) {
              sum[n][y] += __bfloat162float(biases[n][y]);
            }
            C[m + y + n * M] = __float2s<scalar_t>(sum[n][y]);
          }
        }
      }
    } else {
  #ifdef __HIP__GFX9__
    #pragma unroll
      for (int n = 0; n < N; n++) {
    #pragma unroll
        for (int y = 0; y < YTILE; y++) {
          /*float accm1 = 0;
           for (int i=0; i<64; i++)
              accm1 += __shfl(sum4[n][y][i%4], i);
          sum4[n][y][0] = accm1;*/
          float accm = sum4[n][y][0];
          accm += __builtin_amdgcn_mov_dpp(sum4[n][y][1], 0x101, 0xf, 0xf,
                                           1);  // row_shl1
          accm += __builtin_amdgcn_mov_dpp(sum4[n][y][2], 0x102, 0xf, 0xf,
                                           1);  // row_shl2
          accm += __builtin_amdgcn_mov_dpp(sum4[n][y][3], 0x103, 0xf, 0xf,
                                           1);  // row_shl3
          accm += __builtin_amdgcn_mov_dpp(accm, 0x104, 0xf, 0xf,
                                           1);  // row_shl4
          accm += __builtin_amdgcn_mov_dpp(accm, 0x108, 0xf, 0xf,
                                           1);  // row_shl8
          accm = __builtin_amdgcn_mov_dpp(accm, 0x11f, 0xf, 0xf,
                                          1);  // row_shr15
          accm += __builtin_amdgcn_mov_dpp(accm, 0x142, 0xf, 0xf,
                                           1);  // ROW_BCAST15
          accm += __builtin_amdgcn_mov_dpp(accm, 0x143, 0xf, 0xf,
                                           1);  // ROW_BCAST31

          sum4[n][y][0] = accm;
        }
      }
      if (threadIdx.x == (THRDS - 1)) {
        scalar_t biases[N][YTILE] = {};
        if (BIAS)
          for (int n = 0; n < N; n++) {
            for (int y = 0; y < YTILE; y++) {
              biases[n][y] = BIAS[(m + y) % Bx + (n % By) * Bx];
            }
          }
        for (int n = 0; n < N; n++) {
          for (int y = 0; y < YTILE; y++) {
            sum4[n][y][0] += __bfloat162float(biases[n][y]);
            C[m + y + n * M] = __float2bfloat16(sum4[n][y][0]);
          }
        }
      }
  #endif  // __HIP__GFX9__ (MFMA path)
    }
  }
}

template <typename scalar_t, int THRDS, int WvPrGrp, int A_CHUNK, int N>
__global__ void __launch_bounds__(WvPrGrp* THRDS)
    d23_grouped(const int K, const int Kap, const scalar_t* __restrict__ A,
                const scalar_t* B0, scalar_t* C0, const int M0, const int V0, const int T0,
                const scalar_t* B1, scalar_t* C1, const int M1, const int V1, const int T1,
                const scalar_t* B2, scalar_t* C2, const int M2, const int V2, const int T2,
                const int _WvPrGrp, const int nWaves) {
  constexpr int max_lds_len = LDS_SIZE / 2;
  #if defined(__HIP__MI3XX__)
  constexpr bool use_mfma = (std::is_same_v<scalar_t, __hip_bfloat16>);
  #else
  constexpr bool use_mfma = false;
  #endif
  using scalar8 =
      __attribute__((__vector_size__((A_CHUNK / 2) * sizeof(float)))) float;
  using half4 =
      __attribute__((__vector_size__((A_CHUNK / 2) * sizeof(__bf16)))) __bf16;
  union bigType {
    scalar_t h[A_CHUNK];
    float f[A_CHUNK / 2];
    float2 f2[A_CHUNK / 4];
    double d[A_CHUNK / 4];
    half4 h4[A_CHUNK / 4];
    scalar8 h8;
  };

  //----------------------------------------------------
  // Reserving 64/160 KB of LDS to have 1 WG / CU
  // Goal is to bring the activation matrix A to the LDS
  // and use it across the lifetime of the work group
  // TODO: When activation matrix is larger than 64 KB
  //	     then this is not going to work!
  //----------------------------------------------------
  __shared__ scalar_t s[max_lds_len];

  //----------------------------------------------------
  // Fetch the activation matrix to LDS
  // Loop iteration:
  // - Each thread (lane) is fetching 8 elements (A_Chunk)
  // - Each wave will fetch 64*8=> 512 elements
  // - Each WG will fetch 512 * 16 => 8K elements
  // - Then the WG will move to another 8 K elements
  // TODO: Logic below will only work when K is multiple of 8
  //----------------------------------------------------
  for (uint32_t k = (threadIdx.y * THRDS + threadIdx.x) * A_CHUNK;
       k < min__(Kap * N, max_lds_len); k += THRDS * WvPrGrp * A_CHUNK) {
  #if defined(__gfx950__)
    __builtin_amdgcn_global_load_lds((int*)(&A[k]), (int*)(&s[k]), 16, 0, 0);
  #else
    *((bigType*)(&s[k])) = *((bigType*)(&A[k]));
  #endif
  }
  __syncthreads();

  if (threadIdx.y >= _WvPrGrp) return;

  const uint32_t total = (uint32_t)(T0 + T1 + T2);
  for (uint32_t t = blockIdx.x * _WvPrGrp + (threadIdx.y % _WvPrGrp); t < total; t += (uint32_t)nWaves) {
    const scalar_t* Bx_; scalar_t* Cx; int Mx, Vx; uint32_t lt;
    if (t < (uint32_t)T0) { Bx_ = B0; Cx = C0; Mx = M0; Vx = V0; lt = t; }
    else if (t < (uint32_t)(T0 + T1)) { Bx_ = B1; Cx = C1; Mx = M1; Vx = V1; lt = t - T0; }
    else { Bx_ = B2; Cx = C2; Mx = M2; Vx = V2; lt = t - T0 - T1; }
    switch (Vx) {
      case 12: d23_tile<scalar_t, THRDS, 1, WvPrGrp, A_CHUNK, 2, N>(K, K, Kap, Mx, 1, 1, Bx_, nullptr, Cx, s, lt * 1); break;
      case 14: d23_tile<scalar_t, THRDS, 1, WvPrGrp, A_CHUNK, 4, N>(K, K, Kap, Mx, 1, 1, Bx_, nullptr, Cx, s, lt * 1); break;
      case 41: d23_tile<scalar_t, THRDS, 4, WvPrGrp, A_CHUNK, 1, N>(K, K, Kap, Mx, 1, 1, Bx_, nullptr, Cx, s, lt * 4); break;
      default: __builtin_trap();
    }
  }
}

template __global__ void d23_grouped<__hip_bfloat16,32,16,8,1>(const int,const int,const __hip_bfloat16* __restrict__,const __hip_bfloat16*,__hip_bfloat16*,const int,const int,const int,const __hip_bfloat16*,__hip_bfloat16*,const int,const int,const int,const __hip_bfloat16*,__hip_bfloat16*,const int,const int,const int,const int,const int);
