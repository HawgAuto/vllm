// qsa_union_build_v1: exact native replacement for direct_qsa_union_kernel_v1.build_union (QG = 4)
//   = torch.zeros bitmap [groups, NB] + _scatter/_scatter_lead (atomic OR) + _count (per-1024-block tile counts)
//     + _compact (ascending-block ranks, keys/masks for rank < MAXU)
// Original semantics (verbatim from the Triton source):
//   entry (row, j), j < TOP, idx = I[row*SI + j]: valid = idx >= 0 && idx < LIMIT && token_to_req[row] == 0
//   bitmap[row // 4][idx // 4] |= 1 << ((row % 4) * 4 + idx % 4)
//   counts[g][t] = #{x in [1024 t, 1024 t + 1024) n [0, NB) : bitmap[g][x] != 0}
//   rank(x) = #{x' < x : bitmap[g][x'] != 0};  keys[g][rank] = x, masks[g][rank] = bitmap[g][x]  for rank < MAXU
// Here the per-group bitmap never exists: a 1-bit presence map over NB blocks lives in LDS (NB <= 33792), ranks come
// from a popcount prefix scan, and the 16-bit masks are OR-accumulated per rank in LDS (MAXU <= 12288).  Integer OR is
// associative/commutative/idempotent, so the result is independent of scatter order: outputs are bit-identical.
// keys/masks beyond the group's count are left untouched (the original allocates them with torch.empty).
#include <hip/hip_runtime.h>
#include <stdint.h>

#define NT_ 256
#define NBW_MAX 1056   // 33792 blocks
#define MAXU_MAX 12288   // = 4 x 3072 (production rows are 2051 wide: MAXU 8204)

template <typename IDX>
__device__ __forceinline__ void scan_entries(const IDX* __restrict__ I, long long SI, const int* __restrict__ R, int rows,
                                             int top, long long limit, int g, unsigned* P, const int* WOFF, unsigned* M,
                                             int maxu, bool second) {
  const int tid = threadIdx.x;
  for (int qs = 0; qs < 4; ++qs) {
    const int row = g * 4 + qs;
    if (row >= rows) break;
    if (R[row] != 0) continue;
    const IDX* base = I + (long long)row * SI;
    for (int j = tid; j < top; j += NT_) {
      const long long idx = (long long)base[j];
      if (idx < 0 || idx >= limit) continue;
      const int b = (int)(idx >> 2);
      const int w = b >> 5, s = b & 31;
      if (!second) {
        atomicOr(&P[w], 1u << s);
      } else {
        const int rank = WOFF[w] + __popc(P[w] & ((1u << s) - 1u));
        if (rank < maxu) atomicOr(&M[rank], 1u << (qs * 4 + (int)(idx & 3)));
      }
    }
  }
}

template <typename IDX>
__device__ __forceinline__ void union_body(const IDX* __restrict__ I, long long SI, const int* __restrict__ R, int rows,
                                           int top, long long limit, int nb, int nt, int maxu,
                                           int* __restrict__ keys, int* __restrict__ masks, int* __restrict__ counts) {
  __shared__ unsigned P[NBW_MAX];
  __shared__ int WOFF[NBW_MAX];
  __shared__ unsigned M[MAXU_MAX];
  __shared__ int SCAN[NT_];
  __shared__ int TOTAL;
  const int tid = threadIdx.x;
  const int g = blockIdx.x;
  const int nbw = (nb + 31) >> 5;
  for (int w = tid; w < nbw; w += NT_) P[w] = 0u;
  for (int k = tid; k < maxu; k += NT_) M[k] = 0u;
  __syncthreads();
  scan_entries<IDX>(I, SI, R, rows, top, limit, g, P, WOFF, M, maxu, false);
  __syncthreads();
  // exclusive prefix of popcounts over words: thread tid owns words [tid*c, tid*c + c)
  const int c = (nbw + NT_ - 1) / NT_;
  const int w0 = tid * c;
  int local = 0;
  for (int w = w0; w < w0 + c && w < nbw; ++w) local += __popc(P[w]);
  SCAN[tid] = local;
  __syncthreads();
  for (int off = 1; off < NT_; off <<= 1) {  // Hillis-Steele inclusive scan
    const int v = (tid >= off) ? SCAN[tid - off] : 0;
    __syncthreads();
    SCAN[tid] += v;
    __syncthreads();
  }
  int run = SCAN[tid] - local;
  for (int w = w0; w < w0 + c && w < nbw; ++w) { WOFF[w] = run; run += __popc(P[w]); }
  if (tid == NT_ - 1) TOTAL = SCAN[NT_ - 1];
  __syncthreads();
  const int total = TOTAL;
  for (int t = tid; t < nt; t += NT_) {
    const int lo = t * 32, hi = min(lo + 32, nbw);
    const int a = (lo < nbw) ? WOFF[lo] : total;
    const int b = (hi < nbw) ? WOFF[hi] : total;
    counts[(long long)g * nt + t] = (hi > lo) ? (b - a) : 0;
  }
  scan_entries<IDX>(I, SI, R, rows, top, limit, g, P, WOFF, M, maxu, true);
  __syncthreads();
  for (int w = tid; w < nbw; w += NT_) {
    unsigned bits = P[w];
    int r = WOFF[w];
    while (bits) {
      const int s = __ffs(bits) - 1;
      if (r < maxu) {
        keys[(long long)g * maxu + r] = w * 32 + s;
        masks[(long long)g * maxu + r] = (int)M[r];
      }
      bits &= bits - 1u;
      ++r;
    }
  }
}

extern "C" __global__ __launch_bounds__(NT_) void qsa_union_build_v1_i32(
    const int32_t* __restrict__ I, long long SI, const int* __restrict__ R, int rows, int top, long long limit, int nb, int nt,
    int maxu, int* __restrict__ keys, int* __restrict__ masks, int* __restrict__ counts) {
  union_body<int32_t>(I, SI, R, rows, top, limit, nb, nt, maxu, keys, masks, counts);
}

extern "C" __global__ __launch_bounds__(NT_) void qsa_union_build_v1_i64(
    const int64_t* __restrict__ I, long long SI, const int* __restrict__ R, int rows, int top, long long limit, int nb, int nt,
    int maxu, int* __restrict__ keys, int* __restrict__ masks, int* __restrict__ counts) {
  union_body<int64_t>(I, SI, R, rows, top, limit, nb, nt, maxu, keys, masks, counts);
}
