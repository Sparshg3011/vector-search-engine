// K2: k smallest values per row of an (nq, n) fp32 matrix, ascending,
// with their column ids. One block per row. Each thread keeps a sorted
// shortlist over a strided slice, then the block merges the shortlists
// with k rounds of argmin.

#include "config.cuh"

#define K2_THREADS VSG_K2_THREADS
#define K2_WARPS (VSG_K2_THREADS / 32)
#define K2_FULL_MASK 0xffffffffu

__global__ void vsg_k2_topk(const float* __restrict__ dmat,
                            int* __restrict__ out_ids,
                            float* __restrict__ out_dists, int n, int k) {
  const float INF = __int_as_float(0x7f800000);
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const float* row = dmat + (long long)blockIdx.x * n;

  __shared__ float warp_best_d[K2_WARPS];
  __shared__ int warp_best_t[K2_WARPS];
  __shared__ int round_winner;

  float cand_d[VSG_MAX_K];
  int cand_i[VSG_MAX_K];
  for (int j = 0; j < k; j++) {
    cand_d[j] = INF;
    cand_i[j] = -1;
  }
  for (int i = tid; i < n; i += K2_THREADS) {
    float d = row[i];
    if (d < cand_d[k - 1]) {
      int j = k - 1;
      while (j > 0 && cand_d[j - 1] > d) {
        cand_d[j] = cand_d[j - 1];
        cand_i[j] = cand_i[j - 1];
        j--;
      }
      cand_d[j] = d;
      cand_i[j] = i;
    }
  }

  int cursor = 0;
  for (int r = 0; r < k; r++) {
    float v = (cursor < k) ? cand_d[cursor] : INF;
    int owner = tid;
    for (int off = 16; off > 0; off >>= 1) {
      float ov = __shfl_down_sync(K2_FULL_MASK, v, off);
      int oo = __shfl_down_sync(K2_FULL_MASK, owner, off);
      if (ov < v) {
        v = ov;
        owner = oo;
      }
    }
    if (lane == 0) {
      warp_best_d[warp] = v;
      warp_best_t[warp] = owner;
    }
    __syncthreads();

    if (tid == 0) {
      float best = warp_best_d[0];
      int best_t = warp_best_t[0];
      for (int w = 1; w < K2_WARPS; w++) {
        if (warp_best_d[w] < best) {
          best = warp_best_d[w];
          best_t = warp_best_t[w];
        }
      }
      out_dists[(long long)blockIdx.x * k + r] = best;
      round_winner = best_t;
    }
    __syncthreads();

    if (tid == round_winner) {
      out_ids[(long long)blockIdx.x * k + r] = cand_i[cursor];
      cursor++;
    }
  }
}
