// K1: out[q * n + b] = dist(queries[q], base[b]) for fp16 rows of length
// stride (pad columns are zero). Squared l2, or negated dot product for
// ip. Block = 16x16 threads owning a 16x16 patch of out; the dim axis is
// walked 32 at a time through shared memory.

#include "config.cuh"
#include <cuda_fp16.h>

#define K1_TILE VSG_K1_TILE
#define K1_CHUNK (2 * VSG_K1_TILE)

__global__ void vsg_k1_distances(const __half* __restrict__ queries,
                                 const __half* __restrict__ base,
                                 float* __restrict__ out, int nq, int n,
                                 int dim, int stride, int metric) {
  (void)dim;

  // +1 skews rows across banks; btile[tx][t] is a column walk
  __shared__ float qtile[K1_TILE][K1_CHUNK + 1];
  __shared__ float btile[K1_TILE][K1_CHUNK + 1];

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int qi = blockIdx.y * K1_TILE + ty;
  const int bi = blockIdx.x * K1_TILE + tx;
  const int brow = blockIdx.x * K1_TILE + ty;

  float acc = 0.0f;
  for (int d0 = 0; d0 < stride; d0 += K1_CHUNK) {
    const int d = d0 + 2 * tx;
    float2 qv = make_float2(0.0f, 0.0f);
    float2 bv = make_float2(0.0f, 0.0f);
    if (d < stride) {
      if (qi < nq) {
        qv = __half22float2(*reinterpret_cast<const __half2*>(
            queries + (long long)qi * stride + d));
      }
      if (brow < n) {
        bv = __half22float2(*reinterpret_cast<const __half2*>(
            base + (long long)brow * stride + d));
      }
    }
    qtile[ty][2 * tx] = qv.x;
    qtile[ty][2 * tx + 1] = qv.y;
    btile[ty][2 * tx] = bv.x;
    btile[ty][2 * tx + 1] = bv.y;
    __syncthreads();

    int span = stride - d0;
    if (span > K1_CHUNK) span = K1_CHUNK;
    if (metric == VSG_METRIC_L2) {
      for (int t = 0; t < span; t++) {
        float diff = qtile[ty][t] - btile[tx][t];
        acc += diff * diff;
      }
    } else {
      for (int t = 0; t < span; t++) {
        acc += qtile[ty][t] * btile[tx][t];
      }
    }
    __syncthreads();
  }

  if (qi < nq && bi < n) {
    out[(long long)qi * n + bi] = (metric == VSG_METRIC_L2) ? acc : -acc;
  }
}
