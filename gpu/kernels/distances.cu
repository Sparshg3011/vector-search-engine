// K1 - batched distance matrix. TODO(sparsh): this is yours.
//
// CONTRACT (the launcher and the tests hold you to exactly this):
//
//   out[q * n + b] = distance(queries[q], base[b])   fp32, row-major
//
//   queries: (nq, stride) fp16, row-major, rows 16-byte aligned
//   base:    (n,  stride) fp16, row-major, same stride
//   dim:     true dimensionality; columns dim..stride are ZERO on both
//            sides, so looping to stride instead of dim is safe and
//            usually faster (no ragged tail).
//   metric:  VSG_METRIC_L2 -> squared l2, no sqrt (ranking is the same
//                             and the cpu FlatIndex does the same)
//            VSG_METRIC_IP -> NEGATED dot product, so lower = closer
//                             everywhere in this codebase
//
// NON-NEGOTIABLE: accumulate in float, never in __half. sift components
// are 0..255, so a squared-l2 sum reaches ~8.3e6 while fp16 tops out at
// 65504 - a half accumulator silently becomes inf and the ranking
// collapses. Load as __half (or __half2), convert, accumulate in fp32.
//
// Tolerance you are held to: rtol 1e-3, atol 1e-2 against a fp64 numpy
// reference computed on the same fp16-quantized inputs.
//
// SHAPE OF THE PROBLEM: this is memory-bound (~0.5 flops/byte). Every
// base row is read by every query, so the win is in reuse, not math.
// The usual structure is a tiled kernel: one thread per (query, base)
// pair, each block staging a TILE x TILE chunk of both operands through
// shared memory so a global value is read once per block instead of
// once per thread. Watch for: coalescing (consecutive threads should
// read consecutive addresses), shared-memory bank conflicts (pad the
// tile row by 1), and keeping the metric branch uniform across the
// block so it costs nothing.
//
// Handle nq and n that are not multiples of your tile: out-of-range
// threads must still reach every __syncthreads() in the loop, so guard
// the loads (read zeros) and the store, not the barriers.

#include "config.cuh"
#include <cuda_fp16.h>

__global__ void vsg_k1_distances(const __half* __restrict__ queries,
                                 const __half* __restrict__ base,
                                 float* __restrict__ out, int nq, int n,
                                 int dim, int stride, int metric) {
  // TODO(sparsh): K1 goes here.
  //
  // suggested skeleton:
  //   __shared__ float qtile[TILE][TILE + 1];   // +1 dodges bank conflicts
  //   __shared__ float btile[TILE][TILE + 1];
  //   int qi = blockIdx.y * TILE + threadIdx.y;
  //   int bi = blockIdx.x * TILE + threadIdx.x;
  //   float acc = 0.0f;                          // fp32, always
  //   for (int d0 = 0; d0 < stride; d0 += TILE) { load; __syncthreads();
  //                                               accumulate; __syncthreads(); }
  //   if (qi < nq && bi < n) out[(long long)qi * n + bi] = finish(acc);
  //
  // note the (long long) on the output index: nq * n reaches 5e8 with
  // the launcher's chunking, which still fits in int, but the habit is
  // free and the next size up is not.
  (void)queries;
  (void)base;
  (void)out;
  (void)nq;
  (void)n;
  (void)dim;
  (void)stride;
  (void)metric;
}
