// K2 - per-row top-k selection. TODO(sparsh): this is yours.
//
// CONTRACT:
//
//   for each row of dmat (one row = one query's distances to all n
//   base vectors), write the k smallest values and their column ids,
//   CLOSEST FIRST:
//
//   out_dists[q * k + j] = j-th smallest distance in row q   (ascending)
//   out_ids  [q * k + j] = its column index
//
//   dmat: (nq, n) fp32 row-major, produced by K1 or the cublas path
//   k:    1 <= k <= VSG_MAX_K (128). The launcher rejects larger.
//   grid: one block per row (blockIdx.x == the row), so nq blocks.
//
// Ties: rows can hold duplicate distances (integer datasets do this a
// lot). Any consistent choice among equal values is fine - the tests
// compare ids as sets and treat two ids at the k boundary as
// interchangeable when their distances agree to 1e-3 relative. What is
// NOT fine is emitting the same id twice.
//
// WHY NOT JUST SORT: n is up to a million per row and k is 10. Sorting
// costs O(n log n) and touches everything; selection is O(n) with a
// tiny working set. The usual structure is two phases:
//
//   1. each thread scans a strided slice of the row (thread t takes
//      columns t, t+T, t+2T, ...) keeping its own sorted top-k in
//      registers. Most values lose to the current worst in a single
//      compare, so the insertion path is rarely taken.
//   2. the block merges the T sorted lists into one. A k-round
//      min-reduction over each thread's next unconsumed candidate is
//      the simple version; warp shuffles (__shfl_down_sync) make the
//      reduction cheaper than shared memory.
//
// Why the merge is correct: a true top-k element can only be evicted
// from a thread's local list by k values that also beat it globally,
// and fewer than k such values exist - so it survives in some list.
//
// Watch for: every thread must reach every __syncthreads() (do not
// return early from a partial block); k rounds of reduction means the
// scratch arrays are rewritten each round, so sync after reading the
// winner as well as after writing.

#include "config.cuh"

__global__ void vsg_k2_topk(const float* __restrict__ dmat,
                            int* __restrict__ out_ids,
                            float* __restrict__ out_dists, int n, int k) {
  // TODO(sparsh): K2 goes here.
  //
  // the row this block owns:
  //   const float* row = dmat + (long long)blockIdx.x * n;
  // per-thread state:
  //   float cand_d[VSG_MAX_K]; int cand_i[VSG_MAX_K];  // sorted, INF-filled
  // build +inf without math.h (nvrtc-safe, and free here):
  //   const float INF = __int_as_float(0x7f800000);
  (void)dmat;
  (void)out_ids;
  (void)out_dists;
  (void)n;
  (void)k;
}
