// K3 - warp-per-query HNSW beam search on layer 0. TODO(sparsh): this
// is the main event, and the one an interviewer will read line by line.
//
// CONTRACT:
//
//   for each query, run the layer-0 best-first search starting from
//   entries[q] (the host already walked the upper layers), and write
//   the k nearest, closest first:
//
//   out_dists[q * k + j] ascending, out_ids[q * k + j] their node ids
//
//   vectors:   (n, stride) fp16 row-major, pad columns zero
//   adjacency: (n, width) int32, row i holds degrees[i] neighbor ids
//              then -1 padding. width == 2M (32 for M=16).
//   entries:   (nq,) int32, one layer-0 start node per query
//   visited:   (gridQueries, VSG_VISITED_SLOTS) int32 scratch in GLOBAL
//              memory, pre-filled with VSG_VISITED_EMPTY by the
//              launcher. Query q owns row q.
//   ef:        beam width, k <= ef <= VSG_MAX_EF (256)
//   metric:    VSG_METRIC_L2 / VSG_METRIC_IP, same meaning as K1
//              (fp32 accumulation, ip negated)
//
// THE SHAPE OF THE PARALLELISM (this is the thesis of the project):
// the walk itself is sequential - step two starts wherever step one
// landed, and no amount of hardware changes that. So we do not
// parallelize the walk. We parallelize (a) INSIDE each step: one warp
// owns one query, and its 32 lanes split the fan-out of computing
// distances to up to `width` neighbors of `dim` values each; and (b)
// ACROSS queries: thousands of independent warps, each crawling, which
// is what keeps the machine busy and hides memory latency. Single-query
// latency does not improve. Throughput does.
//
// Keep the traversal logic WARP-UNIFORM: every lane evaluates the same
// loop condition and the same branch, then lanes diverge only over
// data (which neighbor, which dimension). Reduce with __shfl_down_sync
// / __reduce_min_sync rather than shared memory where you can, and
// remember every lane must reach __syncwarp() together.
//
// THE THREE DATA STRUCTURES (sized from measurement - gpu/notes.md):
//
// 1. visited set. Open-addressing hash in GLOBAL memory, one row of
//    VSG_VISITED_SLOTS (8192) int32 per query, empty == -1. Insert with
//    a linear probe; the table is a power of two so the probe mask is
//    (VSG_VISITED_SLOTS - 1). Sized from the real sift-1M numbers: a
//    query at ef=200 visits ~2800 nodes (p95 3547), NOT the "few
//    hundred" the first plan assumed - and a full open-addressing table
//    does not degrade, it spins forever. Global memory rather than
//    shared because an honest 32 KB per query would leave room for one
//    or two warps per SM and destroy the occupancy that hides the
//    latency in the first place.
//    PLANNED A/B (write this second, measure both): the same table in
//    SHARED memory at a size that fits, evicting on collision instead
//    of probing. Forgetting a visit only costs duplicate work, never
//    correctness - provided the result-list insert below rejects an id
//    it already holds. That safety property is the whole reason the
//    variant is allowed.
//
// 2. candidate/result list. ONE combined array of length ef in shared
//    memory: (dist, id, expanded flag), kept sorted by distance. Pop
//    the closest unexpanded entry, expand it, insert its neighbors,
//    drop anything past ef. Measurement says expansions per query are
//    almost exactly ef (203 at ef=200), so one ef-length structure is
//    the right size and a separate candidate heap buys nothing.
//    Insertion is warp-cooperative: a shift-insert into a sorted array
//    that all 32 lanes perform in lockstep beats a clever branchy heap
//    here, because divergence costs more than the extra compares.
//
// 3. the k results are just the first k entries of that array when the
//    search stops - no separate structure.
//
// TERMINATION: stop when the closest unexpanded candidate is farther
// than the current worst result. Expanding it could only add nodes you
// would immediately throw away. (This is exactly _search_layer's break
// condition in vecstore/hnsw.py - read it before you start.)
//
// CORRECTNESS GATE: mean recall@10 over >=200 queries must land within
// 0.01 of the cpu index at the same ef, at ef=50 and ef=200. A
// structural gap is a bug in this file, not "the gpu is different".
// Debug ritual when it is wrong: shrink to ~100 nodes and one warp,
// printf from lane 0, and diff the visit order against the cpu trace.

#include "config.cuh"
#include <cuda_fp16.h>

__global__ void vsg_k3_hnsw_search(
    const __half* __restrict__ vectors, const int* __restrict__ adjacency,
    const int* __restrict__ degrees, const __half* __restrict__ queries,
    const int* __restrict__ entries, int* __restrict__ visited,
    int* __restrict__ out_ids, float* __restrict__ out_dists, int nq, int n,
    int dim, int stride, int width, int k, int ef, int metric) {
  // TODO(sparsh): K3 goes here.
  //
  // one warp per query:
  //   int lane = threadIdx.x & 31;
  //   int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  //   if (warp >= nq) return;              // whole warp leaves together
  //   int* my_visited = visited + (long long)warp * VSG_VISITED_SLOTS;
  //
  // shared per warp (blockDim.x / 32 warps per block):
  //   extern __shared__ char smem[];       // launcher passes the size
  (void)vectors;
  (void)adjacency;
  (void)degrees;
  (void)queries;
  (void)entries;
  (void)visited;
  (void)out_ids;
  (void)out_dists;
  (void)nq;
  (void)n;
  (void)dim;
  (void)stride;
  (void)width;
  (void)k;
  (void)ef;
  (void)metric;
}
