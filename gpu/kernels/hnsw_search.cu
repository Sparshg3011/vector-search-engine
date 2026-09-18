// K3: layer-0 HNSW beam search, one warp per query. Entry points come
// from the host's upper-layer descent. The beam is a sorted list of ef
// (dist, id, expanded) entries in shared memory; the visited set is a
// per-query open-addressing table in global memory, owned by one warp,
// so it needs no atomics. Every branch below is warp-uniform.

#include "config.cuh"
#include <cuda_fp16.h>

#define K3_FULL_MASK 0xffffffffu

static_assert((1 << VSG_VISITED_BITS) == VSG_VISITED_SLOTS,
              "VSG_VISITED_BITS must be log2(VSG_VISITED_SLOTS)");

__device__ __forceinline__ float k3_distance(const __half* __restrict__ q,
                                             const __half* __restrict__ v,
                                             int stride, int metric,
                                             int lane) {
  float acc = 0.0f;
  for (int d = 2 * lane; d < stride; d += 64) {
    float2 a = __half22float2(*reinterpret_cast<const __half2*>(q + d));
    float2 b = __half22float2(*reinterpret_cast<const __half2*>(v + d));
    if (metric == VSG_METRIC_L2) {
      float dx = a.x - b.x;
      float dy = a.y - b.y;
      acc += dx * dx + dy * dy;
    } else {
      acc += a.x * b.x + a.y * b.y;
    }
  }
  for (int off = 16; off > 0; off >>= 1) {
    acc += __shfl_down_sync(K3_FULL_MASK, acc, off);
  }
  acc = __shfl_sync(K3_FULL_MASK, acc, 0);
  return (metric == VSG_METRIC_L2) ? acc : -acc;
}

// returns true if node was not yet visited (and marks it). 32 slots are
// probed per step. a full table reports every node as new; the list's
// dedup and worst-distance checks keep that safe.
__device__ __forceinline__ bool k3_visit(int* table, int node, int lane) {
  unsigned h = ((unsigned)node * 2654435761u) >> (32 - VSG_VISITED_BITS);
  for (int w = 0; w < VSG_VISITED_SLOTS / 32; w++) {
    int slot = (int)((h + lane) & (VSG_VISITED_SLOTS - 1));
    int cur = table[slot];
    if (__ballot_sync(K3_FULL_MASK, cur == node)) return false;
    unsigned empty = __ballot_sync(K3_FULL_MASK, cur == VSG_VISITED_EMPTY);
    if (empty) {
      if (lane == __ffs(empty) - 1) table[slot] = node;
      __syncwarp();
      return true;
    }
    h += 32;
  }
  return true;
}

__global__ void vsg_k3_hnsw_search(
    const __half* __restrict__ vectors, const int* __restrict__ adjacency,
    const int* __restrict__ degrees, const __half* __restrict__ queries,
    const int* __restrict__ entries, int* __restrict__ visited,
    int* __restrict__ out_ids, float* __restrict__ out_dists, int nq, int n,
    int dim, int stride, int width, int k, int ef, int metric) {
  (void)n;
  (void)dim;
  const float INF = __int_as_float(0x7f800000);
  const int lane = threadIdx.x & 31;
  const int wib = threadIdx.x >> 5;
  const int q = blockIdx.x * VSG_K3_WARPS_PER_BLOCK + wib;
  if (q >= nq) return;

  extern __shared__ __align__(16) unsigned char k3_smem[];
  unsigned char* mine = k3_smem + (size_t)wib * VSG_K3_SMEM_PER_WARP(ef);
  float* cd = reinterpret_cast<float*>(mine);
  int* ci = reinterpret_cast<int*>(mine + (size_t)ef * sizeof(float));
  int* cf = reinterpret_cast<int*>(mine + (size_t)ef * (sizeof(float) + sizeof(int)));

  int* table = visited + (long long)q * VSG_VISITED_SLOTS;
  const __half* qv = queries + (long long)q * stride;

  const int entry = entries[q];
  k3_visit(table, entry, lane);
  const float d_entry =
      k3_distance(qv, vectors + (long long)entry * stride, stride, metric, lane);
  if (lane == 0) {
    cd[0] = d_entry;
    ci[0] = entry;
    cf[0] = 0;
  }
  __syncwarp();
  int size = 1;

  for (;;) {
    const int live_chunks = (size + 31) >> 5;
    // the list is sorted, so the first unexpanded entry is the closest
    int idx = -1;
    for (int c = 0; c < live_chunks; c++) {
      int i = c * 32 + lane;
      unsigned m = __ballot_sync(K3_FULL_MASK, i < size && cf[i] == 0);
      if (m) {
        idx = c * 32 + __ffs(m) - 1;
        break;
      }
    }
    if (idx < 0) break;
    // the ballot syncs execution, not memory: order every lane's read of cf
    // above before lane 0 overwrites it
    __syncwarp();
    if (lane == 0) cf[idx] = 1;
    __syncwarp();

    const int node = ci[idx];
    const int deg = degrees[node];
    const int* nbrs = adjacency + (long long)node * width;

    for (int j = 0; j < deg; j++) {
      const int nb = nbrs[j];
      if (!k3_visit(table, nb, lane)) continue;
      const float d =
          k3_distance(qv, vectors + (long long)nb * stride, stride, metric, lane);
      if (size == ef && d >= cd[ef - 1]) continue;

      int pos = 0;
      unsigned dup = 0;
      for (int c = 0; c < live_chunks; c++) {
        int i = c * 32 + lane;
        bool live = i < size;
        pos += __popc(__ballot_sync(K3_FULL_MASK, live && cd[i] < d));
        dup |= __ballot_sync(K3_FULL_MASK, live && ci[i] == nb);
      }
      // only reachable once the visited table is allowed to forget
      if (dup) continue;

      float sd[VSG_K3_CHUNKS];
      int si[VSG_K3_CHUNKS];
      int sf[VSG_K3_CHUNKS];
#pragma unroll
      for (int c = 0; c < VSG_K3_CHUNKS; c++) {
        int i = pos + c * 32 + lane;
        if (i < size) {
          sd[c] = cd[i];
          si[c] = ci[i];
          sf[c] = cf[i];
        }
      }
      __syncwarp();
#pragma unroll
      for (int c = 0; c < VSG_K3_CHUNKS; c++) {
        int i = pos + c * 32 + lane;
        if (i < size && i + 1 < ef) {
          cd[i + 1] = sd[c];
          ci[i + 1] = si[c];
          cf[i + 1] = sf[c];
        }
      }
      __syncwarp();
      if (lane == 0) {
        cd[pos] = d;
        ci[pos] = nb;
        cf[pos] = 0;
      }
      __syncwarp();
      if (size < ef) size++;
    }
  }

  for (int c = 0; c < (VSG_MAX_K + 31) / 32; c++) {
    int i = c * 32 + lane;
    if (i < k) {
      bool have = i < size;
      out_ids[(long long)q * k + i] = have ? ci[i] : -1;
      out_dists[(long long)q * k + i] = have ? cd[i] : INF;
    }
  }
}
