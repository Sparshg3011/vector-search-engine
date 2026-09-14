#pragma once

// Constants shared by the kernels and the host launchers. The launchers
// size workspaces from these and the kernels index with them, so a
// change here is a change on both sides - keep them in sync with
// gpu/SPEC.md.

// k2: per-query result capacity. Bounds the per-thread candidate lists.
#define VSG_MAX_K 128

// k3: candidate-list capacity (ef). 256 entries of (float dist, int id)
// plus a flag byte is ~2.3 KB of shared memory per query.
#define VSG_MAX_EF 256

// k3: per-query visited-hash slots, in GLOBAL memory. Measured on the
// real sift-1M index: ef=200 visits a median of 2792 nodes, p95 3547,
// max ~3700 (gpu/notes.md). 8192 is ~2x the p95 with room for the
// power-of-two mask; an open-addressing table that fills up spins
// forever looking for a free slot, so the headroom is not optional.
#define VSG_VISITED_SLOTS 8192
// log2 of the above: the hash keeps this many high bits of the mixed key
#define VSG_VISITED_BITS 13

// k3: an empty visited slot. Node ids are non-negative.
#define VSG_VISITED_EMPTY -1

// metric codes on the device. No cosine: angular data is L2-normalized
// at export, after which ip reproduces the cosine ranking.
#define VSG_METRIC_L2 0
#define VSG_METRIC_IP 1

// Launch geometry. The launchers compute grids from these, so changing
// a tile size in a kernel means changing it here too - that coupling is
// deliberate, it is the one thing that silently corrupts results.

// k1: square thread tile, block is (TILE, TILE).
#define VSG_K1_TILE 16

// k2: threads per block, one block per query row.
#define VSG_K2_THREADS 128

// k3: warps per block; blockDim.x is this times 32.
#define VSG_K3_WARPS_PER_BLOCK 4

// k3: dynamic shared memory per warp. The candidate list is three
// parallel arrays of ef entries: float dist, int id, int expanded-flag,
// in that order. Change this if the layout changes.
#define VSG_K3_SMEM_PER_WARP(ef) ((int)((ef) * (2 * sizeof(int) + sizeof(float))))

// k3: the list is scanned 32 entries per warp step, so this many steps
// cover MAX_EF. Sizes the per-lane shift buffers.
#define VSG_K3_CHUNKS (VSG_MAX_EF / 32)
