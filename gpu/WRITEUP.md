# porting vecstore to cuda — writeup (skeleton)

skeleton only. each section header carries a one-line prompt for what goes
there; prose gets written from `notes.md` once the numbers exist. do not
fill a section before its measurement is in the log.

## origin story

recap the cpu chapter in one paragraph — what vecstore is, the honest
single-thread numbers (qps, recall@10 at each ef), and why "make it faster"
points at the gpu.

## the tension

hnsw search is memory-latency-bound and the walk is inherently sequential;
state the two-level answer — parallelize within a step (distance evals across
a node's neighbors) and across queries (warp per query) — and say plainly
that the walk itself is never parallelized.

## what does not get faster

batch-1 latency is expected to lose to the cpu and that is the honest
headline, not a footnote — a single query cannot fill a gpu, pcie transfer
alone can exceed the cpu search time; show the crossover batch size.

## k1 — brute-force distances

the hand-rolled tiled kernel vs cublas gemm: the measured gap, what the
hand-rolled version teaches (tiling, coalescing, fp16 loads / fp32 math),
and why the gemm decomposition (‖q‖²+‖x‖²−2q·x) wins.

## k2 — top-k selection

how k ≤ 128 selection works on the gpu, what was tried, and where it sits
vs the k1 cost — is selection ever the bottleneck?

## k3 — the visited-set problem, worked

the full arc from `notes.md` 2026-08-30: the measurement (sift-1M ef=200
visits p95 3547, max 3699), why the 512-slot shared-memory hash was ~7x
undersized and live-locks when full, the global-memory 8192-slot redesign,
and the evict-on-collision a/b with its measured verdict.

## results

the curves (placeholders until `bench/results/` has data): recall@10 vs qps
per dataset, batch-size sweep per backend, kernel-ms vs wall-ms side by side.

## gap analysis vs faiss-gpu and cagra

decompose the remaining gap with profiler evidence — achieved bandwidth vs
peak, occupancy, top stall reasons per kernel — and attribute each slice
to a named cause, not "they optimized more".

## methodology appendix

batch axis {1, 32, 128, 512, 2048}; medians of ≥20 timed runs after ≥3
warmups; kernel time (cuda events) and wall time reported separately and
never blended; clocks caveat on rented gpus (log gpu name, driver, sm clock
per run); ground-truth provenance per dataset.

## cut list

what was descoped and why — each cut named with the reason and what it
would have cost to keep.
