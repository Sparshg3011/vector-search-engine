# engineering log — gpu chapter

running log of measurements, bugs, and decisions. the gap analysis and the
final writeup get written from this file, so entries record what was measured
and what it forced, not what was hoped.

## 2026-08-30 — instrumented the cpu search before designing k3

added counters to `HNSWIndex._search_layer` and ran the real saved indexes
(fashion-mnist-60k and sift-1M, both M=16), 1000 queries each, layer-0 search
only. visited = unique nodes touched, pushes = candidate-heap pushes, hops =
pops that got expanded.

```
sift-1M (M=16, layer 0)
  ef=50   visited p50  894 / p95 1063 / max 1129   pushes p95 207   hops p95  59
  ef=100  visited p50 1558 / p95 1959 / max 2015   pushes p95 385   hops p95 105
  ef=200  visited p50 2792 / p95 3547 / max 3699   pushes p95 622   hops p95 203

fashion-mnist-60k (M=16, layer 0)
  ef=50   visited p50  408 / p95  591 / max  680
  ef=100  visited p50  678 / p95  952 / max 1036
  ef=200  visited p50 1059 / p95 1489 / max 1707
```

### what the numbers forced (now in SPEC.md, "K3 design constraints")

- **visited set v1 goes to global memory, 8192 slots per in-flight query.**
  the original plan was a 512-slot shared-memory hash. at 1M / ef=200 that is
  ~7x undersized (p95 visited 3547), and the open-addressing design live-locks
  when the table fills — every probe loops forever looking for an empty slot.
  sizing it honestly (8192 x 4B = 32KB per query) blows the shared-memory
  budget and kills occupancy, so v1 allocates the workspace in global memory,
  `(grid_queries, VISITED_SLOTS)`, VISITED_SLOTS = 8192 (~2x p95 headroom
  over the max observed 3699).
- **shared-memory evict-on-collision is the planned a/b**, not the baseline.
  a small shared table that evicts on collision can only cause re-visits,
  never drops — safe because result-list insertion dedups by id. worth
  measuring against v1 once v1 is correct, not before.
- **one combined candidate list of length ef is enough.** expansions ≈ ef
  (hops p95 203 at ef=200) and pushes ≈ 3x ef, so a single sorted list of
  (dist, id, expanded-flag) in shared memory covers it. MAX_EF = 256 keeps
  it ≤ 2KB + flags per query.

### other standing decisions

- **no cosine kernel.** angular datasets get L2-normalized at export
  (`export.py --normalize`); after that l2 and ip both reproduce cosine's
  ranking. every cpu baseline must run on the same normalized data or the
  recall comparison is meaningless.
- **fp32 accumulation is an overflow requirement, not a nicety.** sift
  squared-l2 sums reach ~8.3M; fp16 max is 65504. accumulate a sift distance
  in `__half` and it saturates to inf — the ranking collapses silently.
  storage fp16, math fp32, everywhere.
- **glove-100 stride pads 100 -> 104** (dim rounded up to a multiple of 8
  for 16-byte rows). pad values are 0.0 so they contribute nothing to l2 or
  ip; kernels may loop to stride.
- **brute-force distance matrix chunked at 2GB.** batch=2048 x 1M x fp32 is
  8GB — never materialize it whole. `brute_force` chunks internally.

### timeline, re-anchored

- d1 = aug 30: scaffold + pod phase-0 gates.
- gate 1 (~sep 3): exact gpu search (k1+k2) benchmarked against faiss-gpu flat.
- gate 2 (~sep 7): decision — is k3 viable, or does the graph chapter become
  a documented negative result?
- ship ~sep 14.

### open items

- **ncu counter access must be verified before renting by the hour.**
  rentals often block gpu performance counters (`ERR_NVGPUCTRPERM`); gate 5
  in `check_env.py` exists for this. no counters = no gap analysis = wrong
  provider.
- **768-d dataset rule needs a decision**: gist-960 (published ground truth,
  wrong dim) vs a self-computed-GT 768-d embedding set. leaning self-computed
  since `brute_force` gives exact answers anyway, but provenance must be
  stated in the writeup either way.
