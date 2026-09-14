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

## 2026-09-13 — kernels written, verified off-gpu

k1, k2 and k3 are in and enabled. nothing has run on a gpu yet. what has
been checked on the mac:

- structural compile of the whole device side against stub cuda headers
  (`gpu/tools/precheck.py`): clean. catches typos, arity, braces — not
  semantics.
- thread-accurate emulation of each kernel's index math against numpy:
  - k1: 17 cases across tile boundaries (stride 8, 16, 24, 32, 40, 64,
    104; nq and n not multiples of 16), both metrics, plus sift-scale
    integers. all within tolerance, every output cell written.
  - k2: 408 trials incl. heavy ties, all-identical rows, k=128, n<128.
    exact match to np.sort, no duplicate ids, ascending.
  - k3: real 5k-node index (M=16). with the cpu's own fp32 rows and fp64
    distances the emulation reproduces `HNSWIndex.search`'s result sets
    on 200/200 queries at ef 50 and ef 200. with fp16 rows and fp32
    accumulation (what the gpu sees) the recall gap to the cpu is 0.0005
    at both ef, against the 0.01 gate. visited table checked separately:
    3700 inserts, a 40-way hash-collision chain, no false hits.

### design decisions

- **k1**: 16x16 output patch per block; the dim axis is walked 32 at a
  time through `[16][33]` fp32 shared tiles, one `__half2` load per
  thread per chunk (a warp's 32 lanes fetch two whole 64-byte row
  chunks). conversion happens on the way into shared memory, so the
  inner loop never touches a half. `+1` on the tile row so the
  `btile[tx][t]` column walk is bank-conflict-free.
- **k2**: per-thread sorted shortlist over a strided slice (coalesced
  reads), then k rounds of argmin: `__shfl_down_sync` inside each warp,
  four warp minima merged by thread 0, winner broadcast through shared
  memory. candidate arrays are indexed by a runtime k, so they live in
  local memory — the cost is on the rare insert path and the merge only.
  if ncu shows it mattering, template the kernel on k.
- **k3**: one warp per query. the visited table is warp-private, so it
  needs no atomics: 32 slots are probed per step with a ballot, insert
  goes to the first empty. the beam is one sorted list of ef entries
  with an expanded flag; expand the first unexpanded entry, stop when
  none is left. this expands exactly the set the cpu's two-heap loop
  expands (a candidate the cpu would pop and expand is one still inside
  its results heap), which is why the emulation reproduces it exactly.
  neighbor distances are warp-per-distance — 32 lanes split one vector,
  `__half2` per lane, shfl tree reduce, broadcast. thread-per-neighbor
  is the planned a/b.
- **k3 list insert**: `pos` = popcount over a ballot of entries `< d`;
  the shift of `[pos, size)` up by one goes through per-lane register
  buffers — read everything, `__syncwarp`, write everything — so no lane
  overwrites an entry another lane has not read yet. when the list is
  full the last entry falls off via the `i + 1 < ef` guard.
  dedup-on-insert stays in even though the exact table makes it
  unreachable: it is the property that makes the evict-on-collision
  table variant safe to try.

### open

- nothing has been through nvcc. first pod run: `bash gpu/setup_pod.sh`,
  then `pytest gpu/tests -v`.
- first two things to read off ncu: k2 local-memory traffic, and k3
  occupancy (per-lane shift buffers cost ~24 registers on top of the
  per-warp shared slice).
