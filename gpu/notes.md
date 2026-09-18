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

## 2026-09-17 — first gpu runs, USC Discovery

first compile through nvcc 12.6 (gcc 13.3) was clean. all 76 tests pass on
an A40 and an L40S; all six phase-0 gates pass, including ncu counter
access, so the profiling half of the plan works on carc without root.

### sift-1M on one L40S (job 12146981, commit 04a1ff6 + b079dea)

```
method                         batch   recall@10   ms/q wall   ms/q kernel      qps
cpu hnsw (numpy), ef=200           1       0.995      2.975           -           336
cpu hnsw (numpy), ef=50            1       0.938      0.878           -         1,138
gpu hnsw (k3), ef=200              1       0.995      2.743        2.567          365
gpu hnsw (k3), ef=200           2048       0.991      0.112        0.0025       8,958
gpu hnsw (k3), ef=50            2048       0.932      0.110        0.0009       9,095
gpu flat, k1 + k2               2048       0.999      0.062        0.0596      16,129
gpu flat, cublas + k2           2048       0.999      0.031        0.0286      32,162
```

- **batch 1: gpu ≈ cpu** (2.74 vs 2.98 ms at ef=200). the thesis, measured.
- **batch 2048: k3's kernel is 0.0025 ms/query, but wall is 0.112** — the
  host `descend()` (a python loop over the upper layers) is ~97% of it.
  next fix is on the host, not the gpu: vectorize the descent across the
  batch, or move it onto the device.
- **batched brute force beats cpu hnsw outright**: cublas + k2 is 96x the
  cpu's qps at ef=200, at 0.999 recall. hand-written k1 is ~2x slower than
  the cublas path — first thing to take into ncu.
- **k3 recall matches the cpu on the full query set**: replayed the cpu
  search over all 10,000 queries and scored it on exactly the query sets
  each batch size used; |gpu - cpu| <= 0.0005 everywhere (gate: 0.01). the
  apparent recall drop from batch 1 to 2048 is the query subset, not the
  batching — batch 1 scores queries 3..202, batch 2048 all 10,000.
- **brute force recall is 0.999, not 1.0, because of ties**: sift
  distances are exact integers (fp16 holds 0..255 exactly, sums stay under
  2^24), and in 137 of 10,000 queries the 10th and 11th neighbors tie. a
  differently broken tie scores as a miss: at most 0.0014 of recall.

### what the checks caught

- **racecheck, k3**: write-after-read on the expanded flags — every lane
  reads them inside the ballot that picks the next node, lane 0 then marks
  it. `__ballot_sync` orders execution, not memory. harmless in practice
  (tests and recall unaffected); fixed with a `__syncwarp` (2278cb9).
  re-check on the fixed build (job 12147440): memcheck 0 errors,
  racecheck 0 hazards, all three kernels.
- **cpu index bug, surfaced by the gpu run**: 8 of 10,000 queries return
  fewer than k results (two return 2). their layer-0 entry points are
  exact duplicate vectors — sift has 29,076 rows with an exact twin.
  `_select_neighbors` rejects a candidate when a chosen neighbor is `<=`
  as close; after the twin is chosen at distance 0 every other candidate
  ties and is rejected, so twins link only to each other: a closed
  2-node island. hnswlib and faiss both use strict `<`. not fixed yet — it
  changes the cpu graph, so it means a rebuild and re-measuring the cpu
  chapter's published numbers. the benchmark now pads short results with
  -1 (5e5db24) so the cpu baseline runs at every batch size.

## 2026-09-17 — the duplicate-vector fix, and the corrected numbers

### the fix

`_select_neighbors` now rejects a candidate only when a chosen neighbor is
*strictly* closer to it than the query (`<`, as hnswlib and faiss do). On
SIFT-1M that turns the two-node islands back into normal nodes: queries
returning fewer than k results 8 -> 0, layer-0 nodes with no incoming link
23 -> 3, median degree unchanged at 19. on fashion-mnist (no exact
duplicates to speak of) the graph changed by 7 links out of 740k.

both indexes rebuilt (sift: 1414 s vecstore, 120 s faiss) and re-measured on
an idle machine. cpu tables in the README regenerated from results/*.json.

```
sift-1M, cpu, this mac      recall@10 (old -> new)   ms/query (old -> new)
  ef=10                     0.638 -> 0.647           0.16 -> 0.20
  ef=50                     0.916 -> 0.921           0.47 -> 0.56
  ef=200                    0.990 -> 0.993 (=faiss)  1.66 -> 1.97
  exact                                              7.2  -> 7.6
```

recall is up at every point; latency ~15% up because the graph keeps the
links it used to prune, and the machine measures ~10% slower than in july
(faiss moved the same way). the clustered-data claim in the README
(0.75 -> 0.96) re-measured at 0.750 -> 0.964, unchanged.

### descend() batched

the host-side upper-layer walk was a python loop per query per hop, ~97%
of the batched gpu wall time. it now walks the whole batch with array ops
over compact per-layer fp32 tables built once per loaded index (profile of
the loop version: 68% of the time was gathering neighbor rows out of the
fp16 array and converting them). same entries on all 10,000 sift queries;
0.066 -> 0.016 ms/query at batch on the mac. still ~3x the k3 kernel time
on an a40; moving the upper-layer walk onto the device is the next step
if that ever matters.

### sift-1M on one A40, corrected index (job 12147871, main @ c8b8732)

```
method                         batch   recall@10   ms/q wall   ms/q kernel      qps
cpu hnsw (numpy), ef=200           1       0.995      3.822           -           262
gpu hnsw (k3), ef=200              1       0.995      4.430        4.066          226
cpu hnsw (numpy), ef=200        2048       0.993      3.512           -           285
gpu hnsw (k3), ef=200           2048       0.993      0.0227       0.0062      44,060
gpu hnsw (k3), ef=50            2048       0.936      0.0184       0.0019      54,488
gpu flat, k1 + k2               2048       0.999      0.148        0.146        6,772
gpu flat, cublas + k2           2048       0.999      0.054        0.052       18,405
```

- batch 1: gpu loses (4.43 vs 3.82 ms). batch 2048: 155x the cpu at the
  same recall. wall/kernel gap at 2048 is now 0.0227 vs 0.0062 ms —
  descend 0.016, the rest transfers.
- recall parity, rebuilt index: cpu search replayed over all 10,000
  queries and scored on exactly the query subset each batch used —
  |gpu − cpu| ≤ 0.0005 in every cell, 0.0000–0.0001 at batch 2048.
- hand-written k1 is 2.7x slower than the cublas path on the a40 (2.0x on
  the l40s earlier). first ncu target.
- the l40s run of the corrected index is queued (job 12147828); the README
  table will switch to it when it lands.

### sift-1M on one L40S, corrected index (job 12147828, same code as the a40 run)

```
method                         batch   recall@10   ms/q wall   ms/q kernel      qps
cpu hnsw (numpy), ef=200           1       0.995      2.988           -           335
gpu hnsw (k3), ef=200              1       0.995      2.907        2.647          344
cpu hnsw (numpy), ef=200        2048       0.993      2.904           -           344
gpu hnsw (k3), ef=200           2048       0.993      0.0178       0.0026      56,265
gpu hnsw (k3), ef=50            2048       0.936      0.0161       0.0009      62,188
gpu flat, k1 + k2               2048       0.999      0.062        0.059       16,186
gpu flat, cublas + k2           2048       0.999      0.031        0.029       32,146
```

- batch 1: a wash (2.91 vs 2.99 ms). this node's cpu is faster than the
  a40 node's for the same numpy walk (2.99 vs 3.82 ms), so the batch-1
  comparison moves with the host; the shape of the result does not.
  batch 2048: 163x the cpu at the same recall.
- kernel time at 2048 is 0.0026 ms/q against 0.0178 wall — descend 0.015,
  the rest transfers. the host walk is now ~6x the kernel; moving it onto
  the device is the obvious next step.
- cublas flat 32,146 qps @ 0.999 (93x the cpu); hand k1 2.0x slower than
  cublas, same as the first l40s run (2.7x on the a40).
- the README table and headline carry this run; the a40 job stays in
  gpu/bench/results/ for the cross-card comparison.
