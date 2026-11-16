# Recording a scaling campaign

Use `campaigns/<site>/<workload>/<UTC-date>-<experiment>/`. Use distinct workloads for `cpp-read`, `sim10-staging`, and `sim10-xpp-cube`. A campaign is one controlled comparison with explicitly identified allocation blocks; a synthesis can reference several campaigns but must retain their cohort boundaries.

## Required record

Each campaign has a readable `README.md` and a `manifest.json` indexed in `catalog.json`. Manifest paths are repository-root-relative. Pin the framework commit separately from the harness commit, and preserve per-file hashes for packaged or untracked sources. Record the runtime/install receipt, environment versions, dataset identities and raw location. Never substitute the results-branch HEAD for the code that ran.

Record site, compute node, CPU/GPU models, MPI roles, active and allocated GPUs, BDs/GPU, workers, physical/logical CPU affinity, batch/stream/bulk/task/bounce parameters, input budget, calibration/cube options, D2H/export policy and GDS/compatibility state. Record stripe size, count and placement for each file; count 1 per file does not imply one OST for the node.

Define the timing bracket, event and byte denominators, warmup, repetitions, randomized ordering and cache checks. Separate full cases, correctness pilots, anchors and diagnostics. For new campaigns, retain one row per measured case in `runs.csv` with at least:

```text
case_id,job_id,block,repeat,phase,stage,site,node,storage,
gpus_allocated,gpus_active,bds,workers_per_bd,cpu_policy,stripe_count,
events_global,requested_bytes,loop_seconds,whole_case_seconds,payload_gib_s,
cache_state,nic_rx_bytes,nic_tx_bytes,raw_path,raw_sha256
```

Use explicit null/blank values for unavailable metrics; zero means measured zero. Keep unknown/failed cases out of summaries rather than plotting zero throughput. Store any per-rank/pixel arrays outside Git and retain a raw index with locations, sizes and SHA256s. Scratch paths may expire; hashes alone cannot recreate lost data.

For psana Sim10, payload GiB/s is globally requested input bytes / 2**30 / loop seconds; the five-stream fixture has 3,358,200 bytes/event. C++ whole-file byte counts include other file bytes and have their own denominator. NIC traffic, expanded image bytes, calibration bytes and cube exports are separate metrics.

## Acceptance and comparison

Record terminal job/step accounting, correctness and resource gates, raw-evidence audit and independent review separately. A completed scheduler job is not enough to accept throughput. An implementation review does not constitute a results review. Preserve failed attempts and their causes without promoting them into full-run tables.

Describe median versus mean, sample count, observed range and paired speedup formula explicitly. Keep nodes/allocations and workload stages separate; do not pool cohorts or divide unmatched absolute rates to claim an optimization. Three repeats support descriptive spread, not a storage ceiling or an isolated bottleneck claim.

For full Sim10/XPP-cube, record S/E/EC/ECC separately, expansion shape and dtype, calibration definition, dense-bank bins/dtype, nonempty/empty-bin counts, the exact validation policy, GPU/host/pinned peaks and end-to-end total. Record numerical-reference checks and export/reduction only when performed; the current prepared bootstrap deliberately performs neither, retaining distributed GPU cubes with shape/content/bin-count checks. Current staging batch and sharing choices are not established application optima.

## Updating this branch

1. Keep execution scripts on the implementation branch; finish its intended review and checks there. New job submission follows the task's authorization, not this catalog.
2. Preserve an exact script snapshot or immutable source commit with its dependency/bundle manifest. Keep frozen relative hierarchies under the archive; never execute an archived launcher as a relocated installation.
3. Copy only compact reports, CSV/JSON receipts, review records and PDF/SVG figures. Keep XTC data, native installs, profiler traces and full cubes in site storage with hashes/paths.
4. Add a new dated record from the template. Link existing authoritative evidence instead of introducing edited copies. If correcting a report, state the correction and retain the native receipt.
5. Run `python -B psana/psana/gpu/scripts/performance/analysis/check_catalog.py`. Recompute or replay analysis with the documented tools where available. A catalog pass verifies archived records; it does not replace the original raw audit.
6. Commit the record to `results/psana2-gpu-scaling`. Publish/push only as requested. Do not merge changing framework code into this branch just to update a result.

Historical SDF imports contain the full normalized report tables and plotting code, but not every original cohort source CSV or linked report. Their origin and availability are explicit in the campaign record. Archive replay is a derivative calculation, not a claim that the original raw data have been re-audited.

## Catalog schema 2

Catalog paths (`record_path`, `manifest`, `report`, `script_roots`) are repository-root-relative.
Each manifest repeats its safe unique campaign ID, kind and exact script roots.
Kinds are scaling, regression, application and diagnostic. Historical roots may
be shared; new orchestration belongs in scripts/campaigns/<id>/ beneath GPU.
A synthesis may have no scripts. Unknown schema/kind/analysis values fail closed.

`lifecycle` separates execution, correctness, results_review and archive_verification;
archive migration never upgrades scientific status. `analysis` dispatches a named,
versioned offline validator. Existing numeric profiles preserve their case counts
and table precision. `documentary-v1` and `artifact-only-v1` verify structure,
links and artifacts only; neither asserts numerical performance acceptance.
A new campaign does not require the five historical entries to be present.

Use [the common template](templates/manifest.json) and [kind-specific requirements](templates/README.md).
Pin runtime, harness, application and analysis independently; preserve resolved
configuration, immutable inputs, review states and artifact retention limits.
The catalog and templates are source-only, excluded from installed packages.
