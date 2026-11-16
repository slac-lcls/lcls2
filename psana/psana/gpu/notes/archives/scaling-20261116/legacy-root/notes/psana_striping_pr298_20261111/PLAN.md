# Proposed psana2-GPU stripe-count sweep with PR298

**Historical plan: the unexecuted 72-case screen is superseded by the
[November 15 configuration selection](selection-20261115/README.md).** The new
proposal fixes four BDs, four workers/BD, batch 200 and one CUDA stream/BD,
leaving 18 stripe cases. Its harness adaptation and local qualification are pending.

Status: planning only; no new build, data preparation or jobs submitted.
Goal: identify the best tested stripe layout for psana-GPU input staging at
controlled reader concurrency, then confirm candidates on another placement.
The prior standalone POSIX screen informs candidates but does not select a
psana optimum. All six existing stripe counts remain in this new screen.

## Pinned runtime and previous evidence

Use PR298 head **67c327a148a08b2aaf972e67e1e02e367c2b298f**, fetched and inspected
from monarin/lcls2, base e032e0d562f39ecfee735ebeae8b395f28d4b08e.
PR was open at discovery. Do not silently follow a later head or use GitHub's
synthetic merge SHA. Metadata and PR performance notes are in evidence/.

After execution is authorized, create an isolated checkout/build/install of
that exact head using checkout-local build_psana.sh. Verify available build
options, compiler/environment and the installed Python/native import paths;
record hashes for every loaded framework module and native extension. Use a
matching Perlmutter MPI runtime and KvikIO26.10.00/CuPy14.0.1/CUDA12.9 stack if
compatible with the PR build. If compatibility requires a change, record and
revalidate it before freezing the experiment. Do not label the old three-file
research overlay or install-B as a PR298 build.

The original research implementation improved two-stream 10k staging from
1.2656 to2.1841GiB/s; one stream measured2.2871GiB/s after the fix. These results
were on b95b789ca, not this extracted PR tree. The PR notes explicitly say its
exact new-base GPU runtime has not been rebenchmarked. Both stream counts must
therefore be measured, not assume two is faster. It fixes repeated input-pool
bookkeeping, not a new physical read/coalescing algorithm.

Evidence checked:
- ../sim10_bulk_target_20261109/README.md: 512KiB outperformed4/8MiB targets;
  larger targets changed callback sizes and restricted live input groups.
- ../sim10_stream_length_20261109/README.md: Perlmutter controls were batch100,
  SMD1000,6GiB/BD,8workers,1MiB tasks,16MiB bounce,fallbackON,directOFF.
- ../sim10_100k_bd_sweep_20261109/README.md: 1–4BD scaling was measured on Weka,
  with the old two-stream penalty; its rates are not Perlmutter predictions.
- ../mfx_kvikio_read_20261108/README.md: standalone4-worker/4MiB-task results
  had different cache/access patterns; do not call them psana-optimal defaults.
- evidence/pr298-performance.md: exact source and historical fix evidence.
- ../sim10_striping_20261110/reader/RESULTS.md: reviewed POSIX stripe screen.

These are proven usable controls and the best tested bulk target, not a jointly
optimized set of batch/KvikIO parameters on PR298.

## Fixed parameters for the main screen

| Parameter | Value |
|---|---|
| Dataset | run10 sim10,100,000 unique event timestamps,all five JF streams005–009 |
| Native payload | 32panel seed descriptors; simulated compression; payload cycles repeat |
| Workload | Public GpuTask field-descriptor staging; read/parse/descriptor preparation |
| Excluded processing | Dense gather,expansion,calibration,cube,user payload D2H |
| gpu_bulk_read | True |
| gpu_bulk_target_bytes | 524288 (512KiB); this is not the total GPU memory budget |
| batch_size | 100 |
| PS_SMD_N_EVENTS | 1000 |
| PS_SMD_CHUNKSIZE | 268435456 (256MiB) |
| gpu_memory_budget_gb | 6 per BD (GiB); check aggregate admission and actual device headroom |
| KvikIO compat_mode | ON: CPU fallback, not GDS |
| KvikIO num_threads | 8 per BD |
| KvikIO task_size | 1048576 (1MiB) |
| KvikIO bounce_buffer_size | 16777216 (16MiB) |
| KvikIO auto_direct_io_read | False |
| Lustre stripe size | 1048576 (1MiB),original pool,raid0; existing verified placement |
| GPU | One physical A100-SXM4-40GB,UUID fixed throughout the full screen; helpers GPU-masked |
| Node | One exclusive GPU node; four reserved GPUs,one active |
| CPU | Fixed56 physical cores,one logicalCPU/core,shared affinity set and NUMA interleave |
| MPI roles | One SMD0,one EB,1or4BDs on the same GPU; zero SRV |
| Other controls | PS_PARALLEL=mpi,PS_FAKESTEP_FLAG=0,BLAS/OpenMP threads1 |

Resolve historical API calls against PR source before implementation. Effective
KvikIO defaults must be set and read back before handles open; capture requested
and observed framework settings. n_gpu_streams is execution-slot depth, distinct
from physical XTC streams and KvikIO worker/copy streams. No independent raw-slot
knob is swept: PR298 derives capacity from depth,batch and file configuration.
At4BDs,6GiB per-BD limits total24GiB. CUDA contexts,shared allocations and all
unbudgeted overhead still need measured headroom; limits alone do not prove fit.
Stop any pilot that cannot satisfy admission without pressure drains or OOM.

The56-core choice is a new fixed allowance sufficient for up to32KvikIO workers
plus rank/control work. It differs from the historical24-core Perlmutter test;
new results compare within this campaign, not claim a controlled speedup over
historical numbers. Record physical core/NUMA mapping and exact rank masks.
Same shared core set at1and4BD; increasing BDs changes contexts and total workers
from8to32, intentionally testing aggregate reader concurrency, not process count
in isolation. No cross-topology rate ratio is labeled a pure BD effect.

## Parameter sweep and order

| Stage | Stripes/file | CUDA streams/BD | BDs/active GPU | Events/case | Repeats | Timed cases |
|---|---|---|---|---|---|---|
| Native pilot | 2 | 1,2 | 1,4 | 1000 | 1 | 4 |
| A: one-reader screen | 1,2,4,8,16,32 | 1,2 | 1 | 100000 | 3 | 36 |
| B: higher-concurrency screen | 1,2,4,8,16,32 | 1,2 | 4 | 100000 | 3 | 36 |

A and B labels describe strata, not chronological execution. Run three complete
randomized blocks of all24 stripe×depth×BD configurations, with fresh MPI
processes for every case. Build the canonical list using ascending stripe count,
then depth1/2,then BDs1/4. Use one Python random.Random(20261111) generator and
shuffle a fresh copy of that list sequentially for each of the three blocks;
there is no additional rotation. Save the exact schedule before launch. Run all
blocks within one allocation on the same node/GPU with fixed affinity. Retain all
valid slow trials. If interrupted, preserve the partial campaign; resumption on
another allocation is a new series with an explicit node/job factor and requires
a revised analysis plan, not silent pooling. All three complete blocks are
required for ranking/finalist selection; otherwise report screening-incomplete
and descriptive individual measurements only. All72 full cases use the same exact event set, no prefix
selection. Each uses the intended stripe files. No old POSIX result substitutes
for a missing psana case. Pilot1000 cases are excluded from rankings.

Before the timed pilots, execute CPU unit tests from the exact PR tree and
native ownership/transition/task acceptance on its matching installation.
The six required files under psana/psana/tests/gpu/integration are:
- test_gpu_input_group_device.py
- test_gpu_input_window_device.py
- test_gpu_group_ownership_device.py
- test_gpu_group_schedule_device.py
- test_multiowner_transitions.py
- test_gpu_task_batch_device.py

All six exist at the pinned PR head. Run the full current GPU unit suite,
including test_gpu_input_group.py and test_gpu_slot_selection.py. The static
source test inventory is in evidence/pr298-test-inventory.json. Before timed
pilots, implementation review must include actual pytest collection/node IDs
(including parameterized cases), expected counts, environment requirements and
the matching test-source hashes. Static function names are not collected counts.
Record all passes/skips/failures. Any required test
missing/skipped/failing blocks performance acceptance. Exercise delayed consumers,
partial submission,cleanup,slot reuse,transitions and GPU/CPU sampled seed-value
witnesses. Use a non-dividing event count of1003 in separate correctness
pilots to validate a tail for both depths and both topologies. Correctness
samples are untimed and cache is reestablished afterward. No benchmark threshold
is inserted into unit tests. Do not reuse research tests' old overlay path guards.

Future sensitivity, NOT part of the initial72-case screen: if framework
submission/CPU limits hide any stripe effect, consider batch_size50/100/200,
KvikIO workers4/8,and task_size1/4MiB while retaining512KiB bulk and16MiB bounce.
That is12 configurations at a frozen stripe/topology/depth,36trials for3repeats.
Propose a separate bounded matrix then, including a subsequent stripe comparison
under the selected controls. Do not tune those controls separately per stripe or
combine settings changes with an apparent stripe win. Bulk4/8MiB,bounce tuning,
GDS,multiple activeGPUs and multi-node tests are deferred.

## Dataset binding and cache discipline

Existing source: /pscratch/sd/m/monarin/psana2-gpu/datasets/mfx101555026-r0010-sim10-100k-jf.
Stripe copies: /pscratch/sd/m/monarin/psana2-gpu/sim10-striping-20261110/attempt01/datasets/count-{01,02,04,08,16,32}.
Every layout contains the same five full bigdata files,335820951264bytes total.
These are100k events, not400k. There is one placement per count. All30 were
independently SHA256 verified during staging. Recheck current identities and
actual layouts against the receipts; rehash if identity changed, not on every run.

The count directories currently contain five bigdata files plus verified.json;
SMD metadata is not installed there. After authorization, create one immutable small run view per stripe count,
reused unchanged by fresh case processes, with five payload symlinks to the selected existing count directory, copied
matching five SMD files, original generation.json, and an explicit relocated
subset manifest. Keep existing datasets unchanged. Do not hardcode the original
source root in the runner. Validate every SMD offset/length/timestamp against the
selected target and compare panel membership/Configure and full100k event set.
SMD byte offsets remain valid because payload copies are byte-identical.
Keep original generation.json byte-identical; it also mentions nontransferred
ancillary streams, so the subset manifest governs five-JF-file selection.

A run-view symlink is a reference to an already independently written copy, not
new capacity or a placement replicate. Cache code resolves each link once and
opens the allowlisted real file with O_NOFOLLOW; verify device/inode/size/owner
and getstripe. Capture fstat(device,inode,size) from the actual open KvikIO file handles and
DgramManager bigdata descriptors, with their resolved paths for all five streams,
asserting they equal the selected count receipt. Fail if resolver fallback reads
the original single-stripe source or another count. Before timed pilots, a cheap untimed100-event binding preflight at1BD/1stream
must pass for each of the six stripe counts. It must exercise payload opens,
not only SMD discovery. Where KvikIO has multiple direct/buffered descriptors,
record every underlying input descriptor. Implement a tested capture mechanism
using the installed handle API or descriptor census with unambiguous attribution;
a path string or separately reopened file is insufficient. If the real handles
cannot be audited, stop before timing. Fresh DataSource processes and the six
immutable per-count views prevent reuse of previous file-resolution state.

For each case, warm compilation independently, prepare DataSource/config/SMD,
then owner-scoped fadvise eviction and two zero mincore scans of all30 campaign
payload files, followed by an immediate zero-residency check on the selected5.
No global cache drop. All files and all nodes that could read must pass (one
node in this plan). BDs wait at a host barrier without issuing timed payload
reads until the coordinator completes the gate. Capture MemAvailable,Cached,
Dirty,Writeback and available Lustre cache/readahead settings before/after.
Do not change kernel tuning between cases. If a gate fails after bounded
retries, retain the failed receipt and stop the block; no automatic warm fallback.

Only local page cache is controlled. SMD,server/backend caches and shared
filesystem activity are uncontrolled. Capture llite.*.max_read_ahead*,bdi
read_ahead_kb and available OSC/NIC counter snapshots without root or resets.
Unavailable counters remain explicit; llite application read_bytes is not proof
of physical remote traffic. No storage-wire bandwidth claim without validated
counter semantics. All30-file eviction adds setup time outside the loop.

## Metrics and acceptance

Primary: aggregate completed useful JF datagram GiB / global loop elapsed time;
secondary: completed events/s,per-BD times/work,read-wait diagnostics,and fullcase
wallseconds. Start at synchronized release; end after last BD read/parser/task
completion and explicit GPU drain, including barrier/drain/fill overhead.
Use maximum rank elapsed time covering this common interval, not sum or mean
of overlapping rank/stage times. Report start skew and global completion bound.
No warmup event exclusion in full runs; lazy per-process setup inside the event
loop remains measured. Keep Nsight/detailed timing in separately labeled cases.

Expected full-case global totals at fixed512KiB/batch100:100000 distinct events,
500000 reads,335820000000datagram bytes (312.756747GiB),1000 callbacks of100events.
Full-file POSIX bytes were335820951264, a different denominator. Exact input
range/timestamp/panel sets must match globally once and remain ordered perBD;
perBD event counts can differ. Derive expectations independently from SMD/manifest
and inspect callback/range histograms on the PR build. If the actual contract
changes, stop and explain it; do not relax expectations merely to admit a result.
Both requested CUDA stream IDs must be exercised by every nonemptyBD atdepth2;
record stream use without claiming that it proves overlap. No CPU EventManager
payload reads,pressure-driven cache trims,unfinished requests,active slots,held
budget or leftover rank failures may be accepted. Each BD must report the expected device peer count1or4 for its topology.
PR discover_peers must accept the explicit6GiB-per-BD budget and record the
aggregate6GiB/24GiB reservation against discovered usable device capacity,
plus shared/reserved/unbudgeted memory and actual headroom. GPU UUID and peer
membership must agree across ranks. Full input metadata/layout identities must
match before/after. Require Slurm
job and every step COMPLETED0:0 plus the campaign completion marker.

Read-only descriptor throughput is not cube throughput or a storage ceiling.
Use full cube and a matched exact-read-pattern control before transferring any
stripe recommendation to the XPP end-to-end benchmark or multi-node scaleout.

## Selection, confirmation and cost

For each (BDs,depth) stratum, report all six medians,min/max and block-paired
ratios tocount1. Preserve separate preferred stripe counts if rankings differ;
do not hide a high-concurrency win in an average across unlike workloads.
Three repeats support screening, not narrow confidence/optimality claims.

Retain count1 plus up to two non-1 finalists: rank by geometric mean of the
12paired ratios percount (4strata×3blocks); also retain distinct stratum-specific
winners outside that set. If that would exceed two finalists, report workload-
specific choices and propose a larger confirmation matrix instead of dropping
one silently. The old POSIX finalists1,2,4 do not constrain this new selection.

Confirmation is a later proposed stage, after screening and storage/time review:
second independently allocated five-file set for each retained count,1MiBstripes,
full SHA/size/actual-OST verification. Retain original and new placements and
report OST/OSS overlap. At the preselected target of4BDs/oneGPU with2CUDAstreams/BD run5randomized
blocks of up to3counts×2placements =30fullcases. Select the smallest count with
confirmed median within5% of best, qualify with paired uncertainty and stability
across placements; otherwise report inconclusive. This target is selected now to stress higher input concurrency with the
requested two-stream fix; do not switch it after seeing the screen. Preserve
all other stratum results. More target topologies require separate confirmation
strata and a revised budget. No automatic confirmation
copy creation is authorized by this planning turn.

Initial proposed allocation cap:6node-hours total for native correctness,pilots
and72-case screen, on lcls_g/premium. Historical corrected1BD rate~2.2GiB/s
suggests~142s per100k loop, but the exact new build is unmeasured. Rough planning
range is4–6hours including cold conditioning and preflights, not an ETA. Measure
whole-case pilot durations plus one separate full1BD/depth2/count2 calibration
case after authorization (charged to the6hourcap, excluded from rankings), revise
explicit walltime within cap, and stop with partial evidence if remaining budget
cannot fit the next block. Do not increase cap automatically or skip slow valid
points. CPU build happens outside compute allocations using the checkout-local
build helper; no GPU benchmark runs on a login node.

If charged against the earlier12node-hour campaign envelope,8,807node-seconds
(2.446h) have already been used;6additional hours leaves3.554h before confirmation
or extra preparation. Check that envelope against any other task charges before
claiming it available. Iris allocation balance is separate: refresh before each
submission and alert below20% personal/project remaining. Refresh scratch quota
and retained datasets before any new large copies; the initial screen reuses
existing payloads and needs only small metadata/log storage.

After the parameter list is presented: implement/adapt the harness in an isolated
work area, review its code and provenance, validate the exact PR build and data
binding, then submit only the agreed stage. This document review is not runtime
acceptance. No jobs or build were run while preparing this plan.
