# Perlmutter100k S/E/EC/ECC comparison01

Conditional submission is authorized after the implementation-review loop passes
without a user decision. One exclusive node/four GPUs, lcls_g/premium,4h10m;
all35 cases stay in that allocation. No retries, tuning, extra cases or automatic
successor job. Keep the reused PR298 native install and fixed count2/1MiB view.

## Evidence and sizing

Bootstrap job61387931 completed0:0 in22m21s on nid001033. Coordinator re-audited
all11 result hashes/receipts, identity/range/byte/memory gates, native31tests
and10guards on each GPU, and immutable source/external pins. Compact evidence:
evidence/job61387931-results.json. Independent assessment accompanies review.

| Profile |100k staging median processing|Slowest whole step|
|---|---:|---:|
|1BD/1GPU/batch50|151.851s|198.252s|
|4BD/4GPU/batch200|61.612s|110.602s|
|4BD/1GPU/batch200 bridge|61.378s|109.151s|

Small ECC cases:201events1BD processed in0.834s, whole step20.252s;
1003events4BD processed in1.203s, whole step23.863s. All bank content checks
passed. Minimum per-process free memory:27.397GiB1BD,12.809GiB4BD; sampled
whole-device extrema are in the evidence. These short timings do not establish
100k compute throughput or linear scaling. Retain runtime memory gates.

Whole-case S limit240s retains the tested bound,41.7s beyond the slowest
observed1BD step. Compute limits are600s1BD and360s4BD: conservative stop bounds
covering unmeasured full processing plus setup/conditioning/checks, not runtime
predictions. Crude short-ECC extrapolations (~415s and120s processing) inform
these ceilings only; batching/cache/startup differences make them unreliable.
A timeout fails the job and requires a revised plan, never a larger automatic
budget. New GPU checks have executed; job61373306's removed export-path stall
still has no proven root cause.

## Matrix and deadline budget

Two short ECC checks first. For each of3 repeats: run the100k staging-only
bridge, then both target profiles. Each profile block starts with full S, runs
E/EC/ECC in cyclic order, and ends with a separate S anchor. Profile order is
1BD,4BD /4BD,1BD /1BD,4BD. All target cells use100k global unique events.
This yields24 main cells,6 S anchors,3 bridges and2 acceptance cells=35.
The historical8 traced4000-event pilots are not scheduled: the requested
comparison is untraced100k work. Do not claim kernel attribution/syscall proof.

|Work|Stop budget|
|---|---:|
|Runtime/hardware/affinity preflight|300s|
|31native tests+10guards on each GPU|480s|
|Two ECC checks|600s|
|Six main S +six S anchors +three bridges|15×240=3600s|
|1BD E/EC/ECC,three repeats|9×600=5400s|
|4BD E/EC/ECC,three repeats|9×360=3240s|
|Case audits|35×20=700s|
|Final complete-matrix re-audit and summary|120s|
|Safety/startup/final checks reserve|560s|
|Total|15000s=04:10:00|

Maximum reservation4.167node-hours/16.667allocatedGPU-hours before premium
accounting conversion; not an Iris charge prediction. Use scheduler UTC EndTime
and refuse a step unless its whole limit plus120s remains. Stop at first failure.

## Controls, validation and interpretation

Four KvikIO workers/BD, one CUDA stream,1MiB tasks,16MiB bounce/worker,512KiB
bulk,6GiB framework input ceiling/BD, SMD1000, compatibility ON/direct OFF,
PS_JF_SHARE_DERIVED=0, PS_GEO_SHARE=0; frozen pedestals/gain/offset. Target
profiles1BD/1GPU/batch50 and4BD/4GPU/batch200. Never share GPU for compute.

Retain srun --mpi=cray_shasta, explicit3/6ranks,120/tasks CPU slots,
--cpu-bind=none/--gpu-bind=none and numactl --interleave=all. Probe allowed masks,
bind3,3,7 or3,3,7,7,7,7 distinct physical cores; retain first-BD CPU/GPU placement.
Helpers see no GPUs; actual allocated UUID/PCI mapping is checked. All cases
run fresh MPI processes on the same node. No claim of optimized NUMA locality.

Recondition only qualified owned inputs after setup; require zero local resident
pages. Preserve available Lustre counters; backend/NIC/OST traffic remains
unverified. Coordinate active count2 users before submit; cooperative lock alone
does not constrain old campaigns. Refresh Iris personal/project balances.

Retain native tests, errors, actual-open identities, global event union, exact
bytes/ranges, single-stream/ownership/drain, >=4GiB physical memory headroom and
sampled whole-device NVML. ECC validates each actual GPU bank's dimensions,
float32 dtype, bytes, counts, finite every pixel, nonzero populated bins and zero
empty bins. No cube payload D2H, global cube merge, files or CPU reference.
Only scalar flags/counts return; diagnostic JSON/logs retained with4GiB reserve.

The processing timer includes staging/delivery/drain and existing memory probes.
Setup/JIT/constants, cold conditioning, post-loop cube verification and summary
are excluded. Per-rank monotonic progress markers stay outside processing.
A stall in its ending barrier/allreduce remains indistinguishable from processing
until that marker is written; this known diagnostic limit is unchanged.

comparison.py re-audits every result against the same hardware/affinity map,
checks hashes/audits/memory and same job/node, and refuses partial matrices.
It reports median/min/max for8profile-stage groups; per-repeat compute/S ratios
and differences; post/pre S drift; bridge separately. Acceptance and anchors
never enter main stage statistics. No cross-node pooling, isolated-kernel or
GPU-only scaling claim (BD count and batch also change across profiles).

The bootstrap's measured audit-step durations are recorded in the evidence.
Maximum100k audit was 1.017s;33times that maximum is 33.545s versus120s summary
ceiling. This supports retaining the budget but does not guarantee runtime.
If only the final summary fails, preserve all failed-job receipts and rerun
comparison.py offline against that immutable attempt after diagnosing the error;
no compute rerun is needed. An offline summary does not change Slurm failure
status or replace independent result review.
