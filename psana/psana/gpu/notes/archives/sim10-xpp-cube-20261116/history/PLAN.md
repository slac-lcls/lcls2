# Perlmutter100k S/E/EC/ECC comparison02 — script review

Current authorization is implementation and review of three cache-conditioning
attempts. No replacement submission in this turn. Proposed future allocation:
one exclusive node/four GPUs, lcls_g/premium,5h55m, same35 cases. Preserve
comparison01 and its failed job61391587. No scheduler retries or automatic
successor jobs. PR298 build, workers and count2 inputs remain fixed.

## Cache-conditioning recovery

Job61391587 passed its first seven cases, including the full100k1BD S/E/EC/ECC
set, then failed the first S anchor's cold-cache gate before processing. Exact
reason for incomplete eviction is unknown; counts were not persisted. See
RESULTS-61391587.md. This change tolerates transient failures without assuming
their cause or claiming that retries will resolve persistent residency.

Each case conditions the five qualified count2 files after setup while other
ranks wait. At most three attempts: request DONTNEED and scan, wait0.1s, scan
again without eviction. Both scans must show zero resident pages for every
file. Wait1s before retrying a failed residency pair. All scans, elapsed times,
paths, page counts and failures are written atomically to <case>.json.cache.json
before the next gate. Successful history is embedded in the hashed case report
and independently audited, including both final scans, file coverage and limits.
I/O/identity errors fail immediately; they are not retryable residency results.

Conditioning deadline180s uses monotonic time, checked before and after scans
and pauses. Synchronous scans cannot be preempted by this helper; enclosing
process and srun timeouts remain the hard bound. A late scan is saved and fails,
even if zero. Deadline includes receipt writes; processing starts only after a
successful gate and MPI barrier. Conditioning is excluded from processing time.
Same policy for every stage/profile; retain retry histories to assess effects.
Only node page-cache residency is checked, not backend Lustre cache coldness.

Add180s per case conservatively to every previous whole-case ceiling. The old
ceiling already included one conditioning pass; this deliberate over-reserve
avoids cutting processing allowance to fund retries. No automatic enlargement.

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

Whole-case S limit420s; ECC acceptance480s; compute780s1BD and540s4BD.
These retain original stop ceilings plus180s conditioning reserve. Failed job
61391587 measured first1BD processing150.052/151.240/168.811/186.531s for
S/E/EC/ECC, but no repeated or4BDcompute results. Limits are stop bounds, not
predictions. A timeout fails the job; no automatic processing retry.

## Matrix and deadline budget

Two short ECC checks first. For each of3 repeats: run the100k staging-only
bridge, then both target profiles. Each profile block starts with full S, runs
E/EC/ECC in cyclic order, and ends with a separate S anchor. Profile order is
1BD,4BD /4BD,1BD /1BD,4BD. All target cells use100k global unique events.
This yields24 main cells,6 S anchors,3 bridges and2 acceptance cells=35.
The historical8 traced4000-event pilots are not scheduled: the requested
comparison is untraced100k work. Do not claim kernel attribution/syscall proof.

The original case ceilings below plus the separate conditioning reserve sum
to the actual revised case timeouts; do not add that reserve twice.

|Work|Stop budget|
|---|---:|
|Runtime/hardware/affinity preflight|300s|
|31native tests+10guards on each GPU|480s|
|Two ECC checks|600s|
|Six main S +six S anchors +three bridges|15×240=3600s|
|1BD E/EC/ECC,three repeats|9×600=5400s|
|4BD E/EC/ECC,three repeats|9×360=3240s|
|Additional conditioning reserve for all35cases|35×180=6300s|
|Case audits|35×20=700s|
|Final complete-matrix re-audit and summary|120s|
|Safety/startup/final checks reserve|560s|
|Total|21300s=05:55:00|

Maximum reservation5.917node-hours/23.667allocatedGPU-hours before premium
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
and differences; post/pre S drift; bridge separately. Per-case conditioning
attempt counts and seconds accompany the summary to expose uneven retries. Acceptance and anchors
never enter main stage statistics. No cross-node pooling, isolated-kernel or
GPU-only scaling claim (BD count and batch also change across profiles).

The bootstrap's measured audit-step durations are recorded in the evidence.
Maximum100k audit was 1.017s;33times that maximum is 33.545s versus120s summary
ceiling. This supports retaining the budget but does not guarantee runtime.
If only the final summary fails, preserve all failed-job receipts and rerun
comparison.py offline against that immutable attempt after diagnosing the error;
no compute rerun is needed. An offline summary does not change Slurm failure
status or replace independent result review.
