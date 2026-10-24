# Device peer discovery and CUDA-IPC constant sharing

Integration evidence for `features/psana2-ipc` (issues #155, #168) on real
A100s with real CUDA IPC. Unit tests inject fakes for the identity table, the
launcher environment, the communicator and CuPy; these cases use none of that.

Cases are driven by `run_cases.py`, one MPI job each with a private PRRTE
session, because a shell loop of back-to-back `mpirun` calls hung: consecutive
launches share session state and an aborting case can block the next. Each case
is hard-bounded by a timeout and its outcome parsed from its log, since `mpirun`
exits 0 even when ranks report failures.

Collective-order checks supplement exit status and timeout checks. A driver
that wraps `device_comm` in the test recorder
(`psana/psana/tests/gpu/collective_recorder.py`) has each peer compare its
sequence with the others; `collective_order_diverged` must be 0 in every case.

Run from the repository root after building:

    sbatch validation/ipc-integration-20261007/run.sbatch    # 13 cases, 1 node, 2 GPUs
    sbatch validation/ipc-multinode-20261007/run.sbatch      # 4 cases, 2 nodes

## Cases

Thirteen cases. Five discovery topologies, because the defect being fixed was
peer counts
derived from EB-group arithmetic: `bd_comm` is split per EB group, so the count
was wrong whenever groups did not map onto devices. `discovery-uneven` is the
case where rank arithmetic would give ranks on one device *different*
counts.

The standalone discovery driver runs one CPU-only participant and eight GPU
workers (seven for `discovery-uneven`). Its `--n-eb` and `--contiguous` options
change only the hypothetical EB-group arithmetic used for comparison; they do
not launch real EBs or reproduce psana's EB/BD assignment. The GPU placement
stays the same across these comparisons. Real EB roles are asserted in the
DataSource event-loop cases.

| Case | Ranks | Asserts |
| --- | --- | --- |
| `discovery-eb1` | 9 | 2 devices, 4 peers each, aggregate claim 1.0 |
| `discovery-eb2` | 9 | same; hypothetical two-group arithmetic would claim 2.0x |
| `discovery-eb2-contig` | 9 | same; hypothetical contiguous group assignment |
| `discovery-eb3` | 9 | same; hypothetical three-group arithmetic (3.0x claim) |
| `discovery-uneven` | 8 | 3 and 4 peers, both claim 1.0 |
| `calib-shared` | 4 | 1 smd0 + 1 EB + 2 BDs on one GPU; pixel-exact against `det.raw.calib()` |
| `calib-2-gpus` | 9 | 1 smd0 + 1 EB + 7 BDs; 4 and 3 BDs per GPU, mask left wide |
| `calib-reversed-mask` | 6 | the same under `CUDA_VISIBLE_DEVICES=1,0`, where ordinals and physical cards disagree |
| `calib-budget-fits` | 4 | an explicit `gpu_memory_budget_gb` the device can honour is accepted **and applied** — `usable_bytes == budget x peers` |
| `calib-budget-over` | 4 | a budget no device can honour aborts **every** role, smd0 and EB included |
| `fail-select-device` | 5 | a rank that cannot select a device aborts the **job** |
| `multirun-3-runs` | 6 | known second-run hang, tracked separately in P2 #209 |
| `fail-follower-import` | 4 | a failed IPC import aborts **every** peer, not just the one that failed |

The two-node suite (`../ipc-multinode-20261007/run.sbatch`) adds four cases at
10 ranks across 2 hosts:

| Case | Asserts |
| --- | --- |
| discovery, hypothetical 1-EB comparison | 2 hosts, 4 devices, no group spans a host |
| discovery, hypothetical 2-EB comparison | the same, so EB topology does not reach grouping |
| event loop, 2 EBs | 1 smd0 + 2 EBs + 7 BDs; 2 and 5 BDs per GPU; 13 pixel-exact events |
| event loop, 1 EB spanning 2 nodes | 1 smd0 + 1 EB + 8 BDs; 3 and 5 BDs per GPU; 13 pixel-exact events |

The event-loop cases narrow the mask to one card per node so the workers on a
node are peers. With a wider mask, workers can spread over more devices;
the observed peer counts, rather than the mask alone, establish sharing. The
mask is exported with `mpirun -x`, because a prefix assignment
reaches only the local rank and the remote node would keep Slurm's own.

`aggregate_claim` is the sum of per-rank budgets over device capacity. It must
be <= 1.0: above 1.0 the ranks on that device have collectively claimed more
memory than exists. Each discovery record also reports `current_believed_peers`
and `current_aggregate` computed from EB-group rank arithmetic, so the log
shows the difference identity-based grouping makes rather than only that it
passes.

The two failure cases are what unit tests structurally cannot cover: `FakeComm`
drives peers sequentially, so it has no notion of two ranks blocked in
*different* collectives.

The `calib-*` cases are the same program in three layouts, and they are what
covers the branch's machinery. A real psana job -- smd0, EBs and BD workers as
psana assigns them -- calibrates real Jungfrau data on the GPU through
`GpuTask` and compares every event against `det.raw.calib()` on the CPU in the
same loop.

Pixel equality establishes calibration correctness, but can also pass when
workers hold private constant copies. The original driver captured `sharing`
before `run.events()` initialized the live constants, so its `off` field is
an early snapshot and cannot establish whether IPC was used. The independent
[live IPC audit below](#live-ipc-sharing-evidence-october-23) verifies owner
allocations, follower imports, and budget charges during processing.

`fail-select-device` checks that a device-selection failure aborts the job,
including CPU roles. `calib-budget-over` separately exercises the job-wide
budget verdict. `judge_abort` requires a nonzero exit and completion within
its timing bound so that a timeout is not mistaken for a successful abort.

`fail-follower-import` requires the abort to be uniform. The capability flag
is reduced over the group, so a rank that carried on would read an allocation
its peers have already freed — silent corruption rather than a hang. The
assertion is therefore on `allgather`ed raise-or-not verdicts rather than on
each rank alone, and the case stays `expect='clean'` because the driver catches
the `SharedConstantsError` to check things an exit status cannot show: that
every peer raised, that the message names `PSANA_GPU_SHARE_CONSTANTS=0`, and
that nothing was left mapped or charged.

### The two explicit-budget cases

`gpu_memory_budget_gb` had unit coverage only: `mpi_placement_discovery.py`
always passes `explicit_limit_bytes=0`. It is a different code path, not a
parameter — discovery validates `budget x peers` against **this device's**
capacity and replaces `usable_bytes` with the claim.

`calib-budget-fits` checks the budget was *applied*, not merely accepted: a
job that ignored the ceiling calibrates the same pixels, so
`--expect-budget-honoured` asserts `usable_bytes == budget x peers`. This
checks the configured budget independently of pixel equality. In single-node
job 41602553, 2 GiB per BD times two peers produced a 4.00 GiB device claim
and 13 pixel-exact events. The two-node cases in job 41519162 used automatic
budgets (`budget_gb: null`); they did not run this explicit-budget case.

`calib-budget-over` tests the collective. smd0 and EB cannot evaluate any
device's claim and take the verdict from the `discovery/budget-verdict`
`MAX`; if that reduction were dropped they would continue into the event loop
while the workers aborted — a hang, not a failure.

Its 40 GiB is chosen so the over-commit holds whatever the peer count and
card: `40 x 2 = 80` GiB exceeds both a 40 GB card's ~34.8 usable and an 80 GB
card's ~72. A smaller figure passes vacuously — `15 x 2 = 30` *fits* in 34.8.
`judge_abort` also requires `usable on that device` in the log, not bare
`exceeds`, which matches `allocation exceeds admission estimate` and two
others; such a log was measured passing the case.

`multirun-3-runs` is `expect='hang-known'`: psana2 hangs at the second run of
the tested multi-run MPI jobs, including on the base branch. The case
passes while the bug is present and **fails if the job completes**, so the
suite says when to promote it. See
[P2 #209](https://github.com/slac-lcls/lcls2/issues/209) and
[docs/multirun_hang.md](../../psana/psana/gpu/docs/multirun_hang.md).

## Results

**Production code: `b0103a3ad`; recorded validation revision: `c9d3b8327`.**
The original runs used a working tree carrying the validation changes later
committed at `c9d3b8327`; production GPU and psexp code was unchanged from
`b0103a3ad`. The October 23 audit also used production head `c9d3b8327`, with
local test-only instrumentation described below. This report update changes
no production or test code.

| Suite | Job | Node(s) | Result |
| --- | --- | --- | --- |
| single node | 41602553 | sdfampere004 | **13/13 expected outcomes**, including known multi-run hang |
| two nodes | 41519162 | sdfampere038, sdfampere040 | **PASS**, 4/4 |

Two nodes, job 41519162: both EB topologies pass, and the two-EB one is
**observed** rather than requested — `roles: {bd: 7, eb: 2, smd0: 1}` with
`eb_nodes: 2` at 10 ranks, against `{bd: 8, eb: 1, smd0: 1}` for the
single-EB case. 13 events and `all_pass` in both. Real two-EB coverage comes
from multi-node event-loop cases: `PS_EB_NODES` is capped at the distinct
host count for mfx experiments, so a single-node allocation cannot produce it however many are
requested.

### Per case, single node

| Case | Ranks | Expect | Result | Elapsed |
| --- | --- | --- | --- | --- |
| `discovery-eb1` | 9 | clean | PASS | 10.4s |
| `discovery-eb2` | 9 | clean | PASS | 9.2s |
| `discovery-eb2-contig` | 9 | clean | PASS | 9.3s |
| `discovery-eb3` | 9 | clean | PASS | 9.2s |
| `discovery-uneven` | 8 | clean | PASS | 8.7s |
| `calib-shared` | 4 | clean | PASS | 195.3s |
| `calib-2-gpus` | 9 | clean | PASS | 195.5s |
| `calib-reversed-mask` | 6 | clean | PASS | 195.4s |
| `calib-budget-fits` | 4 | clean | PASS | 194.7s |
| `calib-budget-over` | 4 | abort | PASS | 190.0s |
| `multirun-3-runs` | 6 | hang-known | Known hang reproduced (#209) | 150.0s |
| `fail-select-device` | 5 | abort | PASS | 10.9s |
| `fail-follower-import` | 4 | clean | PASS | 6.7s |

The multi-run result is not successful multi-run processing: run 15 delivered
20 events, run 16 delivered none and hung, and run 17 was not reached before
the 150-second timeout. It does not establish rediscovery or cleanup across
all three runs. Fixing P2 #209 is separate from this PR.

The recorder reports `collective_order_diverged: 0` for its observed
sequences; that does not establish completion of the known-hanging case.

A run can also stop short: the suite emits `CASE_SKIP` rather than be killed
mid-case and lose the VERDICT. **A `SKIP` is neither a failure nor a pass** —
the verdict reports `INCOMPLETE` and exits nonzero. A healthy ~20-minute run
never skips.

`calib-budget-over` passed in 190.0s against a 300s budget, exit 1, with the
expected error on the BD workers —

    GpuPlacementError: gpu_memory_budget_gb=40.00 GiB x 2 peers on
    sdfampere004 ... = 80.00 GiB exceeds the 34.80 GiB usable on that device

which also shows why the figure is 40 and not 15: the card reports
**34.80 GiB** usable to **2** peers, so `15 x 2 = 30` would have *fit* and
the case would have passed without provoking an abort.

The measured `calib-budget-over` duration is 190.0s from launch to exit.
Its fatal error is not timestamped, so the log does not establish the exact
abort time or partition the remaining time into discovery and MPI teardown.
The fits case records setup at t+90.7s and placement at t+92.4s; the over case
records setup at t+90.5s. Those observations explain why a short timeout can
expire during DataSource setup, but do not prove an abort at t+92s.
`fail-select-device` calls discovery directly and completed in 10.9s in this
suite. The configured budget-case timeout is 300s.

This run used `SBATCH_ACCOUNT=lcls:prjdat21 SBATCH_QOS=normal`. Earlier
attempts on the default `preemptable` QoS were evicted three times mid-suite.

Earlier runs are not cited as evidence for this head. Jobs 40888078, 40889090
and 40889879 (Oct 15) predate `4b068a867`, `b5cda0dae` and `b0103a3ad`, and
the last of those moved the capacity `MIN` off `psana_comm` and replaced the
peer-count `MAX` with the budget-verdict `MAX` — a change to the setup
collective sequence, which is exactly what a hardware run is for. Job
40888078 also **FAILED** (`sacct` exit 1:0), so the previous "13/13 PASS"
line citing it was wrong on its own terms.

Discovery: two devices, `aggregate_claim` 1.0 on each, with `discovery-uneven`
giving groups of 3 and 4. Every record also reports what EB-group rank
arithmetic would have produced, which reaches 2.0x the device at
`PS_EB_NODES=2`.

The value check, 13 events per case, every pixel equal to `det.raw.calib()`.

`calib-shared` has four total ranks: one smd0, one EB, and two BDs sharing
one GPU. `calib-2-gpus` has nine total ranks, with seven BDs split four and
three across the GPUs. `calib-reversed-mask` reverses the card order. These
cases establish pixel agreement; the live audit below establishes actual
constant sharing. The fixture requests four arrays totaling 960 MiB; the
older 384 MiB example counts only pedestals and gain.

All three run with ONE event builder. `PS_EB_NODES` is capped at the distinct
host count for mfx experiments -- the cap lives in `_force_mfx_overrides`,
which returns early for any experiment whose name does not start with `mfx`,
and this fixture is `mfx100848724` -- so a single-node allocation cannot have
more. The 2-EB topology is covered by the multi-node suite, which has the
hosts for it; `--expect-eb-ranks` asserts the count in both places rather
than leaving it to the case name.

Those timings are why the `calib-*` cases carry a 420s timeout rather than the
180s default -- `hybrid_det` mirrors the bigdata streams through both the CPU
and GPU paths, which is what makes `det.raw.calib()` available for the
comparison, so the I/O is done twice.

### Live IPC sharing evidence, October 23

Independent audit against production head `c9d3b8327`, with test-only
instrumentation kept outside this PR:

| Layout | Job | Nodes | Result |
| --- | --- | --- | --- |
| 1 GPU, 2 BDs | 41748018 | sdfampere030 | PASS, 100 pixel-exact events |
| 2 GPUs, 2 BDs each | 41748018 | sdfampere030 | PASS, 100 pixel-exact events |
| 2 nodes, 2 EBs, 2 + 5 BDs | 41748646 | sdfampere010/011 | PASS, 100 pixel-exact events |

Both accepted Slurm jobs completed `0:0`, and each MPI launch exited zero.
Every BD received pixel-checked callbacks. Each GPU held one set of constants:

| Measurement | Owner | Each follower |
| --- | ---: | ---: |
| Owned constants | 960 MiB | 0 |
| Imported constants | 0 | 960 MiB |
| Constant-storage charge | 960 MiB | 0 |
| Allocation calls | 4 | 0 |
| IPC import calls | 0 | 4 |

Imported constants are mappings of the owner's GPU allocations into the
follower's CUDA context. They are not another allocation or copy. The zero
follower charge concerns constant storage, not context or event-processing
memory. The total comprises pedestals, gain, and offset at 192 MiB each, plus
uint64 pixel status at 384 MiB.

Follower IPC-handle fingerprints matched the owner's exports, and live
`batch.calibconst()` pointers matched the observed allocations or mappings.
Allocation history recorded no temporary private upload or extra owner
allocation. Automatic limits matched integer-byte calculations exactly:
`follower = (usable - shared) // peers`; `owner = follower + shared`.

The first two-node attempt (41748020) failed the audit's coverage check:
one SMD batch routed all 100 events to one EB, leaving a follower without a
callback. It is not counted as an acceptance pass. Setting and exporting
`PS_SMD_N_EVENTS=10` exercised both EBs in 41748646, with 50 events per EB
group and callbacks on all seven BDs. Production and audit code were unchanged
between attempts. Seven CPU-only evidence-validator checks also passed.

The dataset was mfx100848724 run 51, Jungfrau, `batch_size=1`, two GPU streams,
bulk reads, and automatic budgets. Hardware was A100 40 GB; KvikIO used CPU
fallback. Python 3.9.20, CuPy 13.6.0, and mpi4py 4.0.1 were used. Native
extensions were reused from Seema's installation, with tracked Python sources
overlaid from the reviewed revision. Runtime hashes and exact commands are
retained with the evidence. This measures correctness and constant ownership,
not throughput, total process VRAM, or true-GDS performance.

Detailed results and artifact locations are recorded in the
[PR review comment](https://github.com/slac-lcls/lcls2/pull/187#issuecomment-6371369199).
The local audit commit `c97533d71` and its patch are retained for reproduction;
the instrumentation is not required to merge this PR. On SDF:

- Scratch root: `/sdf/scratch/users/m/monarin/gpu-validation/pr187-ipc-evidence-20261023`.
- Accepted logs and JSON: `runs/41748018-single/` and `runs/41748646-multi/`
  under that root; rejected attempt: `runs/41748020-multi/`.
- Durable report: `/sdf/home/m/monarin/lcls2_worktree/psana2-gpu/psana/psana/gpu/notes/pr187_ipc_sharing_evidence_20261023.md`.
  Its linked evidence files contain the collected JSON and audit patch.

The 2-versus-5 BD split also produces unequal GPU sharing and per-BD budgets.
Topology balancing is separate psana work, not an acceptance requirement here.

## Not covered

MIG, which is unsupported and untested: no MIG hardware is advertised on S3DF,
and IPC cannot span instances in any case. Instances of one card share a PCI
bus id, so they group together and the IPC import fails, which aborts.
`PSANA_GPU_SHARE_CONSTANTS=0` gets the job running, but the peer count is still
inflated by the merged instances and every budget is divided by it, so it is
not a MIG configuration. Deferred per issue #155. See
[docs/device_placement_and_shared_constants.md](../../psana/psana/gpu/docs/device_placement_and_shared_constants.md).

No throughput claim. What is measured is device memory, not events/s.

**Historical observation (not validation of this revision):** selecting a
device under a wide mask was measured not to leave a stray CUDA
context on ranks that do not use the GPU (job 40151108), which is why the mask
is left as the launcher set it.
