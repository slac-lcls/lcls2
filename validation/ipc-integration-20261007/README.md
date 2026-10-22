# Device peer discovery and CUDA-IPC constant sharing

Integration evidence for `features/psana2-ipc` (issues #155, #168) on real
A100s with real CUDA IPC. Unit tests inject fakes for the identity table, the
launcher environment, the communicator and CuPy; these cases use none of that.

Cases are driven by `run_cases.py`, one MPI job each with a private PRRTE
session, because a shell loop of back-to-back `mpirun` calls hung: consecutive
launches share session state and an aborting case can block the next. Each case
is hard-bounded by a timeout and its outcome parsed from its log, since `mpirun`
exits 0 even when ranks report failures.

Every hang in this work was a collective-path divergence, and nothing else
detects one. A driver that wraps `device_comm` in the test recorder
(`psana/psana/tests/gpu/collective_recorder.py`) has each peer compare its
sequence with the others; `collective_order_diverged` must be 0 in every case.

Run from the repository root after building:

    sbatch validation/ipc-integration-20261007/run.sbatch    # 11 cases, 1 node, 2 GPUs
    sbatch validation/ipc-multinode-20261007/run.sbatch      # 4 cases, 2 nodes

## Cases

Thirteen cases. Five discovery topologies, because the defect being fixed was
peer counts
derived from EB-group arithmetic: `bd_comm` is split per EB group, so the count
was wrong whenever groups did not map onto devices. `discovery-uneven` is the
case where rank arithmetic would give ranks on one device *different*
counts.

| Case | Ranks | Asserts |
| --- | --- | --- |
| `discovery-eb1` | 9 | 2 devices, 4 peers each, aggregate claim 1.0 |
| `discovery-eb2` | 9 | same, with the EB split that caused the 2.0x over-commit |
| `discovery-eb2-contig` | 9 | same, contiguous rank assignment |
| `discovery-eb3` | 9 | same, 3 EB groups (the 3.0x case) |
| `discovery-uneven` | 8 | 3 and 4 peers, both claim 1.0 |
| `calib-shared` | 4 | **the value check**: GPU calibration pixel-exact against `det.raw.calib()`, constants shared over IPC |
| `calib-2-gpus` | 9 | the same across 2 devices with the mask left wide, so most groups are solo |
| `calib-reversed-mask` | 6 | the same under `CUDA_VISIBLE_DEVICES=1,0`, where ordinals and physical cards disagree |
| `calib-budget-fits` | 4 | an explicit `gpu_memory_budget_gb` the device can honour is accepted **and applied** — `usable_bytes == budget x peers` |
| `calib-budget-over` | 4 | a budget no device can honour aborts **every** role, smd0 and EB included |
| `fail-select-device` | 5 | a rank that cannot select a device aborts the **job** |
| `multirun-3-runs` | 6 | per-run rediscovery — reproduces a known pre-existing hang |
| `fail-follower-import` | 4 | a failed IPC import aborts **every** peer, not just the one that failed |

The two-node suite (`../ipc-multinode-20261007/run.sbatch`) adds four cases at
10 ranks across 2 hosts:

| Case | Asserts |
| --- | --- |
| discovery, 1 EB spanning 2 nodes | 2 hosts, 4 devices, no group spans a host |
| discovery, 2 EB groups | the same, so EB topology does not reach grouping |
| event loop, 2 EB groups | real events and `batch.calibconst()`, one owner per host |
| event loop, 1 EB spanning 2 nodes | identical device groups under either topology |

The event-loop cases narrow the mask to one card per node so the workers on a
node are peers; with it left wide they spread over both cards and most groups
are solo. The mask is exported with `mpirun -x`, because a prefix assignment
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

They assert on PIXELS, not on internals. Device discovery, budget sizing,
constant sharing over IPC and batch routing have all happened by the time a
pixel exists, so if any of them were wrong the values would be wrong. A
follower calibrates from the owner's imported mapping, so a shared view with a
wrong offset or truncated length shows up as mismatched pixels rather than as
nothing at all; a rank that followed the CUDA ordinal instead of the device
would read the wrong card's memory under `calib-reversed-mask`.

This replaced five white-box drivers that probed `_task_constants`, placement
records and peer counts -- 1,828 lines for 196. The earlier `realpath-*` and
`shared-calibration` cases were among them.

`fail-select-device` is the only test of an abort reaching EB and smd0 ranks. The
`allreduce` inside `discover_peers` covers only GPU workers on the node, so
other roles can be reached only by `gpu_error_handler` aborting the
communicator. Timing is part of the contract: a correct abort and a regressed
deadlock produce near-identical logs, so `judge_abort` requires a nonzero exit
*and* completion well inside the timeout. Without the timing clause a hang
would pass.

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
`--expect-budget-honoured` asserts `usable_bytes == budget x peers`. The one
place a case reads an internal.

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
any multi-run MPI job, which is pre-existing on the base branch. The case
passes while the bug is present and **fails if the job completes**, so the
suite says when to promote it. See
[docs/multirun_hang.md](../../psana/psana/gpu/docs/multirun_hang.md).

## Results

**Production code: `b0103a3ad`. Harness: this commit.** Both, because they
differ: the runs used a tree already carrying these validation changes, so
`b0103a3ad` alone will not reproduce the table. Nothing under
`psana/psana/gpu/` or `psana/psana/psexp/` differed when they ran, which is
what the production figures rest on.

| Suite | Job | Node(s) | Result |
| --- | --- | --- | --- |
| single node | 41602553 | sdfampere004 | **13/13** |
| two nodes | 41519162 | sdfampere038, sdfampere040 | **PASS**, 4/4 |

Two nodes, job 41519162: both EB topologies pass, and the two-EB one is
**observed** rather than requested — `roles: {bd: 7, eb: 2, smd0: 1}` with
`eb_nodes: 2` at 10 ranks, against `{bd: 8, eb: 1, smd0: 1}` for the
single-EB case. 13 events and `all_pass` in both. This is the only place the
two-EB path runs: `PS_EB_NODES` is capped at the distinct host count for mfx
experiments, so a single-node allocation cannot produce it however many are
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
| `multirun-3-runs` | 6 | hang-known | PASS | 150.0s |
| `fail-select-device` | 5 | abort | PASS | 10.9s |
| `fail-follower-import` | 4 | clean | PASS | 6.7s |

`collective_order_diverged: 0` in every case.

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

**Why an abort case takes ~190s.** Discovery is reached ~90s in and decides
in under 2s, so nearly all of the elapsed time is either side of it. Measured
back-to-back on `sdfampere004` in job 41602553:

| | setup | discovery | case total |
| --- | ---: | ---: | ---: |
| `calib-budget-fits` | t+90.7s | placement t+92.4s | 194.7s |
| `calib-budget-over` | t+90.5s | aborts ~t+92s | 190.0s |

The two spend about the same ~100s after discovery, but `fits` spends it
calibrating 13 events while `over` has already aborted — so for `over` that
window is MPI/PRRTE teardown and process exit, not a rank waiting. Nothing
waits out `SETUP_TIMEOUT_S`: a collective that slow logs a warning, and the
log has none. `fail-select-device` aborts in ~11s because it calls
`discover_peers` directly and never builds a `DataSource`.

That is why `ABORT_SETUP_S` exists: promptness has to be judged from when the
abort becomes reachable, not from launch. The case first ran with a 120s
timeout copied from `fail-select-device` and was killed mid-setup, reporting
a hang that was only the timeout expiring — and `judge_abort`'s
half-the-timeout clause made it unpassable anyway, since 60s is below the
setup cost.

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

`calib-shared` narrows the mask to one card, so all four BD workers are peers
and the pedestals and pixel_gain -- 384 MiB for this detector -- come from a
single device allocation over IPC. `calib-2-gpus` leaves the mask wide, so
the workers split across cards and most groups are solo. `calib-reversed-mask`
presents the cards in reverse, where ordinals and physical devices disagree.
All three produce the same pixels, which is the point: the sharing path and
the private path have to agree with the CPU and with each other.

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

`fail-select-device` aborted in 7.2s against a 90s budget, exit 1, `survived:
false`.

Two nodes, job 40889879: **PASS** for both EB topologies. One device group per
host, one owner each, and each follower importing 384 MiB:

    sdfampere034:0:01:00.0  peers 3  owners 1
    sdfampere038:0:01:00.0  peers 5  owners 1

Note the grouping key. Both nodes' cards sit at bus address `0:01:00.0`, so a
summary keyed on the PCI id alone merges the two groups and then reports the
merge as a cross-host group with two owners. Production groups on
`(hostname, pci)`; the driver's summary now does too.
    imported   {2: 0, 3: 384MiB, 4: 384MiB, 5: 0, 6..9: 384MiB}

That is the property only a multi-node run can show: CUDA IPC cannot cross
hosts, so the job must end up with one shared copy per node rather than one
for the job. The device groups are identical at `PS_EB_NODES` 1 and 2, which
is the point — EB topology does not reach device grouping.

## Not covered

MIG, which is unsupported and untested: no MIG hardware is advertised on S3DF,
and IPC cannot span instances in any case. Instances of one card share a PCI
bus id, so they group together and the IPC import fails, which aborts.
`PSANA_GPU_SHARE_CONSTANTS=0` gets the job running, but the peer count is still
inflated by the merged instances and every budget is divided by it, so it is
not a MIG configuration. Deferred per issue #155. See
[docs/device_placement_and_shared_constants.md](../../psana/psana/gpu/docs/device_placement_and_shared_constants.md).

No throughput claim. What is measured is device memory, not events/s.

Selecting a device under a wide mask was measured not to leave a stray CUDA
context on ranks that do not use the GPU (job 40151108), which is why the mask
is left as the launcher set it.
