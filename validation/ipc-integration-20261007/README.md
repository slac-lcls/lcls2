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

Twelve cases. Five discovery topologies, because the defect being fixed was
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
| `calib-2-gpus-eb2` | 9 | the same across 2 devices at `PS_EB_NODES=2`, so most groups are solo |
| `calib-reversed-mask` | 6 | the same under `CUDA_VISIBLE_DEVICES=1,0`, where ordinals and physical cards disagree |
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

`multirun-3-runs` is `expect='hang-known'`: psana2 hangs at the second run of
any multi-run MPI job, which is pre-existing on the base branch. The case
passes while the bug is present and **fails if the job completes**, so the
suite says when to promote it. See
[docs/multirun_hang.md](../../psana/psana/gpu/docs/multirun_hang.md).

## Results

Single node, jobs 40888078 and 40889090 on an A100 pair: **13/13 PASS**,
`collective_order_diverged: 0` in every case.

Discovery: two devices, `aggregate_claim` 1.0 on each, with `discovery-uneven`
giving groups of 3 and 4. Every record also reports what EB-group rank
arithmetic would have produced, which reaches 2.0x the device at
`PS_EB_NODES=2`.

The value check, 13 events per case, every pixel equal to `det.raw.calib()`:

    calib-shared      4 ranks, 1 GPU   13 events   191s   work split 8/5
    calib-2-gpus-eb2  9 ranks, 2 GPUs  13 events   199s   work split 5/3/5

`calib-shared` narrows the mask to one card, so all four BD workers are peers
and the pedestals and pixel_gain -- 384 MiB for this detector -- come from a
single device allocation over IPC. `calib-2-gpus-eb2` leaves the mask wide, so
the workers split across cards and most groups are solo. Both produce the same
pixels, which is the point: the sharing path and the private path have to
agree with the CPU and with each other.

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

MIG, which is untested: no MIG hardware is advertised on S3DF, and IPC cannot
span instances in any case. Instances of one card share a PCI bus id, so they
group together and the IPC import fails, which now aborts with a message
naming `PSANA_GPU_SHARE_CONSTANTS=0`. With sharing disabled the job runs,
though the peer count is still inflated by the merged instances, so a large
constant set can exhaust the per-rank limit. See
[docs/device_placement_and_shared_constants.md](../../psana/psana/gpu/docs/device_placement_and_shared_constants.md).

No throughput claim. What is measured is device memory, not events/s.

Selecting a device under a wide mask was measured not to leave a stray CUDA
context on ranks that do not use the GPU (job 40151108), which is why the mask
is left as the launcher set it.
