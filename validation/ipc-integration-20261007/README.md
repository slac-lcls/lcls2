# Device peer discovery and CUDA-IPC constant sharing

Integration evidence for `features/psana2-ipc` (issues #155, #168) on real
A100s with real CUDA IPC. Unit tests inject fakes for the identity table, the
launcher environment, the communicator and CuPy; these cases use none of that.

Cases are driven by `run_cases.py`, one MPI job each with a private PRRTE
session, because a shell loop of back-to-back `mpirun` calls hung: consecutive
launches share session state and an aborting case can block the next. Each case
is hard-bounded by a timeout and its outcome parsed from its log, since `mpirun`
exits 0 even when ranks report failures.

`PSANA_GPU_CHECK_COLLECTIVES=1` is set throughout. Every hang in this work was
a collective-path divergence, and nothing else detects one; `collective_order_
diverged` must be 0 in every case.

Run from the repository root after building:

    sbatch validation/ipc-integration-20261007/run.sbatch    # 12 cases, 1 node, 2 GPUs
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
| `sharing-2-peers` | 3 | owner charged, follower imports, values read back |
| `sharing-4-peers` | 5 | as above at 4 peers |
| `realpath-2-gpus` | 5 | placement through the real DataSource -> Run -> manager wiring |
| `realpath-1-gpu` | 6 | the same with every worker on one device, so all are peers |
| `shared-calibration` | 4 | real Jungfrau calibration while the constants are shared |
| `fail-select-device` | 5 | a rank that cannot select a device aborts the **job** |
| `fail-follower-import` | 4 | a failed IPC import degrades the group **uniformly** |

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

The `realpath-*` cases go through `DataSource` -> `Run` ->
`GpuEventManager` and touch none of the placement helpers, so they cover
whether a correct selection actually reaches a running job. Their decisive
assertion is cross-rank: a wrongly placed rank is still self-consistent --
it reports a device, allocates on it, and its current device matches -- so
only comparing the devices used against the number CUDA offers catches
workers crowded onto one of several GPUs.

`shared-calibration` runs the real Jungfrau kernel while the constants are
shared, with every panel compared to a CPU reference. A follower calibrates
from the owner's imported mapping, so a shared view with a wrong offset or
truncated length shows up as mismatched pixels rather than as nothing at all.

`fail-select-device` is the only test of an abort reaching EB and smd0 ranks. The
`allreduce` inside `discover_peers` covers only GPU workers on the node, so
other roles can be reached only by `gpu_error_handler` aborting the
communicator. Timing is part of the contract: a correct abort and a regressed
deadlock produce near-identical logs, so `judge_abort` requires a nonzero exit
*and* completion well inside the timeout. Without the timing clause a hang
would pass.

`fail-follower-import` requires the degrade to be uniform. If the owner falls
back to private copies and a follower does not, the follower reads memory the
owner has freed — silent corruption rather than a hang, which is why the
assertion is on `allgather`ed shared-selector counts rather than on each rank
alone.

## Results

Single node, job 40800404 on an A100 pair: **12/12 PASS**,
`collective_order_diverged: 0` in every case.

Discovery: two devices, `aggregate_claim` 1.0 on each, with `discovery-uneven`
giving groups of 3 and 4. Every record also reports what EB-group rank
arithmetic would have produced, which reaches 2.0x the device at
`PS_EB_NODES=2`.

Sharing at 4 peers, 12 MiB intersection plus a 3 MiB owner-private selector:

    charged   {owner: 15728640, 2: 0, 3: 0, 4: 0}
    imported  {owner: 0, 2: 12582912, 3: 12582912, 4: 12582912}

The owner carries the shared bytes plus its private selector; followers are
charged nothing, because their views are non-owning and charging them would
shrink their real budget by memory they do not own.

`shared-calibration` ran the real Jungfrau kernel on 13 events with pedestals
and pixel_gain shared: **384 MiB shared, 384 MiB saved**, every panel equal to
the CPU reference, and a constant held across events still valid at the end.

`fail-select-device` aborted in 7.2s against a 90s budget, exit 1, `survived:
false`. `fail-follower-import` degraded uniformly and values stayed correct
through the private path.

Two nodes, job 40801805: **PASS** for both EB topologies. One device group per
host, one owner each, and each follower importing 384 MiB:

    0:01:00.0  sdfampere014  ranks [2,3,4]      owners 1
    0:41:00.0  sdfampere015  ranks [5,6,7,8,9]  owners 1
    imported   {2: 0, 3: 384MiB, 4: 384MiB, 5: 0, 6..9: 384MiB}

That is the property only a multi-node run can show: CUDA IPC cannot cross
hosts, so the job must end up with one shared copy per node rather than one
for the job. The device groups are identical at `PS_EB_NODES` 1 and 2, which
is the point — EB topology does not reach device grouping.

## Not covered

MIG, which is neither supported nor tested: no MIG hardware is advertised on
S3DF, and IPC cannot span instances in any case. With every device visible
instances of one card share a PCI bus id, so they group together and the IPC
import fails; the group degrades, but with a peer count inflated by the merged
instances, so a large constant set can be refused outright. See
[docs/device_placement_and_shared_constants.md](../../psana/psana/gpu/docs/device_placement_and_shared_constants.md).

No throughput claim. What is measured is device memory, not events/s.

Selecting a device under a wide mask was measured not to leave a stray CUDA
context on ranks that do not use the GPU (job 40151108), which is why the mask
is left as the launcher set it.
