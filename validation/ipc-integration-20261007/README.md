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

    sbatch validation/ipc-integration-20261007/run.sbatch    # 9 cases, 1 node, 2 GPUs
    sbatch validation/ipc-multinode-20261007/run.sbatch      # 2 nodes

## Cases

Five discovery topologies, because the defect being fixed was peer counts
derived from EB-group arithmetic: `bd_comm` is split per EB group, so the count
was wrong whenever groups did not map onto devices. `discovery-uneven` is the
case where ranks on one device previously computed *different* counts.

| Case | Ranks | Asserts |
| --- | --- | --- |
| `discovery-eb1` | 9 | 2 devices, 4 peers each, aggregate claim 1.0 |
| `discovery-eb2` | 9 | same, with the EB split that caused the 2.0x over-commit |
| `discovery-eb2-contig` | 9 | same, contiguous rank assignment |
| `discovery-eb3` | 9 | same, 3 EB groups (the 3.0x case) |
| `discovery-uneven` | 8 | 3 and 4 peers, both claim 1.0 |
| `sharing-2-peers` | 3 | owner charged, follower imports, values read back |
| `sharing-4-peers` | 5 | as above at 4 peers |
| `fail-verify-pin` | 5 | a rank on the wrong device aborts the **job** |
| `fail-follower-import` | 4 | a failed IPC import degrades the group **uniformly** |

`aggregate_claim` is the sum of per-rank budgets over device capacity. It must
be <= 1.0: above 1.0 the ranks on that device have collectively claimed more
memory than exists. Each discovery record also reports `current_believed_peers`
and `current_aggregate` from the old arithmetic, so the log shows what was
fixed rather than only that it passes.

The two failure cases are what unit tests structurally cannot cover: `FakeComm`
drives peers sequentially, so it has no notion of two ranks blocked in
*different* collectives.

`fail-verify-pin` is the only test of an abort reaching EB and smd0 ranks. The
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

Job 40264688, 2026-10-08. **9/9 PASS**, `collective_order_
diverged: 0` in every case.

Discovery: 2 devices x 4 peers (3+4 for `discovery-uneven`), `aggregate_claim`
1.0 throughout, against `current_aggregate` up to 2.0x under the old
arithmetic.

Sharing at 4 peers, 12 MiB intersection plus a 3 MiB owner-private selector:

    charged   {owner: 15728640, 2: 0, 3: 0, 4: 0}
    imported  {owner: 0, 2: 12582912, 3: 12582912, 4: 12582912}

The owner carries the shared bytes plus its private selector; followers are
charged nothing, because their views are non-owning and charging them would
shrink their real budget by memory they do not own.

`fail-verify-pin` aborted in 7.2s against a 90s budget, exit 1, `survived:
false`. `fail-follower-import` degraded uniformly and values stayed correct
through the private path.

Multi-node: job 40264689, 2 nodes. PASS. Four device
groups, none spanning a host, `aggregate_claim` 1.0 on each, against
`current_believed_peers: [2, 3]` — peers disagreeing — and `current_aggregate:
1.1667` under the old arithmetic. This case matters because the hostname half
of the grouping key is constant on one node, so a hostname that is ignored or
inconsistently formatted is invisible in single-node runs.

## Not covered

MIG, which is neither supported nor tested: no MIG hardware is advertised on
S3DF, and IPC cannot span instances in any case. With every device visible
`select_device` refuses rather than guessing between instances that share a PCI
bus id; with the mask narrowed to an instance UUID the rank ends up unpinned and
warns, because NVML enumerates parent GPUs only. See
[docs/device_placement_and_shared_constants.md](../../psana/psana/gpu/docs/device_placement_and_shared_constants.md).

No throughput claim. What is measured is device memory, not events/s.

`probe_late_leak.py` is a standalone check that the late discovery path leaves
no CUDA context behind on ranks that do not use the GPU.
