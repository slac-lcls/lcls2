# Device placement and shared calibration constants

Implemented for issues #155 (discover actual GPU peers) and #168 (restore
CUDA-IPC sharing of requested constants). Describes the source, not the
development stages; the working notes and measurement history are in Git.

## Why both exist

**Peer counts came from EB-group arithmetic.** `bd_ranks_sharing_gpu()` derived
how many BD workers shared a GPU from `bd_comm`, which is split per EB group.
With `PS_EB_NODES=2`, eight ranks on one device each believed they had three
peers and claimed a quarter of the card: a measured **2.0x** over-commit, and
**3.0x** with three groups. In uneven layouts ranks on the same device computed
*different* counts, so their budgets did not even agree with each other.

**Every BD rank uploaded its own constants.** Four BDs requesting 1 GiB of
identical pedestals consumed ~4 GiB, subtracted from every rank's automatic
budget. CUDA-IPC sharing existed before `137e2902f` but was removed with the
built-in calibration it served.

Both are fixed by answering one question properly: *which ranks share this
physical device?*

## Placement: two phases

`gpu_placement.py`. The phases are separate because their prerequisites are
incompatible.

```
   pin_device()        choose a device from the launcher environment
        |              no communicator; no CUDA call
        v
   ... MPI starts, CuPy is imported ...
        |
   discover_peers()    make it current, verify it, group ranks by identity
                       collective over psana_comm
```

### Phase 1 — choose, do not mask

`pin_device()` records a choice and deliberately does **not** narrow
`CUDA_VISIBLE_DEVICES`. It cannot: `import psana` loads mpi4py, whose
CUDA-aware MPI calls `cuInit` at library load, and after that a mask write is
silently ignored. Measured: after `import psana` the driver already reports
CUDA initialised, so the branch that wrote the mask was unreachable in any real
job — and the previous `init_gpu_rank()` wrote it anyway, leaving every rank on
device 0 with only a warning.

Selecting instead of masking is sufficient. Measured with 4 ranks over 2 GPUs,
both devices visible throughout: every rank's allocations landed on the device
it selected, and the driver reported each PID resident on exactly one GPU.

The node-local rank comes from the launcher environment, because no
communicator exists yet:

```
   OMPI_COMM_WORLD_LOCAL_RANK   Open MPI / PRRTE
   PMIX_LOCAL_RANK              PMIx
   SLURM_LOCALID                srun
   MV2_COMM_WORLD_LOCAL_RANK    MVAPICH2
   PMI_LOCAL_RANK               MPICH / Intel MPI
```

The first one set wins. None set is a configuration error worth warning about,
not a silent default to 0 on every rank.

### Identity: PCI bus id, with the UUID as the grouping key

| Value | Source | Used for |
| --- | --- | --- |
| PCI bus id | NVML or nvidia-smi | Identifying the device; `select_device` and `verify_pin` match on it |
| UUID | NVML or nvidia-smi | Peer grouping, where it distinguishes MIG instances sharing a bus id |

The UUID is **never** read back from CUDA. `getDeviceProperties()['uuid']` is a
16-byte array that CuPy surfaces truncated at the first NUL, so a UUID
containing a zero byte yields a short, wrong value — measured on sdfampere014,
where `c9ef90546100fd2e...` arrived as five bytes and hashed to eight
characters.

A launcher may still narrow the mask (`srun --gpus-per-task=1`, or a wrapper
setting `CUDA_VISIBLE_DEVICES` from `OMPI_COMM_WORLD_LOCAL_RANK` before Python
starts). That is defence in depth and needs no separate code path: a mask of
one device is a permitted set of size one.

### Phase 2 — verify, then group

`discover_peers()` makes the chosen device current with `Device.use()`, then
verifies by PCI bus id. The visible-device count is recorded for the placement
log but is **not** a correctness check, since the mask is never narrowed.

Grouping is by driver-reported `(hostname, uuid)`, so peer counts are
independent of EB topology by construction — no rank arithmetic participates.
Communicators are built where every member of the parent participates:

```
   psana_comm        every psana rank (smd0, EB, BD, srv)
        |  split by hostname
   node_comm         every psana rank on this node
        |  split by is_gpu_worker      non-GPU roles take MPI.UNDEFINED
   node_gpu_comm     GPU-capable BD ranks on this node
        |  split by (hostname, uuid)
   device_comm       ranks sharing one physical device; rank 0 owns
```

Non-GPU roles must call every split and reduction a GPU worker reaches, in the
same order. They contribute neutral values and receive `COMM_NULL`. Skipping a
call does not raise — it hangs the others, and a CPU-only rank that returned
early was measured racing a peer's `MPI_ABORT` rather than participating in the
failure.

### Budgets

```
   per_rank_limit = (usable_bytes - shared_bytes) / n_device_peers
```

`usable_bytes` is 90% of *free* device memory, reduced with `MPI.MIN` over
`psana_comm`: the CUDA context and the Jungfrau shared caches are allocated
before discovery, and `cudaMemGetInfo` is not time-invariant, so peers sampling
it independently derive limits that sum to slightly more than the device.

`shared_bytes` is subtracted once, as device overhead, not per rank. The owner's
limit adds it back, because `_OwnedBlock` already charges it to the owner's
budget and subtracting it from the owner's share too would charge it twice.

Both adjustments are applied **before** the intersection is allocated, through
a `sizing` callback that `SharedRequestedConstants` invokes once the
intersection is known. Applying them afterwards meant the owner allocated
against `usable / peers`: a 12 GiB intersection on a 40 GiB four-peer device
fits this accounting (owner 7 + 12 = 19 GiB) but was charged against 10 GiB, so
the shared copy was refused, the group degraded, and the private copy was then
refused by the same limit.

An explicit `gpu_memory_budget_gb` is validated against the group total; `N`
ranks each claiming the whole device is otherwise accepted silently. The claim
is checked against the **busiest** device in the job, reduced with `MPI.MAX`,
not against this rank's own peer count: in an uneven 4+3 layout a budget that
over-commits the 4-peer device but fits the 3-peer one would otherwise make
one device raise while the other carried on, leaving ranks on different paths
racing the abort instead of agreeing on it. Rejecting job-wide is also the
right answer — if any device cannot honour the budget, the job cannot run with
it.

## Shared constants

`gpu_shared_constants.py`. `SharedRequestedConstants` keeps
`RequestedConstants`' surface — `get`, `refresh`, `close` — so
`batch.calibconst()` and the task contract are unchanged.

### Intersection, not all-or-nothing

Peers declaring different selector sets share what they have in common and
privately upload the remainder. The intersection is computed by `allgather`
rather than dictated by the owner, so every rank derives the same shared set
and none is surprised by a manifest entry it did not request.

| Disagreement | Outcome |
| --- | --- |
| Selector **sets** differ | Degrade: share the intersection, privately copy the rest |
| **Shape or dtype** differs for a shared selector | Abort on every rank |
| **Content** differs for a shared selector | Abort on every rank |

Content disagreement is fatal because sharing there would hand a follower the
owner's array under the follower's own name — wrong results with no error.
Set difference is benign: a selector outside the intersection is simply not
shared.

### Un-pooled owner allocation

`cudaIpcGetMemHandle` needs a `cudaMalloc` base pointer. CuPy's default pool
returns sub-blocks of a larger segment, and `gpu_allocation.py` mandates that
pool so allocation capacity is exactly predictable for admission.

Measured: exporting a handle for a pooled pointer **succeeds**. That is worse
than failing — the handle refers to the segment base, so for an array at a
nonzero offset the importer reads the wrong data and neighbouring pooled blocks
are exposed to peers. Shared constants are therefore allocated outside the pool
with `cp.cuda.runtime.malloc` and charged to the budget explicitly. They are a
handful of allocations per run, so bypassing the pool costs nothing.

### One protocol shape

Every path goes through `_exchange` and ends in `_settle`:

```
   _exchange(hosts)    publish (owner) or subscribe (follower)
                       NEVER raises; returns (content_errors, capability_error)
                       the owner ALWAYS broadcasts -- an empty manifest is the
                       failure marker, so followers do not wait in bcast
        |
   _settle(...)        one Allreduce(MAX) over [content, capability]
                       content  -> raise on every rank
                       capability -> degrade the whole group
```

Nothing between the broadcast and the agreement may raise. An exception that
skips a collective leaves peers blocked — and because MPI pairs collectives by
call order rather than by operation, a rank reaching a different one gives
undefined results before hanging.

`_release_shared()` holds the one ordering CUDA requires: importers close, a
barrier, then the owner frees. It is used by `close`, `_reestablish` and the
fallback, so a fourth caller cannot get it wrong. **`close()` is collective** —
every peer must call it; a second call performs no collective.

### BeginStep

The case is agreed, not dictated: each rank proposes one and the ordinals are
reduced with both `MIN` and `MAX`. Unequal means peers hold different values
for a shared selector, which is fatal. Agreeing on the case is *not* agreeing
on content — if the owner and a follower both change to different values, both
propose B — so case B also compares digests for the changed selectors.

| Case | Condition | Action |
| --- | --- | --- |
| A | Nothing changed | No upload, no allocation, no synchronisation |
| B | Values changed, layout identical | Owner writes in place after a barrier; imported views stay valid |
| C | Shape or dtype changed | Importers close, barrier, owner reallocates and re-exports |

Case C's close-before-free is mandatory: CUDA requires imported mappings closed
before the exporter deallocates.

### Accounting

```
   owner    : budget.reserve(shared + its own private bytes)
   follower : charges nothing; records imported_bytes
```

A follower's views are non-owning, so charging them would shrink its real
budget by memory it does not own — the opposite of what sharing achieves. The
bytes are recorded separately so a memory report stays complete.

Measured at 4 peers with a 12 MiB intersection and a 3 MiB owner-private
selector: owner charged 15 MiB, three followers charged 0 while each importing
12 MiB.

### When sharing does not happen

Capability failures degrade; agreement failures abort. A rank that cannot share
loses memory, a rank that shares the wrong array loses correctness.

| Condition | Behaviour |
| --- | --- |
| One peer on the device, or no communicator | Private copies |
| MIG instance | Private copies — IPC does not span MIG instances |
| `ipcGetMemHandle` / `ipcOpenMemHandle` fails | Whole group degrades to private copies |
| Empty intersection | Private copies, no error |
| Shape, dtype or digest mismatch on a shared selector | `SharedConstantsError` on every rank |
| Explicit budget x peers exceeds the device | `GpuPlacementError` |

A fallback is reported at WARNING with the reason and the memory it now costs,
and `placement.describe()` carries `sharing=on|off|fallback(<reason>)`. Without
that the only trace is `shared_bytes` dropping to zero while the device quietly
holds n copies.

## Bounded collectives

`gpu_collectives.py`. Setup collectives poll a non-blocking request rather than
blocking forever, because a hung job holds its nodes until the wall clock
expires with no diagnostic.

| | Default | Override |
| --- | --- | --- |
| Warn | 60 s | `PSANA_GPU_COLLECTIVE_WARN` |
| Abort | 1800 s | `PSANA_GPU_COLLECTIVE_TIMEOUT` (`0` disables) |
| Discovery steps | 120 s abort | per-call `timeout=` |

Two tiers because a deadline only separates *hung* from *slow* where ranks
arrive together. `release-shared-closed` is the first collective in teardown,
so ranks arrive whenever their last batch finished; `case-B-drained` waits on
each rank's file I/O. Minutes of legitimate skew must cost a log line, not the
job. Discovery has no such skew, so it aborts sooner.

The abort targets `MPI.COMM_WORLD`: the standard only promises a best effort on
the given communicator's group, and the communicator here is often
`device_comm`. Tests intercept it through `_abort_hook`.

Not bounded — three calls carry Python objects, which mpi4py can only send
non-blocking through a two-step size-then-data `Ibcast`: the manifest `bcast`,
`_intersect`'s `allgather`, and the case B digest `bcast`.

`PSANA_GPU_CHECK_COLLECTIVES=1` wraps `device_comm` in `RecordingComm` and
compares each rank's call sequence at the end of establish, each refresh path
and close. Path divergence is what every hang in this work turned out to be,
and nothing else detects it. Enabled in the validation harnesses.

## Limits

- **MIG: not supported, untested.** No MIG hardware is advertised on S3DF. Two
  distinct behaviours, neither validated:

  With every device visible, `select_device` refuses rather than guessing,
  because MIG instances of one card share a PCI bus id and taking the first
  match would hand this rank another rank's memory.

  With the mask narrowed to a MIG instance UUID, that UUID is **not** in the
  node device map — NVML and `nvidia-smi --query-gpu` enumerate parent GPUs
  only — so the rank is unpinned, `is_mig` stays false, and grouping falls back
  to the PCI bus id that instances share. Ranks on different instances then
  look like peers: sharing is attempted, fails when IPC cannot span instances,
  and degrades to private copies, while the budget is divided by too many
  peers. Safe, but wasteful; `pin_device` warns.
- **No cross-node sharing.** IPC is single-node, so a 2-node job gets one
  shared copy per node. Grouping across nodes is tested; sharing across them is
  impossible.
- **MPI only.** `RunSerial` has no peers, so sharing does nothing there.
- **The automatic budget takes the job-wide minimum**, so one busy GPU, or a
  mix of 40 GB and 80 GB cards, lowers every rank's limit. Correct for
  validating an explicit budget, pessimistic for automatic sizing.
- **No throughput claim.** The benefit measured is device memory, not
  events/s; whether reclaimed VRAM converts into throughput depends on the
  configuration being memory-bound.

## Source map

| Area | Implementation |
| --- | --- |
| Device choice and peer discovery | [gpu_placement.py](../gpu_placement.py) |
| Shared constants | [gpu_shared_constants.py](../gpu_shared_constants.py) |
| Bounded collectives, order checker | [gpu_collectives.py](../gpu_collectives.py) |
| Budget | [gpu_budget.py](../gpu_budget.py) |
| Manager wiring | [gpu_events.py](../gpu_events.py) |
| Discovery call site | [mpi_ds.py](../../psexp/mpi_ds.py) |

## Validation

Unit tests need no GPU; the identity table, launcher environment, communicator
and CuPy are all injected.

```bash
python -m pytest psana/psana/tests/gpu/unit/test_gpu_placement.py \
                 psana/psana/tests/gpu/unit/test_gpu_shared_constants.py \
                 psana/psana/tests/gpu/unit/test_gpu_placement_wiring.py \
                 psana/psana/tests/gpu/unit/test_gpu_collectives.py

sbatch validation/ipc-integration-20261007/run.sbatch    # 9 cases, 1 node
sbatch validation/ipc-multinode-20261007/run.sbatch      # 2 nodes
```

Integration coverage: five EB topologies (1, 2 and 3 groups, even and uneven),
sharing at 2 and 4 peers, and two failure injections — a rank given a device it
did not select, which must abort the whole job promptly rather than hang, and a
follower whose IPC import fails, which must degrade the group uniformly.

Multi-node matters because the hostname half of the grouping key is constant on
one node, so a hostname that is ignored or inconsistently formatted is
invisible there. Measured across two nodes: four device groups, none spanning a
host, aggregate claim 1.0 on each.
