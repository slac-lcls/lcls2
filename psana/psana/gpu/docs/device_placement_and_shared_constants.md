# Device placement and shared calibration constants

Implemented for issues #155 (discover actual GPU peers) and #168 (restore
CUDA-IPC sharing of requested constants). Describes the source, not the
development stages; the working notes and measurement history are in Git.

## Why both exist

**Peer counts must not come from EB-group arithmetic.** Deriving how many BD
workers share a GPU from `bd_comm` does not work, because `bd_comm` is split
per EB group.
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

## Placement: one phase

`gpu_placement.py`. `discover_peers()` selects this rank's device and groups
ranks that share a physical GPU, collective over `psana_comm`.

The device is made current with `Device.use()` under whatever mask the
launcher set. Narrowing `CUDA_VISIBLE_DEVICES` from Python has no effect:
`import psana` loads mpi4py, whose CUDA-aware MPI calls `cuInit` at library
load, and the variable is read once at driver initialisation — measured on
sdfampere002.

`use()` under a wide mask is sufficient. On sdfampere014 with 4 ranks over
2 GPUs, every rank's allocations landed on the device it selected and the
driver reported each PID resident on exactly one GPU.

### How many ranks, and which are BD workers

There is no variable that asks for BD workers. The count is a residual:

```
   bd = world_size - PS_SRV_NODES - 1 - PS_EB_NODES
```

| Knob | Default | What it sets |
| --- | --- | --- |
| MPI launch size (`srun -n`) | — | total ranks, and so the BD count |
| `PS_EB_NODES` | 1 | event-builder ranks |
| `PS_SRV_NODES` | 0 | SmallData writers, carved off the top of world size |

`PS_EB_NODE_LOCAL=1` overrides `PS_EB_NODES` to one EB per compute node. For
any `mfx*` experiment, `_force_mfx_overrides` may also cap it to the detected
node count; it logs the change on rank 0. `DataSource(auto_tune=True)` is
documented as tuning these but is stored and never read — it has no effect.

psana enforces one constraint (`mpi_ds.py`): `world_size > PS_EB_NODES + 1`,
so at least one BD worker exists. Nothing relates the rank count to the GPU
count; device grouping is derived from the driver afterwards.

Roles are assigned before any GPU code runs (`node.py`): world rank 0 is smd0,
then `color = bd_main_rank % PS_EB_NODES` picks each rank's EB group and
`bd_rank == 0` within a group becomes that group's EB.

### Worked example: 9 ranks, one node, 2 GPUs, `PS_EB_NODES=2`

```
world:   0      1     2     3    4    5    6    7    8
role:    smd0   eb0   eb1   bd   bd   bd   bd   bd   bd
```

Three splits, each over a communicator whose every member participates:

```
   psana_comm        0  1  2  3  4  5  6  7  8      every psana rank
        |  split by hostname
   node_comm         0  1  2  3  4  5  6  7  8      all on sdfampere034
        |  split by is_gpu_worker      non-GPU roles take MPI.UNDEFINED
   node_gpu_comm           3  4  5  6  7  8         smd0/eb0/eb1 -> COMM_NULL
                           0  1  2  3  4  5         <- renumbered, no gaps
        |  split by (hostname, device_pci)
   device_comm       [3,5,7] and [4,6,8]            rank 0 of each owns
```

The renumbering is the step that matters: `node_gpu_comm` numbers the six BD
workers 0-5 in world order, leaving no gaps for smd0 or the two EBs.

Every BD rank here sees the same two devices, so the least-loaded pass (see
[Assignment](#assignment)) alternates between them in `node_gpu_comm` order:

```
   world 3 -> gpu_rank 0 -> 0:01:00.0        world 6 -> gpu_rank 3 -> 0:41:00.0
   world 4 -> gpu_rank 1 -> 0:41:00.0        world 7 -> gpu_rank 4 -> 0:01:00.0
   world 5 -> gpu_rank 2 -> 0:01:00.0        world 8 -> gpu_rank 5 -> 0:41:00.0
```

Each rank then calls `Device.use()` on its own ordinal for that PCI id and
reads back `deviceGetPCIBusId()`.
Grouping uses **that** value, not the ordinal, so a group reflects what the
driver says the rank actually holds.

| Device | Peers (world) | Owner | Per-rank budget |
| --- | --- | --- | --- |
| `0:01:00.0` | 3, 5, 7 | world 3 | 12.00 GiB |
| `0:41:00.0` | 4, 6, 8 | world 4 | 12.00 GiB |

`usable = 40 GiB x 0.9 = 36 GiB`, divided by 3 real peers: **aggregate claim
1.00x per device**.

This layout is why the count cannot come from rank arithmetic. `bd_comm` is
split per EB group, so deriving peers from it would give eb0's BDs `[3,5,7]`
and eb1's `[4,6,8]` only their own three workers each: ~2 peers per device,
18 GiB claimed each, **2.0x** the card. Counting from an `allgather` of
driver-reported identity keeps EB topology out of it entirely.

### Worked example: 2 nodes, uneven groups

```bash
export PS_EB_NODES=2
export PS_SRV_NODES=1
srun -N 2 -n 13 --gres=gpu:a100:2 python my_analysis.py
```

```
   13 world - 1 srv = 12 psana ranks = 1 smd0 + 2 eb + 9 bd
   9 BD over 4 GPUs -> device groups of 3, 2, 2, 2
```

Groups are uneven and that is fine: each is sized from its own membership, so
each claims its own card exactly once. Confirm placement from the log line
every GPU worker emits — `peers=` is the figure that has to match reality:

```
gpu placement host=sdfampere034 pci=0:01:00.0 ordinal=0 visible=2 peers=3 \
    owner=True limit=12.00GiB sharing=on
```

Multi-node matters because the hostname half of the grouping key is constant on
one node, so a hostname that is ignored or inconsistently formatted is
invisible in single-node runs.

### Assignment

#### Masks: what each rank can see

A rank's *mask* is the `CUDA_VISIBLE_DEVICES` it was started with. It decides
which physical GPUs the process can see and the order CUDA numbers them in.
Under `CUDA_VISIBLE_DEVICES=2,3` a process sees two devices: ordinal 0 is
physical GPU 2, ordinal 1 is GPU 3, and GPUs 0 and 1 do not exist for it.
CUDA reads the mask once, at `cuInit`, which is why psana cannot change it.

Masks come from the launch line, not from psana:

| Source | Example | Masks the ranks get |
| --- | --- | --- |
| Slurm GPU allocation | `--gres=gpu:a100:4`, `--gpus=4` | every task on the node gets the same mask, e.g. `0,1,2,3` (the common psana case) |
| Slurm per-task binding | `--gpus-per-task=1`, `--gpu-bind=closest`, `--gpu-bind=map_gpu:...` | a different mask per task, e.g. `{A,B}` for ranks near socket 0 and `{C,D}` near socket 1 |
| User or wrapper script | `export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID` | whatever the script computes, possibly reordered, e.g. `2,3` on one rank and `3,2` on another |

With device cgroups, Slurm may also number the mask relative to the job's own
allocation, so mask numbers need not match `nvidia-smi` indices. A PCI bus id
names the same card in every case, which is why assignment works only on PCI
ids. To see what a launch line produces:

```bash
srun <options> bash -c \
  'echo rank=$SLURM_PROCID local=$SLURM_LOCALID mask=$CUDA_VISIBLE_DEVICES; \
   nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader'
```

psana runs under any of these, so assignment cannot assume every rank sees the
same devices, in the same order, or at all.

#### The algorithm

A deterministic least-loaded pass over every node worker's visible PCI ids,
gathered once (`gpu_placement.py`, in `discover_peers`):

```
   node_sets = node_gpu_comm.allgather(sorted(my_pcis))
   load = {}
   for peer, peer_pcis in enumerate(node_sets):
       if not peer_pcis:
           continue
       pick = min(peer_pcis, key=lambda p: (load.get(p, 0), p))
       load[pick] = load.get(pick, 0) + 1
       if peer == node_gpu_comm.Get_rank():
           target = pick
   ordinal = my_pcis.index(target)
   cp.cuda.Device(ordinal).use()
```

The `allgather` sends each BD worker's sorted PCI list to every BD worker on
the node. Before it, a rank knows only its own list; after it, every rank
holds the same list of lists, indexed by `node_gpu_comm` rank. With masks
exposing `{A,B}`, `{C,D}`, `{A,B}`, `{C,D}`:

```
   node_sets = [ [A,B],   # node GPU rank 0
                 [C,D],   # node GPU rank 1
                 [A,B],   # node GPU rank 2
                 [C,D] ]  # node GPU rank 3
```

Every rank then simulates the whole node's assignment and keeps only its own
row. `load` counts how many ranks have been placed on each device so far;
each peer takes the least-loaded device it can see, ties going to the lower
PCI id:

| Step | Peer | Visible | `load` before | Pick | `load` after |
| --- | --- | --- | --- | --- | --- |
| 1 | 0 | A,B | — | A (tie, A < B) | A1 |
| 2 | 1 | C,D | A1 | C (tie, C < D) | A1 C1 |
| 3 | 2 | A,B | A1 C1 | **B** (A already has 1) | A1 B1 C1 |
| 4 | 3 | C,D | A1 B1 C1 | **D** (C already has 1) | A1 B1 C1 D1 |

One rank per device. No further communication is needed: `allgather` gives
every rank identical input, and the loop visits peers in a fixed order with a
deterministic tie-break, so every rank builds the identical table.

The final step, `my_pcis.index(target)`, turns the chosen PCI id back into
*this* rank's ordinal, which can differ between ranks. A rank under mask `3,2`
has `my_pcis = [pci3, pci2]`; if its target is `pci3` it calls
`Device(0).use()`.

#### Why not a simple index

Two simpler rules were tried; each fails on a mask psana can be given.

**Indexing ordinals.** CUDA numbers visible devices in mask order, so the same
ordinal means different cards under different masks. With `rank % 2`:

| Rank | Mask | Ordinal 0 | Ordinal 1 | `rank % 2` | Lands on |
| --- | --- | --- | --- | --- | --- |
| 0 | `2,3` | GPU 2 | GPU 3 | 0 | **GPU 2** |
| 1 | `3,2` | GPU 3 | GPU 2 | 1 | **GPU 2** |

GPU 3 sits idle. Sorting by PCI id fixes this case, since both ranks then see
`[GPU 2, GPU 3]`.

**Indexing the rank's own sorted list.** Sorting is not enough when ranks see
*different* sets. With `{A,B}`, `{C,D}`, `{A,B}`, `{C,D}`, each rank decides
alone:

| Rank | Sorted own list | `rank % 2` | Lands on |
| --- | --- | --- | --- |
| 0 | [A,B] | 0 | A |
| 1 | [C,D] | 1 | D |
| 2 | [A,B] | 0 | A |
| 3 | [C,D] | 1 | D |

A and D carry two ranks each; B and C sit idle. Ranks 0 and 2 both pick A
because neither knows the other did. The `allgather` is what supplies that
knowledge: in the table above, by the time the loop reaches peer 2, `load`
already shows A taken, so peer 2 goes to B.

Budgets stay correct under either simple rule, because grouping uses the PCI
id actually in use. The cost is throughput and per-rank memory: half the
devices do double duty.

#### Communicator rank, not launcher rank

The index is the rank in `node_gpu_comm`, which contains only BD workers and
numbers them without gaps. A launcher node-local rank such as `SLURM_LOCALID`
counts smd0 and EB ranks too:

| `SLURM_LOCALID` | Role | `SLURM_LOCALID % 2` | `node_gpu_comm` rank |
| --- | --- | --- | --- |
| 0 | smd0 | — | — |
| 1 | EB | — | — |
| 2 | BD | 0 | 0 |
| 3 | EB | — | — |
| 4 | BD | 0 | 1 |

On 2 GPUs the launcher rank puts both BD workers on device 0; the
communicator rank spreads them.

#### Error paths

A rank whose selection already failed (empty mask, `getDeviceCount` raising,
PCI ids unreadable) still calls the `allgather`, contributing an empty list.
Every GPU worker therefore issues the same collectives in the same order; the
loop skips empty entries, and the error reaches every rank through the shared
`errors` path after the next reduction, so the job aborts instead of hanging.

#### Limit: greedy in rank order

The pass is balanced for identical masks, reordered masks and disjoint sets
such as `{A,B}`/`{C,D}`. With partly overlapping masks it can still be
uneven: with `{A,B}` on rank 0 and `{A}` on rank 1, rank 0 takes A on the
tie-break and rank 1 can only use A, leaving B idle. Visiting the most
constrained ranks first, or a bipartite matching, would avoid it. Budgets
remain correct; only balance suffers, and such masks are unusual.

### Identity: the PCI bus id from the driver

The grouping key is `deviceGetPCIBusId()`, normalised so nvidia-smi's
eight-digit domain and CUDA's four-digit one compare equal.

The UUID is **never** read from CUDA. `getDeviceProperties()['uuid']` is a
16-byte array that CuPy surfaces truncated at the first NUL, so a UUID
containing a zero byte yields a short, wrong value — measured on sdfampere014,
where `c9ef90546100fd2e...` arrived as five bytes.

MIG instances of one card share a bus id, so they group together here as peers
although their memory is separate; sharing across them then fails at the IPC
import, which aborts the job. MIG is unsupported and no environment variable
makes it correct — see [Limits](#limits).

### Non-GPU roles participate

smd0, EB and srv ranks must call every split and reduction a GPU worker
reaches, in the same order. They contribute neutral values and receive
`COMM_NULL`. Skipping a call does not raise — it hangs the others, and a
CPU-only rank that returned early was measured racing a peer's `MPI_ABORT`
rather than participating in the failure.

### The collectives discovery makes

What matters is the sequence on `psana_comm`, because that is the
communicator every role holds and MPI pairs calls by order rather than by
operation. It is identical for both roles:

| On `psana_comm` | Both roles |
| --- | --- |
| `allgather` | hostnames, to colour the node split |
| `Split` | node communicator |
| `allreduce` SUM | `discovery/job-agreement` — a selection failure anywhere |
| `allreduce` MAX | `discovery/budget-verdict` — an explicit budget over-commits some device |

Every role also splits the node communicator, which is not a `psana_comm`
call and so pairs only with itself:

| On `node_comm` | Both roles |
| --- | --- |
| `Split` | GPU-worker communicator; non-GPU roles pass `MPI.UNDEFINED` and get `COMM_NULL` |
| `Free` | released immediately, in a `finally` |

GPU workers additionally make calls on communicators the CPU roles do not
hold, which therefore cannot pair with anything of theirs:

| Communicator | Call |
| --- | --- |
| `node_gpu_comm` | `allgather` of visible PCI sets, for the assignment |
| `node_gpu_comm` | `allreduce` SUM, `discovery/node-agreement` |
| `node_gpu_comm` | `allgather` of `(host, pci)` keys, then `Split` |
| `device_comm` | `allreduce` MIN, `discovery/sharing-permitted` |
| `device_comm` | `allreduce` MIN, `discovery/budget` — free memory on *this* card |

The last one agrees `PSANA_GPU_SHARE_CONSTANTS` across the device group. It
is read from each rank's own environment, so one rank acting on its own
reading would skip the establishment `allgather` its peers enter — an
unbounded call, so the job would hang with no diagnostic. `MIN` means one
rank's opt-out disables sharing for the whole group. The agreed answer is
`GpuPlacement.sharing_permitted`, which `can_share` folds in; callers should
use that rather than reading the variable themselves.

There is a measured precedent for peers disagreeing about an environment
variable: the multi-node harness set `CUDA_VISIBLE_DEVICES` without
`mpirun -x`, and the two nodes' ranks ended up with different masks.

### Budgets

```
   per_rank_limit = (usable_bytes - shared_bytes) / n_device_peers
```

`usable_bytes` is 90% of *free* device memory, reduced with `MPI.MIN` over
`device_comm`. `cudaMemGetInfo` is not time-invariant -- peers sample it at
slightly different moments while each other's CUDA contexts are still settling
-- so limits derived independently sum to slightly more than the device. The
minimum gives every peer on one card one figure. (The Jungfrau shared caches do
not enter this: `MPISharedMemory` is host memory, not device.)

Over the **device group**, not the job. Capacity is a property of a card and is
spent by that card's peers alone, so a job-wide minimum charged every rank for
the busiest card in the job: 4 peers on an idle 80 GiB card alongside a loaded
40 GiB card were sized at 10 GiB each rather than 20.

`shared_bytes` is subtracted once, as device overhead, not per rank. The owner's
limit adds it back, because `_OwnedBlock` already charges it to the owner's
budget and subtracting it from the owner's share too would charge it twice.

Both adjustments are applied **before** the shared copy is allocated, through
a `sizing` callback that `SharedRequestedConstants` invokes once the group has
agreed to share. The ordering matters: a 12 GiB set on a 40 GiB four-peer
device fits this accounting (owner 7 + 12 = 19 GiB) but not the unadjusted
`usable / peers` of 10 GiB, so allocating first would refuse the shared copy.
The callback is not invoked at all when the group shares nothing: the budget is
already `usable / peers`, which is correct for a private copy per rank.

An explicit `gpu_memory_budget_gb` is validated against the group total,
since `N` ranks each claiming the whole device must not pass silently. The
**claim is per device and the verdict is job-wide**, which are two separate
steps:

1. each device's peers check `budget x n_device_peers` against **their own**
   `usable_bytes` — both figures reduced over the same `device_comm`;
2. the outcome is reduced with `MPI.MAX` over `psana_comm`
   (`discovery/budget-verdict`), so a budget that over-commits any card stops
   every rank.

Validating per device matters because capacity and peer count can come from
different cards. With an explicit 9 GiB per worker, a 40 GiB card holding 4
workers claims 36 GiB and a 20 GiB card holding 1 claims 9 — both fit. A
job-wide capacity `MIN` paired with a job-wide peer `MAX` evaluates `9 x 4 > 20`
and aborts a placement no device objects to. Combining the two reductions
across devices is the bug; combining the *verdict* is required, because a
device that carried on while another aborted would leave ranks on different
collective paths, racing the abort rather than agreeing on it.

Non-GPU roles hold `psana_comm` but not `device_comm`, so they cannot evaluate
any device's claim. They join the verdict `MAX` with a neutral 0 and raise with
the workers, instead of continuing while their peers abort. The two messages
differ: a rank on the offending device reports the figures to fix, while a rank
on a device that fits says so, since its own numbers add up and reporting them
as the cause would read as a contradiction.

## Shared constants

`gpu_shared_constants.py`. `SharedRequestedConstants` keeps
`RequestedConstants`' surface — `get`, `refresh`, `close` — so
`batch.calibconst()` and the task contract are unchanged.

### All or nothing

Every peer must declare the same selector set. Peers on one device run the same
`GpuTask`, so they normally do; a group that disagrees shares nothing and each
rank uploads its own copy. The verdict comes from one `allgather` that every
rank makes, so all of them reach it independently and none is surprised by a
manifest entry it did not request.

| Disagreement | Outcome |
| --- | --- |
| Selector **sets** differ | No sharing: a private copy per rank |
| **Shape or dtype** differs | Abort on every rank |
| **Content** differs | Abort on every rank |
| `ipcGetMemHandle` / `ipcOpenMemHandle` fails | Abort on every rank |

Content disagreement is fatal because sharing there would hand a follower the
owner's array under the follower's own name — wrong results with no error.

Sharing the intersection of unequal sets was supported earlier and removed: it
cost four code paths and their tests for a case that does not arise in
practice, since peers on one device run one task. Set difference is still
benign — it costs memory, not correctness — so it disables sharing rather than
aborting.

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
                       content    -> raise on every rank
                       capability -> release, then raise on every rank
```

Nothing between the broadcast and the agreement may raise. An exception that
skips a collective leaves peers blocked — and because MPI pairs collectives by
call order rather than by operation, a rank reaching a different one gives
undefined results before hanging.

`_release_shared()` holds the one ordering CUDA requires: importers close, a
barrier, then the owner frees. Both callers — `close` and the capability abort
in `_settle` — go through it, so neither can get the order wrong, and the abort
releases through barriers every peer reaches rather than raising on whichever
rank noticed first. **`close()` is collective** — every peer must call it; a
second call performs no collective.

### One upload per run

One calibration-constant set per run is the supported model, and BeginStep does
not fetch the calibration DB, so `refresh` establishes the shared copies on
its first call and returns `False` on every later one. No collective, no scan,
no reupload.

That is a correctness property, not only an efficiency one. Replacing a shared
allocation mid-run means freeing memory that backs every peer on the device,
and no rank can know when another has stopped reading it — the agreement
protocol can make peers decide *together*, but it cannot tell them *when it is
safe*. Not replacing at all removes the question.

A group that shares nothing uploads privately in that same first call, by the
same establishment path, and a later call returns before reaching it — under
one constant set per run there is nothing new to upload either way.

### Accounting

```
   owner    : budget.reserve(shared + its own private bytes)
   follower : charges nothing; records imported_bytes
```

A follower's views are non-owning, so charging them would shrink its real
budget by memory it does not own — the opposite of what sharing achieves. The
bytes are recorded separately so a memory report stays complete.

Measured at 4 peers with a 12 MiB constant set: owner charged 12 MiB, three
followers charged 0 while each importing 12 MiB.

### Lifetime: valid for the whole run

**A shared constant array stays valid until the run is torn down.** The
allocation is written once at setup and never rewritten or replaced while the
run is live, so a reference a callback keeps across events and steps is safe
and keeps its values.

Two consequences worth stating, because a user can only discover them by
being surprised:

- **Host mutation does not propagate.** `refresh()` runs once, at run setup.
  Changing `dsparms.calibconst` after that does not update the device copy —
  BeginStep drains and applies the host transition but does not fetch the
  calibration DB, so there is nothing for a rescan to find. One constant set
  per run is the supported model; dynamic mid-run updates are deferred (#168).
- **Read-only is a contract, not enforcement.** CuPy arrays carry no
  NumPy-style WRITEABLE flag and psana cannot intercept native writes, so
  nothing stops a kernel writing into a constant. Doing so corrupts the
  snapshot for every later batch on this rank *and* for every peer rank that
  imported it over IPC — one writer, n corrupted readers, with no diagnostic.
  That blast radius is why the contract is stated rather than left implicit.

It is **not** valid after the run. Teardown frees the owner's allocation and
closes the importers' mappings, so a reference held past that point dangles.
Copy with `.get()` to outlive the run.

Python offers no construct that prevents a caller retaining a reference — any
returned object can be assigned, and a read-only view, a context manager or a
weakref each control something else. The alternative would be to hand out a
revocable wrapper rather than a CuPy array, which would break every kernel
that passes constants straight to `RawKernel`. So the contract is that
retention during the run is *safe*, which is what one upload per run buys,
rather than that it is forbidden.

### When sharing does not happen

Conditions known before any IPC call disable sharing silently; an IPC call that
fails aborts.

| Condition | Behaviour |
| --- | --- |
| One peer on the device, or no communicator | Private copies |
| Peers declared different selector sets | Private copies, logged at WARNING with the memory cost |
| `PSANA_GPU_SHARE_CONSTANTS=0` on **any** peer | Private copies on the whole device group — the variable is reduced with `MIN` into `sharing_permitted`, so one opt-out disables sharing for every peer sharing that card |
| `ipcGetMemHandle` / `ipcOpenMemHandle` fails | `SharedConstantsError` on every rank |
| Shape, dtype or digest mismatch | `SharedConstantsError` on every rank |
| Explicit budget x peers exceeds the device | `GpuPlacementError` |

`placement.describe()` carries `sharing=on|off`.

An IPC failure used to degrade the whole group to private copies. It now
aborts, for two reasons: degrading hid a misconfiguration behind an n-times
memory cost that only `shared_bytes` dropping to zero revealed, and its budget
reset — the limits are set for one device copy, so every rank holding a full
copy needs them returned — was the source of two separate defects.

The abort names `PSANA_GPU_SHARE_CONSTANTS=0`, which is the deliberate way to
run where IPC cannot work, at n times the device memory for constants.

**It is not a MIG mode.** Instances of one card share a PCI bus id, so
discovery groups ranks on different instances as peers and the import
necessarily fails; the opt-out gets past that failure but leaves the peer count
and every rank's budget wrong, because the grouping it is working around is
still in effect. MIG is unsupported and deferred — see [Limits](#limits) and
issue #155.

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
so ranks arrive whenever their last batch finished. Minutes of legitimate
skew must cost a log line, not the job. Discovery has no such skew, so it
aborts sooner.

The abort targets `MPI.COMM_WORLD`: the standard only promises a best effort on
the given communicator's group, and the communicator here is often
`device_comm`. Tests intercept it through `_abort_hook`.

Not bounded — two calls carry Python objects, which mpi4py can only send
non-blocking through a two-step size-then-data `Ibcast`: the manifest `bcast`,
and `_agree_on_set`'s `allgather`.

The test recorder (`psana/psana/tests/gpu/collective_recorder.py`) wraps a
communicator and compares each rank's call sequence at the end of establish and
close. Path divergence is what every hang in this work turned out to be, and
nothing else detects it. It lives in the test tree rather than in
`gpu_collectives.py`: it is a debugging aid, and keeping it there means
production holds no communicator wrapper and reads no environment variable to
decide whether to use one. `SharedRequestedConstants` calls `check_agreement`
on its communicator when the method is present, so wrapping is all a driver has
to do.

## Limits

- **MIG is unsupported, and there is no flag that makes it work.** No MIG
  hardware is advertised on S3DF. Grouping keys on the PCI bus id, which
  instances of one card share, so ranks on different instances look like
  peers. Sharing is then attempted, the IPC import fails — CUDA IPC cannot
  span instances, which have separate memory — and the job aborts.

  `PSANA_GPU_SHARE_CONSTANTS=0` gets past that abort and the job runs, but it
  does **not** correct the peer count or the budget: two single-worker
  instances with 9 GiB usable each should budget 9 GiB apiece, and being
  counted as two peers gives them 4.5 GiB. So it is not a MIG configuration,
  supported or otherwise. Treat MIG as unsupported; the work is deferred per
  issue #155, which records it as follow-up before any MIG deployment rather
  than a blocker for the current non-MIG scope.

  Separating instances needs an identity neither CUDA nor plain NVML
  enumeration provides: `getDeviceProperties()['uuid']` is truncated at the
  first NUL, and `nvmlDeviceGetUUID` / `nvidia-smi --query-gpu=uuid` list
  parent GPUs only. `nvmlDeviceGetMigDeviceHandleByIndex` would, and querying
  NVML from inside discovery is cheap now that CUDA is already initialised —
  the reason it is not done is that there is no MIG node to test it on. The
  grouping comment in `discover_peers` carries the same note.
- **No cross-node sharing.** IPC is single-node, so a 2-node job gets one
  shared copy per node. Grouping across nodes is tested; sharing across them is
  impossible.
- **MPI only.** `RunSerial` has no peers, so sharing does nothing there.
- **The automatic budget takes the minimum over each device group**, so the
  busiest peer on a card sets the figure for its peers. A mix of 40 GB and
  80 GB cards no longer drags the roomier card down, but one rank holding a
  stale pool still lowers its own card's peers.
- **No throughput claim.** The benefit measured is device memory, not
  events/s; whether reclaimed VRAM converts into throughput depends on the
  configuration being memory-bound.

## Source map

| Area | Implementation |
| --- | --- |
| Device choice and peer discovery | [gpu_placement.py](../gpu_placement.py) |
| Shared constants | [gpu_shared_constants.py](../gpu_shared_constants.py) |
| Bounded collectives | [gpu_collectives.py](../gpu_collectives.py) |
| Collective-order recorder (tests) | [collective_recorder.py](../../tests/gpu/collective_recorder.py) |
| Budget | [gpu_budget.py](../gpu_budget.py) |
| Manager wiring | [gpu_events.py](../gpu_events.py) |
| Discovery call site | [mpi_ds.py](../../psexp/mpi_ds.py) |

## Validation

Unit tests need no GPU; the communicator and CuPy are both injected, so
device selection and grouping are exercised without a device.

```bash
python -m pytest psana/psana/tests/gpu/unit/test_gpu_placement.py \
                 psana/psana/tests/gpu/unit/test_gpu_shared_constants.py \
                 psana/psana/tests/gpu/unit/test_gpu_placement_wiring.py \
                 psana/psana/tests/gpu/unit/test_gpu_collectives.py

sbatch validation/ipc-integration-20261007/run.sbatch    # 13 cases, 1 node
sbatch validation/ipc-multinode-20261007/run.sbatch      # 4 cases, 2 nodes
```

Integration coverage, single node: five EB topologies (1, 2 and 3 groups, even
and uneven), sharing at 2 and 4 peers, two cases through the real
`DataSource` -> `Run` -> `GpuEventManager` wiring, one running the real
Jungfrau calibration while the constants are shared, one reproducing the
pre-existing multi-run hang, and two failure injections — a rank whose
`Device.use()` fails, which must abort the job rather than hang, and a
follower whose IPC import fails, which must abort every peer rather than
only the one that failed.

Across two nodes: the same discovery checks plus the event loop with
`batch.calibconst()` read inside a real callback, at `PS_EB_NODES` 1 and 2.
This is where the single-node suite cannot reach — CUDA IPC does not cross
hosts, so the job must produce one device group and one shared copy *per
node*, with the groups identical under both EB topologies.

Measured after the all-or-nothing and abort-on-IPC-failure changes: 632 unit
tests; 13/13 single-node cases (jobs 40888078 and 40889090) and all four
multi-node cases (job 40889879), with zero collective-order divergence.

Across two nodes, both EB topologies gave one device group and one owner per
host, each follower importing the full 384 MiB:

    sdfampere034:0:01:00.0  peers 3  owners 1
    sdfampere038:0:01:00.0  peers 5  owners 1

`multirun-3-runs` is expected to hang and does: it reaches run 16 after
processing run 15, which is the known pre-existing failure recorded in
[multirun_hang.md](multirun_hang.md).
