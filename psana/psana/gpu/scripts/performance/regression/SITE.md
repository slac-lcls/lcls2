# Allocation and placement preparation

`prepare_site.py` runs **directly in a single-node sbatch script**, outside any
srun step. It creates a fresh `perlmutter-raw-site-v1` manifest for that allocation.
It never submits a job or enables campaign qualification flags. A site manifest
expires when its allocation ends; repeat this preparation in each campaign job.

Use a reviewed, captured isolated runtime from [RUNTIME.md](RUNTIME.md). Freeze
these six source files together: `prepare_site.py`, `site_probe.py`,
`perlmutter_adapter.py`, `native_worker.py`, `native_contract.py`,
`prepare_runtime.py`. Verify their hashes before launch. Load the same compiler
and CUDA modules used during runtime preparation; the adapter applies the pinned
runtime environment. Preserve scheduler/Cray transport variables. The receipt
records inherited environment key names and a full-value digest, without dumping
arbitrary environment values.

```bash
# Within an authorized single-node GPU sbatch allocation, after module setup:
/absolute/pinned/python -B -s /frozen/prepare_site.py \
  --runtime /scratch/runtime.json --runtime-sha256 FULL_SHA256 \
  --output /new/scratch/site.json \
  --cores-per-rank 4 --step-timeout 180 --controller-timeout 900
```

`--controller-timeout` checks the intended future campaign timeout; it does not
bound this preparation command. For batch walltime, allow at least
`4 * (step_timeout + 200) + 120` seconds, plus slow filesystem/setup margin.
This conservative allowance includes sequential scheduler queries with their
individual deadlines (accounting can lag), launch cleanup and setup. For 90-second
steps choose at least 22 minutes; for the default 180 seconds choose 28 minutes.
A walltime kill can still prevent receipts; final scheduler audit is mandatory.

Request sufficient physical CPUs and one GPU for the four-rank step (16 distinct
cores at the default). Retain both requested and actual resources: a premium
Perlmutter allocation may reserve the full 128-logical-CPU/4-A100 node (64 physical cores). Run no competing
steps. The preparation chooses ascending Linux CPU IDs, one thread per physical
core, from the batch process's allowed affinity. Ranks 0/1/2 retain the same cores
when rank 3 is added. This is deterministic placement, not a claim of optimal
NUMA/GPU locality. Full GPU inventory and per-rank physical core/GPU PCI/NUMA
observations are retained. The lowest-index allocated A100 is required; Slurm
selecting another device fails rather than silently changing the manifest.

Four tiny MPI probes use the **shared native adapter launcher**: 3/4 ranks,
CPU `--gpus-per-node=0` and one shared GPU, each with
`--exclusive --exact --cpu-bind=mask_cpu`, with
`--distribution=block:block:block --threads-per-core=1` to allocate the physical
CPU prefix required by the masks. An initial hardware attempt using the previous
default distribution failed before rank launch with CPUs outside the requested
masks; preserve that failure when reporting the successful configuration. They require exact actual affinity,
MPI collective success, CPU/service ranks without CUDA imports/mappings, and one
identical physical A100 on all GPU BD ranks. GPU ranks perform a tiny array sum.
CPU/CUDA absence is enforced by the in-rank import/mapping guard; the
`cpu_no_cuda` row is its completion assertion, not an independent measurement.
Before placement assertions, each rank records actual affinity, physical topology,
allowed CPU mask and scheduler GPU/CPU visibility; GPU ranks also retain CUDA
identity or its query error before checking the expected UUID.
Loaded runtime files/dependencies are checked against the captured manifest.
This does not execute DataSource iteration or validate all six workload paths.

Select only `hsnN` interfaces backed by distinct PCI devices with Ethernet byte
counters. Record driver, PCI path, operational state and before/after counters;
exclude bonds and virtual interfaces. Missing/reset counters fail publication.
These node-wide observations do not identify storage traffic or demonstrate a
NIC bandwidth ceiling. Unexpected interface naming/topology requires explicit
review; do not broaden selection to all interfaces.

The controller timeout must be strictly greater than `2 * step_timeout + 120`.
Each probe requires srun zero exit, no matching active step, and sacct
`COMPLETED|0:0`. The existing launcher targets only its uniquely named step on
failure. Timeout/signal fault-injection qualification remains a separate step.
A final allocation audit is still required if interrupted, killed or cleanup is
uncertain. Place an EXIT accounting trap before setup/checksum commands in job
scripts, and obtain final sacct confirmation after the batch job ends.

`site.json.capture/` retains the draft, runtime/source hashes, allocation,
topology/NIC inventory, requests, per-rank evidence, launch/scheduler receipts,
NIC deltas and an evidence index. Only after all checks succeed is `site.json`
atomically published without overwrite (not a crash-durability guarantee).
Failure retains `failed.json` when possible, and publishes no final manifest.
The draft has an unaccepted schema (`perlmutter-raw-site-draft-v1`).
Never use `site-draft.json` as an accepted artifact. Keep failed directories and
start a new output path for any retry. The final manifest links the capture index by absolute path and SHA256.
Preserve and pin both site.json and index.json with campaign qualification
evidence; the native adapter does not independently validate that extra link.

Agent next steps: obtain independent code/evidence review, qualify disposable
step timeout/signal cleanup, then run the six short native workload smoke cases
and qualify cache/byte accounting, instrumentation overhead and component bounds.
Only after those reviews may a new profile enable its qualification flags and
proceed through correctness, timed A/A and A/B. Placement success alone supplies
none of those approvals or performance claims.
