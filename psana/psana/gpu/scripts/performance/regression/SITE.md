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

Request sufficient physical CPUs and one GPU for the four-rank step (16 distinct
cores at the default). Retain both requested and actual resources: a premium
Perlmutter allocation may reserve the full 128-core/4-A100 node. Run no competing
steps. The preparation chooses ascending Linux CPU IDs, one thread per physical
core, from the batch process's allowed affinity. Ranks 0/1/2 retain the same cores
when rank 3 is added. This is deterministic placement, not a claim of optimal
NUMA/GPU locality. Full GPU inventory and per-rank physical core/GPU PCI/NUMA
observations are retained. The lowest-index allocated A100 is required; Slurm
selecting another device fails rather than silently changing the manifest.

Four tiny MPI probes use the **unchanged native adapter launcher**: 3/4 ranks,
CPU `--gpus-per-node=0` and one shared GPU, each with
`--exclusive --exact --cpu-bind=mask_cpu`. They require exact actual affinity,
MPI collective success, CPU/service ranks without CUDA imports/mappings, and one
identical physical A100 on all GPU BD ranks. GPU ranks perform a tiny array sum.
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
Never use `site-draft.json` as an accepted artifact. Keep failed directories and
start a new output path for any retry. Preserve/hash the capture index with the
campaign qualification evidence; the manifest itself contains placement only.

Agent next steps: obtain independent code/evidence review, qualify disposable
step timeout/signal cleanup, then run the six short native workload smoke cases
and qualify cache/byte accounting, instrumentation overhead and component bounds.
Only after those reviews may a new profile enable its qualification flags and
proceed through correctness, timed A/A and A/B. Placement success alone supplies
none of those approvals or performance claims.
