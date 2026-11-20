# SDF execution path

The maintained runner supports one-node SDF Ampere A100 campaigns with Open MPI 5.
The workload, cache verification, event/byte accounting, paired scheduling,
5% timing policy and result validation are shared with Perlmutter. The historical
`perlmutter_adapter.py` filename is retained, but it dispatches from the frozen
site manifest; there is no copied SDF workload implementation.

Implementation support is distinct from campaign qualification. The shipped
profile remains unqualified; each actual PR needs compatible builds, site/data
receipts, component bounds and correctness -> A/A -> A/B. Small launch/NIC
validation does not establish run51 throughput, observer overhead or PR acceptance.

## Select and announce the cluster

From the runner directory, run `python -B select_site.py`. On sdfiana it reports
`execution_site: sdf`; on Perlmutter it reports `perlmutter`. Recognized hostname
and scheduler/environment identity must agree. A task explicitly targeting the
other cluster uses `--site sdf` or `--site perlmutter`. When `remote_required` is
true, connect to that target and recheck identity there before preparing/running.
Unknown automatic selection fails. Never silently fall back to another cluster.
Announce the detected host and chosen execution site to the user; no extra
approval question is needed for the already requested local-cluster campaign.

Sdfiana is a login host, not the execution node. Use a single-node Slurm Ampere
allocation with an A100, exclusive node use, enough physical CPUs for four ranks,
and a walltime derived from the campaign budget. With four cores/rank, request
one outer task with 16 CPUs (`--ntasks=1 --cpus-per-task=16`) for up to four MPI
ranks; the launcher reserves 12 or 16 CPUs within that allocation. Existing SDF studies used
`lcls:data@ampere`, `ampere`, normal QoS; validate account/access before submission.
Do not inherit Perlmutter account or premium QoS. The task's explicit allocation
instructions take precedence. Do not run benchmarks or full data hashes on login.

## Prepare on SDF

1. Build/pin both revisions with [RUNTIME.md](RUNTIME.md) in matching SDF dependency
   environments. Open MPI 5, Python 3.9+, KvikIO 26.10 and the required private APIs
   must pass the existing probes. Prior environments are candidates, not automatic
   qualification. The pinned `mpiexec` must belong to that runtime. A single Slurm step
   (`srun --mpi=none`) contains local Open MPI ranks, launched with PRRTE
   `plm=ssh` and `--host localhost`, with the remote agent set to `/usr/bin/false`
   and Slurm resource discovery disabled. No nested scheduler steps or remote MPI
   daemons are allowed; attempts to launch remotely fail.
   `sdf_rank.py` checks the Open MPI local/global rank mapping and applies the
   frozen CPU mask inside the scheduler CPU allowance before execing the worker.
   GPU BD ranks select the allocated physical GPU by its frozen UUID, avoiding
   CUDA/NVML ordinal-order differences; native checks verify that actual UUID.
   The original Slurm visibility is retained in the rank-launch evidence.
   SDF exclusive allocations may grant all four GPUs despite requesting one.
   GPU steps reserve the frozen allocation GPU set, then each BD exposes only
   the selected UUID. This prevents Slurm changing the physical GPU across steps;
   it does not turn the workload into a four-GPU test. CPU steps request zero GPUs.
2. Find run51 at the SDF location, initially `/sdf/data/lcls/ds/prj/public01/xtc`.
   Verify file identity/selected streams, not merely directory existence.
   Reuse a verified owner-private run51 copy or use [PREPARE.md](PREPARE.md) on
   compute. Input preparation is shared and now accepts SDF Ampere allocations.
   Set `dataset.sites.sdf` to the private root; retain independent event/range and
   manifest receipts. The public path is not a cache-eviction target. Promoted
   Sim10 files under `prjdatpsana` are a different workload.
3. Inspect physical NICs on the actual compute node. Select storage-facing physical
   interfaces with `ethtool -S` hardware `rx_bytes_phy` and `tx_bytes_phy` counters.
   Freeze the list; no bond/virtual aliases, duplicate PCI endpoints, netdev/RDMA
   counter mixing or fallback to unavailable-as-zero. Ensure `ethtool` is on the
   pinned runtime PATH. These counters are bytes, not RDMA four-byte words.
4. Run `prepare_site.py` directly from the single-node batch shell, outside srun:

   ```bash
   python -B prepare_site.py --site sdf \
     --runtime /sdf/path/runtime.json --runtime-sha256 HASH \
     --output /sdf/path/new-site.json --cores-per-rank 4 \
     --nic-interfaces PHYSICAL_INTERFACE --sdf-mpiexec /sdf/path/bin/mpiexec
   ```

   Replace placeholders from actual inventory. The advertised `pmix_v2` plugin
   failed to load in the initial native probe; this path uses the runtime's
   `mpiexec` inside a tracked `--mpi=none` step instead. Four CPU/GPU probes must
   succeed. The outer task reserves the union of rank CPUs; Open MPI slot
   oversubscription permits multiple ranks within that one Slurm task, while
   the wrapper enforces disjoint physical CPU masks (no physical oversubscription). SDF reserves four physical
   CPUs per rank by default; exact affinity, physical-core disjointness and shared
   GPU UUID are checked. No assumption of Perlmutter's 64-core CPU topology.
5. Run [cleanup checks](CLEANUP.md) using `--site sdf --sdf-mpiexec /sdf/path/bin/mpiexec` and
   [smoke/full accounting checks](SMOKE.md) using `--site sdf --sdf-mpiexec /sdf/path/bin/mpiexec
   --nic-interfaces PHYSICAL_INTERFACE`. The full accounting path uses the same
   idle/cold/warm traffic controls with SDF physical counters. The separate Cray
   MPI warning experiment is not applied to Open MPI. Existing psana correctness
   tests and normal completion/cleanup checks still apply.
6. Set the new profile's `site` to `sdf`, use the SDF dataset root and artifacts,
   supply evidence-backed qualification receipts, freeze it, and run the existing
   correctness/A/A/A/B commands within one allocation. No automatic flag changes.

## Required network and cache evidence

SDF site manifests use `schema: sdf-raw-site-v1`, the shared job/node/GPU/CPU/timeout
fields, plus `job_gpus`, `step_gpu_count` (the allocated GPU set reserved by GPU steps),
`mpi_launcher: openmpi-local-v1`, absolute `mpiexec`, its
`mpiexec_sha256`, `nic_backend: ethtool-physical-v1`, and `nic_devices`
(mapping each selected interface to its resolved PCI sysfs path).

Each measured sample retains raw before/after hardware counter output, interface
and device identity, byte units, per-interface sampling intervals, RX/TX deltas
and rates in `resources.json`. The result's `nic` totals use this physical backend.
The adapter checks counter availability, identity and monotonicity and requires
RX to cover at least 90% of the selected data bytes, as in the existing full-prefix
traffic check. Background traffic can inflate this ratio; it is supporting
coverage evidence, not proof of storage-only attribution. A selected interface
that misses storage traffic must fail the cold/warm controls, not be excused.

The final report must include RX/TX bytes and RX/selected-byte ratios per sample,
plus per-interface rates from the saved counter intervals. `campaign.py summarize`
includes network samples beside each case's timings. Do not compare NIC totals
as though SDF and Perlmutter counter implementations had identical semantics.

Private-file `posix_fadvise` attempts and two zero-residency `mincore` observations
are required before each timed case on both sites. This proves the specified
client page-cache condition only. Weka/Lustre server and SMD caches are uncontrolled.
If that condition or mandatory NIC evidence cannot be established, retain the
failed attempt and stop; do not silently relabel it as cold or accept missing data.

Freeze `sdf_rank.py` as the `sdf_rank` artifact beside the worker and helper.
The shared preparation/smoke drivers do this automatically. The rank wrapper
records original Slurm launcher rank, Open MPI ranks, step identity and before/
after affinity before mapping rank variables for the shared worker. Launcher
bytes are checked against the site manifest before every launch. Scheduler
success and cleanup still refer to the one uniquely named sample step.

`mpiexec_sha256` pins only the resolved launcher front-end binary. Runtime
capture records observed MPI libraries, but PRRTE/PMIx helper daemons and their
full transitive library set are not individually pinned by the site manifest.
This is a provenance limitation: retain the matching Open MPI installation and
site package/module records for the campaign; do not change them between samples.

The maintained SDF `prepare_runtime.py capture`, `prepare_site.py`/`site_probe.py`,
`run_smoke.py`/`accounting_probe.py` and `native_worker.py` paths have not yet
executed on SDF hardware. Native launch evidence used a notes-only probe sharing
`launch()` and `sdf_rank.py`; the maintained cleanup driver did run successfully.
The first SDF campaign must qualify these remaining entry points before accepting
performance results. Small launch/NIC checks do not replace those qualifications.
