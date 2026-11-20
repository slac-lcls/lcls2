# Native raw adapter: Perlmutter and SDF

**Implemented; full native workload qualification remains pending.** The
controller's shipped qualification flags remain false. The shared implementation
selects Perlmutter/Cray MPICH or SDF/Open MPI 5 from the request and matching site
manifest; see [SDF.md](SDF.md). Hardware/site evidence never transfers implicitly.
Runtime and placement preparation have separate hardware receipts. The
[placement validation](../../../notes/pr_regression_site_20261117/README.md)
checks launch/CPU/GPU/NIC behavior only, without run51 iteration, cache conditioning,
instrumentation overhead, component bounds or a timed regression campaign.

The source API was inspected at `features/psana2-gpu` commit `199575205` (PR298).
The worker on both sites requires the KvikIO **26.10** `defaults.get/set` API. The older PR299
Python 3.9 installation is not automatically a compatible runtime. Build and
qualify both selected revisions independently; the manifest `api_base` states an
API contract, not proof that a candidate or its installed files match Git.
A/A may use the same reviewed runtime manifest for both sides. This step does
not select a future PR or claim any existing install has been qualified.

## Work and measurement

- Six existing raw cells: CPU events/steps with 1/2 BDs; GPU events with 1/2 BDs
  sharing one A100. One SMD0, one EB; no service rank. Profile settings stay fixed.
- CPU materializes selected-stream raw datagrams through `run.events()` or nested
  `run.steps()/step.events()`; byte counts come from delivered datagram lengths.
- GPU uses `GpuTask(callback, ['jungfrau.raw'], [])`. This **includes dense input
  preparation/gather**, framework kernels and reads. The callback only records
  batch size; preflight/diagnostic phases also check input dtype/size. It launches
  no user kernel, publishes no outputs and performs no payload D2H. It is not a
  descriptor-only I/O microbenchmark or numerical/raw-pixel verification.
- Separate warmup process before each measured process. Within measured MPI setup,
  rank 0 attempts private-copy eviction and checks measured-range residency twice
  with `mincore`, after every rank has completed setup and before loop release.
  Failure to reach zero aborts. No system-wide flush; SMD/server caches uncontrolled.
- Setup is DataSource/Run construction; first batch is the maximum BD time to the
  first window end containing L1 events (`Events._emit_batch_end`, including a
  finish drain). GPU deferred retirement can yield packet N's events while
  processing N+1, so this is not latency to the first EB packet or user kernel.
  EB packet counts, L1 events per delivery window (including empty windows),
  transitions by service per window, final L1/transition drain counts, and GPU
  callback execution sizes are separate counters. Only L1 window populations
  plus L1 drain must equal delivered user events. Transition-only windows cannot
  set first-batch time. The observer classifies native envelopes with the current
  first_service API and a once-bound `TransitionId.isEvent` classifier, counting
  both L1Accept (12) and L1Accept_EndOfBatch (11) as events. This adds
  per-envelope cost that must be overhead-qualified. Loop spans
  synchronized iteration and draining. Per-rank local loop, host CPU seconds and explicit shared-memory close
  time are separate. Process time ends at measured srun return, excluding scheduler-accounting lag but including
  provenance/cache work and teardown. Full adapter elapsed including warmup is separate. None are sums of unrelated maxima.
- GPU I/O accounting sums disjoint pre-reset snapshots and one final snapshot
  per constructed reader. CPU/GPU delivered-event identities and selected L1
  datagram/read bytes must match an independent reference. Physical GPU reads
  must equal useful bytes for this fixed workload; new amplification fails.
- NIC observations use the predeclared physical interface list and byte units;
  SDF requires hardware `ethtool -S` `rx_bytes_phy`/`tx_bytes_phy` counters and
  refuses unavailable/reset/duplicate endpoints. Raw snapshots, PCI identities,
  per-interface deltas/rates and node totals are retained; no mixing counter families.
  Campaign summaries include per-sample RX/TX and RX/selected-byte ratios.
  The following netdev behavior remains for the legacy Perlmutter protocol:
  reset fails, unavailable interfaces produce null totals, not zero. Interface
  selection requires site qualification to avoid double-counting virtual devices.
- Resources retain physical CPU-core maps, actual GPU UUID/PCI/model/NUMA, per-rank
  RSS high-water, host CPU time, and GPU pool end snapshots. **GPU/pinned peaks,
  candidate/live-window visits, occupancy scan units and kernel counts are not
  implemented measurements**; they remain explicitly unavailable.
- `profile` alone enables per-rank cProfile and writes call counts and host wall
  self/cumulative times for psana functions. This exposes pool polls, issue/plan
  calls, CPU scheduling and task calls where Python-visible. It does not count
  iterations inside a function or separate pressure from shutdown drains. Nested
  cumulative times overlap. No statistical component-timing claim from one pair.
- All phases use envelope-classification, batch-end and read-reset observation plus event identity
  and byte accounting. Their overhead still requires measurement; offline tests
  do not establish that clean timings are unbiased.

## Frozen artifacts and schemas

Add these source files as artifact keys `adapter` (`perlmutter_adapter.py`),
`native_worker` (`native_worker.py`), and `native_contract` (`native_contract.py`).
The controller freezes them beside each other as `adapter.py`, `native_worker.py`
and `native_contract.py`; source imports do not depend on the checkout or archives.
For SDF also freeze `sdf_rank.py` with artifact key `sdf_rank`.
Use an absolute adapter Python executable and:

```json
["/absolute/python", "-B", "-s", "/absolute/perlmutter_adapter.py", "{request}", "{result}"]
```

Each artifact has the controller's `{ "path": "...", "sha256": "..." }` wrapper.
In addition to the controller's qualification receipts, supply:

| Key | Required content |
| --- | --- |
| `event_reference` | `schema: run51-raw-reference-v1`, `events: [[timestamp, native_step_index, selected_L1_datagram_bytes], ...]`, ordered unique 10k prefix. Compute independently from SMD offsets/lengths and transitions, not the benchmark iterator. Event digest is canonical compact JSON of timestamps only, SHA256. |
| `dataset_manifest` | `schema: run51-private-input-v2`, `private_copy: true`, absolute `root`, `event_reference_sha256`, five `files` entries ordered streams 3/5/7/8/9. Each has `stream`, `bigdata`, `smd`. Bigdata files live directly under `root`; SMD files under `root/smalldata`, and both opened lists must match in stream order. Each file record has absolute `path`, `size`, `mtime_ns`, `device`, `inode`, `sha256`. Bigdata additionally has ordered disjoint `ranges: [[start, length], ...]` and total `range_bytes` for the selected L1 prefix; its hash covers the concatenation of those ranges in file order, excluding transition gaps. Legacy v1 single-range manifests remain supported. SMD hash covers its whole file. |
| `baseline_runtime`, `candidate_runtime` | `schema: psana-native-runtime-v1`, exact `commit`, `install`, `api_base: 199575205`, `python`, `python_version`, `pythonpath`, absolute psana `package`, `files` mapping every package `.py`/`.so` relative path to SHA256; `dependencies` mapping absolute loaded numpy/mpi4py/CuPy/cupyx/KvikIO module paths to hashes; `native_libraries` mapping absolute relevant MPI/CUDA/NVRTC/cuFile/KvikIO library paths to hashes; `cupy_version`, `kvikio_version`, numeric `cuda_runtime_version`, `cuda_driver_version`; `environment` below. Retain compiler/build/source correspondence and existing unit/device receipts with the campaign. |
| `native_site` | `schema: perlmutter-raw-site-v1` or `sdf-raw-site-v1` (additional SDF fields in [SDF.md](SDF.md)), string `job_id`, actual compute `node`, full `gpu_uuid`, `cpu_affinity` mapping strings `1`/`2` to lists of CPU lists for ranks 0..BD+1, `nic_interfaces` list, `step_timeout_seconds`. Optional `step_cpus_per_task` reserves more CPUs per task than the bound mask (defaults to mask size for older manifests). Perlmutter reserves 16 physical CPUs per rank and binds four; SDF reserves four per rank inside one combined launcher step. All CPU sets have the same size; physical core sets must not overlap or contain SMT siblings. |
| `component_contract` | Reviewed metrics/denominators/bounds and required observations per the component policy. Retain this even when bounds remain unqualified; a structurally accepted sample is not component acceptance. |

Runtime `environment` must contain exactly `PATH`, `LD_LIBRARY_PATH`, `CUDA_HOME`,
`CUDA_PATH`, `CONDA_PREFIX`, `LCLS_CALIB_HTTP`. Pin appropriate site values.
Both runtime manifests must agree on Python/version, dependency and relevant native
library hashes, the six environment values, and GPU package/CUDA versions. Use
isolated psana prefixes in the same pinned dependency environment. An intentional
change to these controls requires a separately reviewed adapter/profile.
Workload PS/PSANA, KvikIO, CUDA, CuPy, OMP and Python-path overrides are cleared
before launching; the adapter applies its profile and pinned runtime values.
Scheduler/Cray transport variables remain inherited and need qualification in
the selected allocation. GPU ranks preserve Slurm's single-device visibility and query CUDA to match the
frozen physical UUID; CPU/service ranks hide GPUs. KvikIO automatic direct I/O read
and write are explicitly disabled and checked, in addition to fallback ON.

The selected `dataset.sites[site]` path in the new campaign profile must be the
prepared private directory. Never evict the shared public input or use a symlink
view/hardlinks as a cache-isolated copy. Files must be owned by the user and have
link count one. These filesystem checks cannot prove exclusive use; the reviewed
private-copy/compute-access receipt establishes that. The adapter rechecks small
SMD hashes and bigdata metadata per process, not hundreds of GB of bigdata hashes
per sample. Independently verify used-range hashes during preparation, retain the
receipt, and protect input files against modification for the entire cohort.
The adapter supports exact disjoint selected-event ranges in each JF file.
Do not include transitions/gaps or broaden ranges to make byte checks pass.
The [input preparation tool](PREPARE.md) derives these ranges independently from
SMD and verifies source/copy hashes and selected bigdata headers on compute.

Metadata such as available-event count, other-site paths and qualification status
remain frozen declarations supported by reviewed receipts. Native event/byte
counts, effective DataSource/KvikIO settings, environment, selected files and
placement are checked separately; an echoed JSON object alone proves nothing.

## Qualification sequence for the next agent

1. Select and record exact baseline/candidate SHAs. Start with same-baseline A/A
   qualification before any PR comparison. Review the source API on both sides;
   do not claim support for arbitrary earlier/future revisions.
2. On authorized compute resources, prepare the private input prefix and independent
   event/range reference, verify source/copy hashes, freeze manifests and ownership.
   Pin real run51, not the promoted Sim10 `prjdatpsana` inputs. Do not hash/copy the
   full data on a login node. Reuse the existing run51 pixel-status audit only for
   its documented scope; this raw workload does not exercise calibration.
3. Build isolated matching installs, verify loaded Python/native dependencies,
   retain exact-revision unit/device receipts, and create runtime manifests. Run
   short native smoke cases to check all six paths, byte/cache accounting and
   observed CPU/GPU placement. A changed API or unsupported package must fail.
4. Freeze allocation and physical placement. A new allocation needs a fresh
   campaign/cohort. The controller timeout must exceed **twice the per-step timeout
   plus 120 seconds**; the adapter rejects insufficient cancellation headroom.
   For the shipped 900-second timeout choose at most 389 seconds per step.
   For Perlmutter, specifically test `--exclusive --exact --cpu-bind=mask_cpu` together with
   `--distribution=block:block:block --threads-per-core=1` and
   `--gres-flags=disable-binding` (also request disable-binding at allocation): Slurm's
   selected step CPUs must contain the frozen masks. Also verify CPU steps accept
   `--gpus-per-node=0` on this installation. These flags are unqualified launch
   viability items; do not bypass the worker's exact affinity/GPU checks.
5. Qualify timeout/signal cleanup on a disposable step: the adapter targets only
   its uniquely named step, checks it disappears, and records failures. Scheduler
   cancellation has a six-second total deadline; local launcher waits add at most
   one second, within the controller's ten-second SIGTERM grace. If the scheduler
   is unavailable or the step survives, failure.json says no_active_steps=false;
   audit the allocation before further work. This is bounded best-effort cleanup,
   not a guarantee that remote ranks died when the scheduler cannot confirm it. Normal
   success requires srun exit zero, no active step and sacct `COMPLETED|0:0`.
   Accounting is polled within a 30-second deadline, retaining responses.
   Lag beyond that deadline fails; it is not permission to bypass evidence. SIGKILL (including the controller's forced kill), a blocked filesystem
   write, or node loss can prevent cleanup receipts; audit allocation state.
6. Measure instrumentation overhead, qualify component bounds, and supply missing
   metrics when the reviewed component contract makes them required. Do not mark
   native_adapter qualified merely because these offline tests pass. Run the
   independent implementation/evidence review before enabling qualification flags.
7. Prepare a fresh frozen campaign, then workload preflight, A/A, A/B in that order.
   Run `profile` separately. Preserve all failures and review actual results before
   publishing a performance claim. No submit script is added by this step.

Each accepted sample has four distinct required receipts, full per-rank/native
files, and a native index embedded in the hashed resources receipt. The controller
checks indexed files at acceptance and summary. This detects later mutations;
rewriting every receipt is outside this integrity model.

Offline checks (Python 3.9+, standard library):

```bash
python -B psana/psana/gpu/scripts/performance/regression/test_native.py -v
python -B psana/psana/gpu/scripts/performance/regression/test_campaign.py -v
```

The [runtime preparation driver](RUNTIME.md) builds isolated exact revisions and
captures manifests. New `isolated-install-v1` manifests pin the common six-key
dependency environment; the launcher prepends the selected install bin/lib paths.
Their `install_files` map pins support files beyond the psana package. Runtime
preparation and manifest compatibility do not imply native qualification.

The [allocation and placement driver](SITE.md) creates a fresh `native_site`
artifact after four tiny adapter-launched CPU/GPU placement probes. It supplies
launch evidence separately from workload, cache, cleanup-fault and performance
qualification; the shipped profile remains blocked.
