# Perlmutter native raw adapter

**Implemented, not hardware-qualified.** This is the next implementation layer
under the controller. No new benchmark measurement, qualified profile, copied
input, scheduler submission, or performance acceptance accompanies it. Leave the
shipped profile's qualification flags false. SDF is rejected explicitly.

The source API was inspected at `features/psana2-gpu` commit `199575205` (PR298).
The worker requires the KvikIO **26.10** `defaults.get/set` API. The older PR299
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
  first nonempty `Events._emit_batch_end`; loop spans synchronized iteration and
  draining. Per-rank local loop, host CPU seconds and explicit shared-memory close
  time are separate. Process time spans the measured srun step through scheduler completion, including
  provenance/cache work and teardown. Full adapter elapsed including warmup is separate. None are sums of unrelated maxima.
- GPU I/O accounting sums disjoint pre-reset snapshots and one final snapshot
  per constructed reader. CPU/GPU delivered-event identities and selected L1
  datagram/read bytes must match an independent reference. Physical GPU reads
  must equal useful bytes for this fixed workload; new amplification fails.
- NIC observations use the predeclared physical interface list and byte units;
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
- All phases use low-overhead batch-end/read-reset observation plus event identity
  and byte accounting. Their overhead still requires measurement; offline tests
  do not establish that clean timings are unbiased.

## Frozen artifacts and schemas

Add these source files as artifact keys `adapter` (`perlmutter_adapter.py`),
`native_worker` (`native_worker.py`), and `native_contract` (`native_contract.py`).
The controller freezes them beside each other as `adapter.py`, `native_worker.py`
and `native_contract.py`; source imports do not depend on the checkout or archives.
Use an absolute adapter Python executable and:

```json
["/absolute/python", "-B", "-s", "/absolute/perlmutter_adapter.py", "{request}", "{result}"]
```

Each artifact has the controller's `{ "path": "...", "sha256": "..." }` wrapper.
In addition to the controller's qualification receipts, supply:

| Key | Required content |
| --- | --- |
| `event_reference` | `schema: run51-raw-reference-v1`, `events: [[timestamp, native_step_index, selected_L1_datagram_bytes], ...]`, ordered unique 10k prefix. Compute independently from SMD offsets/lengths and transitions, not the benchmark iterator. Event digest is canonical compact JSON of timestamps only, SHA256. |
| `dataset_manifest` | `schema: run51-private-input-v1`, `private_copy: true`, absolute `root`, `event_reference_sha256`, five `files` entries ordered streams 3/5/7/8/9. Each has `stream`, `bigdata`, `smd`. Each file record has absolute `path`, `size`, `mtime_ns`, `device`, `inode`, `sha256`. Bigdata additionally has `range_start`, `range_bytes` for the contiguous selected L1 prefix; its hash covers that exact range. SMD hash covers its whole file. |
| `baseline_runtime`, `candidate_runtime` | `schema: psana-native-runtime-v1`, exact `commit`, `install`, `api_base: 199575205`, `python`, `python_version`, `pythonpath`, absolute psana `package`, `files` mapping every package `.py`/`.so` relative path to SHA256; `dependencies` mapping absolute loaded numpy/mpi4py/CuPy/cupyx/KvikIO module paths to hashes; `native_libraries` mapping absolute relevant MPI/CUDA/NVRTC/cuFile/KvikIO library paths to hashes; `cupy_version`, `kvikio_version`, numeric `cuda_runtime_version`, `cuda_driver_version`; `environment` below. Retain compiler/build/source correspondence and existing unit/device receipts with the campaign. |
| `native_site` | `schema: perlmutter-raw-site-v1`, string `job_id`, actual compute `node`, full `gpu_uuid`, `cpu_affinity` mapping strings `1`/`2` to lists of CPU lists for ranks 0..BD+1, `nic_interfaces` list, `step_timeout_seconds`. All CPU sets have the same size; physical core sets must not overlap or contain SMT siblings. |
| `component_contract` | Reviewed metrics/denominators/bounds and required observations per the component policy. Retain this even when bounds remain unqualified; a structurally accepted sample is not component acceptance. |

Runtime `environment` must contain exactly `PATH`, `LD_LIBRARY_PATH`, `CUDA_HOME`,
`CUDA_PATH`, `CONDA_PREFIX`, `LCLS_CALIB_HTTP`. Pin appropriate site values.
Workload PS/PSANA, KvikIO, CUDA, CuPy, OMP and Python-path overrides are cleared
before launching; the adapter applies its profile and pinned runtime values.
Scheduler/Cray transport variables remain inherited and need qualification in
the selected allocation. GPU ranks set visibility to the frozen UUID and query
CUDA to check it; CPU/service ranks hide GPUs. KvikIO automatic direct I/O read
and write are explicitly disabled and checked, in addition to fallback ON.

The `dataset.sites.perlmutter` path in the new campaign profile must be the
prepared private directory. Never evict the shared public input or use a symlink
view/hardlinks as a cache-isolated copy. Files must be owned by the user and have
link count one. These filesystem checks cannot prove exclusive use; the reviewed
private-copy/compute-access receipt establishes that. The adapter rechecks small
SMD hashes and bigdata metadata per process, not hundreds of GB of bigdata hashes
per sample. Independently verify used-range hashes during preparation, retain the
receipt, and protect input files against modification for the entire cohort.
The current adapter requires a contiguous selected-event range in each JF file;
verify that from SMD first. Do not include transitions/gaps or silently broaden
ranges to make byte checks pass. No input-reference generator is implemented here.

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
5. Qualify timeout/signal cleanup on a disposable step: the adapter targets only
   its uniquely named step, checks it disappears, and records failures. Normal
   success requires srun exit zero, no active step and sacct `COMPLETED|0:0`.
   Accounting lag is a failure requiring investigation, not permission to bypass
   evidence. SIGKILL/node loss cannot run cleanup handlers; audit allocation state.
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
