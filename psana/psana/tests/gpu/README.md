# psana GPU tests

Run from the repository root in the locally built psana environment. Tests
are organized by the contract they protect, not by implementation stage.

| File | Responsibility |
|---|---|
| `unit/test_core.py` | Public exports, exclusive/hybrid routing, Event envelopes, transitions, MPI transport, slot ownership, subbatch splitting, and memory accounting |
| `unit/test_gpudgram.py` | Stream-indexed Configure tables, numeric field handles, named-field and adapter-input selection, and read-descriptor translation |
| `unit/test_gpu_input.py` | Event/stream dgram mapping, canonical segment order, multiple detectors/fields, segment-preserving shape/dtype access, and input-consumer leases |
| `unit/test_gpu_result_lifetime.py` | Result copies/views, completion-token tracking, D2H handoff, host caching, budgets, and safe slot retirement |
| `unit/test_gpu_smoke.py` | Manual-smoke reporting, incomplete participation, and launcher argument/exit-status handling without acquiring GPUs |
| `integration/test_gpudgram_device.py` | Real CUDA XTC walking and field consumption, stream-scoped NamesId resolution, and reusable parser tables |
| `integration/test_pixel_exact.py` | Two fast locator-to-adapter CUDA tests plus six slow DataSource raw/calibration acceptance cases |

## Running the tests

CPU-only unit tests use lightweight CUDA stand-ins where necessary:

```bash
python -m pytest -q psana/psana/tests/gpu/unit
```

On a CUDA node, run the five fast device tests:

```bash
python -m pytest -q -rs -m "gpu and not slow" psana/psana/tests/gpu/integration
```

The parser tests use the tracked fixture
`psana/psana/tests/test_data/chunking/xpptut15-r0014-s000-c000.xtc2`.
They do not depend on generated `tests/.tmp` files or external experiment
data. Missing tracked data is an error, not a data-availability skip.
The CPU harness frames dgrams and obtains independent expected values;
the GPU walks the XTC and resolves payload addresses. Tests may copy metadata
back for assertions after device consumption. The no-round-trip contract
applies to the device consumer, not test assertions or high-level Python
shape materialization.

Run the six slow acceptance cases separately:

```bash
python -m pytest -q -rs -m slow psana/psana/tests/gpu/integration/test_pixel_exact.py
```

These cover single-event delivery, batched slot reuse and partial tails,
three automatic-D2H chunk sizes, and mirrored `hybrid_det` routing. They
preserve pixel-exact raw/calibration comparisons and exercise public named
field access through `on_cpu`, `on_gpu`, and `on_gpu_view`. Slot-backed result
ownership remains checked; parser internals are tested in the device suite.

The acceptance dataset defaults to public Lysozyme Jungfrau data,
`mfx100848724` run 51, under `/sdf/data/lcls/ds/prj/public01/xtc`. Override
with `PSANA_GPU_TEST_EXP`, `PSANA_GPU_TEST_RUN`, and `PSANA_GPU_TEST_DIR`.
The reference must have nonzero calibrated pixels and multiple gain-bit
values. Runs 77/78 are unsuitable defaults because their effective masks
make calibrated output entirely zero. Common-mode correction is disabled
for the CPU reference to match the GPU calibration adapter.

The default pytest configuration excludes `slow`; GPU tests skip without
CUDA, and slow acceptance tests also skip without their external dataset.
Check the skip summary before claiming device or DataSource coverage.

## Scope and maintenance

- GPUBAT1 descriptors remain the active read/transport contract; they are
  not the removed fixed-stride raw-field layout.
- Exclusive `gpu_det` and mirrored `hybrid_det` routing have different
  ownership rules. Their rejection/acceptance tests are intentional.
- Automatic calibration-adapter input selection may require one unique
  array. General named-field access supports multiple arrays and scalars.
- Subbatch splitting tests protect event indivisibility and order. They do
  not grant permission to allocate beyond the device-memory quota.
- CPU CUDA stand-ins test bookkeeping but cannot prove real asynchronous
  execution ordering; retain the real-device and D2H acceptance cases.
- Add coverage for independent failure modes, not redundant external-data
  shape/dtype/NaN smoke checks. Keep performance thresholds out of pytest.

## Manual MPI/GPU transport smoke check

Run this after MPI transport, GPU routing, or GPU-assignment changes. It is
not required for every parser edit. It exercises one node with one SMD0,
one EB, and at least two BDs, with one BD per GPU. Multi-node, multiple-EB,
and GPU-sharing topologies are outside this check's scope.

Activate the Python environment used for your local build, then launch:

```bash
source setup_env.sh
bash psana/psana/gpu/scripts/run_multi_gpu_test.sh --max-events 50 --batch-size 5
```

The launcher sources `install_psana/activate.sh` and verifies the imported
psana is under that prefix. Set `PSANA_GPU_TEST_PREFIX` for a different local
build. Missing setup or a wrong install is fatal. Python comes from the
activated environment, not a hard-coded executable. No output is filtered.

From a login node it requests an A100 allocation; inside an existing Slurm
allocation it starts a step. With the default two GPUs, the allocation must
provide one node, four tasks, and two CPUs per task. Set `N_GPUS_PER_NODE`
to request more GPUs/BDs. The default time limit is 15 minutes; override with
`PSANA_GPU_TEST_TIME`. Slurm bounds hangs and the launcher returns `srun`'s
exit status, including failures and timeouts.

The workload defaults to public Jungfrau `mfx100852324` run 77. Override
`PSANA_GPU_TEST_SMD_GLOB` to select one experiment/run/directory. Its zero
calibration mask is acceptable here because this check does not compare
pixel values, shapes, NaNs, or performance. Numerical correctness belongs
to the separate pixel-exact acceptance suite above. Each delivered event
must expose `jungfrau.raw`; the smoke check does not copy those pixels.

Every rank is reported, including BDs with zero events. GPU identity comes
from the actual CUDA device's PCI bus address together with its hostname,
not an inferred rank or `CUDA_VISIBLE_DEVICES` ordinal.

| Result | Meaning |
|---|---|
| `PASS` / exit 0 | Expected event count, unique timestamps, distinct BD devices, every requested BD processed events, and MPI completed |
| `FAIL` / nonzero | Event delivery or device-placement failure, missing GPU result, or runtime/launcher error |
| `INCOMPLETE` / script exit 2 | Delivery checks passed, but at least one requested BD/GPU received no events |

Incomplete participation can occur legitimately with very short runs. Rerun
with more events or smaller batches; do not force event destinations or treat
it as a data-loss error. For example, `--max-events 1 --batch-size 1` cannot
exercise both BDs and should report incomplete coverage, not a full PASS.

This smoke check does not prove numerical equality, true GDS operation,
performance, or the proposed user-task C ABI. Benchmark scripts likewise do
not replace tests. The small CPU tests of reporting/launching are collected
by pytest; the real multi-rank GPU run remains manual.
