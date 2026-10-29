# Issue159 Stage 6: integrated correctness reconciliation

Stage 6 starts from merged PR259, `a4293d71f`, on the separate branch
`test/issue159-stage6-correctness`. The test correction is `71f5c04a2`.
Production Python/CUDA/native code is unchanged. This step requires independent
review before merging into `fix/gpu-unaligned-reads-159`; Stage 7 and the final
feature-target PR remain separate. Issue159 stays open.

## Coverage and evidence reuse

The following maps the implementation plan to concrete tests and evidence.
Counts from overlapping selections are not summed into a distinct-test total.
The new broad run does not replace the focused sanitizer or MPI evidence.

| Requirement | Coverage | Evidence and disposition |
| --- | --- | --- |
| Safe dense gather at actual addresses | `test_gather_alignment.py`: uint16/float32, every base/offset residue, exact allocation tails at 1/255/256/257/300 pixels, floating object bits including NaNs, invalid/absent zeroing and slot reuse | PR246 jobs 42247538 (333 CUDA cases) and 42247695 (315 focused memcheck cases, zero errors); gather code/test unchanged; included in new core run |
| Dense preparation and slot reuse | `test_raw_preparation.py`: exact Jungfrau raw/gain bits and presence mask, malformed/missing segments, canonical ordering, consumer completion, one gather per batch and budget admission before growth | Included in PR246 device checks and the new core run; raw preparation code/test unchanged |
| Borrowed fields are typed-load safe | `test_gpu_field_views_device.py`: ten types, scalar/2D/empty shapes, offsets 0–7, shifted unaligned and shifted-but-aligned bases; real typed CUDA loads; direct-view pointer identity and cached copy identity | Reviewed empty-field fix at `405f868ed`: job 42254009, 302 CUDA + 302 memcheck, zero errors; current runtime and device test match its recorded hashes |
| Async ownership and retirement | Input-copy lifetime, input-window, group ownership, raw preparation and retirement tests; delayed producer/two consumer streams, root/child ordering, ownerless retention, failure/retry | PR246 device/memcheck selections plus CPU coverage; included in new core run; no ownership change since those checks |
| Budget and failures | Rounded allocation charge once; cache-hit reuse; insufficient capacity; allocation/copy/context-entry failures; partial initialization; empty/absent fields need no alignment storage; release only when reclaimable | Field CPU/device tests plus MPI reporting below; current field-copy implementation matches reviewed hash |
| Existing `.on_cpu` / `.on_gpu` compatibility | Exact bits/segments/shapes, independent copies, CPU caching, copy counts, nondefault-stream readiness, retained outputs, retired first-access rejection | Stage 5 `4a4df550b`: 393 CPU, job 42391066: 60 CUDA + 60 memcheck, zero errors; current tests/runtime unchanged and included in new core run |
| MPI/constants/reporting | Six ranks, one EB/four BDs; sharing on/off, automatic/explicit budget, bulk off/on, normal/early close; unaligned scalar/array/absent fields; two consumers; private charge versus imported constants | Job 42250696 at integrated runtime `31c4c2f2`: all 16 cases and two real-Jungfrau 20-event comparisons passed; qualified reuse described below |
| Real detector field access and hybrid behavior | `test_pixel_exact.py`: five explicitly selected slow cases, each 13 events; single-event, slot reuse/partial tail, hybrid, bulk exclusive/hybrid; exact raw pixels, scalar fields, borrowed views, retained host copies | Explicit new Stage 6 pixel run; not silently inferred from the default suite, which deselects slow tests |
| Broad integration | Both required groups, `psana/psana/tests/` and `psana/psana/tests/byhand_*`, from the frozen repository root; fresh process per group | New Stage 6 runs below, including all non-slow GPU tests on an actual A100 |

The initial PR246 device and MPI evidence used runtime merge `31c4c2f2`,
validated against independent trial tree `88d0fd3f`; initial PR head was
`f2b3c2d4e`. The only subsequent production behavior change is the empty-field
fast path in `gpu_input.py` (`nbytes == 0`); `gpu_events.py` also received a
reporting docstring clarification. The focused 302-case review run covers that
empty-field change. The MPI fixture's materialized fields are nonempty (257
uint64 elements and a scalar); its absent segment is not an empty field. Its
constant/IPC, budget and retirement paths are unchanged, so that evidence is
reused without a new MPI campaign. Empty *constants* remain distinct from empty
*fields*.

Stage 5 added an optional `defer_retirement` fixture argument, preserving its
previous default. Existing view tests still use that default. New accessor tests
exercise immediate retirement explicitly and have their own sanitizer evidence.
The Stage 5 sanitizer used `--show-backtrace device` after controlled diagnosis
of host-backtrace retention; no memory-check suppression or weakened assertion
was used. See the [Stage 5 report](issue159_stage5_compatibility_20261029.md).

The MPI scalar-constant fixture was restored at the GPU-manager boundary because
of the inherited host scalar-shape issue [#247](https://github.com/slac-lcls/lcls2/issues/247).
This proves GPU field/constant ownership, not an end-to-end fix for that host
transport issue. Real Jungfrau comparisons used normal calibration setup.

Evidence links: [PR246 CUDA/memcheck](issue159_parallel_validation_20261028.md),
[review delta](issue159_review_followup_20261028.md),
[MPI/reporting](issue159_mpi_reporting_20261028.md), and
[Stage 5 compatibility](issue159_stage5_compatibility_20261029.md).

## First broad run and narrow corrections

Job **42394606**, `sdfampere010`, tested the unmodified Stage 6 base `a4293d71f`.
Core result: **1,642 passed, four failed, five skipped, seven deselected**.
The separate byhand group had **four passed and one failed**. This attempt is
preserved and is not counted as acceptance.

- Two D2H delivery tests got only single-event batches. Collection of `test_xtc`
  imported `run_loop_callback`, whose top-level assignment set
  `PS_SMD_N_EVENTS=1` for the whole process. Move that assignment to the helper's
  CLI entry point; the serial callback test now scopes it with `monkeypatch`.
  The CLI/byhand multi-EB behavior remains the same. This explains why the two
  historical Stage 2 failures passed in isolation.
- The GPU task setup test expected batch size 20 while omitting `batch_size`.
  MFX intentionally defaults to one. Request 20 explicitly, retaining the
  assertion and the test's dense-input/constant setup purpose. No default policy
  changes.
- The smalldata test passed its file-data portion but received no shared-memory
  events, so `/oneint` was missing. Preserve the failure and retry with full
  system access; do not classify it as the historical MODE/SRV-core error.
- The byhand MPI test stopped in `user_loops.py` during a calibration HTTP lookup:
  the remote end closed the connection without a response (`RemoteDisconnected`). The other four byhand tests passed. Retry the
  group without skipping calibration or changing assertions.

The historical BeginStep constant failures and MPI-lookahead attribute failure
from Stage 2 did not reproduce on this integrated runtime. The original Stage 2
triage remains [recorded](issue159_stage2_20261019.md); this is current evidence,
not a claim that all historical failures were alignment defects.

### PR260 review clarification: preserved calibration traceback

Claude's review of `3f6b8d1ab` found no blockers and requested the source of the
HTTP failure attribution. The traceback is in the original
`attempt01/job-42394606/byhand.log`, under **Captured stderr call**, lines 70–177.
Its SHA-256 still matches the value committed before review:
`e26b78c0fff1d7f6678fe9229978ceddbf836b007453a4f8197ce21dcc0ba9ab`.
The log contains one NUL byte, so default `rg` treats it as binary. Use text mode:

```bash
stage6_artifacts=/sdf/scratch/users/m/monarin/gpu-validation/issue159-stage6-20261029
rg -a -n -C 3 'Captured stderr|RemoteDisconnected|_setup_run_calibconst|ConnectionError' "$stage6_artifacts/attempt01/job-42394606/byhand.log"
```

Selected original lines (intervening lines omitted):

```text
70: ----------------------------- Captured stderr call -----------------------------
95: http.client.RemoteDisconnected: Remote end closed connection without response
132:   File "/sdf/scratch/users/m/monarin/gpu-validation/issue159-stage6-20261029/attempt01/source/psana/psana/tests/user_loops.py", line 72, in <module>
138:     self._setup_run_calibconst()
140:     calib_const = wu.calib_constants_all_types(
154:   File "/sdf/scratch/users/m/monarin/gpu-validation/issue159-target-merge-review-20261028-r1/trial/install/lib/python3.11/site-packages/psana/pscalib/calib/MDBWebUtils.py", line 292, in find_docs
177: requests.exceptions.ConnectionError: ('Connection aborted.', RemoteDisconnected('Remote end closed connection without response'))
```

The pytest failure summary contains only the subprocess `CalledProcessError`;
the captured stderr in the full log supplies the calibration lookup traceback.
This establishes a connection closure during that lookup, not its underlying
cause or a proven TCP reset. The evidence JSON now records the locator and
excerpt. The original log is unchanged; no new test run is needed for this
clarification. The unexplained smalldata failure remains separately described.

## New validation results

Job **42396585**, `sdfampere002`, completed **0:0** in **9m28s** at
`71f5c04a2`, on A100-SXM4-40GB with driver 575.57.08.

| Group | Result | Harness elapsed |
| --- | --- | --- |
| Required core suite | **1,646 passed, five skipped, seven deselected**; includes all **555 GPU integration cases**, no GPU skips/failures | 185.15 s |
| Required byhand/MPI suite, fresh process | **Five passed**, no skips/errors/failures | 157.68 s |
| Explicit slow real-Jungfrau field/pixel suite | **Five passed**, no skips/errors/failures; 13 events per case | 221.60 s |

All 3,723 frozen source/install hashes matched before and after the run. Origin
audits were clean for 38 core processes, 78 byhand/MPI processes and one pixel
process. The [machine-readable evidence](evidence/issue159_stage6_20261029.json)
records commands, summaries, environment, device identity, hashes and the first
attempt's failures.

The core skips are unchanged: missing calibration pickle, missing optional
Roentdek library, the existing timetool calibration skip, and two explicitly
skipped serial callback/timestamp-filter tests. They are not alignment/device
acceptance. The repository's default `-m "not slow"` deselects five Jungfrau
cases plus live transfer and shared-memory calibration-prefetch tests. The five
Jungfrau cases were explicitly run with `-m slow`; the other two unrelated slow
tests remain unrun, outside the default required groups and #159's focused scope.
The pixel invocation deselects one non-slow gain-mode reference test, which
already passed in the core group.

All four first-attempt core failures and the byhand calibration-connection failure
passed in the retry. No smalldata or HTTP runtime/test code was changed; the
no-event shared-memory failure's underlying cause is not established by this
retry. Preserve it as a validation limitation rather than claiming a fix.
The test-only corrections pass within the complete collection/order that exposed
them; CLI callback behavior also passes the byhand serial/multi-EB checks.

Stage 6's technical checks are complete and ready for independent review. The
branch is not yet merged; no Stage 7 run or final feature PR is included.

## Reproducibility

Artifacts are under
`/sdf/scratch/users/m/monarin/gpu-validation/issue159-stage6-20261029/`.
`attempt01` freezes the base and `attempt02` freezes `71f5c04a2` using `git archive`.
Each attempt includes source/environment manifests, 3,723 source/install hashes,
commands, pytest XML/logs, and module/native-origin audits for Python processes
including subprocess/MPI descendants. Frozen input hashes are checked before
and after the run. The independent native/non-GPU installation remains at its
original path:
`/sdf/scratch/users/m/monarin/gpu-validation/issue159-target-merge-review-20261028-r1/trial/install`.

The frozen GPU package/tests overlay that unchanged installation. Relevant
non-GPU source and native inputs are unchanged from the validated integration;
no copied/relocated native library or new dependency is used. Python is the
KvikIO 26.10 validation venv; CuPy 14.0.1, KvikIO 26.10.0, NumPy 2.3.5, mpi4py 4.1.1, h5py 3.16.0 and
pytest 9.1.1. The harness uses MPI mode with OB1,
disables the smcuda transport, respects the Slurm GPU mask, and launches MPI
locally inside the allocation. KvikIO compatibility mode is ON. No performance
numbers are collected in these correctness jobs.

## Review and next steps

Review `a4293d71f..HEAD`: confirm the test isolation fixes preserve CLI behavior
and explicit batch assertions; inspect the complete first-attempt failures and
final dispositions; check the coverage/revision mapping and qualified reuse.
No production or performance claim should be inferred from a skip. No new cold
A/B or A/A is justified by these test-only changes. Stage 7 will consolidate the
accepted performance evidence under the revised plan's change triggers.

[#256](https://github.com/slac-lcls/lcls2/issues/256) retains field-copy retirement
wait/accumulation work; [#257](https://github.com/slac-lcls/lcls2/issues/257) retains
per-event API policy and optional copy optimization. These accepted deferrals
remain separate from this correctness step. Do not merge the final feature PR
or close #159 as part of Stage 6.
