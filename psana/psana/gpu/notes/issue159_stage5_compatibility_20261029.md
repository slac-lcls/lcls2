# Issue159 Stage 5: field copy-accessor compatibility

Stage 5 adds focused accessor regressions and usage guidance. Production runtime
is unchanged. The separate branch `test/issue159-stage5-accessors` starts at
`1a3efd86b`, the merged PR258 plan; tests and user guidance are committed at
`4a4df550b`. This step requires independent review before merge into
`fix/gpu-unaligned-reads-159`. Stages 6–8 and the final feature-target PR remain
separate work.

## Audit and coverage

Both `GpuFieldResult` copy properties already use `on_gpu_view()` / `_slot_views()`.
Actual-address alignment and lazy aligned field materialization therefore cover
both properties. No direct-byte D2H optimization or runtime repair was needed.

| Contract | Existing evidence reused | Stage 5 addition |
| --- | --- | --- |
| Actual-address alignment | Full ten-type, offset-residue, scalar/array/empty borrowed-view matrix; shifted bases | Real copy-property checks for uint16, uint32 and float64; scalar, 2D and empty shapes; aligned, shifted-unaligned and shifted-but-aligned addresses |
| Exact values and segment identity | `test_gpu_input.py`; real Jungfrau/scalar CPU fields in `test_pixel_exact.py` | Exact payload bytes through both copy properties; GPU ragged/scalar mapping with an absent segment |
| CPU cache/independence | Aligned real-data CPU snapshots usable after input retirement | Mutation isolation, cached object identity, all-segment versus selected-segment host caches in either access order, shared aligned device copy |
| GPU independence | Borrowed-view tests do not establish this contract; other result classes are distinct | Two independent outputs, mutation isolation from input/cache/other output, live caller outputs with no remaining field-copy charge after safe retirement |
| Copy counts | View alignment-copy count/cache tests | Six CPU cases: both properties on aligned, first-unaligned and primed-unaligned inputs; repeated CPU access adds no transfer, GPU access always copies |
| Empty/absent and retirement | View empty-resource and root/child lifetime/failure tests | Empty/absent properties at zero alignment budget; first access after retirement rejects for both properties; cached CPU and owned GPU outputs survive input reuse |
| Stream readiness | Delayed producer/copy/view consumers and second-stream view cache tests | Both copy properties on a nondefault current stream after a delayed producer, aligned and unaligned; CPU result checked without an explicit host wait |

New cases: **16 CPU**, appended to `unit/test_gpu_field_views.py`, and **60 CUDA**
in `integration/test_gpu_field_accessors_device.py`. Logical copy counts use the
injected CPU allocator; CUDA cases verify actual arrays, values, alignment,
mutation isolation, stream behavior and rounded allocation charges. They do not
require CuPy to implement an independent copy with a particular memcpy API.

Repository caller inspection found field `.on_cpu` in real-data pixel checks,
MPI input/publication checks and input lifecycle/ownership tests. The main GPU
README uses published task output `.on_cpu`, a different result class. This is a
bounded compatibility inventory, not evidence that external field consumers do
not exist; the long-term consumer/API inventory remains #257.

The parser guide now gives the per-segment copy table, mutable host cache
semantics, stream ordering, caller-owned GPU memory costs, absent/empty behavior
and retirement contract. Both user guides direct routine detector processing to
batched tasks. APIs remain available; there is no removal, rename or deprecation.

## Validation and provenance

- CPU: **393 passed**, no failures/errors/skips, on `sdfiana025` (6.43 seconds).
  The six focused files cover fields, input-copy lifetime, input windows,
  retirement, input APIs and memory reporting. The previous 377 cases plus the
  16 new cases passed together.
- Initial GPU job **42390626**, `sdfampere014`: **60 passed** normally. With
  default sanitizer backtraces, 53 passed and seven budget-release assertions
  failed; memcheck reported **zero device-memory errors**. This attempt is not
  counted as sanitizer acceptance. See the diagnosis below.
- Final job **42391066**, `sdfampere002`, A100-SXM4-40GB, completed **0:0**
  in **16 seconds**: **60 CUDA + 60 memcheck passed**, zero failures/errors/skips
  and **zero sanitizer memory errors**. Device tests took 6.63 seconds and
  memcheck 5.88 seconds. CuPy 14.0.1; driver 575.57.08.
- Loaded module/library origin audits were clean for CPU, CUDA and memcheck;
  988 file hashes verified before and after each accepted run. The standalone
  harness emitted one pre-existing unregistered `gpu` marker warning; no test
  was skipped or filtered because of it.
- [Machine-readable evidence](evidence/issue159_stage5_20261029.json) preserves
  exact commands, counts, source hashes, origin summaries and the initial failure.

Artifacts:

- `/sdf/scratch/users/m/monarin/gpu-validation/issue159-stage5-20261029/attempt01`
  contains CPU evidence, the initial GPU failure, and diagnostic runs.
- `/sdf/scratch/users/m/monarin/gpu-validation/issue159-stage5-20261029/attempt02`
  freezes committed tests/runtime at `4a4df550b` for final validation.
- Existing native/non-GPU install:
  `/sdf/scratch/users/m/monarin/gpu-validation/issue159-target-merge-review-20261028-r1/trial/install`.
- Python:
  `/sdf/scratch/users/m/monarin/gpu-validation/kvikio-2610-validation-20261015-r1/venv/bin/python`.

The harness verifies 988 frozen/install file hashes and audits loaded module and
native-library origins. Test files in both attempts are byte-identical to
`4a4df550b`; the first attempt was frozen before that commit. No native rebuild
or dependency change is involved. CPU evidence is reused across attempts because
the test/runtime bytes are unchanged. Saved status JSON includes exact commands,
counts and elapsed times; metadata includes source identity, environment and
file hashes.

## Sanitizer backtrace retention diagnosis

The seven initial failures demanded zero field-copy charge after retirement
while independent GPU outputs remained alive. These assertions pass normally.
Under default CUDA 12.8 Compute Sanitizer, a retired aligned array remained
reachable through a `GpuFieldData` mapping in a completed `on_gpu` Python frame,
even after GC and dropping the independent output. The lease had already cleared
its retained-copy/cache references. Repeated copies reproduced the retention;
it was not confined to first-use compilation.

Diagnostic jobs 42390755 / 42390831 / 42390891 recorded the reference chain.
Job **42390945** compared normal execution and sanitizer backtrace modes on the
same node/process environment: native, `--show-backtrace device`, and
`--show-backtrace no` all released each copy immediately, including repeated
calls. This isolates host-backtrace collection as the trigger in this environment;
it does not establish a general diagnosis for every sanitizer version.

The final harness uses the documented `--show-backtrace device` setting. Device
backtraces, memcheck and `--error-exitcode 99` remain enabled; only host backtraces
are omitted. See NVIDIA's [backtrace option documentation](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html#command-line-options).
No runtime code, allocator behavior, GC policy or test assertion was changed to
work around the retention. This instrumentation limitation is recorded rather
than counting the initial failing run as a pass.

## Performance and remaining work

No production Python/CUDA/native code changed, so the accepted batched cold A/B
and field-view measurements remain applicable. No new performance campaign or
MPI rerun was launched. This stage does not claim new end-to-end throughput or
repeat the real-data acceptance campaign.

#256 retains the field-copy retirement/slot accumulation follow-up; #257 retains
API policy and optional copy optimization; #247 remains the separate host scalar
transport issue. After this Stage 5 PR is independently reviewed and merged,
reconcile the remaining Stage 6 correctness requirements, then prepare final
performance/provenance documentation and the feature-target PR. #159 stays open.

## Review prompt

Review `1a3efd86b..HEAD` on this branch. Confirm the added tests exercise
`GpuFieldResult`, not only borrowed views or task outputs; check that coverage is
bounded, copy/cache/stream/retirement expectations match the current APIs, and
user guidance distinguishes batched task output from per-event field copies.
Review the sanitizer failure and diagnostic evidence before accepting the
backtrace-mode remedy. Verify runtime is unchanged and accepted performance
reuse is justified. Report remaining blockers before merging this step; do not
start Stage 6 or merge the final feature-target fix as part of this review.
