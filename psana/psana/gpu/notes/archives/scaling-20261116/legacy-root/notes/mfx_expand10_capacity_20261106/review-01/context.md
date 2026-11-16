# Dense capacity and profiling review

Review exact commit supplied by helper; base 60b69ff08. Read scope.md, README.md,
dense_cube.py, measure.py, validate.py, runner.py, test_dense_cube.py, run.sbatch,
status.json and validation-summary.json under notes/mfx_expand10_capacity_20261106.
All three full GPU reports, controlled test report, seed-cache manifest and log
are copied INSIDE this checkout in build/capacity-review-evidence/ with a copy
index. Original scratch paths are provenance only outside the read sandbox.

User authorized configurable bin/state limits, bounded final export, 30/100/200
bins on existing 1000 events with one A100/two streams, correctness/memory/phase
timings and independent review. No runtime core, XTC, DB, cache or distributed
routing changes. New Cube derives from Stage3A Cube: init, profiling callback and
finish_to are new; _identify/_bank/add/close and all count/sum kernels inherited
unchanged. Stage2 adapter/calibration unchanged. Review those dependencies as
needed. Ordinary callable GpuTask, explicit finish after EOF, no managed hooks.

Check integer budget admission and actual free-memory preflight; bank ownership
before async fills, max two streams, repeated bins and duplicate identities;
profiling wrappers preserve original kernel arguments/lifetimes and do not add
hot-path synchronizations; CUDA events owned through completion; measured read
stats aggregate across resets without duplicate counts. Timings are intervals,
not exclusive kernel/IO time or sustained throughput; calibration interval can
include setup/host gaps and first stage/stream invocation is excluded only from
the labelled warm summary. KvikIO compatibility ON and fixture caches warmed.

finish_to must never allocate the full host cube: reusable float64 one-bin and
float32 transfer arrays plus small counts; join then transfer/merge/fsync each
file, write success manifest last; idempotence and failed/incomplete export
semantics. No GPU bank reuse until work completes. Tests cover third-stream
rejection without mutation (prior F-008 gap), >30 bins, exact budget edge,
repeated/invalid event accumulation, reusable output array, partial sink failure,
run isolation and abort. Full native validation covers all 30/100/200-bin pixels
against independent CPU-v3 calibration from CPU-read seeds and exact counts.
CPU verification keeps only one bin's reference/absolute sums and panel-sized
error/bound scratch; same gamma_n*sum(abs(input))+1e-5 bound as Stage3A.

Node-local full outputs are temporary: all bins written/read back, 3 bins per
case copied/hash-checked to shared scratch, all pixel/file hashes retained;
local generated outputs deleted only after pass. No old artifacts deleted.
Scope needs bounded host memory as well as GPU memory; sampled driver/pool
measurements and VmHWM are labelled accurately. The run does not establish
35kHz, native multi-step, MPI, actual GDS or arbitrary scientific bin mapping.

Coordinator executed shell/host tests, one-A100 campaign and artifact/hash checks.
Reviewer only reads files, must not claim to execute tests/hashes or inspect
arrays not provided. Exact source+evidence hashes bind to the reviewed commit.
No delegation or recursive review loop. Max three rounds; start finding IDs at
CAP-001. Return exact SHA, verdict and stable findings, distinguish blockers from
optional future hardening beyond this fixed campaign.

Recovery evidence: first job 43533593 (sdfampere034) passed 30/100 then was
PREEMPTED during 200-bin CPU validation. Its incomplete 200 report is not accepted.
Second job 43533968 (sdfampere015), resume200.sbatch, reruns only 200 with fresh
seed cache and same runner/core/source hashes. Read interruption.json and the
second launcher.json/copy index. This is not a same-node scaling comparison.
Review summarize.py's explicit recovery selection and provenance checks too.
