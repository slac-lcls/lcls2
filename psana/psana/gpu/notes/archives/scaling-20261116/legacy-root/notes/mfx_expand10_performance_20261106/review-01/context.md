# Independent review: sample10 performance preparation

Base: d91085f653c54940ee5eb63462efab7aaced1617.
Review exact commit: 5727e727fd18de513988acaba66251a681dda25c.
Checkout: /tmp/mfx-performance-review-20261106-r1 (detached).

User approved preparation for the 10k single-node campaign and explicitly
requires implementation/review/response/re-review in this session. Scope is
notes/mfx_expand10_performance_20261106 plus the proposal's follow-up paragraph.
Read scope.md, README.md and all new Python files/run.sbatch. No psana core,
calibration DB, source fixture or reviewed kernel changes. No 10k performance
campaign yet. One-GPU 1k smoke and small controlled arithmetic tests are in scope.

Focus on whether the preparation runner measures completed GPU work, correctly
rounds/excludes warmup, avoids per-event D2H and diagnostic hot-loop timers in
throughput mode, safely retains input/output owners, and preserves arithmetic
and bin semantics. Validate that profile wrappers do not alter lifetime/stream
behavior, do not include generator suspension, and restore original methods.
Check count-based rejection, failure reporting, bounded optional export, fixture
provenance and runtime fingerprinting. Challenge unsupported measurement claims.

Project guidance: KvikIO submission is host-side; the installed runtime uses
compatibility mode ON/CPU fallback, not GDS. Multiple CUDA streams may overlap;
CUDA interval sums and nested CPU totals are not loop wall time. All borrowed
arrays must be kept alive until stream/lease completion. Persistent cube sums
belong to each actual execution stream; explicit finalization occurs after EOF.
Do not assume a managed GpuTask lifecycle or MPI support. Nominal batch size is
not actual callback size. Cube output is 64 MiB/bin/stream, independently of the
framework's byte budget. Sim10 reads seed payloads through field descriptors,
not a dense seed gather; normal `jungfrau.raw` input does use a dense gather.

Dependencies to inspect as needed: notes/mfx_expand10_capacity_20261106/
dense_cube.py and measure.py; notes/mfx_expand10_stage3a_20261105/cube.py;
notes/mfx_expand10_stage2_20261105/adapter.py, gpu_accept.py, runtime.sh;
psana/psana/gpu/examples/jungfrau_calibration.py; current gpu_events.py,
gpu_stream.py, gpu_task_batch.py and gpudgram/batch.py. The executed psana is a
frozen B/install runtime, not a fresh rebuild of this checkout. Its paths/hashes
are recorded; coordinator also checked the frozen `_trim_gpu_caches` callsite
is only the admission-failure path, as in this checkout.

Validation evidence committed in this stage:
- Seven host tests passed, including callback-boundary rounding/tail inclusion,
  failed join/incomplete run rejection, no-CUDA-event disabled recorder,
  suppressed publications with keepalive, generator timing/close, hook restore,
  10k receipt/size/full-hash mismatch detection. job-43548178.err holds the log.
- Four controlled GPU cases (profile on/off, invalid panel on/off) passed exact
  per-bank sum comparison and count rejection: device-checks.json.
- Slurm 43548178 completed exit 0 in 46s, sdfampere035 A100-SXM4-40GB,
  account lcls:data@ampere, normal QOS. One-stream and two-stream clean 1k runs
  plus two-stream profile all passed 1000 event/bin counts; 100 warmup/900 steady.
  Two banks handled 501/499 events, 158 callbacks, sizes 1,2,5,6,7.
- validation-summary.json, profile-summary.json, slurm-accounting.txt,
  execution-source-hashes.json and artifacts.json. Coordinator rehashed every
  executed Python source reported by all three runs; all match this commit.
- Coordinator ran check_10k against the actual project-scratch fixture: expected
  generation SHA c45d191bc92bab1036caf24a4d92039aebe2036d00fd11cbd562da0f84551bcf,
  10000 unique ordered events and all 20 file sizes matched. No fresh 37.7GB
  file rehash in this step; accepted Stage1 file hashes are explicitly historical.
- All commit hooks passed including Python AST and whitespace.

Full reports and intervals are copied/hash-checked into this review checkout's
ignored build/preparation-evidence/ (copy-index.json). These are evidence, not
new source. They avoid needing read access outside the detached checkout.

Known limitations: this is preparation/smoke, not stable throughput or 10k GPU
numerical acceptance. Earlier full-pixel 1k correctness is the kernel baseline;
new checks cover counts plus small controlled sums. No Nsight capture yet; bin
H2D and null-stream waits are included in larger spans, not separately timed.
Metadata/parser CUDA spans include host gaps and waits, not pure kernel costs.
142 steady profile cache trims followed admission pressure with the inherited
4GiB framework budget; tuning is deferred to the next campaign. Throughput
mode intentionally has no I/O-delta monkeypatch. No new per-pixel science claim.

Preserve failure history: job43548142 failed the controlled profile test because
Recorder.device_call's `stream` parameter collided with a forwarded `stream=`
kernel keyword. The parameter is now execution_stream and a host regression
assertion covers forwarding. The passing job above reran all device/native tests
after that fix. Earlier queue-switch attempts 43547849/43547876 were canceled
or collided with the new-output guard; no acceptance is based on them.

Review independently using only your restricted read tools. Do not run tests,
launch jobs, access MCP, edit files or delegate. Report unavailable reads as
limitations, not passes. No prior finding IDs. Give stable IDs for findings and
review the latest implementation, with explicit reasons for any required fixes.
