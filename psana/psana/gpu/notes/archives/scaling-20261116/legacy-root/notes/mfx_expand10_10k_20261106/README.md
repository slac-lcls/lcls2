# 10,000-event sample10 fixture for single-node measurements

Generation and full CPU validation passed November 6, 2026 in Slurm job
43538554 on sdfmilan271 (COMPLETED, exit 0, 56m46s). Independent Claude review
passed implementation commit `738058473bf73601fd978c67179f2e9d7efff05b` with no
blocking findings. See [review closure](review-01/response.md), status.json and
validation-summary.json for the verdict, limitations and results.

Fixture:
`/sdf/data/lcls/ds/prj/prjdatpsana/scratch/synthetic_mfx/sample10_10k/`

Full evidence:
`/sdf/data/lcls/ds/prj/prjdatpsana/scratch/synthetic_mfx/sample10_10k_evidence_20261106/`

The first 10,000 unique events from public mfx101555026 run 10 retain their
original timestamps, five JF-bearing files, five ancillary files, and all 32
physical panel identities. Native one-step structure is retained. Only the
truncated end needs new timestamps for source-derived Disable/EndStep/EndRun
datagrams. No internal steps or repeated events are inserted. The exact planned
size is 37,682,986,080 bytes (37.683 GB / about 35.1 GiB), excluding JSON evidence;
all actual file sizes matched this prediction exactly.
There are 10,091 datagrams per stream, including transitions and SlowUpdates.

This is project scratch, with source experiment/run filenames retained and
`dir=` required; it is not registered as a new prjdatpsana run. The directory
inherits project group ACLs. A partial directory is not an accepted fixture:
require evidence/status.json with phase complete and accepted true, plus passing
summary/verification reports. The controller marks acceptance after full readback.
Existing output directories are rejected, not overwritten or silently resumed.
On preemption, time-limit termination or SIGKILL, the phase can remain
`generating` or `verifying` with accepted false: check Slurm state and the job
log to distinguish an active job from an interrupted one. Such a directory is
not accepted. The controller does not install signal handlers or auto-resume.

The writer and verifier are frozen copies of the reviewed Stage 2 tools with
the bounded changes described in scope.md. Historical Stage 2 hashes remain
valid. Indexed DataSource seed witnesses avoid quadratic verification work
without removing per-panel comparisons. Progress logs and atomic phase
checkpoints identify interrupted work. CPU Slurm execution uses the same frozen
psana runtime as the 1k campaign, not a fresh rebuild of the current branch.

```bash
stage=notes/mfx_expand10_10k_20261106
runtime=notes/mfx_expand10_stage2_20261105/runtime.sh
bash "$runtime" "$stage/test_fixture.py"
bash "$runtime" "$stage/campaign.py" preflight
sbatch --output="$PWD/$stage/job-%j.log" "$stage/run.sbatch" "$PWD/$stage"
```

Use campaign.py/run.sbatch for the project-shared fixture. The campaign sets
umask 0007 before creation; the low-level fixture.py standalone entry point
does not set it, so callers must arrange the intended permissions themselves.
The accepted job's ACLs are recorded in access.txt. Standalone permission
generalization is deferred; it is not required to use the generated fixture.

Preflight checks source identities, exact per-stream output sizes, available
filesystem space plus 5 GiB headroom, and SHA256/shape/size of frozen pedestals,
pixel_gain and pixel_offset. Project quota was not exposed by the inspected
quota command; filesystem free space is not an allocation promise. Successful
write/fsync/readback is required before acceptance.

Acceptance checks all 20 file hashes, every retained datagram, Names/schema,
SMD references, unchanged ancillary bytes and JF scalars, source-strided seeds
and CPU expansion, timestamps and transitions across ten streams. Serial
DataSource must yield one run/step, exactly 10,000 ordered original events, and
all 320,000 panel-event seed hashes matching independent source readback.
PS_FAKESTEP_FLAG=0, PS_PARALLEL=none, live=False, skip_calib_load=all.
Calibration files are checked separately; no DB writes or live lookup claim.

All acceptance checks passed: 20 file hashes, ten aligned timelines, one native
step with 10,000 unique original events, and 320,000 panel-event source/CPU and
DataSource seed comparisons. All ten focused tests passed. Frozen full-panel
pedestals, gain and offset payload hashes passed. Full verification.json is
150 MiB in project scratch; its SHA256 and compact results are in the committed
validation-summary.json. files.json records the 20 file paths, sizes and hashes.
The coordinator checked both report hashes and all executed source hashes.

This CPU-only stage does not establish GPU correctness on 10k, GPU throughput,
GDS, MPI scaling or cold-cache I/O. The earlier 1k GPU/cube checks remain the
numerical baseline. The old GPU preflight is still capped at 1,000; the next
performance-runner stage must explicitly support this fixture/provenance.
Single-node timing/scaling follows; the repeated 400k dataset remains deferred.
