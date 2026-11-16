# Prelaunch independent review

User authorization: use the implementation/review loop; submit one diagnostic
job only after a pass. Codex coordinates; Claude reviews independently, with
three rounds maximum and a 900-second timeout per invocation. No git commit,
branch, push, core psana edit, runtime rebuild, or input cleanup is authorized.

## Identity exception

The diagnostic is uncommitted and has an explicit frozen dependency manifest.
The skill's clean-commit helper cannot honestly review these files as HEAD.
Use a task-local snapshot adapter retaining the helper's restricted read-only
Claude configuration, schema validation, timeout, and before/after integrity
checks, but identifying the candidate by its manifest SHA256. This is NOT a
review of a clean git commit. The frozen source is separate from the mutable
implementation checkout; no additional copied tree or git object is needed.

## Required inspection

Review all files in source/notes/sim10_storage_gpu_20261114, especially launch,
runner, storage, audit, probe, plan, prepare and run.sbatch. Follow the frozen
baseline support/cpu_plan/hardware/task_evidence/runtime helpers and the pinned
psana source as needed. Inspect supplied test evidence, not just this summary.
Determine prelaunch correctness, experimental validity, scheduler/resource
consistency, executable control flow, fail-closed gates, cleanup ownership and
process lifetime safety. Report specific blocking findings with stable IDs.
Do not execute commands, edit anything, delegate, submit jobs, or start a loop.

## Experiment and acceptance

Fixed 4 total BDs, 4 KvikIO workers per BD and one stream per BD. Paired cold
host-page-cache Weka FFB and NVMe, GPUs 1/2/4 (BDs/GPU 4/2/1). Global full-run
event count 100000; batch200, budget6GiB, bulk512KiB, task1MiB, bounce16MiB,
SMD batch1000. Compatibility ON, auto-direct OFF, no GDS or cube/calibration
kernels. Pinned PR298 runtime, not the implementation checkout HEAD.

Six traced pilots, 18 full measurements, six separate Weka/1GPU anchors;
three repeats with rotated GPU/media order. Two bounded CPU slab probes are
not descriptor-aligned event replay and do not prove storage ceilings. Compare
same-media same-repeat baselines and preserve all evidence before interpretation.
Check rank masking before imports, actual peers/headroom on each physical GPU,
exact global event coverage/bytes/reads, coldness after DataSource setup,
physical-leaf accounting without double counting, NIC evidence, timing windows,
GPU/CPU telemetry and clean drains. No speedup is required for scientific success.

Request: one exclusive sdfampere036, all four A100s, six tasks x six physical
CPUs, hint nomultithread, 192G RAM, three hours. No automatic node fallback.
Historical hardware: two Samsung MZQL27T6HBLA-00A07 7.68TB drives in md0 RAID0,
PCIe4x4/NUMA5, EPYC7542. Live topology/free-space qualification must run inside
Slurm; no direct compute-node PAM bypass. Input stages stay private on FFB and
local NVMe; do not stage 336GB inputs on shared scratch (only ~7.9GiB quota free).

## Evidence and limitations

CPU-only pytest: 42 passed, zero failures/skips, 0.79 seconds. The round context
links its JUnit XML; AST/bash parsing and source/native/KvikIO hash validation
also passed during preparation. No core changes, so full psana suites were not
run. No live multi-GPU/NVMe execution has occurred. This gate approves only
launching the diagnostic; allocation-time 31 inherited GPU tests per GPU,
storage qualification, six pilots and per-measurement I/O/coldness audits are
deliberately inside the authorized job. Their absence is a limitation of a
prelaunch review, not a claim of hardware acceptance. Missing CPU evidence or
missing required source prevents a pass. Assess whether the future gates work.

At 2026-11-14 11:16 PST, 036 was IDLE/no allocated TRES, 018 had all GPUs in
use, 017 was drained. Two test-only estimates predicted Nov30, not immediate
eligibility; cause unknown. Refresh before submission and report pending honestly.
Pass only with no unresolved blocking findings on the exact latest manifest;
review failure/timeout/authentication/denied source reads are never approval.

## Round 1 evidence

Candidate: /sdf/scratch/users/m/monarin/gpu-validation/sim10-storage-gpu-20261114/attempt01
Manifest SHA256: 0492701e59828404a84f9d29a4c37b6a9d4054e994e322ad7676d748d431c45e
JUnit: /sdf/scratch/users/m/monarin/gpu-validation/sim10-storage-gpu-20261114/host-validation01/junit-final.xml
The manifest is appended to this prompt. Work from its source/ directory;
the pinned native receipt's Python source directory and host-validation01
are allowed additional read roots. No prior findings.
