# Runtime driver round2

Review exact95ea0c4ed0fed477c2d42076445e48c5f315c862 after round1
50eabba86f550b14fa7b1d38b1b225265ab96a3a changes_requested. Same public repository
and user scope, restricted file reads only, no raw data/credentials or nested
reviews. Read notes/pr_regression_runtimes_20261117 under psana/psana/gpu:
scope, review-01, response-01, validation-02, capture-01.sbatch and build-validation-01.
Read scripts/performance/regression prepare_runtime.py, native_contract.py,
test_prepare_runtime.py, test_native.py, RUNTIME.md. Review all fixes and regressions.

RUN-001: actual allocation/Slurm/GPU records before srun, EXIT accounting on
failure too; agent will additionally collect sacct after job completion, because
trap observes still-running batch job. Manifest includes step, GPU UUID/model and
CUDA visibility. RUN-002: GPU module namespace and mapped-library CPU guard,
with failure tests. RUN-003: environment key names plus hash, no secret values.
RUN-004: exact coverage and exclusions documented; no full-dependency-closure claim.
RUN-005: all post-mkdir operations guarded; early errors retained; parent rechecks
build and child association; fsynced inode exclusively hard-linked to output,
never overwrite or publish partial JSON. RUN-006: same-manifest A/A; separate,
nonnested install/source and rooted package/pythonpath for different revisions;
mixed-policy/psalg/native-library/nested and mutation tests. RUN-007: literal
code hashes in launcher, pre-submission Git/blob readback planned.

70 offline tests PASS, bash syntax and diff check PASS. Real build passed under
initial driver (build logic unchanged); capture not yet run. No job submitted.
Review whether blockers are resolved and bounded one-rank GPU validation can run.
Actual result/evidence review planned as round3. No claims of dataset/6-case/native
qualification. Shared dependency environment stays fixed; default profile and
archived campaigns unchanged. Stable IDs, exact reviewed SHA required.
