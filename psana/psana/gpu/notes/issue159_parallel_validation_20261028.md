# Issue159: review and integration validation in parallel

The user authorized independent review alongside validation on October 28.
PR246 stays open; the main fix branch is unchanged. No Stage 5 implementation
or PR merge is authorized by this validation work. The reviewer receives the
[full prompt](issue159_review_prompt_20261028.md), covering cumulative Stages
1–4 before reviewing the dependency integration.

## Review follow-up update

Claude completed the initial review at `e7c700ba2` without blockers. The
[small follow-up fixes](issue159_review_followup_20261028.md) passed 377 CPU
tests and 302 CUDA tests plus 302 memcheck cases with zero sanitizer errors.
The new empty-field guard changes the B candidate: freeze the reviewed follow-up
head and verify a separate install before timing. Earlier results below remain
at their stated revisions. Claude's follow-up review accepted all six dispositions. Mona authorized the
[matched cold screen](issue159_pr246_cold_20261028.md), submitted as job 42255812
with fresh independent builds at A `8cf05e86a` and B `405f868ed`. Results, the
separate field-view benchmark, PR merge and Stage 5 remain pending. The scalar transport issue is tracked as
[#247](https://github.com/slac-lcls/lcls2/issues/247).

## MPI/reporting completion update

Job **42250696** on `sdfampere001` passed all 16 combined field/reporting
cases and two real Jungfrau pixel comparisons (20 events each). See the
[MPI report](issue159_mpi_reporting_20261028.md) for the exact coverage, new
test driver, and inherited host scalar-shape limitation. The focused synthetic
scalar check starts at the GPU-manager input boundary; the real-data pixel
checks use unchanged calibration setup. No production code changed.

The planning sections below describe the original coverage plan. Combined
MPI/reporting is now complete within that stated scope. Independent review and
the matched cold/field performance screen remain pending.

## Fixed code under test

- Review baseline: main fix `87bce00cb`.
- Upstream target: `8cf05e86a` (final PR187 + PR208).
- Integration runtime merge: `31c4c2f2`; initial PR head `f2b3c2d4e`.
- Reused independent installation:
  `/sdf/scratch/users/m/monarin/gpu-validation/issue159-target-merge-review-20261028-r1/trial/install`.
- All production, test, and build inputs at the PR head match validated trial
  tree `88d0fd3f` exactly. Only notes differ. Do not copy native installations:
  preserve their original paths and check mapped-library origins.
- Earlier same-code validation: 990 CPU tests and three CPU accounting probes.
  These are not evidence of CUDA/MPI completion.

## Parallel work and order

| Work | Action | Exit condition |
| --- | --- | --- |
| Independent review | Review Stages 1–4 and then PR246 using the prompt | Separate verdicts, concrete findings resolved or explicitly accepted |
| Focused device checks | Existing actual-address gather, field-view, lease, ownership, input-window and raw-preparation tests; selected memcheck | Exact values, no premature releases, no failures/skips, zero sanitizer errors |
| MPI + reporting | Add the missing combined field/constant coverage; run existing pixel/placement/close drivers where applicable | Correct roles/device, no hang, correct budgets/reporting and safe lifetime through close |
| Matched cold screen | Updated target A versus integrated Stage4 B, separate verified installs, 4 BD / 1 GPU | Correctness and cache gates pass; complete samples with rate/loop sample SD and reviewed regressions |
| Stage 5 | New implementation PR after this integration is reviewed, accepted and merged | Remains pending |

Device checks can run while the reviewer reads the code. Prepare MPI and timing
harnesses in parallel, but do not treat a failing correctness run as permission
to proceed to timing. A runtime change from review requires revalidation of its
affected cases and a newly identified performance candidate.

## Device results and preserved attempts

Job **42247426** on `sdfampere004` passed 318 tests and failed 15 before those
cases reached their assertions. Every failure was `FileNotFoundError` for
`tests/test_data/chunking/xpptut15-r0014-s000-c000.xtc2`: the frozen harness copied
GPU test code but omitted this committed 51 KiB fixture. Sanitizer did not run.
This is a harness packaging failure, not device acceptance or an observed
runtime defect. Preserve its logs unchanged at:
`/sdf/scratch/users/m/monarin/gpu-validation/issue159/pr246-20261028-r1`.

Retry **42247538** adds the committed chunking fixture and hashes it with the
other inputs. Runtime, selected tests, and assertions are unchanged. Artifact
root:
`/sdf/scratch/users/m/monarin/gpu-validation/issue159/pr246-20261028-r2`.

Job 42247538 passed **333 GPU tests with no failures/skips**. Its sanitizer
assertions passed 315 cases, but the process exited 99 after ten UCX
`CUDA_ERROR_INVALID_CONTEXT` errors during incidental MPI import. This matches
the already documented environment issue from Stage 3: the new harness omitted
`OMPI_MCA_pml=ob1`. It is not accepted as a sanitizer pass.

Sanitizer-only retry **42247695**, also on `sdfampere004`, restored the established
`OMPI_MCA_pml=ob1` / `OMPI_MCA_btl=^smcuda` environment. It completed with exit 0,
**315 tests passed, no skips, and `ERROR SUMMARY: 0 errors`**. There were no
runtime/test changes or sanitizer suppressions. The accepted 333-test device
result is reused; only the failed sanitizer group was rerun. Third-attempt root:
`/sdf/scratch/users/m/monarin/gpu-validation/issue159/pr246-20261028-r3`.

The job requests one A100, eight CPUs and 64 GiB, with a one-hour limit. It
honors Slurm's GPU mask, records device PCI/UUID identity, verifies native/Python
origins, checks frozen hashes before and after, rejects skipped tests, and stops
on a failed group. The memcheck selection contains gather residue/exact-tail
checks and field-copy/lifetime tests; it is not the entire integration suite.
No performance numbers are collected in this shared-node correctness job.

## MPI and memory-reporting coverage to prepare next

Use six ranks with one Smd0, one EB and four BDs sharing one physical GPU;
assert the observed roles and device identities instead of relying on launch
labels. Run normal completion and early-close paths, with constants shared and
with sharing disabled. Exercise automatic and explicit budgets. Reuse
`mpi_calib_pixel_exact.py` for real Jungfrau pixel comparison and applicable
placement/budget checks. The existing `mpi_gpu_lifecycle.py` asserts four ranks
(two BDs) and does not itself cover the full proposed six-rank field/IPC case;
do not relabel it as 4-BD coverage.

Add a focused driver that combines actual shared-constant owner/importer state
with a deliberately unaligned event-field fixture using the manager's existing
budget. Force a delayed copy/consumer, request the cached view again, and retire
through normal and early close. Verify exact values, copy-ready ordering,
allocation retention, and completion before teardown. Keep synthetic-field
coverage distinct from the real-data pixel test. Exercise scalar/empty constants
and scalar/absent field cases; the two categories are different.

For PR208, capture snapshots and allocation-ledger entries before materializing,
on a cache hit, while a consumer is pending, and after safe retirement. Require
one rounded field-copy charge per BD, no extra cache-hit charge/copy, correct
owner/imported constant totals, and release after the last consumer. Field copies
belong to private retained memory, not imported borrowed constants. A snapshot
must not mutate budget admission state. Use bounded process timeouts and require
clean exit as well as completion markers; a hang or missing rank is not a pass.
This combined coverage is now implemented and passed; see the completion update
and MPI report above, including the host scalar-shape scope limitation.

## Performance preparation and run

Preserve original frozen A `90c05965c`. New A is the unfixed target `8cf05e86a`;
new B is the reviewed follow-up to integrated Stage4 (original runtime
`31c4c2f2`, now with the empty-field correction documented above). Freeze its exact
revision before running; do not time the older installation as the new B.
Both arms include the final PR187/208 code. The other task's C tree at `701f5ebb1` equals new A: reuse that installation
only after checking build/environment manifests, file hashes and actual native
origins against B. Build fresh if those checks fail; do not patch or relocate an
old install. This is not a new A/A campaign.

Prefer `sdfampere012` for the cold campaign, with a single quiet allocation for
both arms. Record SSD model, RAID layout, NUMA/CPU binding, GPU PCI identity,
versions, KvikIO parameters, effective budgets and requested-constant footprint.
Stage the same private prefixes once on node-local `/lscratch`; evict those files
before every sample and verify at most 1% prefix residency. Keep staging, fixture
preparation and correctness checks outside timing. Retain the existing loop
boundary and record startup separately where available. Compilation warmup is
independent of disk-cache coldness.

Run correctness preflights first, then four balanced AB/BA pairs for each bulk
setting: 16 total 10k-event samples, n=4 per variant per setting. Report individual
samples, mean and sample SD (ddof=1) for events/s and loop seconds, and paired
changes. Do not compare isolated historical Hz across nodes or call this final
Stage5/full-fix acceptance.

Run the existing field-view benchmark separately on the same node/install pair:
aligned direct first/cached access on A/B, unaligned first/cached access on B only,
scalar and large array cases. Include bulk-on unaligned access with delayed
consumers and time retirement separately, to measure the host wait identified
in Claude's review. Compare bulk-off as well; do not infer this cost from the
safe-data dense screen. Report host latency, CUDA duration and retained
bytes separately. The safe-data dense workload does not measure these APIs.
No calibrated-path or PR187 startup campaign is reopened. Review repeatable
regressions before proceeding; there is no new arbitrary pass/fail percentage.

## Current remaining gates

Claude's follow-up review is complete. The matched cold screen is submitted as
job 42255812; its results and the separate field-view benchmark remain pending. The original focused device gate passed 333 GPU tests and 315 selected
sanitizer checks; the review delta passed the focused checks recorded above.
The combined MPI/reporting gate passed 16 field cases and two real pixel checks,
with the separate inherited host scalar-shape finding documented in the MPI
report. These results are not full Stage5/final acceptance.
