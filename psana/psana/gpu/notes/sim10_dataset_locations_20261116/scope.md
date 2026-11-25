# Published Sim10 dataset documentation PR

## Authorization and scope

User requests an implementation/independent-review loop followed by a PR. Codex
coordinates and Claude reviews. Maximum three rounds, 15 minutes per reviewer
invocation, no model override. Target: `slac-lcls/lcls2`, base
`results/psana2-gpu-scaling`, topic `docs/sim10-published-dataset-locations`.

Document published SDF Sim10 runs 3/4, lineage/reader/calibration requirements,
transfer verification, and cleanup boundaries. Link the guide from the current
performance overview, campaign index, and both campaign/regression templates.
Incorporate the current target branch without modifying its real run-51 profile.

## Exclusions

No dataset cleanup or transfer, database mutations, runtime changes, GPU jobs,
new performance claims, synthetic substitution for real run 51, or edits to
frozen execution archives, original hashes, or historical campaign results.
Do not claim an exhaustive live-consumer audit or cross-site qualification.

## Required acceptance

- Five authored Markdown changes agree with existing publication/calibration
  receipts and inspected archived Sim10 reader/generation evidence.
- Explicit `exp=prjdatpsana`, runs 3/4, site-local `dir`, 20 files per run,
  published checksums, stream selection, and Sim10 expansion requirements.
- Preserve source versus published identity, full ten-stream versus PM JF-only
  subset, sampled DataSource versus full-file validation, and retention limits.
- Relative links and whitespace pass; frozen archives and all runtime code are
  unchanged relative to the reviewed base.
- Offline catalog integrity, bookkeeping evidence verification, and maintained
  regression-controller contract tests pass. No native/GPU tests are needed for
  this documentation-only change; no such test is claimed.
- Independent reviewer examines a clean detached checkout at the exact commit
  and returns no unresolved blocking findings on the final documentation commit.
- Publish only the topic branch and open a PR against the results branch after
  review. Preserve compact scope, validation, review, and response records.

## Evidence and storage

Implementation: Perlmutter
`/pscratch/sd/m/monarin/agent-work/lcls2/sim10-dataset-pointers-20261116`.
Review worktrees and logs: SDF shared scratch
`/sdf/scratch/users/m/monarin/agent-work/lcls2/sim10-dataset-pointers-review-20261116`.
Durable local records: `notes/sim10-promotion-20261116/pr-review/`.
Published evidence:
`/sdf/data/lcls/ds/prj/prjdatpsana/results/synthetic_mfx/sim10-promotion-20261116/`.

Starting reviewed base: `ce1b9cdba323574efb76bf7dac2d7660e9e30ef5`, which includes
the maintained PR regression template. Recheck the remote target before PR
creation; re-review any subsequent substantive documentation change.
