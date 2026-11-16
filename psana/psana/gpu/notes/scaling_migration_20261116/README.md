# Scaling catalog migration

Local branch: results/psana2-gpu-scaling. Source: monarin/lcls2
results/psana-scaling at 9674e3e803354b65c86e42d616af50b23d2c6e3a.

[Local campaign index](../../docs/performance/campaigns/README.md),
[catalog](../../docs/performance/catalog.json) and
[performance index](../../docs/performance/README.md) cover five original
scaling/application records plus PR187/208 site/synthesis and PR299 wrappers.
[Relocation ledger](relocation.json) accounts for every original 359-file addition
and inherited dependencies. [Dependency inventory](dependency-closure.json)
uses conservative transitive notes-root references, including historical
provenance; full roots preserve sibling imports and receipts. Selected runtime
source excerpts stay inside the frozen archive and never replace runtime code.

Frozen files retain exact bytes and executable bits, including original paths
embedded in receipts. New manifests resolve those paths with the ledger. Original
catalog review records remain archived and attest only to their original SHA.
Source identity is independent of the hub runtime. All 240 retained numeric rows
are rechecked without pooling cohorts: 156 SDF, 54 C++, 18 stripe, 12 GPU scaling.
SDF replay uses the normalized rows; eight original input files were absent from
the transfer. This migration does not reconstruct or freshly verify those inputs.
The cube record remains the historical pending snapshot, not a live job update.

Run from the source repository with Python 3.11 (NumPy/Matplotlib for replay):

```bash
python -B psana/psana/gpu/notes/scaling_migration_20261116/check_migration.py --source-git
python -B psana/psana/gpu/scripts/performance/analysis/check_catalog.py
python -B -m unittest discover -s psana/psana/gpu/scripts/performance/analysis -p test_archive_checks.py -v
python -B psana/psana/gpu/scripts/performance/analysis/replay_sdf_report.py --output /NEW-SCRATCH-DIRECTORY --render
python -B psana/psana/gpu/notes/scaling_migration_20261116/check_isolation.py .
```

Keep MPLCONFIGDIR and TMPDIR on writable scratch; replay refuses existing output
and output inside the source checkout. No job launch or raw dataset access is
part of these offline checks. Full native installation/wheel acceptance remains
outside this migration; package selection is tested with real Meson/Hatch fixtures.

## Completion

Claude's [second review](review-02.json) passed `9c4efbde2109ccccb56fe13edbb47ffbabc3db5d` with no remaining
findings. [Round-1 response](response-01.md) records the added duplicate-repeat
guard. [Initial validation](validation-01.json), [follow-up validation](validation-02.json)
and [completion record](response-02.md) retain the evidence and limitations.
The subsequent evidence-only commit does not change the reviewed implementation.
The local branch is ready for the planned publication step.
