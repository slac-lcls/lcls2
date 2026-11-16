# PR187/208 and PR299 study import

This step imports the two completed documentary packages into
`results/psana2-gpu-scaling`. It does not rerun their campaigns, re-evaluate the
functional PRs, migrate the scaling catalog or implement the future regression
template. Source branch heads and every selected path are in [imports.json](imports.json).

| Study | Source commit | Selected paths | Import policy |
| --- | --- | ---: | --- |
| PR187/208 | `66072235d3cd18c002a13c581af82e111a418758` | 66 | 61 frozen files; five navigation pages reconciled (two remain source-identical) |
| PR299 | `e8d69a7e6604e1d459ad04c633de50a56b46ac9a` | 637 | All exact files, original locations retained |

The 703 selected paths comprise 700 additions and three existing pages with
navigation edits. All campaign scripts and numerical evidence are byte-exact.
The ledger records original Git identity, size, mode, original SHA256 and
imported SHA256. The policy counts are 698 frozen files and five navigation
pages reviewed for reconciliation. Of those five, read_staging.md and
user_kernels.md applied cleanly and remain source-identical: **700 files are
byte-identical to source, and three navigation pages differ**. The ledger
records the pre-import and source delta-base hashes for existing reconciled
pages; both bases are identical for those two clean patches. PR299's
strict archive covers its script/evidence trees and note, so no import records,
configuration files or bytecode are added inside those paths.

Start at the [performance overview](../../docs/performance/README.md), or read
the [PR187/208 report](../../docs/performance/pr187_pr208.md) and
[PR299 report](../pr299_perlmutter_20261115.md). The reports retain the original
qualifications: unresolved remaining-loop equivalence for PR187/208, and
uncontrolled cache, C1 unfixed at the measured head and Run77 deferred for PR299.
The import verifier does not confer new scientific or functional acceptance.

## Offline checks

Run from the repository root with Python 3.11 or newer (for this import's
packaging checks), using `-B` to avoid introducing bytecode in frozen archives:

```bash
python -B psana/psana/gpu/notes/pr_campaign_import_20261116/check_pr_import.py --source-git
python -B psana/psana/gpu/notes/pr_campaign_import_20261116/check_isolation.py .
python -B psana/psana/gpu/scripts/pr299_perlmutter/verify_archive.py
python -B -m unittest discover -s psana/psana/gpu/scripts/pr299_perlmutter -p test_verify_archive.py -v
python -B -m unittest discover -s psana/psana/gpu/scripts/pr187_pr208/perlmutter/scripts -p test_startup.py -v
```

Without `--source-git`, the first command checks imported files, the 47 original
PR187/208 snapshots and report navigation without original Git objects. The
PR299 check verifies the original compact package, its paired summaries and
provenance. Its five tests include missing/extra/modified/symlink/mapping
tampering. PR187's six CPU tests exercise original timing and analysis code;
they do not replay unavailable original raw campaign results.

## Isolation

Root `conftest.py` excludes the two frozen script trees and GPU notes from
routine recursive pytest collection; maintained harnesses remain discoverable.
Ruff and pre-commit code/rewrite hooks exclude these historical sources.
Read-only pre-commit YAML, merge-conflict and size checks remain applicable.
Explicit verifier/unittest commands above remain available.

Meson excludes GPU notes and both frozen script trees from installation.
Hatch excludes them from source and staged-install file selection, including
negative artifact patterns: an `install` artifact would otherwise override
ordinary excludes. Current GPU docs, examples and maintained tools remain
eligible. Packaging verification is an offline file-selection test, not a full
native build or distributed wheel acceptance test.

Full source/build/install trees, original event arrays and all raw monitoring
logs are not reconstructed here. The original reports state their omissions.
Large artifacts remain at their recorded source locations; their continued
availability is not established by this import.

## Completion and review

Complete locally on results/psana2-gpu-scaling. Claude's [second review](review-02.json)
passed implementation commit `4d462252b80370fde35b12af9d490ecef5c28fbc` with no remaining findings.
Both [first-round suggestions](response-01.md) were addressed and re-reviewed.
The [completion record](response-02.md) distinguishes the reviewed implementation
from the subsequent evidence-only commit. See [validation](validation.json) and
[follow-up validation](validation-02.json) for passed checks and limits.

Next is scaling-catalog migration with source/dependency accounting. Publishing,
source-branch retirement and the reusable regression template remain later steps.
