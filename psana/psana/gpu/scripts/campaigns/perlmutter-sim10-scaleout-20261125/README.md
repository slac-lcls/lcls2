# Frozen Sim10 campaign scripts

These are source-only historical execution snapshots for the 2026-11-25 closeout. `stage1`, `stage2` and `stage3` each contain one exact executed bundle; group-specific differences are stored under `overrides/`. `groups/*/bundle-map.json` in the compact archive identifies every file and SHA256 for each of the 12 allocations. Bundles include runners, application callbacks, audits, tests, runtime/dependency pins and shell launchers. Native framework tests remain pinned by source paths/hashes and their measured framework commits, not copied as an additional framework tree.

`stage1/support` and `stage2/support` retain sequential execution controllers/tests. `stage3/support` retains the preparation/qualification/launch source and its 29 CPU tests, including the pinned base and publication receipts required by those tests. These paths preserve source bytes. No launcher has been rewritten to submit from this archive. Exact plans and launch commands remain under the archive's group directories.

From the repository root, validate the compact archive without CUDA or scratch access:

```bash
python -B psana/psana/gpu/scripts/campaigns/perlmutter-sim10-scaleout-20261125/verify_archive.py --repository .
python -B psana/psana/gpu/scripts/performance/analysis/check_catalog.py
```

Restore a group bundle to a new directory (no jobs submitted):

```bash
python -B psana/psana/gpu/scripts/campaigns/perlmutter-sim10-scaleout-20261125/restore_bundle.py --repository . --group stage3-weak-n04 --output /tmp/sim10-stage3-restored
```

Do not execute that relocated bundle directly: paths refer to historical Perlmutter installs, inputs and scratch attempts. Inputs/ranges contain large per-event tables and are retained by raw hash/path, not embedded here. To rerun, restore the pinned framework/install/environment, qualify the data/calibration and layout, adapt paths explicitly, and revalidate the resulting bundle before submission. Runtime JSON pins and per-group plan/source commits document the measured identities. The frozen Stage3 runner accepts only 1/2/4 nodes; extending it to 24 is future implementation work.

The retained CPU suite can run with the original psana-build Python and original calibration snapshot available:

```bash
python -B -m unittest discover -s psana/psana/gpu/scripts/campaigns/perlmutter-sim10-scaleout-20261125/stage3/support -p 'test_*.py'
```

These 29 tests were rerun successfully during closeout. They require original calibration/publication context and are not claimed portable to an arbitrary host. Native GPU tests were not rerun for packaging; their original passing receipts are retained. The offline verifier independently checks arithmetic, case coverage, placement, bundle hashes/syntax, scheduler outcomes and device-test receipts; it cannot reconstruct omitted raw timestamp/pixel arrays.
