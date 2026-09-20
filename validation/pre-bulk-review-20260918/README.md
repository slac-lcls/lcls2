# Pre-bulk combined review cleanup

Changes: report detector routing and pinned row maps, sum detector memory before
recording category high-water marks, correct memory-category documentation, and
remove an unused cached canonical-row mapping. Kernel submission is unchanged.

A copy of install_psana was made in ignored `install/`, then current source
`gpu_detector.py` and `gpu_events.py` were copied into its psana/gpu package.
This preserves every recorded benchmark installation. Native modules are
unchanged. The first source-tree PYTHONPATH attempt failed collection because
that tree does not contain installed native extensions; no tests ran there.

```bash
source setup_env.sh
source install_psana/activate.sh
export PYTHONPATH="$PWD/validation/pre-bulk-review-20260918/install/lib/python3.9/site-packages"
python -m pytest -q psana/psana/tests/gpu/unit -m 'not gpu'
```

Result: **168 passed**, logged in `unit.log`. Parser files and gather-map,
gather-plan and kernel ASTs were checked against the frozen measured G prefix.
No new GPU-performance claim, commit, push, or bulk merge.

Full review: `psana/psana/gpu/docs/batched_pre_bulk_review.md`.
