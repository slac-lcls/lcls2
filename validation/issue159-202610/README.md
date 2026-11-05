# Issue159 alignment validation

Repository-only validation history for the alignment fix in
[PR262](https://github.com/slac-lcls/lcls2/pull/262). These files are not installed
with psana. The production behavior and usage contracts live in the
[GPU parser guide](../../psana/psana/gpu/docs/gpu_xtc_parser.md),
[user-kernel guide](../../psana/psana/gpu/docs/user_kernels.md), and
[callpath/lifetime walkthrough](../../psana/psana/gpu/docs/issue159_callpath_and_lifetime_walkthrough.md).

Start with [Stage 6 correctness](issue159_stage6_correctness_20261029.md) and
[Stage 7 performance](issue159_stage7_performance_20261029.md). The
[implementation plan](issue_159_implementation_plan.md) and earlier reports
retain historical decisions, failed attempts and exact-revision qualifications.
Old pending statuses describe those stages at the time; PR262 is the final
merge/review record. Reviews and development handoffs are preserved in the PR
conversation and Git history rather than shipped as package documentation.

The `evidence/` JSON files retain their original bytes, including absolute
scratch paths, frozen source paths and historical commit identities. Relocation
does not change those provenance records or assert those paths are current.
The offline verifier uses its new repository location:

```sh
python3 validation/issue159-202610/evidence/verify_issue159_stage7_20261029.py /tmp/issue159-stage7-check.json
```

It requires the preserved SDF scratch artifacts named in the evidence. This is
an evidence check, not a new GPU run. Current correctness/performance acceptance
and limitations are unchanged by relocating documentation.
