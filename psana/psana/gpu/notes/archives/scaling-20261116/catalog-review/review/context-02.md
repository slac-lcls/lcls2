# Scaling results catalog review scope

Review the new results/psana-scaling consolidation: five campaign records, SDF heatmaps/data/source imports, Perlmutter C++ and psana stripe/scaling reports, and pending Sim10/XPP-cube record. Check scientific claims, source provenance, review-state distinctions, durability and reproducibility, catalog/link/hash validation and the SDF report replay tool.

Base: 2de8179b8d69a942d120b6c74ce46a11d4c7186a. Initial implementation: 2cd7d69c23551030094ce3a29e029990098dd0ee. Default limit: three review rounds, 900 seconds each. Codex implements; Claude independently reviews through Read/Glob/Grep only.

User authorization: "Let's use review loop on this." This explicitly refers to Claude review of the newly created catalog after the coordinator explained it had not been independently reviewed. Scoped external review, fixes, checks, local commits and re-review are authorized. No job submission, push, deployment, new benchmark, framework refactor or broad re-review of historical implementations is part of this task.

Acceptance: preserved imported bytes and historical provenance; correct scoped numerical tables and paired comparisons; original result-review limitations retained; pending cube results not promoted; readable site/workload/campaign structure; functioning archive checks and heatmap replay with explicit original-input gaps. Validate archive integrity/arithmetic and authored content, not physical storage saturation or new native correctness claims. Existing historical scripts are evidence snapshots, not newly introduced runtime behavior.

## Round 2 re-review

Review exact candidate c55407e1d74daa29d783de2cbbf9c46a5a7c9da7 against initial 2cd7d69c23551030094ce3a29e029990098dd0ee. The overall base remains 2de8179b8d69a942d120b6c74ce46a11d4c7186a. Read benchmarks/scaling/review/review-01.json and response-01.md first. Reuse finding IDs CAT-R1-01 through CAT-R1-05, resolving each on evidence; also check for regressions in the current catalog and tools. Do not start another loop or run tests. The coordinator considers all five addressed, but only your independent verdict closes them.

Required additions: benchmarks/scaling/tools/test_archive_checks.py and updated validation.json. Main fixes are in the cube README/manifest/FORMAT.md, SDF storage/GPU table, failed-job manifests, catalog validation and replay population checks. Source snapshots/native measurements remain byte-for-byte unchanged.

Latest coordinator checks: catalog passes 380 file hashes and 240 case computations, now including authored table values, exact unique summary populations and manifest taxonomy. Five regression tests passed (dropped/duplicate summary cells, incorrect site/workload, wrong README rate, broken source/template links, moved-cohort and duplicate cases). replay02 recomputed all 48 cells and all archived comparisons successfully. Rendering inputs and renderer unchanged; prior full four-page render remains recorded. All original import hashes pass. Whitespace check passed for the entire new diff.

Validation failures not hidden: initial regression runner had two literal-C++ regex matching failures; fixed re.escape and reran all five successfully. Initial sandboxed Matplotlib cache was unwritable and fell back to /tmp; final run used authorized scratch. No runtime jobs, data regenerations, framework changes or native re-audits occurred. Missing SDF historical inputs and limited original review evidence remain explicit. This catalog review must not turn historical pending result reviews or numerical-reference checks into passes.

## Files and evidence to inspect


Checkout root is the supplied detached worktree. All following paths are checkout-relative. Inspect the authored catalog and tools, consulting the original native summaries to check claims. No shell/test tools are exposed; do not claim execution.

Required authored content:
- benchmarks/scaling/README.md, FORMAT.md, catalog.json
- benchmarks/scaling/campaigns/*/*/*/README.md and manifest.json (five entries)
- benchmarks/scaling/tools/check_catalog.py, replay_sdf_report.py and README.md
- benchmarks/scaling/templates/README.md and manifest.json
- benchmarks/scaling/sources/README.md, IMPORTS.json and sdf-input-availability.json
- benchmarks/scaling/validation.json

Scientific source evidence:
- notes/sim10_striping_20261110/reader/RESULTS.md and sweep/{audit.json,trials.csv}; reader count 1 is five files/five OSTs, not one OST.
- notes/psana_striping_pr298_20261111/screen-20261115/RESULTS.md and evidence/screen01/{summary.json,trials.csv,audit.json}; older 7 physical cores/BD, 34 total; no NIC metrics.
- notes/psana_gpu_scaling_20261116/RESULTS.md and evidence/screen01-results/{audit.json,scaling-summary.json,summary.json}; compact 3 physical cores with SMT/BD; 1/2/3/4 GPUs and 4/8/12/16 BDs; paired medians; four Cassini NIC counters are not storage-only.
- notes/psana_gpu_scaling_20261116/evidence/pilot02-results/summary.json (short pilot and separate one-GPU CPU-policy controls, excluded from full screen).
- benchmarks/scaling/sources/sdf-20261115/reference/notes/sim10_parameter_report_20261114/{REPORT.md,SCHEMA.md,build_report.py,validation.json} and artifacts/{runs.csv,summary.csv,comparisons.json,provenance.json}. Seven separate allocation cohorts. Storage/GPU comparison uses four BDs TOTAL across GPUs, unlike Perlmutter scaling. Means for SDF, medians for Perlmutter.
- notes/sim10_perlmutter_baseline_20261115/{PLAN.md,status.json,source-origins.json}. Pending S/E/EC/ECC application campaign is a dated task snapshot. Current prepared input uses stripe count 2; failed bootstrap is not accepted performance.
