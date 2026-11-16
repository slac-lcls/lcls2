# Scaling results catalog review scope

Review the new results/psana-scaling consolidation: five campaign records, SDF heatmaps/data/source imports, Perlmutter C++ and psana stripe/scaling reports, and pending Sim10/XPP-cube record. Check scientific claims, source provenance, review-state distinctions, durability and reproducibility, catalog/link/hash validation and the SDF report replay tool.

Base: 2de8179b8d69a942d120b6c74ce46a11d4c7186a. Initial implementation: 2cd7d69c23551030094ce3a29e029990098dd0ee. Default limit: three review rounds, 900 seconds each. Codex implements; Claude independently reviews through Read/Glob/Grep only.

User authorization: "Let's use review loop on this." This explicitly refers to Claude review of the newly created catalog after the coordinator explained it had not been independently reviewed. Scoped external review, fixes, checks, local commits and re-review are authorized. No job submission, push, deployment, new benchmark, framework refactor or broad re-review of historical implementations is part of this task.

Acceptance: preserved imported bytes and historical provenance; correct scoped numerical tables and paired comparisons; original result-review limitations retained; pending cube results not promoted; readable site/workload/campaign structure; functioning archive checks and heatmap replay with explicit original-input gaps. Validate archive integrity/arithmetic and authored content, not physical storage saturation or new native correctness claims. Existing historical scripts are evidence snapshots, not newly introduced runtime behavior.

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

## Validation performed by coordinator

At clean implementation 2cd7d69c2, check_catalog.py passed: 5 campaigns, 377 file hashes, 240 case-rate calculations (156 SDF including pilots/anchors +54 C++ +18 stripe +12 scaling); authored links resolve. Source summaries recomputed within cohorts, paired scaling ratios verified. SDF replay regenerated all 48 cells and all archived comparisons, then rendered four heatmap pages. Original source and figure bytes remain unchanged. The throughput image was visually inspected. All 44 imported cube execution/test sources byte-match reviewed af6ebc03b86033accc6c0af34bb6e3eb2b85f35c. Evidence is in validation.json.

Known scope limits, which must not be represented as passes: eight of 17 historical SDF source inputs were not transferred; normalized measurements/summary/comparisons/figures are complete. Some original historical report links are consequently unavailable. Replay does not run original load_runs() against absent source CSVs and is not a raw-event re-audit. Native benchmark jobs were not rerun for catalog creation. Import source/figure whitespace is preserved exactly, so unrestricted git diff --check reports historical whitespace; new authored catalog files passed whitespace checks excluding snapshots. The new consolidation has not yet had independent review. Historical implementation or result verdicts apply only to their original scopes.

No unresolved prior finding IDs in round 1. Inspect enough source evidence to make findings concrete. Review read failures must be reported. Do not request new expensive/native benchmarks to validate this archival/documentation scope; request missing compact evidence or scoped report/tool corrections if needed.
