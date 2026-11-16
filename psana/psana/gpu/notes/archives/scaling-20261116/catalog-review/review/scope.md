# Scaling results catalog review scope

Review the new results/psana-scaling consolidation: five campaign records, SDF heatmaps/data/source imports, Perlmutter C++ and psana stripe/scaling reports, and pending Sim10/XPP-cube record. Check scientific claims, source provenance, review-state distinctions, durability and reproducibility, catalog/link/hash validation and the SDF report replay tool.

Base: 2de8179b8d69a942d120b6c74ce46a11d4c7186a. Initial implementation: 2cd7d69c23551030094ce3a29e029990098dd0ee. Default limit: three review rounds, 900 seconds each. Codex implements; Claude independently reviews through Read/Glob/Grep only.

User authorization: "Let's use review loop on this." This explicitly refers to Claude review of the newly created catalog after the coordinator explained it had not been independently reviewed. Scoped external review, fixes, checks, local commits and re-review are authorized. No job submission, push, deployment, new benchmark, framework refactor or broad re-review of historical implementations is part of this task.

Acceptance: preserved imported bytes and historical provenance; correct scoped numerical tables and paired comparisons; original result-review limitations retained; pending cube results not promoted; readable site/workload/campaign structure; functioning archive checks and heatmap replay with explicit original-input gaps. Validate archive integrity/arithmetic and authored content, not physical storage saturation or new native correctness claims. Existing historical scripts are evidence snapshots, not newly introduced runtime behavior.
