# Source-only campaign tools

[Analysis and reproduction](analysis/README.md) covers the maintained catalog
checker, focused CPU tests and SDF normalized-row replay. These tools require a
source checkout; repository discovery uses source markers or --repository,
not a fixed number of parent directories. They are excluded from installation.

[Campaign catalog](../../docs/performance/catalog.json) maps stable IDs to
reports and explicit script roots. Historical execution sources are frozen
under notes/archives and scripts/pr187_pr208 or scripts/pr299_perlmutter.
Do not launch archived Slurm scripts directly. Existing jf_scaling,
stage1_regression, stage5c and feespec_bulk_benchmark tools keep their paths.

The maintained [PR regression template](regression/README.md) provides a controller,
run-51 candidate profile and agent instructions. Native execution requires a
qualified, frozen adapter for the chosen revisions; retention and native A/A
qualification remain explicit gates. Common/site helper extraction still requires
real consumers and validation.
