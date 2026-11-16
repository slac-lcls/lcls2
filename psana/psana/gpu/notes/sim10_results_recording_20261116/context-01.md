Review the completed Sim10 results recording on results/psana2-gpu-scaling.
Base: 39ea002a8ffc95546405139e50f2f45005514ee2
Reviewed commit: bf2ecfa62fdd220051370c78398878161dc8948d
User explicitly requested this review loop and final verdict. Existing scoped external Claude disclosure was approved in this conversation ("Yes, let's review"). Review only this campaign code/evidence. No submissions, pushes or external communication. Do not delegate.

The user requested recording the finished S/E/EC/ECC campaign following the results branch conventions, preserving the old pending snapshot and exact execution sources. Three rounds maximum. Evaluate reporting correctness and packaging, not whether to rerun the experiment. No GPU work is required.

Read:
- psana/psana/gpu/docs/performance/FORMAT.md (branch conventions)
- docs/performance/campaigns/perlmutter/sim10-xpp-cube/2026-11-16-se-ec-ecc-100k/ beneath psana/psana/gpu: README.md, manifest.json, runs.csv, summary.json.
- psana/psana/gpu/scripts/performance/analysis/sim10_cube.py, test_sim10_cube.py, check_catalog.py (new cube-complete-v1 dispatcher).
- psana/psana/gpu/notes/archives/sim10-xpp-cube-20261116/: case-receipts.json, bundle/bundle.json, external-index.json, input-identities.json, raw-index.json; evidence/job61409171-results.json and job61409171-sacct.txt; reviews/comparison02-results-round01.json; history/status.json; execution/results/complete.json and comparison.json. Exact execution scripts are frozen in bundle/.
- psana/psana/gpu/notes/sim10_results_recording_20261116/: scope.md, status.json, verify_raw.py, checks/* (coordinator evidence).
- Updated performance overview, history, campaign index, catalog and offline analysis README.

Acceptance: preserve original pending archive and measurements; exact 49 execution files with seven external pins; all35 ordered cases, 24 full primary cells (3 repeats x2 profiles x4 stages); unpooled35-case CSV, denominator and timing meanings; summary medians/ranges, paired ECC/S, staging drift, bridge; truthful cube/content/reference/export/cache/memory claims; terminal accounting; independent results review distinct from this packaging review; indexed raw retention limitations; source-only archives excluded from runtime tests and packaging. Native summary's pending string is historical and must not be silently rewritten. No numerical CPU reference or scientific correctness claim. Earlier9.842GiB/s is16BD4GPU count1, separate campaign and node.

Checks performed by coordinator (reviewer may inspect receipts but has no command tool):
- Original read-only audit_case applied to all35 original raw results; exact timestamp union/read ranges, mapping, cube per-rank counts, memory receipts; archived compact fields match raw values; native summarize matches saved comparison; all608 indexed raw hashes,49bundle files,7external pins verified. checks/raw-audit.json is pass. No GPU re-execution or XTC dataset rehash.
- Catalog:10campaigns,2081file hashes,275arithmetic cases pass.
- New11 deliberate-corruption tests pass (case drop/duplication, timing/batch, cache, NaN flag, cube counts, byte denominator, authored table, bundle hash, worker count).
- Existing12 archive contract tests pass.
- Source isolation passed actual pytest discovery, sentinel test, Meson installation selection, Hatch source and staged-container selection. Ruff/pre-commit binaries unavailable; checked exclusion config, AST and git whitespace. No native build.
- Initial staged whitespace check found CSV CRLF; corrected to LF in latest reviewed commit, updated manifest hash and rechecked catalog and full diff. Authored files pass whitespace checks. The full diff flags one final blank line in frozen execution/slurm-job.txt; it is deliberately preserved byte-exact and not rewritten.

External raw files and large inventories remain in scratch; do not require their presence in your worktree or infer their raw values beyond supplied audit. Their absence from Git is documented. The per-bin scalar receipts are archived; event arrays/pixel data are not. No durable storage guarantee. Original results reviewer limitations are explicit. Distinguish code defects, false claims and missing required evidence from optional broader baseline qualification work.

Use stable finding IDs SR-R1-01 etc. Report exact reviewed SHA and verdict. Inspect evidence independently; don't just accept this context. Do not run or edit archived scripts.
