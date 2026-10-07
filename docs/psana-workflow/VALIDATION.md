# Prototype validation — 2026-10-07

Baseline: `e2cdad6e18fd9263d7adfeed0ce58d5ad074e070`. PR #151 was OPEN and still at this head when checked before completion. Issues #153, #168 and #170 were OPEN. These are inspection-time facts, not live status.

## Passed

- `build.py --check`: model IDs, source-file/anchor resolution against the pinned Git commit, graph membership/routes, tour coverage, absence of external runtime assets and deterministic generated HTML.
- `browser_check.py`, Chromium `153.0.8010.12`, opening local `file://` HTML with networking disabled: three views, twelve component panels and twelve event-tour steps.
- All visible component navigation, direct component hash, source URLs pinned to the baseline, serial role toggle and reload, previous/next tour navigation, connection table and keyboard activation.
- Downloaded HTML opened independently and remained interactive without network access.
- Mobile layout at 390 × 844: page content fit the viewport; the large map was intentionally scrollable. Diagram labels fit component boxes.
- JavaScript-disabled copy exposed all twelve static component reference sections.
- Zero JavaScript errors and zero external HTTP(S) requests during the browser run.
- Manual visual inspection of the GPU diagram after correcting overflowing node labels.

Machine-readable results and screenshots were saved to `/tmp/psana-workflow-browser-check/`. They are temporary test artifacts; rerun the checker to reproduce them. The browser check uses a separate temporary Python environment and no psana installation.

## Not established by these checks

No psana runtime, data, GPU or MPI jobs were run; no claim of numerical validation is made. Source-anchor checks establish that referenced definitions exist, not semantic correctness of every narrative claim. Safari, Firefox and screen-reader behavior were not tested. The main views are an introductory workflow map; full nested class diagrams and less-common DataSource modes remain future work.
