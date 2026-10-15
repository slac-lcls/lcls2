# Event-flow documentation validation

## Feature-branch documentation home — 2026-10-15

Status: **historical validation evidence**, recorded for this documentation
revision. Reviewed source: `410ebba1ce4d8b2a9d8d7fab56f6dc1fa26a6ef8` on
`features/psana2-gpu`; source paths are listed in the model's
`baseline.source_paths`. PR #151 was merged; issues #153, #168 and #170 were
open when checked. These are dated observations, not live state.

Merged the feature branch cleanly and verified its psana source was identical
to the previous inspected baseline. Moved the viewer into
`psana/docs/architecture/event-flow/` and added documentation navigation,
agent guidance and source-drift validation.

- Generation, commit-pinned source anchors, deterministic HTML and source-drift
  checks passed. A temporary Git repository probe confirmed unchanged files
  pass and both staged and unstaged source changes fail the drift check.
- All 25 local Markdown links in the new docs and agent instructions resolved.
- Chromium `153.0.8010.12` passed all three views, twelve components and twelve
  tour steps, call-tree source clicks, keyboard controls, serial toggle/reload,
  connection table, standalone download, mobile layout and no-JavaScript
  fallback. Provenance survives downloading with scripting disabled.
- Zero JavaScript errors and zero external runtime requests. One intentional
  source-link click used a locally intercepted response, not GitHub access.
- Visually inspected the GPU page. Local artifacts:
  `/tmp/psana-workflow-feature-docs-check/`, including `result.json`, screenshots
  and `downloaded.html`. These temporary files are reproducible with the checker.

The new GitHub Actions workflow runs the same generation/drift/browser checks
and uploads preview artifacts. This local record does not claim a hosted CI run
or a published website. No psana, GPU, MPI or data runtime tests were needed or
run. The earlier records below retain their original dates and scope.

## Call-path update — 2026-10-13

Added source-linked call trees to all twelve component panels, retaining the
same inspected source baseline. Reviewed caller nesting against the checkout,
including separate bulk/per-datagram read and parser paths, manager construction,
lazy KvikIO file/buffer allocation, iterator entry points and teardown.

- `build.py --check` passed for the schema, class-scoped method anchors and
  deterministic standalone HTML. `git diff --check` passed.
- Chromium 153 browser checks passed for all component trees, source links,
  keyboard collapse/expand, all previous navigation/tour checks, offline
  download, mobile layout with the deepest tree, and no-JavaScript trees.
- Tested a call-source link opening the expected commit/line URL in another tab
  using a locally intercepted response; this does not verify GitHub availability.
  No external runtime requests or JavaScript errors occurred.
- Visually inspected the rendered reader-setup tree.
- Artifacts: `/tmp/psana-workflow-call-path-check/`, including `result.json`,
  `input-setup.png`, `input-call-paths.png` and the downloaded standalone copy.

This remains documentation validation, with no psana/GPU/MPI runtime tests.
Source-anchor validation proves definitions exist; call relationships were
reviewed manually and are not a dynamically captured trace.

## Original prototype — 2026-10-07

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
