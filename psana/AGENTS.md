# Psana agent guidance

These instructions apply to source, tests and documentation under `psana/`.
Follow the current task's requirements and applicable higher-priority instructions.

## Establish current behavior

- Confirm the active worktree, branch and revision. Do not assume a document's
  branch or an installed psana package matches this checkout.
- Trace the relevant source, callers and resource ownership before making
  implementation claims or changing behavior. Use tracked files rather than
  stale generated files or `__pycache__` remnants.
- Source is authoritative for **implemented behavior**. Documentation explains
  intent and context but may lag. Tests provide supporting evidence; their
  assumptions may also be outdated.
- If code and docs disagree, state what the code does and what the document
  claims, with the relevant revision and source references. A discrepancy may
  be a bug in code or docs. Resolve intent from the current task and verified
  requirements; seek clarification when an unresolved discrepancy materially
  changes the requested behavior.
- Do not implement a proposal, restore removed code or change runtime behavior
  solely to match a document. Do not treat an old acceptance report or closed
  issue as proof that current behavior is correct.

## Use and maintain documentation

- Start with [`docs/README.md`](docs/README.md). Check each document's status,
  reviewed source revision/date and source paths before relying on it. If those
  are absent, treat the document as unverified context and inspect source first.
- `implemented-reference` describes its inspected revision; `proposal` describes
  intended work; `historical` preserves earlier evidence. Label mixed-status
  sections explicitly. Do not assume implemented means merged into master.
- Compare relevant source paths with the reviewed revision, including local
  edits. For example: `git diff <reviewed-source-commit> -- <source-paths>`.
  Matching files support carrying forward a review; different files require
  inspection of the changed behavior and callers.
- Keep one canonical explanation per topic and link to it. Mark superseded
  material historical, link to its replacement and remove it from current
  navigation. Historical validation remains evidence about its original run.
- New behavior documents record status, reviewed source commit/date and source
  paths. Installation guides also record their tested environment, verification
  commands and date. Advance provenance only after review, never automatically
  during generation or merely to make a drift check pass.

## Interactive event-flow documentation

- Edit `docs/architecture/event-flow/workflow-model.json` and its renderer, then
  regenerate `index.html`. The generated HTML is not an independent editing source.
- Preserve source links pinned to the inspected commit and provenance in the
  standalone download. Keep current limitations and proposed behavior distinct.
- After changing the model or renderer, run these from the repository root:

  ```bash
  python3 psana/docs/architecture/event-flow/build.py
  python3 psana/docs/architecture/event-flow/build.py --check --check-source-drift
  python3 psana/docs/architecture/event-flow/browser_check.py
  ```

  The browser check requires Playwright/Chromium; see the viewer README for setup.
  Source-anchor validation only proves that definitions exist. Review actual
  caller relationships and explanations as well.

## Validation scope

Documentation-only work uses generation, provenance, link and browser checks;
it does not require a psana installation, GPU or experimental dataset.
For runtime changes, run focused tests and the applicable core/MPI/GPU checks
in the matching built environment. Report actual results and environmental
limitations rather than transferring an old document's validation claims.
