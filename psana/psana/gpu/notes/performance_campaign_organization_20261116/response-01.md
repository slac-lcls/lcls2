# Response to Claude round 1

Reviewed proposal: a03bbecdb6eec96944081c8a6ac29056fee47561.

| ID | Response | Evidence/change |
| --- | --- | --- |
| ORG-001 (blocking) | Accepted and fixed | Add explicit default pytest exclusions for all frozen paths, maintained-test allowlists, lint/pre-commit rewrite exclusions, named standalone archive-check commands, and broad-collection verification without modifying archived bytes. |
| ORG-002 | Accepted and fixed | Scope the GPU-relative convention to the tree, declare migration source/destination roots, and spell out repository-root validation/input-bookkeeping-20261110/. |
| ORG-003 | Accepted and fixed | Stable ID is the canonical key; catalog and manifest must agree on record_path/script_roots, with validator checks and explicit historical/multi-site exceptions. |
| ORG-004 | Accepted and fixed | Campaign documents/templates/catalog/FORMAT and experimental tools are source-only; verify both Meson and wheel contents and preserve installed current docs/examples plus working cross-branch links. |

Round 1 could not read external source excerpts or exact-SHA evidence under
the restricted reviewer tool policy. Added source-evidence.md inside the
review checkout, with 11 source selections and line/hash metadata, and retained
the exact round-1 verification receipt. Round 2 must inspect this evidence;
round 1 was not an independent confirmation of the two PR source packages.
No historical payload, runtime, example, test or packaging implementation was
changed. These remain design and acceptance requirements for the migration.
