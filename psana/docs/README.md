# Psana documentation

This is the documentation home for **`features/psana2-gpu`**. It covers the
implementation on that branch; it does not imply that the same features are
available on master or in an installed release.

Status: **implemented-reference**. Reviewed source:
`410ebba1ce4d8b2a9d8d7fab56f6dc1fa26a6ef8`, 2026-10-15.
Source paths: `psana/psana/psexp/`, `psana/psana/gpu/`,
`psana/psana/eventbuilder.pyx`, `README.md`.

| Start here | What it provides |
| --- | --- |
| [Architecture and event flow](architecture/README.md) | CPU/GPU overview, interactive maps and source-linked call trees |
| [Installation](installation/README.md) | Existing build instructions and the location for upcoming offsite/Perlmutter guides |
| [GPU usage](../psana/gpu/README.md) | Current GPU entry point and a short user-task example |
| [External user kernels](../psana/gpu/docs/user_kernels.md) | Task declaration, batch inputs and publication API |
| [GPU limitations](../psana/gpu/docs/limitations.md) | Supported scope and known implementation limits |

## Open the interactive viewer

Download [the standalone HTML](architecture/event-flow/index.html) and open it
in a browser. GitHub's file view displays source, so use its raw-file download
control. The HTML works offline and includes a Download HTML button for sharing
another copy. Source and issue links require internet.

A hosted viewer URL will be added when publishing is configured. The current
documentation checks do not deploy a website.

## Read documentation against the source

An **implemented-reference** explains behavior at its reviewed source revision.
A **proposal** describes intended or possible work; a **historical** document
preserves earlier evidence. A document can contain explicitly labeled proposal
sections without claiming that they are implemented.

Check the active checkout and relevant callers before relying on a description.
When code and docs disagree, report both the observed behavior and the documented
intent. Source shows what is implemented; a discrepancy may still be a code bug.
An issue's status and a successful source-link check do not establish correctness.

## Contribute and maintain

- Keep one maintained explanation per topic and link to it. The GPU API/design
  documents above remain in their existing locations; this index does not copy them.
- Installation guides belong in `installation/`; architecture references in
  `architecture/`. Add `usage/` or `development/` when there is content to move
  or write, rather than creating empty guides.
- Record status, reviewed source commit/date and relevant source paths in new
  behavior documents. Record the tested environment and verification date in
  installation instructions. Review the affected behavior before updating these fields.
- Preserve dated validation as historical evidence. Mark superseded documents
  historical, link to their replacement, and remove them from the current navigation.
- Read [psana agent guidance](../AGENTS.md) before agent-assisted source or doc work.

Viewer maintenance and checks are documented in
[event-flow/README.md](architecture/event-flow/README.md). The viewer model's
source-path list supports a drift check; source changes require review, not an
automatic baseline bump. No psana installation, GPU or experimental dataset is
needed to build or validate these documentation assets.
