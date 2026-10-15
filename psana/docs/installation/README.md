# Installation guides

Status: **proposal** for the guide layout, 2026-10-15. Reference source:
`410ebba1ce4d8b2a9d8d7fab56f6dc1fa26a6ef8` on `features/psana2-gpu`.
Source paths: `README.md`, `build_psana.sh`.

Offsite installation documentation is being prepared separately. Until it is
integrated here, use the repository's existing
[build and Perlmutter instructions](../../../README.md#analysis-only-psana-build).
This page establishes the destination; it does not claim to validate an installation.

The intended guide split is:

| Future guide | Scope |
| --- | --- |
| `offsite.md` | Shared prerequisites, installation procedure, data/calibration prerequisites and a minimal verification |
| `perlmutter.md` | Site environment, storage paths, MPI/GPU launch commands and site-specific verification |
| `troubleshooting.md` | Diagnosed installation failures and recovery steps |

Each completed guide should identify its source revision, tested environment,
verification commands and date. Keep common instructions in `offsite.md` and link
to them from site guides. Separate using an installation from developer rebuilds.
Replace or redirect overlapping root-README instructions when the maintained guide
is ready, so two copies do not diverge.

[Back to psana documentation](../README.md).
