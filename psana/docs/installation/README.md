# Installation guides

Status: **implemented-reference** for the linked Perlmutter guide; remaining
guides are **proposals**. Reviewed 2026-10-15 against source:
`410ebba1ce4d8b2a9d8d7fab56f6dc1fa26a6ef8` on `features/psana2-gpu`.
Source paths: `build_psana.sh`, `pyproject.toml`,
`psana/psana/pscalib/calib/CalibConstants.py`.

Start with [Psana on NERSC Perlmutter](perlmutter.md) for shared activation,
private environments, developer builds and MPI/GPU verification. Its verification
record distinguishes tested configurations from checks readers should perform.
General offsite documentation is still being prepared; the existing
[analysis build instructions](../../../README.md#analysis-only-psana-build)
remain available for that context.

Guide status:

| Guide | Scope |
| --- | --- |
| [Perlmutter](perlmutter.md) | Shared installation, private environments and builds, storage, calibration configuration and MPI/GPU verification |
| `offsite.md` (planned) | Common prerequisites, installation procedure, data/calibration prerequisites and minimal verification |
| `troubleshooting.md` (planned) | Cross-site installation failures and recovery steps; current Perlmutter checks are in its site guide |

Each completed guide identifies its source revision, tested environment,
verification commands and date. Separate using an installation from developer
rebuilds. When `offsite.md` is added, move common instructions there and link from
site guides. Keep site-specific instructions in the Perlmutter guide and its
navigation here; this addition does not require changes to the root README.

[Back to psana documentation](../README.md).
