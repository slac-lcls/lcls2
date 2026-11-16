# PR regression template and run-51 candidate check

The maintained [controller](../../scripts/performance/regression/README.md),
[candidate profile](../../scripts/performance/regression/profile.json) and
[agent instructions](../../docs/performance/templates/pr-regression.md) prepare
new frozen PR comparisons. Existing campaign harnesses remain unchanged.
Implementation review and test outcomes are recorded in status.json and review
receipts at closure. This work produces no hardware performance result.

## Run 51 is not affected by the all-bad status caveat

On November 16 the [Confluence public-practice table](https://confluence.slac.stanford.edu/spaces/LCLSIIData/pages/267391733/psana#psana-PublicPracticeData)
places its g1/g2 nonzero-status caution on **mfx100852324, runs 77–79**.
The following row, **mfx100848724, run 51**, has no such caution. See the exact
row-association and page-hash [receipt](confluence-check.json). Search initially
returned HTTP 429; the user-provided page was successfully read with HTTP 200.

A direct independent-of-page check read all selected Jungfrau Configure identities
and used the installed PR299 runtime's standard calibration selection functions.
Run 51 resolves to cdb_mfx100848724 / jungfrau_000005, document
69d00aad4b71e8288a8c6b0e, valid from run 49 through end, version V2026-04-02.
The uint64 status shape is (3,32,512,1024). All 402,653,184 payload bytes were read;
SHA256 is ff01faf618f88e81453233a02b4c3a0ee1cb5abea0ff4ef864653d03ddb8442e.

| Status plane / combination | Bad | Good |
| --- | ---: | ---: |
| Gain 0 | 6,721 | 16,770,495 |
| Gain 1 | 1,688 | 16,775,528 |
| Gain 2 | 2,170 | 16,775,046 |
| OR of all three, any nonzero bit rejects | 8,825 | 16,768,391 |

Thus 99.9474% of the 16,777,216 physical pixels remain good under this policy;
every panel retains good pixels. [Full compact audit](run51-status-audit.json)
records per-panel counts, value populations, payload hash and Configure hashes.
This checks pixel_status only, not all mask sources, full calibration/numerical
correctness, or future database selections. No constants were modified. The
historical PR299 campaign used raw inputs and skipped calibration; its timings
are not a calibrated regression baseline.

The [frozen audit source](check_run51_status.executed.py.txt) is the executed investigation source,
not a campaign launcher or a reusable CLI. It is stored as text to preserve the
exact executed bytes. It writes next to its source: never execute a copy inside
the repository. For reproduction, first copy it to a new external scratch
directory and activate the recorded native runtime; it downloads about 403 MB.
The [selected document](selected-status-doc.json) is retained, and the
[provenance receipt](audit-provenance.json) records the executed-source hash,
installation, Python and selection-module identity. This receipt is a post-audit
readback, not contemporaneous provenance capture. Its large status payload stays outside Git
at /pscratch/sd/m/monarin/psana2-gpu/pr-regression-template-20261116/.
The [reference summary](run51-reference-summary.json) records the historical
17,872-event input and 10k selection. New native adapters must independently
recheck input identities and derive the expected event/byte reference.

## Qualification boundaries

Run 51 is the selected **candidate**, not a permanently retained/fully qualified
baseline. Public XTC/SMD listing on SDF and readable Perlmutter copies were
verified in the preceding inventory. Current status selection is now audited.
Outstanding before a measured campaign: durable source/retention owner, exact
cross-site file or measured-range identity, current compute-node access, frozen
runtime/adapter receipts, native correctness and A/A noise acceptance. Calibrated
profiles additionally require all constants and independent numerical references.
The default profile remains gated until those artifacts are supplied. No new
native adapter or hardware job is claimed; the controller intentionally accepts
a separately reviewed adapter appropriate to the exact revision pair.
