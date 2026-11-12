# PR #298 input-bookkeeping evidence

Repository-only evidence for [PR #298](https://github.com/slac-lcls/lcls2/pull/298),
addressing review point 2. The [performance note](../../psana/psana/gpu/docs/performance/input_group_bookkeeping.md)
remains the concise interpretation. This directory is outside the installed
`psana`/`psalg` packages; no packaging or runtime change is needed.

## Campaigns and scope

| Campaign | Job | Runtime and measurement | Essential receipts |
|---|---|---|---|
| Historical Perlmutter | 61098604 | Research before/after, 10k events; 1.2656 to 2.1841 GiB/s with two streams | [summary](perlmutter/receipts/native/attempt03/summary.json), [completion](perlmutter/receipts/native/attempt03/complete.json), [provenance](perlmutter/receipts/native/attempt03/provenance.json) |
| Ada 10k | 44185752 | Extracted PR runtime; two-stream median 3.269 GiB/s | [results](ada-10k/receipts/results.json), [runtime](ada-10k/receipts/runtime-evidence.json), [audit](ada-10k/receipts/audit.json) |
| Ada 100k | 44189224 | Extracted PR runtime; two-stream median 2.870 GiB/s | [results](ada-100k/receipts/results.json), [runtime](ada-100k/receipts/runtime-evidence.json), [audit](ada-100k/receipts/audit.json) |

The Perlmutter improvement belongs to the research implementation at
`b95b789ca834e6746870548274d119de09ef86ee`. Both Ada jobs used PR Python at
`67c327a148a08b2aaf972e67e1e02e367c2b298f` over existing native extensions after
source/module compatibility checks, not fresh builds. The [build-config receipt](ada-10k/receipts/build-config-diff.json)
qualifies the packaging-only difference. Ada has no pre-fix control and does not
independently establish the historical 72.6% gain. All campaigns used KvikIO CPU
fallback and descriptor staging, not cube processing or GDS. Two repetitions
per clean configuration remain a limited sample. Local-page-cache-cold does not
mean storage-server/disk-cold; only the Ada campaigns include NIC corroboration.

Each accepted campaign records 31 passing GPU tests; these are not 93 distinct
tests. Earlier failed Perlmutter attempts are retained as separate failure
receipts, not counted as successful acceptance. Original status/receipt text is
historical and has not been rewritten during this import.

## Portable integrity and consistency check

From the repository root, using standard-library Python (3.6 or later):

```sh
python3 validation/input-bookkeeping-20261110/verify.py
python3 -m unittest discover -s validation/input-bookkeeping-20261110 -p test_evidence_bundle.py -v
```

The verifier needs only this directory. It does not need the fork, Git objects,
scratch, psana, CUDA, MPI, credentials or a network connection. It checks:

- Exact archived-file inventory, byte lengths, SHA-256 and original Git blob IDs.
- Historical clean medians from completion receipts and timed summaries from the
  four preserved phase records, without mixing instrumented and clean rates.
- Ada run matrices, read/byte/callback accounting, runtime identity, audit/result
  agreement, clean medians and the preserved GPU-test/completion receipts.

These are checks of saved evidence, not new performance measurements. The
verifier does **not** replay full per-event/raw ownership validation, recompute
timestamp checks from original event arrays, rerun GPU tests, measure current
cache/NIC state, or read the old scratch paths. The original audits performed
those checks where documented; this verifier does not replace them.

## Provenance and archived harnesses

[manifest.json](manifest.json) maps all 61 archived files (592,021 bytes) to
their original repository, commit, path, Git blob SHA-1 and SHA-256. Every
archived file is byte-for-byte identical to its selected source Git blob:

- Perlmutter: `3c3799829c659d739d5927511f86cc968a6fae6e`,
  `notes/sim10_input_bookkeeping_20261110/`.
- Ada: `f888b316e8d940be6cad683d9dc62110938f006f`,
  `notes/sim10_ada_baseline_20261110/` and
  `notes/sim10_ada_100k_baseline_20261110/`.

The originals are not deleted or modified. The manifest proves consistency
against these recorded hashes; it is not a signature or proof against someone
deliberately rewriting both the evidence and manifest.

[Perlmutter harness](perlmutter/harness/), [Ada 10k harness](ada-10k/harness/), and
[Ada 100k harness](ada-100k/harness/) preserve the original launch, GPU-test,
runner and instrumentation/audit code for inspection. They retain site paths,
relative imports and original source/runtime pins. **They are archived source,
not portable launchers for this relocated directory or today's PR head.**
Other source-tree helpers, runtime/test sources, installed environments and
fixtures are not bundled. A full rerun requires those dependencies and a newly
reviewed launch configuration; do not submit the archived `run.sbatch` directly.

The historical summarizer is independently runnable on the preserved phase and
completion receipts, without those site dependencies:

```sh
python3 validation/input-bookkeeping-20261110/perlmutter/harness/summarize.py validation/input-bookkeeping-20261110/perlmutter/receipts/native/attempt03
```

## What remains external

Large raw per-event result files, profiling/log dumps, XTC fixtures, installed
binaries and full review conversations are deliberately not copied. The
[Perlmutter checksum list](perlmutter/receipts/native/attempt03/SHA256SUMS.remote)
and the [10k](ada-10k/receipts/artifact-index.json) / [100k](ada-100k/receipts/artifact-index.json)
artifact indexes retain their original hashes and locations, including absolute
scratch paths. Hashes do not recover files if those external artifacts expire.
The compact results, acceptance/provenance receipts, phase counters and offline
verification remain in this repository even if the personal fork disappears.

Independent-review outcomes remain recorded in the imported status/results;
this import is not a new independent review. Invariant hardening from review
point 3 is intentionally outside this change.
