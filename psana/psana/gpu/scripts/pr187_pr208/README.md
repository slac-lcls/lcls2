# PR187/208 benchmark harness archive

These are byte-for-byte snapshots used for the October 27, 2026 campaigns in
the [performance report](../../docs/performance/pr187_pr208.md). They preserve
the launch, cache controls, timing instrumentation and aggregation code behind
the reported measurements. They are historical campaign artifacts with fixed
site paths, rather than a portable benchmark entry point.

## Contents and provenance

| Campaign | Job | Archived files | Layout |
| --- | --- | ---: | --- |
| Ada A/B/C | 42114353, sdfada016 | 23 | `ada/screen/` driver and calibration measurement; `ada/harness/` raw benchmark and support modules; `ada/metadata/` launch checks; `ada/run.sbatch`. |
| Perlmutter A/B-off/B-on | 60415698 | 24 | `perlmutter/scripts/` driver, timing, analysis and existing CPU tests; `perlmutter/harness/` support modules; prepare/pilot/sweep Slurm launchers. |

The [manifest](manifest.json) records SHA-256, size and original absolute paths
for every snapshot, plus the path and SHA-256 of each campaign's full frozen
manifest. All 47 archived files (184,368 bytes) matched their run-frozen hashes.
The Ada support files were identical across A, B and C: each is stored once
under `ada/harness/`, with all three original paths recorded in the manifest.
Perlmutter helpers retain their own campaign versions.

The manifest covers the archived subset, not every file in the original frozen
runtime. README and manifest are newly authored archive documentation. The
[compact evidence](../../docs/performance/evidence/pr187_pr208_202610.json)
records source versions, result provenance and sample mappings; the
[sample-mapping audit](../../notes/pr187_sample_mapping_20261030.md) traces
Perlmutter logs through these scripts into the published arrays.

## Inspect and verify without GPUs

From this directory, verify the archived bytes:

```bash
python -B - <<'PY'
import hashlib
import json
from pathlib import Path
root = Path('.')
manifest = json.loads((root / 'manifest.json').read_text())
for entry in manifest['files']:
    data = (root / entry['path']).read_bytes()
    assert len(data) == entry['bytes'], entry['path']
    assert hashlib.sha256(data).hexdigest() == entry['sha256'], entry['path']
print('Verified', len(manifest['files']), 'snapshots')
PY
python -B -m unittest discover -s perlmutter/scripts -p test_startup.py -v
```

To replay Perlmutter analysis, supply the original `results.json` and
`provenance.json` from `sweep-60415698` in a separate directory. With Python
3.11.14, this reproduces the original `summary.json` byte for byte:

```bash
PYTHONPATH=perlmutter/scripts python -B - /path/to/copy-of-results <<'PY'
import sys
from analysis import summarize
summarize(sys.argv[1])  # Writes summary.json in the supplied directory.
PY
```

Expected summary SHA-256:
`b954804bfd7da107c65b127c877dd044abaeb8897a25a1aa84f066363c87cfe1`.
Python 3.6 introduces tiny standard-deviation/interval rounding differences;
use the original Python version for byte comparison.

Ada aggregation is inline in `ada/screen/campaign.py`, from `summary=[]`
through the save of `summary.json`. Archive validation executed just that
block and `conditions()` against saved results: all 12 summary records matched
exactly. It did not invoke the campaign's `main()` or launch workloads.
Validation also checked all Python/shell syntax and passed the six existing
Perlmutter CPU tests. No GPU benchmark was rerun.

## Reconstructing a future campaign

The launchers retain their original accounts, node constraints, output paths
and environment setup. A new campaign needs a new scratch root and deliberate
path updates, rebuilt source/install variants, the original input selection
and reference fixtures, and a fresh frozen manifest. Retain these snapshots
unchanged when preparing such a campaign.

For Ada, copy the archived `ada/harness/` contents into each new
`screen/{A,B,C}/scripts/` tree; the manifest gives the exact original mapping.
The `screen/` files, `metadata/` files and launcher otherwise retain their
original relative layout. The driver also checks the prior integration
campaign's frozen installations and external dependencies. Its stage cleanup
and host/GPU assertions are specific to the original allocation.

Perlmutter preserves the original `scripts/` and `harness/` layout. Its
bootstrap expects source/install A and B, the dependency environment, and
provenance files including `states.json` and `HANDOFF.md`. Preparation depends
on the October 26 campaign's staged inputs, references and calibration
constants; the sweep checks inherited frozen runtime hashes. The unused
`stage1_regression` directory on its original `PYTHONPATH` is not archived.

Full logs/results, installed environments, source builds, raw XTC inputs,
reference/pixel fixtures and calibration payloads remain external at the
source paths recorded in the manifest and performance report. These snapshots
preserve the measurement code; they do not package those runtime prerequisites.

## Historical analysis caveats

The snapshot intentionally retains the original `analysis.py` report generator.
Its generated narrative is superseded by the corrected performance report:
missing B-off refresh timers must not be read as measured zero, and the
`within_5pct` flag is meaningful only for percentage differences, not absolute
timings. Use the saved samples and tagged comparisons for interpretation.
The archive adds no psana runtime behavior or new performance claim.
