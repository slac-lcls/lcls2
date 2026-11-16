# Current performance dataset locations

Location update: November 16, 2026. This is a rerun/transfer guide, not a new
performance result, a regression acceptance, or permission to delete data.
Historical reports continue to describe the inputs actually measured.

## Published Sim10 inputs on SDF

Use the explicit XTC directory `/sdf/data/lcls/ds/prj/prjdatpsana/xtc` with
`exp="prjdatpsana"`. The old fixture root is
`/sdf/data/lcls/ds/prj/prjdatpsana/scratch/synthetic_mfx`.

| Old fixture under that scratch root | Published run | Events | XTC plus SMD bytes |
| --- | --- | ---: | ---: |
| `sample10_100k` | 3 | 100,000 | 376,806,534,528 |
| `sample10_400k` | 4 | 400,000 | 1,507,218,362,688 |

Each published run contains ten bigdata files, streams `s000` through `s009`,
and ten matching SMD files in `smalldata/`. For example, run 3 starts with
`prjdatpsana-r0003-s000-c000.xtc2` and
`smalldata/prjdatpsana-r0003-s000-c000.smd.xtc2`. Run 4 uses `r0004`.
There is one native step per run. Existing published runs 1 and 2 are unchanged.

After registering the process-local Sim10 reader, select an input with:

```python
from psana import DataSource

source = DataSource(
    exp="prjdatpsana",
    run=3,
    dir="/sdf/data/lcls/ds/prj/prjdatpsana/xtc",
    live=False,
    max_events=10000,
    use_calib_cache=False,
)
```

Use run 4 for the 400k fixture and choose the event limit for the new campaign.
The explicit directory is required: publication did not add logbook entries.
The original `sample10_10k` was not separately promoted. A newly specified 10k
test can use a prefix of run 3; this is not a byte-identical relocation of the
old 10k files or permission to rewrite their historical checksums/results.

## Identity, reader, and calibration

These lossy synthetic inputs derive from the first 10,000 events of
`mfx101555026` run 10. They preserve the 32 physical Jungfrau panel identities.
Each panel stores `raw.ravel()[::10]` (52,429 uint16 values); the larger fixtures
repeat the seed payloads with unique outer event timestamps.

Promotion copied the accepted sources and changed only fixed-width BeginRun
experiment/run fields, from `mfx101555026`/10 to `prjdatpsana`/3 or /4.
Configure, event payloads, timestamps, transitions, lengths, and SMD offsets
were preserved. Full published-file hashes therefore differ from source hashes.
Do not symlink the old fixture path to the published directory: filenames and
experiment/run checks also need to change in a newly prepared runner.

The `jungfrau_sim10_1_0_0` reader must be registered in each process before
reading seeds via `run.Detector("jungfrau").sim10.seed(evt)`. Expand each panel
with `np.repeat(seed, 10)[:512*1024].reshape(512, 1024)` before applying full-panel
calibration. Sim10 is not ordinary `jungfrau.raw`; changing the validity range
does not make `det.raw.calib(evt)` work directly on compressed seeds.

The existing 13 experiment calibration records in `cdb_prjdatpsana` now cover
`run=1` through `run_end=4` (eight Jungfrau and five PV records). Only `run_end`
changed from 2 to 4; document IDs, payload IDs, and payloads are unchanged.
The frozen calibration manifest SHA-256 is
`90a27eed3679111aa24574c09b5505be6b8775429e5b8de10dc0c9de117b983d`.
Load the published experiment's constants for a new campaign; do not reuse an
old source-experiment selection or silently fall back to detector-DB constants.

Publication acceptance checked complete file hashes/event counts/SMD references,
three sampled DataSource events per staged run, and one event per published run
with live constants and expanded CPU-v3 calibration. This was not a full
100k/400k DataSource scan or a new GPU correctness/performance campaign.

## Transfer and durable evidence

The SDF evidence root is
`/sdf/data/lcls/ds/prj/prjdatpsana/results/synthetic_mfx/sim10-promotion-20261116`.

- `COPY_FILES.txt`: the exact 40 paths, relative to the published XTC directory.
- `SHA256SUMS`: published-file hashes for transfer verification, not source hashes.
- `completion.json`, `r0003-validation.json`, `r0004-validation.json`, and
  `postcheck.json`: publication and sampled DataSource/calibration acceptance.
- `calibration/`: before/after records and selection/payload verification.
- `README.md`, `code/`, and `logs/`: reproducibility and handoff details.

Copy only the listed 40 files, preserving `smalldata/`, to the colleague's
selected Perlmutter project data location. No Perlmutter copy was performed by
the promotion. At this update, the usual
`/pscratch/sd/p/psdatmgr/psdm/prj/prjdatpsana/xtc` directory was absent; do not
advertise it as an available input until the destination and hashes are verified.

The existing private Perlmutter input
`/pscratch/sd/m/monarin/psana2-gpu/datasets/mfx101555026-r0010-sim10-100k-jf`
is a Jungfrau-only source-identity subset, not these published ten-stream runs.
Retain its provenance and any stripe-layout variants used by historical studies.
For a new comparison, specify the stream subset explicitly: replacing five JF
streams with the full ten-stream dataset can change the measured I/O workload.

## Historical pointers and cleanup boundary

The results branch still intentionally contains old dataset and evidence paths:

- `notes/archives/scaling-20261116/legacy-root/notes/mfx_expand10_100k_20261109/`
  preserves source generation, file manifests, and acceptance receipts.
- The same archive's `sim10_100k_bd_sweep_20261109/support.py` hardcodes the SDF
  100k source, and `sim10_stream_length_20261109/support.py` plus
  `sim10_striping_20261110/staging/stage.sbatch` refer to the private PM subset.
- `notes/archives/scaling-20261116/sources/` preserves the SDF source snapshots.
- Repository-root `validation/input-bookkeeping-20261110/ada-100k/harness/support.py`
  and runtime/provenance receipts also retain original paths and source hashes.

The 400k source has no equivalent tracked archive. Under the SDF scratch root
listed above, its generation record is `sample10_400k/generation.json` and its
acceptance receipts are in `sample10_400k_evidence_20261111/`. Preserve the
generation record as well as those sibling receipts before deleting its source
directory.

The `notes/` paths above are relative to `psana/psana/gpu/`. These are frozen
execution packages, not maintained launch defaults. Preserve their bytes and
checksums. Consult this guide and create a new qualified campaign configuration
instead of globally replacing paths in archives or launching frozen scripts.
The [maintained PR regression template](templates/pr-regression.md) provides
comparison orchestration, but still requires a qualified native adapter and
frozen site-specific inputs. Its real-data `mfx100848724` run-51 candidate and
the historical PR299 workload remain separate from Sim10 publication. Do not
substitute a synthetic run or enable qualification flags solely from this guide.

Before deleting either old 100k/400k source directory, check active jobs and
non-archived worktrees/configurations on SDF and Perlmutter, and update any live
consumer's experiment, run, filenames, hashes, reader, and stream selection.
This tracked-branch audit alone does not establish that those sources are unused.
Do not delete the sibling `sample10_*_evidence_*` directories, results provenance,
calibration snapshots, or unrelated scratch artifacts as part of source cleanup.
Keep the old 10k input unless separately reviewed and authorized for removal.

Hidden publication staging directories under `xtc/` are hardlinks to the
published files, not another full data allocation. Removing those links does
not reclaim the published data's space. No cleanup is performed by this update.
