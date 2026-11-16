# Stage 3 synthetic MFX pilot

Stage 3 is complete: the 60-event pilot is generated, validated, independently
reviewed and published. No calibration DB writes or psana core changes.

- Destination identity: `prjdatpsana`, run 1.
- Source: first 60 correlated events of public `mfx101555026`, run 10.
- Ten streams, one chunk each; step counts `[19, 20, 21]`, values `[0, 1, 2]`.
- 76 records per stream; total XTC plus SMD size **2,040,571,066 bytes**
  (2.041 GB / 1.900 GiB).
- Staging: `/sdf/data/lcls/ds/prj/prjdatpsana/xtc/.synthetic-mfx-r0001-stage3-20261104`.
- Final target: `/sdf/data/lcls/ds/prj/prjdatpsana/xtc/`, with SMD in `smalldata/`.
- Durable results: `/sdf/data/lcls/ds/prj/prjdatpsana/results/synthetic_mfx/r0001/stage3-20261104/`.

## Implementation

`xtc_fixture.py` validates the frozen manifest/SMD hashes and source file
identities, indexes source records, checks XTC/Names/ShapesData wire layouts,
and derives the approved schedule. The full-run source index uses recorded
SMD offsets. SlowUpdate lengths follow `Smd::generate()`'s verbatim-copy rule,
and the next L1 offset checks the inferred cursor. Source transition templates
are read from each stream's own bigdata. Only selected bigdata records are
read in full; no 1.224-TB scan is needed.

`make_pilot.py` is deliberately limited to this fixture. It retains source
bytes, appends DgramEdit-produced scan Names/ShapesData to s000, rewrites all
BeginRun identities, copies same-service transition templates and updates
SMD offset/size fields. NamesId 253 is unused in both s000 Configures. Source
SMD and bigdata Configures remain distinct. Every output record is decoded by
psana during generation. Each file is created exclusively, flushed and fsynced.
`generation.json` records the full per-record source/output mapping and source
hashes; it is saved in staging and durable results. The committed summary
references its path/hash rather than duplicating the larger trace.

`verify_pilot.py` reads all output records and source counterparts. It compares
every byte after restoring only the explicitly permitted timestamp, runinfo,
scan append/extent and SMD-offset edits. It independently checks decoded scan
and run metadata, schemas, every SMD offset/size, nested extents, complete
state-machine walks, counts, source timestamp equality, hashes and EOF.

`smoke_pilot.py` uses normal serial `DataSource(exp=..., run=1, dir=...,
live=False, skip_calib_load='all')`. Both `run.steps()` and a separate
`run.events()` pass return the original 60 timestamps. Standard `step_value`
and `step_docstring` detectors return correct values on steps and events.
Calibration is intentionally skipped; this does not complete Stage 4.

`pilot_admin.py` performs authenticated GET-only logbook collision checks and
no-overwrite publication, requiring a passing structural report and a passing
review at an exact implementation commit. It rehashes staged files, checks
implementation bytes against that commit, rechecks disk/logbook collisions,
checks that both target directories exist and are writable/traversable,
compares the recorded generator/validator hashes with the reviewed Git blobs,
and hard-links bigdata before SMD. The twenty-file set is **not atomically
published**. A fsynced intent/completion journal records each link, and a
partial failure must be inspected rather than retried blindly. No existing
files are overwritten or removed. Staging remains as hard links for provenance,
so it does not require another full physical copy of the dataset.

## Correction to the earlier audit

The earlier 32-byte BeginRun experiment-field claim was wrong. Direct
ShapesData inspection found a **16-byte field in all twenty source BeginRun
records** (SMD and bigdata); see `runinfo_layout.json`. The writer derives its
width from the schema/shape and checks that `prjdatpsana` plus NUL fits, preserving
the original field width and padding. The proposal is corrected. All generated
identities decode correctly through psana.

## Checks and reproduction

Run from the repository root using `bash notes/synthetic_mfx_stage3_20261104/runtime.sh`.
This selects the Stage 1 Python 3.11/psana installation and sets
`PS_FAKESTEP_FLAG=0`, `PS_PARALLEL=none` before import. Fresh output paths are
required; do not rerun the generator over existing staging or publication.

```bash
bash notes/synthetic_mfx_stage3_20261104/runtime.sh \
  notes/synthetic_mfx_stage3_20261104/test_fixture.py -v

bash notes/synthetic_mfx_stage3_20261104/runtime.sh \
  notes/synthetic_mfx_stage3_20261104/verify_pilot.py \
  --root /sdf/data/lcls/ds/prj/prjdatpsana/xtc/.synthetic-mfx-r0001-stage3-20261104 \
  --report /tmp/new-pilot-verification.json
```

`unit_checks.txt`: seven checks pass, including second rollover, tight-gap
spacing, insufficient-gap rejection, malformed XTC extent rejection and
preserving damage/control bits. `verification.json`: all ten streams pass.
`smoke.json`: serial discovery, counts, scan state and timestamps pass.
`preflight.json`: no run-1 disk files or logbook entries before generation.
Directory fsync support was checked. Sample output ACLs retain read access for
`prjdatpsana`; actual collaborator-login verification remains Stage 4.

No full psana test suites are claimed: this adds an offline fixture tool and
changes no psana core code. Stage 4 still requires calibrated image/reference
comparisons, supporting-detector/EPICS checks, CPU MPI/batch-boundary coverage,
and collaborator-session readback. Run-2 generation/deployment is outside this
stage. Review results and publication state are recorded in `status.json`.

## Review corrections

Round 1 passed with three nonblocking safeguards, all implemented: generator
and validator provenance must match the reviewed commit; the independent
validator checks source-record timestamps against the index; both destination
directories are checked before publication. `verification_reviewed.json` is
the new all-record pass report and the publication input.
`unit_checks_reviewed.txt` records ten passing tests, including three provenance
rejection subcases and missing/unwritable-directory checks before any
authentication or links. Original reports are retained for the audit trail.
The generated files and their hashes are unchanged. See `responses.md`.

Round 2 passed and closed all three findings at implementation commit
`3c77c8b189fcdbc99a10760325f0727069e4f249`; see
[the final review](round-02/review.json). Subsequent commits record evidence only.

## Publication and handoff

Publication completed at `2026-11-04T18:17:09Z`: all ten bigdata and ten SMD
files have canonical `prjdatpsana-r0001-sNNN-c000[.smd].xtc2` names. The fresh
preflight found no conflicting files or logbook entries. The publisher checked
generator/validator hashes against the reviewed commit and rehashed every
staged file before linking. All published files share device/inode/size with
their staged counterparts. See [the publication journal](publication.jsonl).

[The final-directory smoke test](published_smoke.json) passed normal serial
DataSource discovery, both step/event iteration paths, counts `[19, 20, 21]`,
scan values and original timestamps. It skips calibration and does not replace
Stage 4. Use `exp="prjdatpsana", run=1`, explicit
`dir="/sdf/data/lcls/ds/prj/prjdatpsana/xtc"`, and `live=False`; no logbook
registration was performed. Set `PS_FAKESTEP_FLAG=0` before importing psana.

[Published ACLs](published_acls.txt) show read access for `prjdatpsana` on all
twenty outputs and traverse permission on the experiment/XTC/SMD directories.
Actual collaborator-login readback remains pending. Durable results include
the implementation archives, generation trace, both verification reports,
reviews, journal, smoke results and `SHA256SUMS`.
