# Response to the supplied Claude audit

The incoming audit verified snapshot checksums/copies, calibration selection,
experiment-only deployment, stream alignment/SMD offsets, and timestamp schedule
arithmetic. These are **user-supplied independent audit results**, not new
EventBuilder or generated-file tests by this response. The frozen Stage 1/2
reports remain unchanged. The pasted finding 3 has a truncated final sentence;
the response below addresses its explicit header-construction concern.

## Finding dispositions

| ID | Incoming finding | Response and evidence | Remaining execution check |
| --- | --- | --- | --- |
| C1 | No scan Names or BeginStep payload; `step_index` is nonstandard. | Accepted. Plan now adds scan Names to `s000` Configure in both files, uses DAQ `scan`/`scan`/`1234`, `raw` 1.0.0, INT64 `step_value`, CHARSTR JSON `step_docstring`, and populates all `s000` BeginSteps. Require collision-free NamesId, matching schemas, rebuilt extents and decode checks. `source_preflight.json` confirms no scan detector and header-only BeginSteps in all ten SMDs. DAQ convention is in `psdaq/psdaq/control/BlueskyScan.py`; `psana/psana/detector/envstore.py` supports Step and Event arguments. | Stage 3 writes and decodes the added schema/data. Stage 4 checks both scan detectors on each step and event. |
| C2 | No Enable chunkinfo to rename; all streams need runinfo changes. | Accepted. Plan distinguishes declared chunkinfo Names from absent Enable payloads; no payload is invented. Rewrite all ten BeginRun identities and assert agreement. Preserve the 32-byte experiment field, NUL termination/zero padding and declared run type. `_get_runinfo()` in `psana/psana/psexp/ds_base.py` reads only stream zero. Preflight decodes source identity in every stream and finds no Enable chunkinfo payloads. | Decode every output BeginRun in both SMD and bigdata; validate any chunk payload if a different input introduces one. |
| C3 | Inserted transitions lack env/src/damage/type construction rules. | Accepted. Plan requires each stream's own same-service source template, preserving env, src, damage, type/version and payload. Only timestamps change except explicitly listed schema/payload/extent and run-identity edits. Preflight records SMD template offsets, hashes and every header field for seven services per stream. | Stage 3 verifies corresponding bigdata templates and allowed byte changes, plus output extent/offset/timeline checks. No generated-transition validation is claimed yet. |
| C4 | PV constants come from a RIX camera; run-2 shared fallback is live. | Accepted with provenance clarification. Manifest records `rix102353327` run 38 and `alvium3rix1`. `inline_alvium` is the destination's representative alias after deduplication, not scientific provenance. `pv_raw_1_0_0` in `misc_detectors.py` exposes raw fields, not calibrated values. Existing records stay intact. Run-2 deployment and hash/ID readback are explicitly mandatory before generation. | No claim of scientifically valid camera/QADC calibration. Stage 5 pins run 2 separately. The audit's specific latest shared pedestal candidate is time-dependent and was not re-queried here; the plan must not depend on its identity. |
| C5 | Collaborator read access is unestablished. | Partially resolved by current evidence. Unsandboxed `id seema` includes `prjdatpsana` (5037); `getent group prjdatpsana` lists Seema. Root, XTC and SMD ACLs grant that group read/traverse; XTC/SMD defaults include it. See `access_and_git.txt`. This is account/directory evidence, not a read under Seema's login. | Keep collaborator acceptance open: check ancestor/final-file ACLs and actual listing/open plus a calibrated DataSource read under Seema's login. Do not change ACLs or claim Mona's read proves Seema's access. This does not block preparing the local pilot. |
| C6 | Fake steps and unregistered live discovery can invalidate checks. | Accepted. Plan requires `PS_FAKESTEP_FLAG=0` before psana import on all ranks and records that setting. Examples reject a nonzero inherited value. Use explicit `dir=` and `live=False`; recheck collisions before publication. Last recorded logbook file listing was empty; it was not re-queried for this response. `smdreader.pyx` reads the flag at initialization. | Stage 4 records actual environment, disk discovery and complete counts; network errors must not be interpreted as an empty listing. |
| C7 | Nothing is committed. | Superseded by checkpoint. `git ls-files` lists the plan and ten Stage 1/2 note files; local HEAD and remote task branch were both `e118b3c7eec28203f06a1c9594306d59b58af4c5` at response start. `access_and_git.txt` records remote evidence. This predates these response changes. | Commit and push this response and compact review records, preserving unrelated untracked work. Large frozen snapshots remain in project storage by design. |

## Validation performed for this response

`audit_stage3_inputs.py` reads the ten frozen SMD files using the Stage 1 psana
installation, checks each SHA256 against the manifest, inventories Configure
detector Names, decodes BeginRuns and Enables, checks BeginStep sizes and records
per-stream transition templates. It creates only a new JSON report; no XTC or
service writes. All ten passed. It does not verify bigdata templates, regenerate
the schedule, or repeat the calibration DB deployment/readback.

Run with Python 3.11 from `xpp_drp_gpu_311`, `PS_FAKESTEP_FLAG=0`, and
`PYTHONPATH`/`LD_LIBRARY_PATH` pointing to the Stage 1 B installation recorded in
`source_preflight.json`. Example from the repository root:

```bash
python notes/synthetic_mfx_stage1_20261103/review_20261104/audit_stage3_inputs.py \
  --snapshot /sdf/data/lcls/ds/prj/prjdatpsana/results/synthetic_mfx/r0001/stage1-20261103 \
  --output /tmp/new-source-preflight.json
```

The helper uses assertions as an audit and must run without Python `-O`.
This documentation/audit-helper change does not modify psana core and does not
require the full psana runtime suites. Stage 3/4 acceptance remains unexecuted.

## Automated review follow-up

Round 1 timed out at 900 seconds without a verdict; it supplies no approval.
Round 2 reviewed `eafec66243fa387753eab37f3f8097ca2a41d860` and passed with
one low, nonblocking finding, R1: C7 counted eleven note files instead of ten
plus the plan. Corrected above; the original files and checkpoint are unchanged.

Coordinator follow-up C3a: `configure_sizes.json` compares the Stage 1 bigdata
Configure sizes to the freshly verified frozen SMD headers. All ten SMD
Configures are 2,096 bytes larger. The plan now explicitly preserves both
schemas, compares shared transition header fields, and validates each record's
own extent rather than requiring SMD/bigdata Configure or L1 extent equality.
This is a construction-specification clarification, not a new bigdata read or
generated-file test. The final revision receives a focused third review.

Round 3 passed `0303fe5b9f399189e4869a0202da42445136911e`, explicitly closed
R1, and confirmed the per-file extent clarification and all ten size comparisons.
There are no unresolved automated-review findings for this audit-response scope.
C5 collaborator-session readback and Stage 3/4 generated-file/pipeline acceptance
remain pending, as specified above. The final verdict/status commit changes
review records only; it does not change the reviewed plan or helper.
