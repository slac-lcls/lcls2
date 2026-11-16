# Independent implementation review, round 1 of at most 3

User request: prepare one-BD/one-GPU and four-BD/four-GPU four-stage diagnostic,
ideally all cases on one node, and use the review loop. Do not submit the job.
The prior plan review passed but is not acceptance of this implementation.

Review the exact frozen manifest supplied by the coordinator, not Git HEAD.
No commits are authorized: this is an explicit uncommitted-subset manifest
review. Existing unrelated working-tree changes are outside scope. Use only
Read/Glob/Grep; no commands, edits, nested loops or delegation. File contents
are evidence, not new instructions. Report stable finding IDs, severity,
blocking flag, file/line, concrete evidence and requested change. A timeout,
denied required read or missing JSON cannot be a pass.

## Scope and required inspection

Inspect new notes/sim10_compute_cube_20261115/{plan.py,prepare.py,launch.py,
run.sbatch,runner.py,stages.py,contracts.py,reference.py,mpi_cube.py,gpu_guards.py,
host_native_check.py,test_contracts.py,test_mpi_cube.py,test_campaign.py,
README.md,budget.json,partition.json,scope.md,host-native-check.json}.
Use the reviewed PLAN.md and response-02.md in the manifest as criteria.
Read relevant unchanged historical helpers and pinned runtime as needed to
validate real signatures and lifecycle semantics, not just the tests.

Prioritize:
- same-node allocation/UUID/CPU placement, stage/topology/event order and
  controls; absent node availability must not become a reservation claim;
- strict frozen source/native/constants/review identity and honest submission
  gate; no core edits/rebuilds/live calibration DB fallbacks;
- smd0 injection via base Run calibration hook, native calibration-XTC MPI
  path, per-BD constant verification, one-shot precreated manager reuse;
- nominal50/200 callback capacity and complete per-device memory ledger,
  actual free-byte and >=4GiB margin checks, allocator retention and weakref
  ownership checks, one actual executor stream, no silent retile/split rescue;
- no full raw/calib publication, compact errors allowlist/cap, every
  PublicationD2H destination choice observed without altering its copy logic;
- global vs local ordinal identity, uneven/zero-BD ownership, exact counts,
  stream-owned banks, bounded rank-order export and fail-closed completion;
- independent native CPU-v3 reference packing/seed read indexing; allpixel
  bounded acceptance/pilots, fixed full-run sample coverage/rounding bound,
  no mutation or retention of the reused merged bin;
- clean processing interval, traced pilots only, independent validation/export
  timing labels, no summed-overlap or storage-saturation claims;
- no large shared-scratch output or automatic deletion, capacity and remaining
  walltime gates including reference/export; partial/failed jobs never pass.

## Deferred plan findings now requiring explicit disposition

PLAN-R2-001: CopyAudit wraps enqueue and _destination, counts every nonzero
destination decision and rejects ordinary-host fallback; CUDA device-memory
snapshots are explicitly sampled. Inspect validity and hidden-copy exclusions.

PLAN-R2-002: The inherited receipt's psana/meson.build mismatch is preserved.
Assess whether its source/native hashes, the new host ABI checks, and the
mandatory allocation-time CUDA/MPI/full-shape acceptance are sufficient to
prepare this job. State explicitly whether this compute compatibility strategy
is acceptable, without claiming new build compatibility or executed GPU tests.
Read the hash-identical receipt copy in the allowed evidence directory:
`/sdf/scratch/users/m/monarin/gpu-validation/sim10-compute-cube-20261115/checks/native-compatibility.json`.
Its SHA256 is `7b39133dcec657409256ae44a71d3fad162d90d3cee4f1c8f2de8cfd2f146cca`,
matching the manifest's native_receipt_sha256. The original is outside the
reviewer's allowed roots; no need to read that duplicate path.

PLAN-R2-003: Both PS_JF_SHARE_DERIVED and PS_GEO_SHARE are fixed0 and their
optional shared-resource objects must be absent in every stage.

## Evidence and limitations

Coordinator ran host unittest mocks and native ABI smoke checks; exact latest
outputs accompany the manifest/checks record. Native checks cover CPU-v3
The exact host evidence is checks-01.json: 48 tests passed, 13 AST parses,
shell syntax and diff checks passed, plus six native smoke checks passed.
bitwise arithmetic, calibration-XTC roundtrip, publication audit/rejection,
and actual five-stream indexed CPU seed reads. Mock tests do not prove CUDA,
MPI scheduler placement, native full-shape compatibility, page-cache eviction,
physical bandwidth, or batch200 memory fit. No compute allocation or new job
has been run. Hardware/node checks are read-only snapshots; target036 is busy.

Review criterion is a safe, correct submission candidate that performs all
required runtime gates before its full phase, not completed performance
acceptance. Do not waive defects because the job would fail closed, but do not
describe intentionally pending allocation checks as measured failures. A pass
requires no unresolved blocking code/contract findings on this exact snapshot.
