# Response to round 1

All findings accepted; independent closure requested on the next code commit.

| ID | Response and evidence |
|---|---|
| PM-R1-01 | run.sbatch runs scontrol with TZ=UTC; allocation rejects elapsed or >5400 s remaining. Non-UTC process-timezone regression covers correct UTC and incorrect local/future deadlines. |
| PM-R1-02 | campaign parses normalized UUID/PCI inventory; runner checks actual CUDA PCI against assigned UUID; independent audit repeats check. Corruption and bus-format tests added. |
| PM-R1-03 | evidence/adapter-port.diff records the complete original-to-port diff. The adapter now hashes jungfrau_calibration.py before executing it. Exact pinned PR298 source is reused; numerical code unchanged. |
| PM-R1-04 | Added real tiny-file ECC readback and manifest/size/shape/dtype/pixel/empty/count corruption tests; production full-shape default tested. Added physical-core prefix/SMT/overlap/mask tests, storage allowlist rejection, bundle/file/external/plan tampering, and process timeout termination. |
| PM-R1-05 | Host evidence now records git HEAD, task working status, and all bundled file-type hashes. host-checks05 records the precommit run (90 pass +64 subtests); an exact clean-commit run will be supplied in round-2 context. |
| PM-R1-06 | Nonblocking flock acquisition moved inside failure-status handler. |
| PM-R1-07 | Preparation pins the calibration manifest; existing load_constants hashes all mapped arrays against it before use. |
| PM-R1-08 | PLAN explicitly states budget basis and timeout risk; no per-profile runtime prediction asserted. Reference probe moved before acceptance; rank setup/conditioning progress receipts survive a later timeout. A changed budget requires another reviewed plan/submission approval. |

Additional coordinator finding: AST-extracted device_guards lacked its expand
CPU oracle and would fail on the first valid GPU case. Added the unchanged
original verify.py with source provenance; load_guards extracts expand first.
Host regression resolves every global and tests seed shape/dtype and tail mapping.

No GPU test or compute job was run. The 90-minute proposal and fixed workload
remain unchanged. Scope authorization persists in AUTHORIZATION.md.
