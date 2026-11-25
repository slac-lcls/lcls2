# Round3: completed input evidence and audit fixes

Review exact SHA 6238a8d342edb438899093c0b6109d43ad8d3bfb.
Prior reviewed preparation code SHA199b5091c0482d015609ed728efa1aaaab0240ab
passed round2 and is unchanged. This is the third/final planned round.
Public slac-lcls/lcls2 repository (public visibility verified). Read only scoped
committed code and compact metadata; no detector payloads, credentials or
external private files. No shell, delegation, nested review or job submission.

User scope: prepare private real run51 inputs and independent first10k reference,
with exact source/copy identity and review. No benchmark, runtime build, cache
qualification or enabling profile. See notes/pr_regression_inputs_20261117 under
psana/psana/gpu. Inspect scope.md, audit_prepared_inputs.py, response-02.md,
review-02.json, validation-03.json, audit-guard-validation.json, README.md,
source-scan.json, input-audit-02.json, frozen-verification-02.json,
preparation-started.json, preparation-complete.json, prepare-61480775.log,
artifact-index.json, attempt-01-failure.json and prepare-02.sbatch. Also inspect
prepare_inputs.py/native_contract.py under scripts/performance/regression as
needed. No archived campaign or default-profile changes.

Actual job61480775 completed 0:0 in28:34 on nid002273. All ten full source/copy
hashes matched;50k selected headers and415 SlowUpdate gaps checked by the reviewed
compute code. Metadata/receipt audit PASS after completion, including SMD rehash,
all ten stripe layouts, ownership/stat and frozen code hashes vs reviewed Git.
The audit did not rehash bigdata on login. Receipt/manifest/reference identities
are retained; large plan/reference payloads stay in scratch with hashes.
55 offline tests passed at reviewed199; no preparation/adapter code changed
since that test. Audit guard tested normal help=0, optimized Python refuses=1.
Diff check passes. Final report still says review pending until your verdict.

PREP-005: explicit sys.flags.optimize rejection prevents disabled assertions.
PREP-006: final audit checks started receipt, exact destination/bundle, read-only
bundle/inventory/SHA256SUMS, account/QoS/time/resources. Your suggested allocation
of8CPU/1GPU differs from actual scheduler behavior: requested8CPU/1GPU, allocated
128CPU/4GPU full node under premium, mem32G. Both are explicitly checked and
recorded separately, backed by raw sacct rows. Evaluate this evidence-based
correction rather than silently waiving a mismatch. GPUs hidden/unused.
PREP-001: before/after frozen-code verification vs reviewed Git and exact bundle.
PREP-003: actual stripe checks cover SMD and bigdata.
PREP-004: failed attempt retained invalid; fresh job has successful elapsed,
state and exit evidence. Logs lack timestamps per stream; report does not invent
them or claim controlled measurement of the random-read optimization.
PREP-002: optional gap timestamp and prefix non-L1 validation remains deferred;
claims limited to L1 identities and gap type/length. Single native step, no full
EndRun coverage. Source untouched. Durable retention and SDF qualification remain
separate. Review code, evidence consistency and claims, including new report links
in docs/performance/README.md and templates/pr-regression.md. Return stable finding
IDs and exact reviewed SHA; distinguish supplied execution evidence from checks
you can independently perform with file reads.
