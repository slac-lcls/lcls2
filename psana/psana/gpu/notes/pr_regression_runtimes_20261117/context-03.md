# Runtime driver final implementation/evidence review (round3)

Review exact62fa31992725195cad29ea99e585daf192477174. Prior95ea0c4ed passed round2.
Production prepare_runtime.py/native_contract.py and launcher are unchanged since
that review. Only added tests, docs and evidence. User goal remains reusable
runtime preparation drivers, with real build/probe validation and independent
review; no site driver, benchmark, dataset access or qualification enablement.
Public repository, scoped non-sensitive code/metadata only; no raw data,
credentials, shell/delegation or nested review. This is final planned round.

Under psana/psana/gpu/notes/pr_regression_runtimes_20261117 read README, scope,
review-02, response-02, validation-03, capture-validation-01,
capture-accounting-final.txt, capture-61483852.log, artifact-index,
submission-verification-01, build-validation-01, capture-01.sbatch. Review
scripts/performance/regression test_prepare_runtime.py (new success/mismatch
test), RUNTIME.md boundaries and docs/performance/template links. Revisit driver
and native_contract as needed. Note prior pending status is historical; status
and README intentionally say final evidence review pending until your verdict.

Actual job61483852 COMPLETED0:0 in30sec, step0 COMPLETED0:0 in27sec on nid003109.
Actual allocation128CPU/4GPU/16G, request4CPU/1GPU/16G, step4CPU/1GPU. GPU A10040GB,
Python3.11.14, CuPy14.0.1, KvikIO26.10.00, CUDA runtime12090, driver13000.
CPU no-GPU-module/mapping guard passed. MPICH singleton, NVRTC kernel and exact
4KiB fallback KvikIO read passed; adapter runtime_evidence verifier passed. Source
and installed inventory rechecked before/after; parent association checked.
Manifest6082ff8edb359bd8596cddb6e7bb070b09ddb641ec8800a26a1ace36283a5b35
matches build receipt, code SHA95ea0c4ed and final job/step. Raw manifest/build/log
artifacts stay in scratch; compact hashed summaries committed. No GPU/data
rehashing on login. check-pair using same manifest twice PASS, not timed A/A.

RUN-001: fresh final sacct retained, all relevant job/batch/step rows success.
Optional trap-before-checksum reorder deferred; hash mismatch never accepted.
RUN-005: new fake-child test covers successful publication and wrong commit,
install, receipt association rejection.71 total tests PASS; no production change.
Directory fsync deferred; atomic publication claimed, not crash durability.
RUN-006: docs restrict check-pair to unmodified driver-generated canonical paths.
RUN-007: pre-sbatch code+launcher blobs compared against reviewed Git and frozen
files; post-job manifest code hashes agree. Check summaries vs artifacts for any
unsupported claim. MaxRSS0 in sacct is explicitly not treated as measured zero.

Dependency coverage is limited to named package trees and mapped MPI/CUDA-related
libraries, not whole transitive closure. Inherited environment hashed/key-listed,
not claimed fully qualified. No multi-rank/six-case/cache/timing/numerical result.
Future campaigns freeze exact executed package while maintained driver evolves.
No default profile, old campaigns or input-copy changes. Return exact SHA and
stable finding IDs, distinguishing supplied execution evidence from file review.
