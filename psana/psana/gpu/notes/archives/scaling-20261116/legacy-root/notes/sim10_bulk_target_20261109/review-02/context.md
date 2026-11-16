Independent results review, round2. Read/Glob/Grep only. Do not start another
review or use shell, MCP, delegation. The code-only review in review-01 passed
f849540905a19cc2c18d620702088ade120740fe with hardware pending. Read its findings
and response. Implementation is unchanged. Base for new evidence:35e094197.

User asked to sweep bulk targets512KiB/4MiB/8MiB at batch100, one GPU stream,
reporting actual GPU reader calls and cold Weka FFB throughput. See scope.md,
README.md, validation.json, tests.log, jobs.txt. Verify claims against copied
native evidence in build/bulk-review-evidence/{native,logs}, whose byte/SHA copy
index is build/bulk-review-evidence/copy-index.json. Large caseJSON timestamps
and callbacks can be inspected selectively with Grep; validation.json records
full subbatch and read-size histograms, callback and request counts, peaks, NIC
RX and final residency. README presents all subbatch histograms and limitations.

Job44009313 finished COMPLETED0:0 in5m50s on sdfampere042. Coordinator executed
audit.py successfully and all12 tests passed; exact command/evidence root are in
tests.log. Native sizes/counts match frozen independent expected-reference at
all targets. No new hardware reruns, production edits, calibration, user compute,
or D2H. Changes since code review are evidence/report only.

Acceptance: exact ordered10k events/callback identity,33,582,000,000payloadbytes,
actual completed reader request counts50k/9454/4599 pertarget, full read histogram
matching independent fixture-offset predictor, noCPU payloadreads/admission drains,
resource drain, identical host/device/runtime/settings, one CUDAstream, fixedEB100,
KvikIO8workers/1MiBtasks/16MiBbounce/fallbackON/autodirectOFF,6GiBbudget,SMD1000,
NUMAinterleave, shared56physicalcores. Cold gate covers5payloadfiles aftersetup,
NOT SMD or servercache. NICphysical counters bracketloop; separateidle; noexact
cachehitfraction. NoNVMeinput. Verify cleanup.

SBT-02 from round1 is the key results-report obligation: larger targets reduce
pending requests and split GPU executions into many smaller subbatches. Confirm
this is explicit and does not attribute rate changes solely to read size or
prove a particular component's bottleneck. Only two repeats, no section timings.
SBT-01 and SBT-03 are explicitly deferred low-priority cleanup with same queued
code. Assess any new issues too. The reviewed commit SHA will be appended by the
coordinator after committing this evidence. You cannot execute tests/hashes;
report those independent-verification limitations rather than inventing a pass.

Reviewed commit: f171fdb16c3ec9980a5bbbe60eee50de986645cc.
Note counter semantics: peak outstanding requests means submitted and not yet
retired via wait_batch; some physical I/O may already have finished. This is not
proof that all500requests were simultaneously active at storage. Please assess
whether report wording needs clarification. Coordinator plans to state this
explicitly in the final evidence-only wording.
