Review final code and native result acceptance at 185087d6efcc64d59113595605ce4cd8234bcdbb. Base d7e573f7b. This is round02 (limit3), following the initial readiness review at cfadd496a. Do not delegate, run commands, edit, or invoke another review loop. Read prior perlmutter-review-01/review.json and responses.md, scope.md, final README, implementation scripts and supplied native evidence. Decide whether the code plus measured results satisfy the agreed scope; do not merely repeat the earlier readiness verdict. Identify blocking issues and unresolved finding IDs if any. Do not treat unavailable reads or missing required evidence as approval.

The task is the same accepted five-stream Sim10 fixture on Perlmutter Lustre: one BD/one used GPU, 1/2 CUDA streams x same-file10k/100k, reversed repeats plus pilots. Fixed batch100/SMD1000/bulk524288/6GiB/compatON/8workers/1MiB tasks/16MiB bounce/directIOoff. Metadata-only staging. Explicit local-page-cache-only mode; remote traffic unverified. No framework, installed-runtime or other-checkout edits; no SDF allocation resumed.

Changes since review01: validator distinguishes sim10 Names from config Names; selection comes from the Slurm allocation and each rank verifies original step UUID membership before masking; audit checks relocation mapping and exact 100-event callback histogram; tests expanded to47 with direct fixed-control and native-validation mutations; derived summary checks physical inventory PCI and chronological timing and reports repeats/medians/paired ratios. Extra native and counter probes are outside timed cases. No changes to the16 hashed execution/reference files after 97399922b; later commits add tests, auxiliary probes, reports and native evidence. The test log pins its actual script SHA.

Native results are complete. Job61065065 and all steps COMPLETED0:0 in35m58s on nid001029. Independently SHA256-verified all11 transferred files; all Configure/Names/physical panel order/SMD timestamp+offset checks and full100k DataSource traversal passed. Native Cray MPI/CuPy/KvikIO/CUDA probe passed before pilots. Both pilots and all8 main cases passed. Independent audit passed.47 native mutation tests passed in26.622s. Input size/uid/inode/mtime are unchanged before pilots versus after the sweep; no temporary XTC inputs. These are coordinator-generated native test results, not tests run by the reviewer.

Compact, exact copies under notes/sim10_stream_length_20261109/native/:
- input-validation.json, runtime-probe.json: native data/runtime evidence.
- provenance.json, complete.json, cleanup.json: source identities, exact commands, case list.
- audit.json, summary.json, test-audit.log: recomputed checks, comparisons and47-test result with test script SHA.
- slurm-accounting-utc.txt: every job/step completed0:0 with UTC timestamps.
- input-stat-before.json / input-stat-after.json: unchanged dataset identities.
- counter-probe.log: absolute /usr/sbin/lctl could not expose llite stats; debugfs denied; CXI gives codeword/BER/pause rates. Probe was after complete.json, not a remote-byte measurement.
- artifact-index.json: sizes/hashes and absolute paths of all full receipts/logs.
- commands.txt: exact submission/probe and audit/test invocation descriptions.

Full originals are at /pscratch/sd/m/monarin/psana2-gpu/sim10-stream-length-20261109/attempt03/results/ (pilot-s1.json, pilot-s2.json,00-s1-n10000.json through07-s1-n10000.json and logs). You may Read those artifacts as needed; they contain long timestamp arrays, so prefer targeted sections and compact audited copies. Report any denied access; do not claim to have reread/hashes-tested the335GB dataset. The artifact index supports original-to-copy provenance; the tests supply runtime evidence.

Main medians,1/2 streams:10k2.165964/1.259323GiB/s;100k2.194919/0.968757GiB/s. Full two-stream repeats0.914207 and1.023307 differ~12%. Full/prefix paired ratios1.012/1.015 for1stream,0.728/0.811 for2streams. README deliberately excludes pilots from the stream comparison; startup pilots favored2streams. Review numerical claims, data/measurement scope, source identity and acceptance checks independently. Backend/SMD caches are uncontrolled, later prefixes follow full reads, pipeline startup/fill/drain remain timed, no simultaneous-execution or causal mechanism claim, no direct explanation of SDF/Weka or hardware-ceiling claim.

Prior finding dispositions: PMR01 measured job fits1hour; PMR02 tests expanded+pinned; PMR03 docs updated; PMR04 summary added; PMR05 broader failure-supervision refactor deferred but successful actual Slurm exits/retirement/input readback supplement cleanup code-path assertions; PMR06 original masks+UUIDs+inventoryPCI checked; PMR07/SLN01/SLN04 retained limits. BDS08 now has native evidence and awaits your acceptance verdict. Nonfatal module-telemetry Lua and MPICH communicator warnings are disclosed alongside0:0 exits. Challenge any remaining false-acceptance path or overclaim.

ROUND03 OVERRIDE / ACCESS COMPLETION:
This is round03, the final round, on the SAME unchanged reviewed code commit
185087d6efcc64d59113595605ce4cd8234bcdbb. Round02 returned pass with no blocking
findings, but reads of original per-case receipts were denied. The workflow does
not allow denied reads to count as approval. This invocation therefore adds
--add-dir for exactly /pscratch/sd/m/monarin/psana2-gpu/sim10-stream-length-20261109/attempt03/results
while preserving safe mode, restricted mode, Read/Glob/Grep only, dontAsk and
empty MCP. A separate access-only probe successfully read complete.json with
these same restrictions. The bundled helper was copied into task scratch and
changed only to add this one read directory; the personal helper is unchanged.

Read the full round02 verdict/findings at the now-allowed path
/pscratch/sd/m/monarin/psana2-gpu/sim10-stream-length-20261109/attempt03/results/review-02.json.
The previously unavailable original receipts/logs are in that same directory.
Inspect original raw case evidence as necessary to close the access limitation:
pilot-s1.json, pilot-s2.json, and all00..07 case files listed in complete.json.
Use Grep for line locations and bounded Read sections because timestamp/callback
arrays are long. Confirm original masks/UUIDs, source identities, cold records,
I/O/drain/cleanup and relevant timing against the compact audited results. You
are not expected to manually revalidate every timestamp or rerun tests; inspect
the fail-closed audit and use the provided native test evidence. Do not claim to
have run hashes/tests yourself.

Additional independent coordinator evidence is available at
.../attempt03/results/reviewer-verification.json (use the full root above).
The coordinator reread and SHA256-checked EVERY indexed original receipt/log
against the committed native/artifact-index.json, recomputed current16-source
hashes against provenance, and matched current test_audit.py SHA to the native
47-test log. Status is pass, checked_commit185087d6e. Compact copies were also
independently SHA-compared with their indexed originals. This is supplied
verification evidence, not work the reviewer performed.

Round02 findings: BDS08 resolved, numerical results and code pass; PMR03 stale
status wording to be corrected in documentation-only closure after review;
PMR05 literal cleanup booleans acceptable for this job with actual Slurm exits
and stat readback; PMR08 optional observation of500 pending requests and335820000
bytes (one batch) for BOTH stream counts. We will add that bookkeeping observation
without claiming simultaneous I/O or causal mechanism. PMR07/SLN01/SLN04 remain
explicit interpretation limits. No execution source or data changed after round02.

Your verdict must explicitly address whether the former raw-receipt read denial
is resolved. Report any new denied required read, missing evidence, regression
or overclaim as a limitation/finding, never as approval. Native acceptance is
still pending this final access-complete review. Review only this bounded task;
do not delegate or invoke another loop.
