Independent review of the authorized sim10 bulk-target sweep. Read/Glob/Grep only;
do not start another review, invoke shell/MCP, or delegate. Inspect all maintained
files under notes/sim10_bulk_target_20261109 and relevant reused helpers.

Base:60b7d7ddd. Implementation:f849540905a19cc2c18d620702088ade120740fe.
Scope is scope.md. The user requests varying bulk target at fixed batch100,
one BD/GPU/CUDAstream, reporting actual GPU reader calls and cold Weka bandwidth.
Targets512KiB,4MiB,8MiB. Historical settings8workers,1MiBtask,16MiBbounce,fallbackON,
autodirectOFF,6GiBbudget,SMD1000,sharedphysicalaffinity,NUMAinterleave. Only benchmark
scripts change, not the production framework or frozen B runtime.

The current production bulk grouping changes small-group ownership and execution
cuts as well as read size; assess whether the report keeps these coupled effects
clear and avoids asserting isolated storage causality. Every full case has10k
events and33,582,000,000payloadbytes; expected reader calls50k/9454/4599 computed
from fixture offsets without importing the production planner. Gaps break groups,
including intervening transitions; batches limited to100events. expected-reference
is frozen before hardware. Runtime counters include size histogram and completion
counts. Admission drains/CPU payload reads must be0; resources drained.

Inputs are SHA-verified copies of five accepted sim10 bigdata+SMD files onFFB.
Per-case eviction/mincore covers only5payloadfiles, afterDataSource setup. SMD and
servercache uncontrolled. NoNVMedata. NICphysicalcounters bracket loop withidle
samples; not exact cachehitfraction. Three200eventpreflights warmcompilercache;
lazyprocessallocations remain timed. No kernels, gather, calibration, expansion,
payload D2H. This sweep checks descriptors/identity/IO, not pixel values.

Reused helpers: notes/jf_raw_cache_20261109/{common.py,runtime.sh,reference.json,
pixels.json,expected-sha256.json}; notes/jf_cold_ffb_20261109/run.py(cacheonly);
notes/mfx_kvikio_read_20261108/{network_watch.py,plan_reads.py};
notes/mfx_expand10_performance_20261106/fixture_check.py;
notes/mfx_expand10_stage2_20261105/adapter.py;
notes/jf_framework_cold_20261109/sim10_identity_check.json.
The new bulk_accounting.py adds read size counts to previously reviewed accounting.

Use saved native counters to check claims, not just logs statingpass. Large
caseJSONs have timestamp/callbacklists; targetedgrep plus validation.json compact
summary helps. You cannot execute checks/hashes; disclose this limitation.

REVIEW PHASE: implementation-only while hardware is queued. Reviewed SHA is
f849540905a19cc2c18d620702088ade120740fe. No native hardware evidence exists yet.
Do not require nonexistent README/validation/job output files in this checkout.
Do NOT approve measured results or overall benchmark acceptance. Review the code,
independent expected read prediction, configuration, ownership/counter semantics,
and whether it is ready to run. A pass on code must explicitly say hardware
validation and the complete benchmark remain pending. Python syntax and shell
syntax checks passed; no hardware or post-run audit tests have run yet.
Job44009313 is queued, exclusive700GiBampere, expected start14:03Pacific; min10min,
max20min backfill window. Result review is planned as a second phase after the job.
