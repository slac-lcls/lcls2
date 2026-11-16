# Reader review round1 responses

R1 fixed: reviewed bundle.py creates source snapshots from git show at a full
commit, rejects working-source differences, verifies binary/compiler hashes
against reviewed build-pins.json, records exact key set/flags and all hashes,
and makes bundle read-only. Runtime/audit require exact key set and verify
reader,reader.cpp,compiler pins and inequality to test-binary hash. Unit test
rejects missing source and altered binary even if provenance is updated.
R2 fixed: every case first invokes production --evict-all for all30inputs,
checks every page twice with50ms separation, and validates complete path/id
coverage. Timed reader independently clears its five inputs after worker ready.
Memory and llite.max_cached_mb observations are saved before/after each case
and audited. No claim is made to clear server/other Lustre client caches.
R3 fixed: pilot probe coordinator descriptors use RANDOM advice (worker opens
remain default); positive residency must stabilize across50ms scans before
eviction. Every final cold gate requires two zero scans50ms apart. Bounded
sampling cannot prove absence of unrelated external-client activity.
R4 fixed: test_audit.py builds complete pilot and54-case sweep fixtures, checks
estimate/18summarycells/CSV, and mutates missing receipt,raw hash,command,
schedule,Slurm failure and memory fields. validate_inputs and bundle.verify are
stubbed in audit fixtures; native metadata parser and bundle verifier have
separate tests. These fixture results are not native runtime evidence.
R5 fixed: audit requires saved sacct job/batch/extern/step0 COMPLETED0:0,
correct account/QoS,node match to native scontrol record,status==complete.
R6 fixed: EXIT trap captures counters after either success/failure, preserving
original exit code. Remote bytes still unverified.
R7 documented: submission must override --time from audited pilot estimate and
remaining campaign budget; two endpoints do not bound all18 combinations.
Coordinator will record and verify the actual requested time before sweep.
R8 deferred: failing workers still invalidate trial; other reads may continue
until completion or900s timeout. Optional cancellation doesn't alter acceptance.
R9 fixed: NUMA node per CPU recorded. Explicit locality confound limits cross-
thread-count causal claims; stripe counts compared within fixed worker count.
R10 fixed: verbose22-test log plus test-provenance.json with host,compiler,
production/test SHAs and exact command.12 internal receipt mutations included.

Build04 uses GCC14.3.0,warnings-as-errors.22 tests pass13.349s. Shell syntax,
Python AST,diff checks pass. No native pilot or sweep has been submitted yet.
