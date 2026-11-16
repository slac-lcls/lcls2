# Standalone read pilot and screening

Implementation pending independent review and native pilot. Existing30 copies
are accepted; this phase measures only buffered CPU reads. No psana/MPI/CUDA
in the hot path. This is screening, not a final stripe-count recommendation.

reader.cpp uses independent pread descriptors,1MiB requests, disjoint contiguous
partitions covering each of five files, and5/20/40 pinned threads. Each thread
opens its descriptor and first-touches its buffer before joining the ready
barrier. The coordinator performs DONTNEED and scans every file page with
mincore, allowing3 attempts. A trial starts only when all files report0 resident
pages. Pilot cases additionally read1byte per file and require positive mincore
residency as a positive control. Windowed mincore includes partial EOF pages.
Default kernel readahead is retained; its effects belong to the measured access
pattern. Backend cache state is uncontrolled; remote bytes remain unverified.

Timing starts at barrier release and ends at the last successful worker's
last read, including release skew and excluding setup/cache eviction/cleanup.
Worker CPU times, offsets, bytes, call counts and start/end times are retained.
Zero-length partitions can occur only in tiny correctness fixtures; those
workers still participate in the barrier. Any read/setup/cache error exits1
with a failed record and no throughput. Thread creation/cache failures release
and join existing workers before exiting. Each input identity is checked
through the open descriptors and against the accepted staging receipt.

build.sh produces separate production and READER_TESTING executables. Fault
injection and per-byte checksums exist only in the latter. Production source
and binary hashes, compiler version, explicit flags and the accepted staging
receipt hash are preserved in each immutable bundle. The completed staging
receipt is pinned by SHA256 in campaign.py. Metadata/layouts are checked before
and after each campaign; no additional full SHA256 data pass is included.

campaign.py selects one logical CPU for each distinct package/core pair from
its allocation mask, sorts the mapping and fixes the first40 for the entire
job.5/20-thread cases use the corresponding prefixes. Worker binding,first
and last observed CPU are recorded and verified. Warm/control and timed reads
are serial between cases; no staging or profiling overlaps them.

Pilot: count1 with5workers, count32 with40workers, one full read each; excluded
from rankings. Sweep:3 randomized complete blocks of all18 count/thread
combinations, seed20261111,54 cases. Save the exact schedule before execution.
Each case is limited to900seconds and any failed case stops the campaign;
partial records remain, no complete marker. No automatic retries or selective
outlier removal. SIGTERM/SIGKILL may leave stale progress; require complete.json
AND successful Slurm accounting for acceptance. No interrupted attempt is
resumed or overwritten. Fresh attempts retain any failed history.

audit.py reopens raw receipts, checks schedule/coverage, byte accounting,
partitions,cache,CPU mapping,timing,throughput arithmetic,source/binary identity,
and current input metadata/layouts. It writes audit.json and trials.csv. Pilot
runtime estimate is54 times the slower whole-case wall time, with1.5x margin
plus300s for a suggested sweep walltime. The coordinator must check this against
the remaining12-node-hour campaign envelope before submitting the sweep.
The final sweep report must show medians,ranges,and block-paired ratios;
finalist selection and second-placement confirmation remain later work.

Local tests exercise real buffered reads and mincore on small,fsynced,unequal
fixtures,partial EOF requests,empty partitions,short reads,EINTR,EOF/errors,
setup failure,cache failure,CPU rejection,checksum coverage and12 corrupted
receipt variants. Compilation uses GCC14.3.0 with warnings treated as errors.
These local tests do not establish native compute-node throughput or cache
behavior; pilot acceptance remains mandatory.

Review-driven controls: before every case, the production binary's --evict-all
mode evicts and scans all30 campaign files twice, with50ms between zero scans.
The timed reader independently repeats the five-file cold gate after workers
are ready. Pilot probes use RANDOM advice on separate coordinator descriptors
(workers retain default readahead), require stable positive residency, then two
zero scans before release. These are sampled residency checks, not guarantees
against activity by unrelated external clients. Per-trial MemFree,MemAvailable,
Cached,Buffers,Dirty,Writeback,Slab and llite.max_cached_mb availability/raw output
are saved before/after. The audit requires all30 zero-residency identities and
memory observations; other kernel/Lustre/backend caches remain uncontrolled.

bundle.py creates a fresh snapshot from an exact reviewed Git commit, compares
working sources to git show, and requires the built binary/source/compiler
hashes in the reviewed build-pins.json. Runtime and audit require the complete
file key set and reject the test-binary hash. The coordinator will save sacct
accounting after each job; audit.py requires all job/step exits0:0, correct
account/QoS,node identity and matching status/complete records before pass.

Sweep submission must explicitly pass --time using the pilot audit's suggested
walltime, subject to remaining12-node-hour campaign budget. The two pilot points
do not exhaust all18 cases; the estimate is provisional and the1.5x margin
may not cover the slowest case. Preserve incomplete work if time expires.
CPU map now includes NUMA node;5/20-worker prefixes also change NUMA/CCD locality.
Compare stripe counts within a fixed worker count; do not attribute all scaling
across thread counts solely to parallelism. Optional early abort after a timed
worker fails is deferred; the trial fails, and the900s timeout bounds remaining
reads. lctl before/after snapshots are saved, including failures, with remote
bytes still unverified.
