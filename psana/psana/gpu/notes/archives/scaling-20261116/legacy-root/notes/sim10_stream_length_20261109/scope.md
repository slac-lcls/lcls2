# Perlmutter controlled stream and duration comparison

Continue the transferred task on branch work/perlmutter-sim10-stream-length.
Base d7e573f7b. Do not touch other checkouts or their builds.
One BD/one physical GPU, SMD0+EB helpers masked before MPI/psana/CuPy imports.
Cray MPICH via srun, constant physical CPU affinity and NUMA interleave.
Same transferred five JF files for 10k prefix and 100k full duration; streams 1/2,
two repeats in reverse order after two 1000-event pilots. Fixed batch100,
SMD1000, bulk524288B, budget6GiB, compatON, 8 workers, 1MiB tasks,16MiB bounce,
automatic direct I/O OFF. Descriptor staging only, no calibration or expansion.

Use original generation.json plus explicit transfer relocation manifest; retain
source receipt and independently SHA256 all files on compute. Validate Names,
panel order, SMD timestamps/offsets and complete DataSource subset traversal.
Reuse frozen Perlmutter install-B, verify eight Python identities plus native
imports, Cray MPI and GPU operation. No framework or installed-file edits.

Cold scope is local page cache on Lustre only. Owner/path/size-guarded
fsync/FADV_DONTNEED plus mincore after setup must show zero pages. No global
cache drop, no staging copy or deletion of transferred data. Raw llite/CXI
counter discovery is diagnostic; remote traffic remains explicitly unverified.
Timing includes startup/fill/drain; backend caches remain uncontrolled and later
10k prefixes follow full-file reads. No direct Weka reproduction claim.

Require exact ordered identities, bytes/read counts/histograms/callbacks,
requested streams exercised, GPU masks, one job/node/GPU, frozen runtime,
no pressure drains or CPU payload reads, complete retirement and cleanup.
Independent audit and negative mutations require native receipts; absent native
evidence is not acceptance. Report individual repeats, medians and paired ratios.

Use implement-review-loop, three new rounds for this Perlmutter implementation,
separate detached read-only Claude review. Review native results separately.
All logs, caches and reviewer worktrees under unique scratch task attempts.
