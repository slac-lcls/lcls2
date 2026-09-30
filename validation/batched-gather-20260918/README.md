# Warm B+ versus batched canonical gathering

O means B+ at b0c9c3c02; G adds batched canonical gathering. Before refreshing
this worktree's installation, its validated B+ install was copied (not hard
linked) to `install_baseline`. The five relevant production modules in that
copy were verified byte-for-byte against the commit. G uses `install_psana`.
Generated installation copies, logs, datasets and traces stay local/ignored.

The harness is adapted from the committed one-allocation A/B/B+ comparison.
Both variants use 10,000 mfx101210926/run-387 events, Jungfrau 32 segments,
batch 20, depth one, 8 GiB budget, one BD/GPU, KvikIO compatibility ON (not GDS),
eight threads and 1 MiB tasks. Staging/cache preparation stays outside timers;
each sample requires >=99% residency before and after. No user D2H is requested.

Run `sbatch validation/batched-gather-20260918/run.sbatch` after building G and
verifying the frozen O prefix. Four clean repetitions alternate OG/GO/OG/GO;
three repetitions also have separate CPU/NVTX samples. Four CPU pixel-reference
preflights check both variants with and without timing hooks before timing.
Separate Nsight captures follow. Uniform CPU affinity, GPU identity/runtime,
installed source hashes, dataset timestamps/bytes and memory samples are logged.
No hardware speed threshold is asserted. `audit.py JOB_DIRECTORY` verifies
sample order, correctness preflights, cache guards, placement and provenance.

`correctness.sbatch` runs focused CPU/GPU tests. `lifetime-pixels.sbatch` adds
actual delayed-consumer retirement and six slow real-data acceptance cases.
Main-suite and longer MPI test logs are `core.log` and `byhand.log`.

Initial job 38559942 was interrupted during its final B+ control by
cudaErrorContained; 13 samples completed and no traces ran. The completed retry
is the primary comparison; nodes and unmatched pairs are not mixed.
Retry job 38561649 excludes sdfampere024 and uses four clean repetitions, one
host-timed pair, and two traces. See STATUS.md and the final performance report.

Completed retry: job 38561649, sdfampere004, COMPLETED / 0:0, 48:04 allocation.
All 12 samples and four preflights passed audit; all cache checks are 100%.
Clean medians: O 37.274593 s, G 28.578059 s (23.3% lower elapsed time).
Nsight verifies 320,000 → 500 gather kernels and 341,500 → 22,000 total kernels.
Both captures have collection-completeness warnings retained in the report.
Run `audit.py` and `summarize.py` against `job-38561649-warm-ab` to reproduce.
The full report is `psana/psana/gpu/docs/performance/batched_canonical_gather_sdf.md`.
