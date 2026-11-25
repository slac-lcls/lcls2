# PR regression allocation and placement preparation

Implemented `prepare_site.py` and `site_probe.py`. Claude
[passed the final review](review-03.json) at
`22f0199d3355b180c6c364cc272b1dd4e7b897d8`, with no blocking findings. Use the [agent instructions](../../scripts/performance/regression/SITE.md)
in each new single-node Perlmutter campaign allocation. The driver produces an
accepted site manifest only after four tiny MPI placement probes pass.

Job **61499579** on **nid001613** completed **0:0 in 39 seconds** using production
commit `a589507b07676276c7740476424bd7a438be8e89`. All four steps completed 0:0;
no active steps remained. Coordinator readback verified all 64 capture-indexed
files and the frozen source/runtime associations.

| Probe | MPI ranks | Bound cores/rank | Reserved physical cores | GPU | Step elapsed |
| --- | ---: | ---: | ---: | --- | ---: |
| CPU, 1 BD | 3 | 4 | 48 | none | 6 s |
| CPU, 2 BDs | 4 | 4 | 64 | none | 3 s |
| GPU, 1 BD | 3 | 4 | 48 | one A100 | 12 s |
| GPU, 2 BDs | 4 | 4 | 64 | same shared A100 | 4 s |

These are launch-validation durations, not psana workload throughput or regression
measurements. The scheduler records 96/128 logical CPUs for the 48/64 physical-core
step reservations; each rank's actual affinity remains exactly four physical cores.
Rank masks are CPUs 0–3, 4–7, 8–11 and (for 2 BDs) 12–15. CPU/service ranks passed
the no-CUDA import/mapping guard. GPU ranks observed
`GPU-da185f47-3cc8-142f-a6e9-0b8e993193db`; identity/PCI/NUMA and runtime-file
provenance are in the indexed evidence. This is a reproducible placement policy,
not an assertion of optimal NUMA locality.

The job requested 64 CPUs/1 GPU/16 GiB and received the full 128-logical-CPU,
64-physical-core/4-A100 node with 16 GiB reserved. Step MaxRSS was `0` (CPU steps) or `512K` (GPU steps), implausibly low for
these processes, so step memory usage is **unmeasured**. Batch MaxRSS was `152328K`. The driver identified four distinct PCI-backed
`hsn0..hsn3` interfaces using `cxi_ss1`, all up. Their interface counters increased
by 6,056,264 RX bytes and 7,215,008 TX bytes across preparation. This only validates
counter availability/monotonicity; it neither attributes traffic to storage nor
establishes total RDMA/fabric throughput or a bandwidth ceiling.

## Hardware findings retained

| Job | Result | Finding and correction |
| --- | --- | --- |
| 61498633 | rejected before rank launch | Default Slurm distribution allocated CPUs outside masks. Added block distribution and one thread per core. |
| 61499087 | CPU probe passed; capture rejected | Five-second accounting poll missed final completion. Added bounded 30-second confirmation and retained each response. |
| 61499310 | both CPU probes passed; GPU launch rejected | Slurm selected GPU-local CPUs outside the common masks. |
| 61499458 | same GPU launch rejection | Disabling GRES binding alone did not change that preference. |
| 61499579 | all four probes accepted | Separate 16-core/task reservation from four-core/task actual affinity. |

All failed attempts remain failures; none published an accepted manifest. Final
accounting and failure receipts are retained. No run51 files were read or evicted.
The successful manifest belongs to the now-ended allocation and cannot be reused
for a campaign. Generate a fresh one in the campaign's own allocation.

## Validation and review

- Round 1 requested diagnostic evidence and safeguards; round 2 passed the fixes at
  `b5ddc20f71de447bb8019131d68159e7bf04a6d3` for bounded hardware validation.
- Hardware exposed the launcher changes above; round 3 passed the final code
  and evidence together. The closure commit only records review, corrects RSS
  wording (SITE-008) and polishes report spacing; production code is unchanged.
- The 85-test full offline suite passed with the final reservation implementation.
  An additional reservation-contract test then passed in the 30-test adapter suite.
  Earlier, an unchanged Lustre mincore test intermittently saw 0 instead of 57
  resident bytes; its failure and successful targeted/full reruns are retained.
  This does not qualify cache behavior or justify ignoring a native cache failure.
- Optional SITE-006 walltime margin was addressed in later 25-minute validation
  scripts. Optional SITE-007 GPU-error-path unit coverage remains deferred.

See [hardware summary](hardware-05.json), [final accounting](accounting-final-05.txt),
[capture validation](capture-validation-05.json), [artifact hashes](artifact-index.json),
and [executed source hashes](frozen-05.json). Large receipts remain under
`/pscratch/sd/m/monarin/psana2-gpu/pr-regression-site-20261117`.

Next: disposable timeout/signal cleanup qualification, then the six short native
run51 workload smoke cases, byte/cache accounting and instrumentation/component
qualification. Timed A/A and PR A/B follow only after those checks and review.
The shipped profile remains unchanged and blocked.
