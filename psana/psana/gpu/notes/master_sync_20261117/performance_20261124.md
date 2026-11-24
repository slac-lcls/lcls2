# PR331 Perlmutter performance scope and results

Recorded November 24, 2026 from the completed Perlmutter campaign. This report
adds documentation and analysis of existing measurements; runtime code is unchanged.
The tested candidate remains `5bd5c7ba300cf0a0894b5b904d3975d5ba0b3cad`.

## Agreed performance scope

For the master-to-GPU-branch merge decision, the performed performance work is a
Perlmutter-only, instrumented raw-input A/B regression screen: CPU events with
1/2 BDs, CPU steps with 1 BD, and GPU events with 1/2 BDs sharing one A100.
This follows the user's narrowed A/B campaign scope and CPU steps/1-BD constraint.
Separate A/A was omitted under that revised scope; it is not a passed noise gate.

Calibrated processing, user GPU tasks/kernels, output/D2H costs, larger BD counts,
multi-GPU and multi-node scaling are not measured by these five cases. Treat
those as explicitly deferred follow-up performance work, not results supplied
by this screen. Prior pixel-exact calibration and transport checks are correctness
evidence, not calibrated or scaling performance measurements. The two-BD result
is a per-case baseline/candidate comparison, not a scaling-efficiency study.

Mona accepted this narrower scope on November 24: use the five-case raw A/B
screen for the merge decision and track calibrated/task/broader scaling work
separately under [#330](https://github.com/slac-lcls/lcls2/issues/330). This is an
explicit scope decision, not a pass against the original broader first-sync gates.
The two inconclusive results remain inconclusive; the decision does not turn them
into evidence of equivalence.

## Measured comparison

- Baseline: `546764e0338225d026faa29627297a4a390d3396`.
- Candidate runtime: `5bd5c7ba300cf0a0894b5b904d3975d5ba0b3cad`.
- Template anchor: `9f809fb08468c232e208dcfeb4d8a1a745f5fc96` on
  `results/psana2-gpu-scaling`; actual staged controllers have separate manifests.
- Dataset: original Jungfrau `mfx100848724`, run 51, private first-10,000-event
  prefix, streams 3/5/7/8/9; 335,571,760,000 selected datagram bytes per sample.
  This is neither Sim10 nor the later `prjdatpsana` run-1 correctness fixture.
- Each case: same-allocation baseline/candidate correctness, then six balanced
  paired A/B measurements; 10,000 measured events and a separate 200-event warmup
  per sample. Cases used separate exclusive Interactive allocations, `lcls_g`.
- Batch 20, one GPU stream per BD, 8 GiB/BD, bulk target 1 MiB;
  KvikIO compatibility mode, GDS off. GPU work prepares raw input and observes
  callbacks; calibration, user computation and output are excluded.
- Loop effect: `(candidate / baseline - 1) * 100`, summarized as the mean of
  six paired changes and a 95% paired t interval. Positive means slower.
  Equivalence requires the entire interval inside [-5%, +5%].

| Case | Job | Mean loop change | 95% interval | Result | Execution review |
| --- | --- | ---: | --- | --- | --- |
| gpu-raw-events-bd2 | 61881293 | -0.51% | [-3.43%, +2.40%] | equivalent | pass |
| gpu-raw-events-bd1 | 61885491 | -11.66% | [-30.04%, +6.72%] | inconclusive | pass |
| cpu-raw-events-bd1 | 61889954 | -0.37% | [-3.35%, +2.61%] | equivalent | pass |
| cpu-raw-events-bd2 | 61906060 | +1.31% | [-5.77%, +8.38%] | inconclusive | pass |
| cpu-raw-steps-bd1 | 61917936 | -0.17% | [-1.55%, +1.20%] | equivalent | pass |

Three cases meet loop-time equivalence; two are inconclusive. No case establishes
a slowdown greater than 5%, but the two inconclusive intervals also do not rule
one out. Do not claim universal equivalence, speedup, or merge readiness.
All five cases passed their recorded correctness and terminal integrity audits.
All five independent execution reviews passed for their scoped claims.
The [last two reviews](performance_20261124/execution-review-20261124/review.json)
closed with four nonblocking findings; see their
[dispositions](performance_20261124/execution-review-20261124/response.md). Secondary setup/first-batch/process outcomes remain in
[results.json](performance_20261124/results.json); loop equivalence does not imply end-to-end equivalence.

## Limits and retained exceptions

No separate A/A or established observer-overhead bound; no overhead subtraction.
Shared-filesystem contention is uncontrolled. Cache and NIC evidence are retained,
but node-wide traffic is not storage-only bandwidth or physical-media proof.
Native-adapter/component qualification and unavailable physical-memory peaks,
kernel counts and scan/visit metrics are not silently upgraded to passes.
The original candidate CPU steps/1BD crash (61664705.35) remains **observed once,
not reproduced, root cause unresolved**. All six diagnostic cells (61679475),
the 10k-event steps retries (61710680), and the later steps/1BD A/B campaign
(61917936) completed successfully. The retries did not exactly repeat the original
preceding workload sequence, and no confirmed fix or causal attribution is claimed.
Interrupted job61897961 and all failed/cancelled attempts remain separate; only
complete audited cohorts contribute, without partial-cohort pooling.

Correctness exceptions are tracked separately: inherited CPU failures in
[issue #351](https://github.com/slac-lcls/lcls2/issues/351); the deferred callback
MPI case has related multi-run fix [PR #326](https://github.com/slac-lcls/lcls2/pull/326),
without a confirmed exact-test pass on the combined tree. #299 coverage is
supported by carried-forward recovery tests and the later reviewed uncapped
multi-step/EndRun job61937600; that correctness job is not a timing sample.

## Correctness coverage carried into the disposition

The SDF reconciliation is 2,140 passed, five skipped, two failed/hung and one
explicitly deferred, across attempts. GPU-directory coverage is 1,960 passes,
including 561 GPU-marked cases, with no GPU failures/skips. The two inherited CPU
slow-test failures are tracked in [#351](https://github.com/slac-lcls/lcls2/issues/351).
The five skips remain missing coverage: calibration pickle, roentdek library,
timetool calibration constants, and two explicitly skipped callback/timestamp
filter cases. No skipped or failed test is relabeled as passing.

The deferred `test_fresh_callback_per_run[mpi]` uses `run=[14,14]` and an SMD
callback. [PR326](https://github.com/slac-lcls/lcls2/pull/326) targets stale MPI
messages across runs, tested with callback-free `run=[14,15]` and real CPU/GPU
multi-run data. It is a related fix, not proof that this exact callback test
passes on the combined tree. PR326 was open and unmerged when checked.

For #299, the existing unit recovery test exercises the production retry method
with injected pressure; device tests retain/release GPU owners and verify resumed
event timestamps. The relevant test files match the pinned candidate. Those
passed suite results are carried forward. Perlmutter job61937600 adds five passing
uncapped `prjdatpsana` run-1 cases: CPU steps/1BD, GPU serial events/steps, and GPU
MPI events/steps with two BDs sharing one GPU. All delivered the exact 60 events,
with step populations 19/20/21 and natural EndRun. Claude's supplied-evidence review
passed with six nonblocking findings; its [review and dispositions](performance_20261124/pr299-correctness/review-response.md)
are retained. Configure metadata establishes 32 panels, not callback shape.
Memory-pressure recovery was not forced during this real-data job. The #299 disposition credits
the combined existing recovery and new EndRun/steps evidence for this merge; it does not establish that untested combined pressure scenario.

## Reproducible compact record

- [Results and secondary timings](performance_20261124/results.json): the original
  consolidated result snapshot with execution-review statuses updated.
  [The prior snapshot](performance_20261124/results-before-final-review.json) is preserved;
  numerical results and timing classifications are unchanged.
- [Sample timings](performance_20261124/samples.csv): 70 measured rows (ten
  correctness and sixty A/B), including runtime, pair/order and original result
  hashes. The separate 200-event warmups are not timing rows in this CSV.
- [Paired loop changes](performance_20261124/paired-loop-changes.csv): 30 pairs.
- [Provenance](performance_20261124/provenance.json): per-case frozen profile,
  controller, artifact and runtime hashes; settings, placement, example launch
  argv/environment and requests. Paths describe the original installs, not a
  portable installation or an instruction to rerun an archived launcher.
- Per-job subdirectories retain terminal accounting, audit, source-file hash
  index and execution reviews. The two final reviews share the record linked above.
- [Manifest](performance_20261124/manifest.json) binds the compact files.

Reproduce all loop and secondary means/intervals with standard Python:

```bash
python -B psana/psana/gpu/notes/master_sync_20261117/performance_20261124/verify.py
```

The calculation preserves the original controller's rounded Student-t critical
value **2.571** (six pairs, five degrees of freedom), rather than silently changing
historical intervals. Export verified the source audits against the previously
recorded hashes, sample receipts against each job's frozen index, and referenced
cache/correctness/resource/runtime evidence hashes. It recalculated all twenty
case/metric comparisons from the original sample times; no workload was rerun.
This numerical verification is distinct from the independent execution reviews,
which are now complete for all five cases.

## Retention and follow-up

The report and compact numerical evidence are Git records. Complete per-rank
logs, frozen launchers, datasets and native installs remain in site storage:
`/pscratch/sd/m/monarin/agent-work/lcls2/gpu-master-sync-20261119-attempt02/performance-continuation-01`.
The full evidence has not yet been copied to an agreed persistent archive.
Before scratch expiry, retain the scripts/configuration, per-rank observations,
logs, scheduler accounting, reviews and all failed attempts in persistent project
storage; verify checksums/readback and record the destination, owner and date.
Hashes alone cannot recreate expired evidence. Cross-site recoverability remains
unestablished. Large data, installs and CUDA caches are excluded from this compact
record.

Follow-up under #330: calibrated Jungfrau, representative user-task/D2H/output
workloads, and matched broader scaling profiles. Each needs its own correctness,
noise/observer qualification and statistical criteria. This report does not claim
full PR acceptance or authorize those new campaigns. The two inconclusive cases
and all declared measurement limits remain visible.


## Final execution-review notes

Claude passed jobs61906060 and61917936 with no blocking findings. This review
confirms that the inconclusive CPU2BD result is correctly classified; it does
not make that result equivalent. The review is limited to supplied evidence.

The compact provenance projection reports `before_after_equal=false`: comparison
of all98 measured-rank records found only13 newly loaded modules per record, with
no removed modules or changes to common module entries. Other provenance fields
were unchanged and manifest hashes were verified at both observations. This is
module-set growth, not evidence of installation drift. The comparison receipt is
retained with the review dispositions.

Scheduler MaxRSS and per-rank ru_maxrss disagree; neither is promoted into a
host-memory acceptance result here. Those measurements need reconciliation before
any subsequent memory claim. The reconciled61906060 audit receipt is bound to its
job/auditor by the adjacent readiness receipt and log; original receipts are intact.
