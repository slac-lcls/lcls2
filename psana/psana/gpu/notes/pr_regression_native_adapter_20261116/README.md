# Native PR regression adapter implementation and review

The [Perlmutter adapter](../../scripts/performance/regression/NATIVE.md) is now
implemented for the six run51 raw cases. It remains **hardware-unqualified**.

Reviewed code: `f5f45d34adfd8ce58ca34145764e990b6ef88511`.
Independent Claude review: **PASS**, round 3, no unresolved blockers.
See [final review](review-03.json) and [response](response-03.md).

Validation: 25 offline adapter tests, 20 controller tests; catalog validation
passed for 10 campaigns, 2,082 hashes and 275 arithmetic checks. Profile and
frozen PR187/208, PR299 and archived inputs remain unchanged. No native MPI/GPU
run, build, dataset copy, job submission or performance measurement was made.

Review fixes cover GPU delivery/drain accounting, CPU L1 versus transition
classification, actual SMD selection, bounded step cleanup, scheduler-independent
process timing, runtime/GPU assignment controls and evidence-index integrity.

The non-blocking service11 item (NAT-009) was subsequently resolved by the
[reviewed classifier update](../pr_regression_service11_20261116/README.md).
The original round records below preserve the state at that review closure. The native instructions also
require input/reference and runtime manifests, exact Slurm launch/cleanup tests,
observation-overhead measurements, and qualified component/resource bounds.
No GPU/pinned peak or internal visit counter is invented when unavailable.

The three contexts, reviews, responses and validation records in this directory
preserve the implementation loop. review-artifacts.json hashes the compact
records. Raw provider output remains in task scratch and is not published.
