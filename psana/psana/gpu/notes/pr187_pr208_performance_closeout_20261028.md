# PR187/208 performance investigation closeout — October 28, 2026

The maintained [performance report](../docs/performance/pr187_pr208.md) records
completed measurements. [Hardware](../docs/performance/hardware.md) is the
canonical inventory; retain the qualification notes as dated observations.
Both PRs merged October 27. The investigation was closed October 28.

| Job | Outcome | Contribution |
| --- | --- | --- |
| Ada 42114353 | Completed October 27, 09:34–13:50 PDT | 12 preflights and 96 clean A/B/C samples |
| Ada startup follow-up 42152466 | Canceled pending when moved to Ampere | No timing results |
| Ampere startup follow-up 42153755 | Failed October 27, 17:33 PDT after about six minutes | First A preflight hit harness `Event.query()` incompatibility with CuPy 14; no completed timing sample |
| Corrected Ampere follow-up 42241613 | Canceled pending October 28, 07:51 PDT | Redundant query removed after synchronization; no timing results |
| Perlmutter 60415698 | Started October 28, 00:38:30 per scheduler record; 36 clean startup samples completed; three separate profile attempts failed | First-batch/remainder evidence, no successful profile trace |

Do not present the failed harness attempt as a runtime regression or the
canceled follow-up as an independent confirmation. The docs retain the observed
startup cost and unresolved remaining-loop equivalence. No further jobs were
requested at closeout.
