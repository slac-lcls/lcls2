# GPU performance history

[Performance overview](README.md) · [Hardware inventory](hardware.md)

Entries identify measured revisions and workloads. A merge date alone does not
make an old measurement a benchmark of the new branch head.

| Date | Major change or study | Measured scope and evidence |
| --- | --- | --- |
| October 27–28, 2026 | [PR187](https://github.com/slac-lcls/lcls2/pull/187): discover GPU peers and share calibration constants; [PR208](https://github.com/slac-lcls/lcls2/pull/208): report owned/imported memory | [Ada A/B/C and Perlmutter sharing off/on](pr187_pr208.md). Measured A is [c157b490c](https://github.com/slac-lcls/lcls2/commit/c157b490ccb42940632ac83fce2a33e98335e9ce), the first parent of the PR187 merge. Local B/C trees exactly match published merges [c1ab81fa2](https://github.com/slac-lcls/lcls2/commit/c1ab81fa2aa3ec50f3952d9f41a51cd2ed94f7c4) / [8cf05e86a](https://github.com/slac-lcls/lcls2/commit/8cf05e86a7511f84a24f82b9c9720bc2308c471d). Raw stability, startup cost and remaining-loop uncertainty are recorded separately. Both PRs merged October 27. |
| October 26–27, 2026 | Hardware qualification, no runtime change | [Ampere inventory and actual campaign hardware](hardware.md); [Ampere 013 qualification](../../notes/ampere013_qualification_20261026.md) and [Ada 008 raw-read qualification](../../notes/ada008_raw_read_20261027.md). Hardware classes constrain cold-rate comparisons. |
| October 16–17, 2026 | September versus October stack; pre/post review stack [PR136](https://github.com/slac-lcls/lcls2/pull/136)–[PR151](https://github.com/slac-lcls/lcls2/pull/151) | [Historical revision comparisons](read_staging.md). The October 14 review-stack final merge was `410ebba1c`; the pre/post compared GPU source trees were identical. Do not attribute cross-node cold differences to the stack. |
| September 28–29, 2026 | Batched external science kernels | [Calibration plus radial integration](user_kernels.md), runtime/algorithms `6ba5fa586`; matched scheduling comparison and scaling. This workload differs from calibration-only PR187/208 validation. |
| September 28, 2026 | Full Jungfrau and partial Jungfrau+feespec read/staging baseline | [Raw staging/scaling](read_staging.md), runtime `10df4c6e3`; host-cache cold/warm, bulk off/on and GPU/BD topology matrices. |

Add new studies above prior entries and link their compact evidence. Retain
historical tables, configurations and uncertainties; update the overview to
point readers to the latest relevant measurement.
