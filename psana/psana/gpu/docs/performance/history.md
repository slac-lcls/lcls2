# GPU performance history

[Performance overview](README.md) · [Hardware inventory](hardware.md)

Entries identify measured revisions and workloads. A merge date alone does not
make an old measurement a benchmark of the new branch head.

| Date | Major change or study | Measured scope and evidence |
| --- | --- | --- |
| November 23, 2026 | Perlmutter NIC and Lustre configuration inspection | [Hardware context](hardware.md#perlmutter-network-and-lustre-capacity--november-23-2026): four active 200 Gb/s NICs on nid200276; saved Lustre configuration selects cxi0. Runtime LNet access denied; no new throughput measurement or independent review. |
| November 16, 2026 | Sim10 S/E/EC/ECC completion | [Job61409171](https://github.com/slac-lcls/lcls2/blob/results/psana2-gpu-scaling/psana/psana/gpu/docs/performance/campaigns/perlmutter/sim10-xpp-cube/2026-11-16-se-ec-ecc-100k/README.md): 24 primary 100k cases plus11 controls/checks; four workers/BD, count2. Four-BD S 5.181 GiB/s, ECC 4.780 GiB/s; independent results review passed, basic cube content checks only. |
| November 16, 2026 | Perlmutter Sim10 psana GPU scaling | [100k-event screen](https://github.com/slac-lcls/lcls2/blob/results/psana2-gpu-scaling/psana/psana/gpu/docs/performance/campaigns/perlmutter/sim10-staging/2026-11-16-gpu-scaling/README.md): 5.075 / 7.558 / 9.341 / 9.842 GiB/s for 1/2/3/4 GPUs, stripe count 1 and four BDs/GPU. CPU allocation and workers increase with GPUs; local-cache gates and whole-NIC counters recorded. Separate independent results review is not recorded. |
| November 16, 2026 | Perlmutter Sim10 psana stripe screen | [One GPU, four BDs, 100k events](https://github.com/slac-lcls/lcls2/blob/results/psana2-gpu-scaling/psana/psana/gpu/docs/performance/campaigns/perlmutter/sim10-staging/2026-11-16-stripe-screen/README.md): count 1 led the tested screen at 5.443 GiB/s. Local-cache checks passed; NIC bytes were not collected; independent results review remains pending. |
| November 15, 2026 | [PR299](https://github.com/slac-lcls/lcls2/pull/299): Run51 performance follow-up | [Perlmutter paired baseline/head screen](../../notes/pr299_perlmutter_20261115.md). Measured head `acfd029a8d27898b5f165e02facd3c3bbc9c4ad2`; 48 timed samples, uncontrolled cache, small effects inconclusive. C1 unfixed at that head; Run77 deferred. |
| November 14, 2026 | SDF Sim10 staging parameter synthesis | [Seven allocations, 48 configurations and heatmaps](https://github.com/slac-lcls/lcls2/blob/results/psana2-gpu-scaling/psana/psana/gpu/docs/performance/campaigns/sdf/sim10-staging/2026-11-14-parameter-screen/README.md). The storage/GPU cohort fixed four BDs total: little observed gain from spreading them across more GPUs. Normalized tables are preserved; some original inputs were not transferred. Last three cohort reviews and independent synthesis review remain pending in the source record. |
| November 11, 2026 | Perlmutter C++ reader stripe screen | [54 full-file read cases](https://github.com/slac-lcls/lcls2/blob/results/psana2-gpu-scaling/psana/psana/gpu/docs/performance/campaigns/perlmutter/cpp-read/2026-11-11-stripe-screen/README.md): count 2 reached 12.92 GiB/s and count 1 reached 11.42 GiB/s at 40 threads. Five files span five OSTs at count 1; these are reader rates, not a single-OST ceiling. Independent results review passed; backend cache was uncontrolled. |
| October 27–28, 2026 | [PR187](https://github.com/slac-lcls/lcls2/pull/187): discover GPU peers and share calibration constants; [PR208](https://github.com/slac-lcls/lcls2/pull/208): report owned/imported memory | [Ada A/B/C and Perlmutter sharing off/on](pr187_pr208.md). Measured A is [c157b490c](https://github.com/slac-lcls/lcls2/commit/c157b490ccb42940632ac83fce2a33e98335e9ce), the first parent of the PR187 merge. Local B/C trees exactly match published merges [c1ab81fa2](https://github.com/slac-lcls/lcls2/commit/c1ab81fa2aa3ec50f3952d9f41a51cd2ed94f7c4) / [8cf05e86a](https://github.com/slac-lcls/lcls2/commit/8cf05e86a7511f84a24f82b9c9720bc2308c471d). Raw stability, startup cost and remaining-loop uncertainty are recorded separately. Both PRs merged October 27. |
| October 26–27, 2026 | Hardware qualification, no runtime change | [Ampere inventory and actual campaign hardware](hardware.md); [Ampere 013 qualification](../../notes/ampere013_qualification_20261026.md) and [Ada 008 raw-read qualification](../../notes/ada008_raw_read_20261027.md). Hardware classes constrain cold-rate comparisons. |
| October 16–17, 2026 | September versus October stack; pre/post review stack [PR136](https://github.com/slac-lcls/lcls2/pull/136)–[PR151](https://github.com/slac-lcls/lcls2/pull/151) | [Historical revision comparisons](read_staging.md). The October 14 review-stack final merge was `410ebba1c`; the pre/post compared GPU source trees were identical. Do not attribute cross-node cold differences to the stack. |
| September 28–29, 2026 | Batched external science kernels | [Calibration plus radial integration](user_kernels.md), runtime/algorithms `6ba5fa586`; matched scheduling comparison and scaling. This workload differs from calibration-only PR187/208 validation. |
| September 28, 2026 | Full Jungfrau and partial Jungfrau+feespec read/staging baseline | [Raw staging/scaling](read_staging.md), runtime `10df4c6e3`; host-cache cold/warm, bulk off/on and GPU/BD topology matrices. |

Add new studies above prior entries and link their compact evidence. Retain
historical tables, configurations and uncertainties; update the overview to
point readers to the latest relevant measurement.

The November dates identify campaign records/synthesis, not new measurements of
the current branch head. The Sim10/XPP-cube preparation remains a historical pending snapshot; the
separate November16 completion record above contains accepted application results.

Execution completion, archive verification and independent results review are
separate states. See the [evidence assessment](https://github.com/slac-lcls/lcls2/blob/results/psana2-gpu-scaling/psana/psana/gpu/notes/campaign_evidence_assessment_20261116/README.md)
for retained limitations and follow-up priorities.
