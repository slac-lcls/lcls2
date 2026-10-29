# PR #187 integration: four BDs on one GPU

Status: Slurm job **41310953** completed successfully on **sdfampere030**,
2026-10-19, 08:09:38–09:30:26 Pacific (**1h20m48s**, exit **0:0**).
All five staged input hashes, four pixel correctness preflights, 32 timed
samples, and 1,861 frozen artifact hashes passed. Private staging was removed.

Compare the issue #159 integration base with the preceding `psana2-gpu`
revision. Neither variant contains the alignment fix.

| Variant | Revision | Meaning |
| --- | --- | --- |
| A | `c157b490ccb42940632ac83fce2a33e98335e9ce` | Before PR #187 |
| B | `90c05965cf34869388cc526a50e7e822b50a6eba` | Integration of pinned PR #187 head `4b068a867d9164354ac7b7613d89812f06dc5ee6` |

The workload follows the [staging comparison contract](../docs/performance/read_staging.md):
Jungfrau run 387, full 32-panel data, five streams, dense raw gathering, no
calibration or user kernel. This measures staging overhead associated with
the integration; it does not measure IPC calibration-constant sharing benefits.

## Controls and acceptance

- One exclusive A100 node selected by Slurm, physical GPU 0, four BDs, one EB
  and one SMD0. The node is reserved for isolation; only one GPU processes data.
- Independently built source/build/install trees; clean runtime environment
  for each process, with per-rank Python and native import provenance checks.
- Official KvikIO 26.10, CuPy 14.0.1, Python 3.11, CPU fallback; batch 20,
  execution depth 1, eight read workers, 1 MiB task/group targets.
- **6 GiB explicit budget per BD in both variants.** PR #187 changes automatic
  sizing, so an explicit budget holds effective capacity constant. Actual
  usage must stay within the cap; four peers and device identity are checked.
- Four 200-event correctness preflights, including CPU-reference gathered
  pixels, then 32 fresh-process timed samples of 10,000 events each.
- Cold/warm cache and bulk off/on: four matched pairs per condition, AB then
  BA on alternating repetitions. Stage private node-local prefixes once;
  verify hashes and per-file cache residency before and after timing.
- Time the event loop, including work lazily performed there. Report setup
  separately. Staging/cache preparation and diagnostic preflights are excluded.
- Preserve timestamp identity, exact payload/request counts, peer/device/CPU
  placement, budget high water, read wait, cleanup and artifact hash checks.

Positive paired B/A loop-time change means the integration is slower. Report
individual pairs and their spread, not just aggregate rates. Do not infer
alignment-fix performance from this experiment.

## Reproducibility and results

Both independent builds passed. A → B → A activation verified isolated Python
and native library origins. The 19 existing acceptance-harness tests passed;
1,861 frozen file hashes and the balanced 32-sample schedule were verified.
An obsolete physical-device override in the historical frozen validator was
replaced by the maintained validator in both copies: equivalent for this GPU-0
campaign, and compatible with the existing multi-GPU CPU test fixtures. The
initial failed test log is retained; no runtime code was changed.

| Cache | Bulk | A events/s | B events/s | Median paired time change | Pair range |
| --- | --- | ---: | ---: | ---: | ---: |
| Cold | Off | 260.95 | 258.31 | +1.35% | +0.94% to +3.64% |
| Cold | On | 261.16 | 264.39 | -0.99% | -3.02% to +3.54% |
| Warm | Off | 474.94 | 478.56 | -1.56% | -2.43% to -0.46% |
| Warm | On | 503.56 | 487.42 | +5.00% | -8.93% to +13.81% |

There are four pairs per condition, 320,000 timed events and 800 preflight
events. Rates use each variant's median loop duration. The median paired
percentage is calculated separately and need not equal the ratio of those rates.
Peak tracked allocation was **1.2518 GiB per BD** in both variants. Median
explicit setup was **2.513 s for A** and **2.710 s for B**, excluded from loop time.

No uniform staging slowdown was observed. Cold/bulk-off was slower in B in all
four pairs, a small consistent signal that should remain visible in the record.
Warm/bulk-off was faster in B in all four pairs. Warm/bulk-on had a +5.00%
median with a wide, mixed-sign range; four pairs cannot distinguish a stable
regression from variability or identify its cause. Do not treat that range as
an acceptable regression allowance for #159, or conclude performance equivalence.

The [saved evidence](evidence/pr187_staging_comparison_20261019.json) includes
all pairs, compact per-sample measurements and cache checks, resource counters,
provenance and log hashes. Full logs remain on scratch. The measured integrated
revision `90c05965c` becomes the unchanged reference for alignment comparisons;
future candidates must be measured beside it again on the same node/allocation,
rather than compared directly with these historical rates.

The [implementation plan](issue_159_implementation_plan.md) now places focused
performance checks after each relevant stage. Keep dependency effects separate
from alignment changes, and retain both incremental and cumulative comparisons.

The physical device is NVIDIA A100-SXM4-40GB, UUID
`GPU-59fbc7e4-8129-4a75-df6e-671b17c6251e`, PCI `00000000:01:00.0`, driver
575.57.08. Initial staging was limited by local disk write/flush throughput;
that work is excluded from timed loops.

Scratch campaign:
`/sdf/scratch/users/m/monarin/gpu-validation/pr187-ab-1gpu4bd-20261019-r1/`.

The campaign retains build commands and logs, exact source trees, independent
installs, identical measurement scripts, `plan.json`, `hashes.json`, frozen
reference inputs, runtime verification, and Slurm stdout. Each job writes
`job-<id>/provenance.json`, individual logs, `results.json`, `pairs.json`, and
`summary.json`. A complete summary requires all acceptance checks to pass.

Source input:
`/sdf/data/lcls/drpsrcf/ffb/users/monarin/jf-feespec-bulk-38995226/xtc`.

No production code is changed for this experiment. PR #187 review fixes remain
with Seema. Any inherited failure encountered here must be recorded before
deciding whether a valid comparison can proceed.
