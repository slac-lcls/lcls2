# PR187 Perlmutter sample mapping — verified October 30, 2026

The [performance report](../docs/performance/pr187_pr208.md) now records a
verified mapping from each metric-array entry to its original sample tag and
repetition. This resolves the earlier uncertainty about array order.

## Committed mapping

In the [compact evidence](../docs/performance/evidence/pr187_pr208_202610.json),
each `perlmutter.arms[]` entry has a `sample_metadata` list. Its element `i`
identifies `metrics[*].samples[i]` and `timers[*].samples[i]`. It records:

- Zero-based array index and explicit repetition ID.
- Original tag, for example `B-on-n10000-r1-clean`.
- One-based arm position within the repetition and zero-based position in the
  complete clean execution sequence, including both event counts.
- Tagged BD timing rows, allowing first-batch, remaining and loop maxima to be
  checked directly against the sample arrays.
- SHA-256 hashes of the original sample log and launch-command file.

The evidence retains the complete schedule and the comparison `pairs` records,
which name both tags and their repetition. `sample_mapping_audit` records audit
checks, the original-runtime replay, and all 16 harness-file paths and hashes.
The source list includes results, summary, provenance, frozen hashes and the
campaign execution log. The [harness archive](../scripts/pr187_pr208/README.md)
now preserves the inspected code and its required helpers, with original paths
and frozen-hash matches in its manifest. Full logs and installed runtimes
remain external artifacts.

## Trace through the original code

All paths below are relative to
`/pscratch/sd/m/monarin/psana2-gpu/pr187-ipc-startup-perlmutter-20261027/attempt01/`.
The inspected harness files match the run's `sweep-60415698/frozen-hashes.json`.

| Stage | Original code | Ordering or measurement rule |
| --- | --- | --- |
| Schedule | `scripts/schedule.py:11–31` | Assign repetitions 1–6, balance arm permutations, alternate event-count order and validate coverage. |
| Launch and save | `scripts/campaign.py:161–208`, `223–225` | Execute the schedule; construct `{case}-n{events}-r{repetition}-clean`; save the command; parse one `PERF_RESULT`; attach tag/repetition and append to results in execution order. |
| First-batch boundary | `scripts/startup_timing.py:66–79` | Record a CUDA event after the first callback; synchronize once after its final public event is delivered; measure elapsed time from loop entry. |
| Per-rank timing | `scripts/calibration.py:172–198`, `227–232` | Measure setup separately; each BD's remaining time is its own loop minus its own first-batch time. |
| Rank reduction | `scripts/calibration.py:266–279` | First batch and remaining are BD maxima; setup and whole loop are all-rank maxima. Whole-loop maxima also came from BDs in all 36 clean samples. |
| Array construction | `scripts/analysis.py:23–46` | Exclude diagnostics/traces, assert results follow the schedule, sort each arm by repetition, and build metric arrays and tagged rank records from that same group. `stats()` preserves input order. |
| Pairing | `scripts/analysis.py:47–56` | Zip the two repetition-sorted arms; record repetition and both tags; calculate absolute and percentage differences. |
| Documentation compaction | Original local `/tmp/build_performance_docs.py:35–48` | Copy metric arrays unchanged, but originally omit tagged rank records. This update restores explicit metadata without altering statistics. |

For example, `B-on-n10000-r1-clean` has BD remaining times
58.37602471397258, 58.18044461403042, 58.396057737991214 and
58.32421492901631 seconds. The rank-4 maximum is the first B-on remaining-array
entry. Its paired B-off tag is `B-off-n10000-r1-clean`, with
51.1109240220394 seconds. Their difference is 7.285133715951815 seconds,
displayed as +7.29 seconds in the report.

## Verification

- All 16 retrieved harness files matched the run-frozen hashes.
- All 36 clean `BEGIN`/`PASS` log entries matched the schedule and result order.
- All 36 launch commands matched their arm, event count and sharing switch.
- All 36 raw `PERF_RESULT` objects matched their saved result fields.
- All 144 per-BD remaining-time identities and 288 aggregate timing values
  recomputed from rank rows.
- All 24 tagged pairs and 576 paired values matched; all 72 compact metric
  arrays retained their original values and ordering.
- Replaying the original aggregation with the campaign's Python 3.11.14 in a
  temporary directory reproduced `summary.json` byte for byte, SHA-256
  `b954804bfd7da107c65b127c877dd044abaeb8897a25a1aa84f066363c87cfe1`.

An SDF Python 3.6 replay differed only in 62 standard-deviation/interval rounding
fields below 1e-12; underlying sample arrays and pairings were exact. The
original-runtime replay establishes the byte-identical result. Original
campaign artifacts were not modified and no GPU benchmark was rerun.
