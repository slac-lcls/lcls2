## Motivation

SMD callbacks were recreated at each input chunk. Local `enumerate(run.steps())`
and `enumerate(step.events())` counters therefore restarted at chunk boundaries,
causing step filters to miss data and per-step event windows to repeat. Mikhail's
Jungfrau stage-1 command for `mfx100848724`, run 49, step 2 selected no events on
current master with the effective MFX default `PS_SMD_N_EVENTS=1000`.

Keep one callback iterator per run on each executing EventBuilder rank (or the
serial reader), preserving counters across chunks while resetting event indices
at physical step boundaries. Preserve transition order, destination routing and
cleanup, and add serial/MPI regressions to psana CI. No greenlet dependency.

Long stretches without a callback yield can still delay transition delivery and
accumulate pending transitions. That separate limitation remains tracked by #284;
this PR does not close it. Counters are local to each EventBuilder rank.

## How to test the behavior?

In a locally built psana environment:

```bash
export PS_PARALLEL=mpi
python -m pytest psana/psana/tests/test_callback_chunk_state.py \
  psana/psana/tests/test_callback_iterator.py \
  psana/psana/tests/test_callback_stream.py
python -m pytest psana/psana/tests/
python -m pytest psana/psana/tests/byhand_*
```

Fresh build against master `9ab5c193b`: **142 passed, 5 skipped, 2 deselected** in
the core suite; **4 passed** in the longer MPI group. All 59 callback cases passed.

Protected real-data comparison of Mikhail's exact command, 50 frames across all 32 panels:

| Source | PS_SMD_N_EVENTS | Result |
| --- | --- | --- |
| Current master | Unset (effective 1000) | 0 events; existing empty-step `ievt` logging exception |
| Current master | 10000 | Correct50 events from step2 |
| This fix | Unset (effective 1000) | Same correct50 events; both gate arrays exactly equal to the 10000 control |

Data inputs were checked against a direct-file timestamp oracle. Outputs were
isolated, detector-name lookup readonly, and calibration database write helpers
blocked. No database writes were attempted.

Earlier implementation reviews and experiment records are preserved on the
[audit branch](https://github.com/slac-lcls/lcls2/tree/tests/smd-callback-chunk-state/notes/smd-callback-chunk-state).
[Current-master validation and independent review](https://github.com/slac-lcls/lcls2/tree/tests/smd-callback-chunk-state/notes/smd-callback-chunk-state/pr-preparation)
passed on `7ddd3f4fd`; the reviewer reported no blocking findings.

## Checklist

- [x] Updated `CHANGELOG.md` under Unreleased/Fixed (PR link added after creation).
- [x] Described the problem and solution.
- [ ] All GitHub Actions checks pass (pending PR creation).
- [x] Issue linkage checked: no existing issue tracks this specific counter fix;
  #284 is a deferred follow-up and must remain open, so no `Fix #284` keyword.
