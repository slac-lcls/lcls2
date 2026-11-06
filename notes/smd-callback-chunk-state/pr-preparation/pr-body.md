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
The 20 MPI cases also passed with an explicit two-slot limit using the explicit
Open MPI `mpirun --oversubscribe` launcher option.

Protected real-data comparison of Mikhail's exact command, 50 frames across all 32 panels:

| Source | PS_SMD_N_EVENTS | Result |
| --- | --- | --- |
| Current master | Unset (effective 1000) | 0 events; existing empty-step `ievt` logging exception |
| Current master | 10000 | Correct 50 events from step 2 |
| This fix | Unset (effective 1000) | Same correct 50 events; both gate arrays exactly equal to the 10000 control |

Data inputs were checked against a direct-file timestamp oracle. Outputs were
isolated, detector-name lookup readonly, and calibration database write helpers
blocked. No database writes were attempted.

Earlier implementation reviews and experiment records are preserved on the
[audit branch](https://github.com/slac-lcls/lcls2/tree/tests/smd-callback-chunk-state/notes/smd-callback-chunk-state).
[Current-master validation and independent review](https://github.com/slac-lcls/lcls2/tree/tests/smd-callback-chunk-state/notes/smd-callback-chunk-state/pr-preparation)
passed on final head `38010bfb1`; the second review reported no findings.
Production code is unchanged from the full-suite and real-data validation at
`7ddd3f4fd`; the follow-up adjusts only the MPI test launcher and changelog link.

## Checklist

- [x] Updated `CHANGELOG.md` under Unreleased/Fixed with the PR link.
- [x] Described the problem and solution.
- [x] All GitHub Actions checks pass on `38010bfb1`: [psana](https://github.com/slac-lcls/lcls2/actions/runs/42297812200) (100 passed, 3 skipped), [DAQ](https://github.com/slac-lcls/lcls2/actions/runs/42297812297), and [changelog](https://github.com/slac-lcls/lcls2/actions/runs/42297812205).
- [x] Issue linkage checked: #284 is a separate deferred follow-up and remains
  open; this PR intentionally has no issue-closing reference.
