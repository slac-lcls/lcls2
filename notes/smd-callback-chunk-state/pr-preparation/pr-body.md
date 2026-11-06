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

Minimal reviewer check: activate a build of this PR on a host with access to
`mfx100848724` run 49, use a fresh working directory, and run Mikhail's command
with the default SMD chunk size:

```bash
unset PS_SMD_N_EVENTS
export PS_PARALLEL=none
jungfrau_dark_proc -k exp=mfx100848724,run=49 -d jungfrau -o ./work1 \
  --nrecs 50 --nrecs1 50 --stepnum 2
```

**Expected:** the command exits successfully, processes 50 events from step 2,
and writes the `gate_lo` and `gate_hi` arrays under `./work1`.
Please confirm that this command works in your build of the PR.

Measured before/after with `PS_SMD_N_EVENTS` unset (effective MFX default: 1000):

| Build | Result |
| --- | --- |
| Before: master `9ab5c193b` | 0 events; exits 1 with `UnboundLocalError: local variable 'ievt' referenced before assignment`; no gate files |
| After: this PR | 50 events from step 2; exits 0; writes both gate arrays for all 32 panels |

As a cross-check, the unfixed build succeeds with `PS_SMD_N_EVENTS=10000`.
Its selected timestamps and both gate arrays exactly match the fixed build at
the default chunk size.

These measurements used the command's entry point and identical arguments through
an instrumented harness with isolated outputs, readonly detector-name lookup,
and calibration database writes blocked. No database writes were attempted.
[Validation details and independent reviews](https://github.com/slac-lcls/lcls2/tree/tests/smd-callback-chunk-state/notes/smd-callback-chunk-state/pr-preparation)
include the harness and results. Automated validation also passed: 142 core tests
and 4 longer MPI tests locally; GitHub checks are linked below.

## Checklist

- [x] Updated `CHANGELOG.md` under Unreleased/Fixed with the PR link.
- [x] Described the problem and solution.
- [x] All GitHub Actions checks pass on `38010bfb1`: [psana](https://github.com/slac-lcls/lcls2/actions/runs/42297812200) (100 passed, 3 skipped), [DAQ](https://github.com/slac-lcls/lcls2/actions/runs/42297812297), and [changelog](https://github.com/slac-lcls/lcls2/actions/runs/42297812205).
- [x] Issue linkage checked: #284 is a separate deferred follow-up and remains
  open; this PR intentionally has no issue-closing reference.
