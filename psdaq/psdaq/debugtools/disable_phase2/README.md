# Disable delivery under inhibition (#237)

When Disable is inserted with XPM mode `0x180` (WAIT_FULL), a readout group's
message waits while message inhibition is asserted. On TMO, a controlled
inhibition test twice reproduced the production signature: group 0 received
Disable, groups 2 and 5 did not, and phase 2 timed out after 40 seconds.

This change defaults Disable to `0x000`: insert the transition without waiting
for message inhibition. Control still stops L0 triggers first, completes phase 1,
and requires phase-2 replies from all active DRPs and MEBs. The transition ID,
group mask, and the modes of other transitions are unchanged.
`--disable_msg_mode 0x180` retains the baseline behavior for comparison.

The mode-only candidate passed 20 and then 300 cycles between/after the two
baseline failures. This verifies delivery under programmable inhibition; it
does not yet establish safety with truly full receiver buffers or recorded-data
integrity. Keep those checks open before operational adoption.

## Hardware experiment: TMO, 2026-10-28

All times below are America/Los_Angeles. The experiment used TMO XPM 2,
platform 0, ZMQ (`-X drp-srcf-mon001`), SlowUpdate at 1 Hz (`-S 1`), common
group (`-c`), and a 40000 ms phase-2 timeout. Active groups were 0, 2, 5
(mask `0x25`). Actual L0Delay values were 75, 75, 10.

- Group 2 used event code 10 (120 Hz), group 5 code 13 (10 Hz).
  Group 0 was the common group.
- In group 2 only, inhibit row 0 was enabled with interval 4000, limit 1.
  Other inhibit rows remained disabled.
- Recording was OFF throughout. No retry or cleanup transition was sent after
  either failure; logs and state were preserved before restarting.
- Baseline and candidate were separate launches of the same private installation,
  with the same detector/config snapshots. The mode argument was the only
  baseline/candidate software configuration difference.
- The private installation copied production `lcls2_101226/install`; a comparison
  of 2621 files found only `psdaq/control/control.py` changed. That installed
  patch added the mode option with the old default; candidate launches explicitly
  selected `0x000`. This PR makes that tested selection the new default.
- The existing pyxpm and IOCs were not changed or restarted for the experiment.
  The earlier PR's pyxpm readback instrumentation was not used and has been
  removed from this PR. No MFX hardware test was performed.

| Order | Result file | Mode | Disable attempts | Completed cycles | Outcome |
|---|---|---|---:|---:|---|
| A | `baseline_20261028_122652.jsonl` | `0x180` | 1 | 0 | Groups 2 and 5 missed Disable |
| B | `candidate_20261028_123347.jsonl` | `0x000` | 20 | 20 | Passed |
| A | `baseline_20261028_124401.jsonl` | `0x180` | 1 | 0 | Groups 2 and 5 missed Disable again |
| B | `candidate_20261028_132228.jsonl` | `0x000` | 300 | 300 | Passed in 1573.5 seconds |

A preceding 300-cycle baseline run without programmed inhibition passed, so
the controlled inhibition was necessary to reproduce quickly in this session.
The historical production estimate (21/8815 Disables) did not determine the
stressed test's sample size. The experiment stopped after the repeated
baseline/candidate comparison and 300 additional candidate cycles; it does
not claim to satisfy the earlier 1250/2000-cycle production-rate criterion.

### What was verified

Both baseline failures had the same partial-group signature in DRP logs:
`timing_0` (group 0) logged `PGPReader saw Disable` and sent its contribution
to the TEB; the checked DRPs in groups 2 and 5 logged the control request but
no timing-system Disable. The TEB received only the group-0 contribution.
Control's list of missing acknowledgments includes downstream consequences,
so its first reported alias is not a diagnosis of that detector.

For both candidate runs, all ten DRPs with timing-reader logging had exactly
20 or 300 `PGPReader saw Disable` messages and the same number of
`TebRcvr saw Disable` results. Their ordered timestamp/pulse-ID sequences
matched within each DRP and across all ten DRPs, with no duplicates.
The ten were timing, laser_wav8_ip1, fzp_opal, atm_piranha_ip1, fzp_piranha,
itof_hsd, mbes_hsd_0, mbes_hsd_1, epics, and im5k4.
Control also required phase-2 replies from every active participant,
including those without that logging.

In the 300-cycle run:

- Every group had a positive accepted-trigger increase in every cycle.
- Groups 0 and 2 each accepted 90053 triggers (241–360 per cycle);
  group 5 accepted 7505 (20–30 per cycle).
- Group 2's median sampled DeadTime was 51.74%; 298/300 samples exceeded 40%.
  Samples ranged from 0.053% to 67.61%. These were sampled after Disable,
  not measured at the exact insertion instant.
- The JSONL contained 300 attempts, 300 trigger checks, 300 completed cycles,
  and a successful final EndRun; no error or warning records.
- Control was left Configured with recording OFF. A post-run read confirmed
  group 2 inhibit 0 remained enabled at interval 4000, limit 1.
- TEB/MEB logs had an existing startup `Inadequate RTPRIO limit` message;
  this report does not claim completely clean process logs.

[tmo-20261028.json](tmo-20261028.json) contains per-run outcomes, per-DRP
delivery counts, timestamp agreement, and SHA-256 hashes of the input JSONL
and DRP/control logs. Raw evidence remains on SLAC storage:

```text
/sdf/home/t/tmoopr/daq/disable-test-20261028/
  results/<result file above>
  logs/baseline/2026/10/28_10:52:38_<host>:<process>.log
  logs/candidate/2026/10/28_12:32:13_<host>:<process>.log
  logs/baseline/2026/10/28_12:42:18_<host>:<process>.log
  logs/candidate/2026/10/28_13:19:50_<host>:<process>.log

/sdf/home/m/monarin/lcls2/tmp/tmo-disable-20261028/evidence/
  baseline-inhibit-122658/
  candidate-inhibit-123347/
  baseline-inhibit-124401/
  candidate-inhibit-132228/
```

## Recorded follow-up: tmo101907427 run 275

Later on 2026-10-28, the same candidate launch recorded run 275 after the
operator restored group 2 to event code 136. Control logs show two pause/resume
operations followed by the final stop: three Enable/Disable intervals within
one BeginStep/EndStep, then EndRun (17:12:44–17:13:17 local time).

All 11 full XTC streams and their 11 SMD counterparts were read through their
complete datagram boundaries. Every file contains three Enables and three
Disables, with identical transition timestamps across streams. Every Disable
has zero damage and the full group mask `0x25`; there are no duplicate or
unpaired Disables. EndStep and EndRun are present in all files. The Disable
timestamps (DAQ epoch seconds.nanoseconds) are:

- `1162080775.157645594`
- `1162080785.579961059`
- `1162080795.611422279`

This checks recorded transition delivery and pause/resume. It is **not a full
recorded-event integrity pass**:

- The group 2 streams contain zero L1Accept events. The event-bearing streams
  carry mask `0x21` (groups 0 and 5), consistent with no group 2 events after
  restoring code 136. This run therefore does not test group 2's final events.
- `epics_0` (s007), `timing_0` (s008), and `im5k4_0` (s009) each record
  103, 63, and 59 events in the three enabled intervals (225 total).
- The full XTC `im5k4_0` stream marks all 225 events with damage `0x0040`
  (TimedOut), including events before the first pause. Its SMD top-level
  damage field is zero, so the full XTC check is essential. The other two
  event-bearing full XTC streams have zero top-level event damage. The camera
  timeout cause has not been established; it is not attributed to this fix.
- Event acquisition resumes after each pause, but independent accepted-event
  counters were not captured for this run, so complete final-event retention
  is not established. This was one step, not a next-step test.

The installed `xtcreader -s` independently confirmed 225 damaged events in
s009 and zero events in s000. The run interval's TEB/MEB and participating DRP
logs contained no `<E>`/`<C>`, pulse-ID regression, or out-of-bounds markers;
that does not negate the damage recorded in s009. Control completed all
transitions and returned to Unallocated. A later read confirmed group 2 code
136, inhibit 0 disabled, interval/limit 1/1; these post-run settings do not
measure the inhibit state at the earlier Disable insertion times.

The `recorded_run275` entry in [tmo-20261028.json](tmo-20261028.json) records
file hashes, stream identities, transition results, event counts, and damage.
Detailed local evidence is in the existing evidence root's `run275/` directory.

## Reproduction method

Use an agreed hardware test window and separate launch/build configuration.
A private installation does not isolate shared timing hardware. Set control
to `--disable_msg_mode 0x180` for baseline or `0x000` for candidate; verify
the actual command in its launch log. The stress script's `--label` only labels
the result: it does not select or verify the running control's mode.

1. Launch the selected mode, choose the participating detectors, and keep
   recording OFF. Configure in the GUI.
2. Apply the same trigger selections and programmable inhibition **after**
   Configure, which can overwrite timing-panel changes. At TMO the settings
   were those listed above; enable the inhibit only after setting interval/limit.
3. Run a short baseline attempt and stop on the first failure.
4. Preserve the launch logs and JSONL, inspect per-group delivery, then restart
   with candidate mode and reapply the same settings.
5. Run candidate for 20 cycles, repeat baseline, then run candidate for 300.

Example for the TMO candidate, with the intended test DAQ already Configured:

```bash
python -m psdaq.debugtools.disable_phase2.disable_stress \
    --host drp-srcf-mon001 --platform 0 --xpm-base DAQ:NEH:XPM:2 \
    --groups 0x25 --label candidate --cycles 300 --use-configured \
    --out candidate.jsonl
```

Omitting `--use-configured` requires Connected and performs Configure with
`readout_count=0`. With the option, Configure is skipped to preserve manual
settings. The harness checks that recording is OFF and that the active groups
exactly match `--groups`. It never changes trigger or inhibit PVs.

Each cycle requests Running (BeginRun on the first cycle, then BeginStep and
Enable), acquires for a random 2–3 seconds, requests Paused (**Disable only**),
waits 1.2 seconds for published statistics, verifies accepted-trigger increases
in every group, and requests Starting (EndStep). A completed run requests
Configured (EndRun). It does not use StepDone or event-count termination.

Any error, missing trigger increase, timeout, or Ctrl-C stops with a nonzero
exit status and no retry/cleanup transition. Failure records distinguish Enable,
Disable, trigger verification, and EndStep and count attempts separately from
completed cycles. Inspect the DAQ before recovery; logical state can remain
Running even after triggers have stopped. Success does not imply recording
integrity because recording is prohibited by this harness.

This script ports the private `reproduce.py` used on TMO. The port makes
host/platform/groups/XPM prefix explicit and defaults to 20 cycles.
The hardware results belong to the private script, whose hash is retained in
the evidence; the packaged version is covered by hardware-free tests.

To summarize a failed launch:

```bash
python -m psdaq.debugtools.disable_phase2.check_disable_groups \
    --logdir /sdf/home/t/tmoopr/daq/disable-test-20261028/logs/baseline/2026 \
    10/28_12:42:18
```

Restore the original timing settings when finished. For this TMO session:
disable group 2 inhibit 0 first, restore interval/limit to 1/1, and restore
group 2 event code **136**. Stop the private DAQ with its launcher, return to
the normal release/configuration, and perform the usual operational check.

## Interpretation and remaining checks

The A/B/A/B result strongly supports changing Disable's mode for this
reproducer. It does not establish why WAIT_FULL failed to complete: a held
or displaced message, internal timing, and other firmware mechanisms remain
to be distinguished. The loaded firmware's exact correspondence to the
inspected RTL has not been established.

`0x000` bypasses the message-inhibit check, which can include receiver
backpressure as well as programmable inhibition. Before operational adoption:

- Verify Disable delivery with genuine receiver backpressure and confirm
  transition-buffer capacity/overflow behavior with the firmware owner.
- Extend the recorded test with events in every group: verify final accepted
  events and the next step, and check exactly one Disable per enabled interval
  in each stream with the expected group mask. Resolve or account for the
  camera damage in run 275 and check event-builder errors.
- Inspect for stray late transitions after a failed baseline and restart.
  Duplicate/retry behavior was not tested because this harness never retries.
- Validate other hutch topologies, including MFX, separately.

No hardware message-header readback was taken in this experiment.
The cached PVA `MsgHeader` value is not evidence of the hardware register.
Even a direct readback of `0x189` would establish the observed register value,
not prove WAIT_FULL is the root cause or exclude a clock-domain/timing issue.
For reference, `0x080` is DROP_FULL; LOW_PRIO is `0x100`.

Hardware-free validation: all 11 harness tests and the applicable repository
pre-commit hooks passed. The actual control parser was also exercised with its
manager mocked: the default and explicit candidate resolve to 0x000, and the
baseline resolves to 0x180. Mocked calls to the actual Disable method verified
stop/phase-1/insert/phase-2 ordering, both headers, and propagation of a failed
phase-2 acknowledgment without marking the transition successful.

```bash
PYTHONPATH=psdaq python3 -m unittest discover \
    -s psdaq/psdaq/debugtools/disable_phase2 -p 'test_*.py' -v
```
