# Disable phase-2 failures: test tools and plan (issue #237)

At TMO and MFX, a Disable sometimes reaches only some readout groups. control then logs `disable phase2 failed` and lists most of the DAQ as not responding. This directory has the tools and the plan to reproduce it on a test stand, find the cause, and test a fix.

| File | What it does |
|---|---|
| `check_disable_groups.py` | For each `disable phase2 failed` in a launch, lists which readout groups' DRPs received the Disable |
| `disable_stress.py` | Cycles Enable/Disable many times through `DaqControl`, counts failures, and saves XPM state when one happens |

The same change also adds two things outside this directory:

- `control --disable_msg_mode {0x180,0x080,0x100,0x000}`: sets the XPM message mode bits for Disable. The default is `0x180`, which is the current behaviour.
- pyxpm: after control writes the per-group `MsgHeader` (ZMQ path), each group's header is read back. A mismatch is logged as `msgHdr readback mismatch`.

## What we know

| Fact | Source |
|---|---|
| TMO: **21 failures in 8815 Disables (0.24%)**, 09/17 – 10/27/2026 | all `drp-srcf-mon001:control.log*` under `/sdf/home/t/tmoopr/daq/logs/2026` |
| In 18 of 19 TMO failures with per-group data, group 0 got the Disable and group 2 and/or 5 did not | DRP logs (`check_disable_groups.py`) |
| The XPM generated the Disable for only some groups | XTC `env & 0xffff` on the Disable record: run 105 `09040005` (groups 0, 2; group 5 missing), run 126 `09220021` (groups 0, 5; group 2 missing) |
| MFX: same per-group loss in 10 of 12 failures | `/sdf/home/m/mfxopr/daq/logs/2026/10` |
| TMO and MFX control send XPM writes over ZMQ (`-X drp-srcf-mon001`) | control `CMDLINE` |
| pyxpm writes the headers one group at a time (0, 2, 5), then `GroupMsgInsert` | `pyxpm/pvctrls.py`, `zrcv()` |
| Disable is the only transition sent with `0x180` (`WAIT_FULL`) | `control/control.py`, `condition_disable()` |
| control couples message deadtime across the partition's groups (`L0Groups = groups`) | `control.py` `DaqXPM.allocate()`; `XpmAppMaster.vhd` lines 387 – 389 |
| Per-group DeadTime/DeadFrac stop counting while L0 is disabled, and L0 is disabled just before the Disable is inserted | `XpmL0Select.vhd` lines 205 – 216 |

The last two rows explain why Grafana can show no deadtime in group 5:
- the per-group counter is stopped when it matters;
- deadtime in *any* group holds every group's message.

Per-link `DeadFLnk` keeps counting with L0 disabled, so it is the metric to watch.

## Two explanations

| | Cause | Does Disable `0x000` fix it? |
|---|---|---|
| **A** | The `0x180` "wait while full" handling loses or holds the Disable while there is deadtime | Yes |
| **B** | A group's `MsgHeader` write is lost or late, so it still holds the previous header | No |

**How the pyxpm read-back tells them apart:**
- The group that misses still reads back `0x189` (Disable, `0x180`) → the header is fine, so **A**.
- It reads back `0x08A` (SlowUpdate) or `0x008` (Enable) → **B**.

An explanation consistent with B, not yet tested:
1. Group 5 keeps the last SlowUpdate header (`0x08A`, "drop if full").
2. `GroupMsgInsert` fires during a brief deadtime in a coupled group.
3. Groups 0 and 2 wait and get the Disable.
4. Group 5's stale SlowUpdate is dropped, so it gets nothing.

## Test setup

This follows Matt's three steps on #237.

Use the FEE test stand (`cnf/fee_teststand.cnf`, `DAQ:NEH`, XPM 10). It reports the same XPM firmware version as TMO and MFX (`FwVersion 0x30f0000`).

1. **Allocate many readout groups.**
   - Enable the commented-out fakecam DRPs (`timing_3`, `timing_4`) so there are at least 3 DRPs, each in its own group. Mirror TMO: groups 0, 2 and 5.
   - Set `PART:<g>:L0Delay` to 99, 75 and 10.
2. **Configure triggers to generate deadtime,** in one group (start with group 2), in two ways:
   - **Programmable:** `PART:<g>:InhEnable 1`, with `InhInterval` and `InhLimit` set for 10 – 50% deadtime.
   - **Back-pressure:** fakecam trigger rate above what the DRP can take, so its link reports full.
3. **Cycle through many transitions** with `disable_stress.py`.

Run control as production does: **`-S 1`** (SlowUpdates on; `fee_teststand.cnf` has `-S 0`), and `-X <pyxpm host>` (ZMQ path). Run pyxpm from this branch so the read-back is active.

```bash
# DAQ allocated and connected with the groups and deadtime above
python -m psdaq.debugtools.disable_phase2.disable_stress \
    -C <collect host> -p <platform> -B DAQ:NEH -x 10 --groups 0x25 \
    --cycles 2000 --events 100 --out T1.jsonl

# after a run, per-group delivery for each failure
python -m psdaq.debugtools.disable_phase2.check_disable_groups \
    --logdir <daq log dir> <launch prefix>

# header read-back results
grep 'msgHdr readback' <pyxpm log>
```

## Test matrix

| ID | Groups | Deadtime | `-S` | XPM path | `--disable_msg_mode` | Purpose | If A | If B |
|---|---|---|---|---|---|---|---|---|
| T0 | 1 | none | 1 | ZMQ | `0x180` | Harness check | 0 failures | 0 failures |
| **T1** | 3 | programmable, group 2 | 1 | ZMQ | `0x180` | **Reproduce** | failures | failures |
| T1b | 3 | back-pressure, group 2 | 1 | ZMQ | `0x180` | Reproduce with real "full" | failures | failures |
| **T2** | 3 | as T1 | 1 | ZMQ | **`0x000`** | **Test the fix** | **0 failures** | failures |
| T3 | 3 | none | 1 | ZMQ | `0x180` | Is deadtime needed? | 0 failures | failures |
| T4 | 3 | as T1 | **0** | ZMQ | `0x180` | Is the stale header a SlowUpdate? | failures | group gets a duplicate **Enable** instead of nothing |
| T5 | 3 | as T1 | 1 | **PVA** (no `-X`) | `0x180` | Does the write path matter? | same as T1 | rate changes |
| T6 | 3 | as T1 | 1 | ZMQ | `0x180` | Equal `L0Delay` in all groups (optional) | same as T1 | – |
| T7 | 3 | as T1 | 1 | ZMQ, no `-c` | `0x180` | Like MFX (no common group) | same as T1 | same as T1 |

T6 checks an observation that may be coincidence. At both hutches, the group that misses has a shorter `L0Delay` than group 0:
- TMO: groups 5 (10) and 2 (75) miss; group 0 (99) never does.
- MFX: group 2 (70) misses; groups 0 and 3 (75) never do.

## How many cycles

- At the production rate (0.24%), 2000 Disables gives about 5 failures, even if the deadtime stress has no effect.
- Measure the T1 failure rate, `p1`.
- To call the T2 fix good: **0 failures in at least `3 / p1` cycles, and at least 1250.** 1250 is the 95% confidence figure at the production rate: 0.9976¹²⁵⁰ ≈ 0.05.
- For T3 – T7, run at least as many cycles as T1 needed to see 3 failures.

## Checks that `0x000` is safe (T2)

`0x000` sends the Disable even while a group is in deadtime. Check:

1. every DRP logs `PGPReader saw Disable`, `TebCtrb sent`, `TebRcvr saw` and `MebCtrb sent`, and answers phase 2;
2. no `Pulse ID did not advance`, damage, or `out-of-bounds` errors in TEB/MEB logs;
3. the next step starts and records `--events` events;
4. with recording on, exactly one Disable per step in every stream, with `env & 0xffff` equal to the group mask.

Two more checks:

- **cpo's concern** (the comment above the Disable insert in `control.py`): after a T1 failure, restart and check the new session's first transitions in XTC for a stray Disable.
- **Retries:** after a T1 failure and its retry, check that the duplicate Disable in the groups that got the first one is handled. Run 126 at TMO already contains such duplicates.

## Open question for the firmware

The per-group message FIFO fill count (`msgRdCount`, `XpmAppMaster.vhd` line 293) only goes to a debug ILA. If it could be read as a register, it would show directly, after a failure, whether a group's Disable is still queued.
