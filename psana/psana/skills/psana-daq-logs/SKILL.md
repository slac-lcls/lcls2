---
name: psana-daq-logs
description: Read and search raw LCLS-II DAQ log files on disk for a session in a given time window — live (now) or a specific past date/range. Use for "read DAQ log files", "find error messages in DAQ logs", "why did a DAQ component crash", "current/live run log inspection", "tail DAQ logs", "DRP/TEB/MEB log errors".
---

# Skill: psana-daq-logs — LCLS-II DAQ Log Inspection

Reads raw DAQ log files directly off the filesystem for a session in a
given time window. Self-contained — no log aggregation database or search
index behind this, every command is a direct filesystem operation.

**Related skills:** if arrived here without first checking metrics, load
`psana-daq-monitor` to locate a time window, or `psana-daq` if the report
is still vague. Load `psana-configdb` afterward if a log finding looks
configuration-related.

## Path convention

    /sdf/home/<x>/<hutch>opr/daq/logs/<YYYY>/<MM>/

Verify with `test -d` before assuming a given month exists — don't guess
an alternate path if it's absent. This is the path all four regularly-used
hutches (xpp/tmo/rix/mfx) actually write to today; the launcher's own code
default (`SbatchManager.get_default_output_root()` in `psdaq/psdaq/slurm/
utils.py`) only names this path for xpp specifically, so a hutch using
something else is launching with an explicit `--output` override — if the
convention path is empty, check that session's `control.log` `CMDLINE:`
for the real `--output` value rather than assuming the hutch has no logs.

## Filename grammar

    <DD>_<HH:MM:SS>_<host>:<component>.log[.zst]

Example real filenames:

    15_15:31:17_xpp-daq:ami-client.log
    15_15:31:17_drp-srcf-mon008:control.log

The `<DD>_<HH:MM:SS>` prefix is shared across every process in one DAQ
launch — but it's a **grouping hint, not a unique session ID**: it can
collide across different months/years (an 11-char prefix carries no
month/year information), and a restart can share a prefix with only some
components relaunched. Always pair it with its parent `YYYY/MM` directory.

**Do not just pick the newest prefix** — list sessions and ask, unless the
window was already handed off. The newest session is often a short test
with 0 errors; an earlier one may be the one actually worth investigating.

## If a window was already handed off

If `psana-daq` already established `[start, end]` before invoking this
skill, use it — don't re-ask or default to "today":

- **Window ends at "now"** → the current/newest session is in scope; no
  list needed unless the sweep's own "unexpectedly many sessions" check
  applies.
- **An exact `DD_HH:MM:SS` prefix is already known** → skip selection,
  scope straight to it.
- **A vague window** (resolve first — see "Resolving a vague date" below)
  → run session selection anchored on the *resolved* window, not today.

If no window was handed off (standalone invocation), ask, then resolve the
same way.

**Targeted request vs. sweep context:** "list and ask" below is for a
*targeted* request. As the Logs leg of `psana-daq`'s sweep, don't ask —
every session intersecting the resolved window is investigated without a
per-session prompt (the sweep's own narrow "too many sessions" exception
aside). Both files state this so neither reads as an absolute rule alone.

## Resolving a vague date

Convert phrasing into a concrete `YYYY-MM-DD` date or `[start, end)`
window before selecting sessions:

- **"yesterday" / "last night"** → today minus one calendar day, via real
  date arithmetic (`date -d yesterday +%Y-%m-%d`), not string manipulation
  on the day field — "the 1st minus one day" must land in the previous
  month (and, on Jan 1, the previous year).
- **A bare day-of-month** with no month stated → the current month, unless
  that date is in the future, in which case the previous month.
- **"the last Nh/Nd"** → `[now - N, now)` — may itself span a month/day
  boundary; don't collapse it to a single date if it does.
- **An explicit date/range** → use as given, normalized to `YYYY-MM-DD`.
- **A local time inside a DST fold or gap** → ambiguous or
  self-contradictory; don't silently normalize it. Keep it unresolved
  until an offset or other evidence (e.g. a timestamped log line)
  disambiguates it. Convert to UTC before correlating against logs,
  metrics, or control transitions from a different source.

State the resolved date/window back before proceeding — a wrong resolution
should be visible immediately.

## Session selection

A launch is not time-boxed to the day/month it started in — its output
path is fixed once at launch and keeps writing there past midnight/month/
year rollover (verified: a June-launched xpp session wrote files with a
July mtime). There is no "self-contained month directory" assumption to
rely on.

Use `scripts/daq-logs.py list <log-dir> --start <iso> --end <iso>` — it
implements the interval-overlap test directly: a session is in scope iff
`session_start <= requested_end AND session_last_mtime >= requested_start`,
not "session's mtime falls on the requested day." This is why a request
for "yesterday" must still surface a session that started yesterday and is
still writing today. If the requested window is near a month boundary, run
`list` again against the adjacent `<YYYY>/<MM>` directory too, and merge by
sorting on `(year, month, prefix)` — the script sorts within one directory
only.

```
$ scripts/daq-logs.py list /sdf/home/x/xppopr/daq/logs/2026/09 \
    --start 2026-09-17T00:00:00 --end 2026-09-18T00:00:00
Sessions overlapping 2026-09-17T00:00:00 .. 2026-09-18T00:00:00 (latest first):
  1. 18_10:33:36   2.0m   0 errors
  2. 17_18:40:41   1.2m   15 errors   xpp-monReqServer[...]: <E> Configure ... is missing src 1 (bld_0) ...
  ...
```

Present this list (or a `--json`-produced version of it) and ask which
session, unless invoked as a sweep leg (see above). Narrow further with
`--component GLOB` when the user names a component or host.

## Canonical log read

A session directory mixes `.log` and `.log.zst`, and `.log.zst` is often
the **majority** of files (verified: 3620 vs 2767 in one real xpp month) —
a plain `grep *.log` silently skips over half the evidence, and grepping
`<prefix>_*` without a suffix filter feeds compressed bytes straight to
`grep`, which won't match either. Use `scripts/daq-logs.py grep` for every
session read — it handles both formats transparently via `zstd -dc` (never
`zstdcat`, which passes non-zstd input through unchanged at exit 0 instead
of failing loudly on a corrupt archive):

```
scripts/daq-logs.py grep <log-dir> <session-prefix> --count
scripts/daq-logs.py grep <log-dir> <session-prefix> --first
scripts/daq-logs.py grep <log-dir> <session-prefix> --full --component teb
```

Default filters: `--pattern '<[EC]>'` (Critical/Error), `--exclude
'Inadequate RTPRIO'` (startup noise, see below) — filter-then-exclude,
never reversed (counting first and filtering second is a no-op: `grep -c`
emits one number, which a second grep then filters against nothing).
`--json` for programmatic use.

If working directly in a shell instead, the equivalent is:

```bash
for f in "$dir"/${prefix}_*; do
  if [[ "$f" == *.log.zst ]]; then zstd -dc -- "$f"; else cat -- "$f"; fi
done | grep '<[EC]>' | grep -v 'Inadequate RTPRIO'
```

## Header block

Every log file begins with `#`-prefixed metadata lines — not just
preamble, itself useful diagnostic data. Some files begin with 1-4 lines
of `git describe` stderr before the first `#` line; scan for the first
`#`-prefixed line rather than assuming line 1.

```
# SLURM_JOB_ID:69442
# ID:      ami-meb0
# PLATFORM:0
# HOST:    drp-srcf-mon008
# CMDLINE: monReqServer -P xpp -C drp-srcf-mon008 -M /sdf/group/lcls/ds/daq/prom/xpp -d -n 60 -q 31 -p '0' -u ami-meb0
# TESTRELDIR:/sdf/group/lcls/ds/ana/sw/conda2/rel/xpp/lcls2_090826/install
```

`CMDLINE` reveals the actual runtime flags for that process — cross-check
what `psana-daq-monitor` shows in Grafana (current metric values) against
what was actually configured (the flags here), not the same thing.

## Log line grammar

    <hutch>-<process>[<pid>]: <L> <message>

`<L>`: `<C>` Critical, `<E>` Error, `<W>` Warning, `<I>` Info.

    xpp-teb[1788610]: <C> Inadequate RTPRIO limit: got 0, require 99

**Lead with a Critical/Error grep, excluding the RTPRIO startup line** —
`Collection.cc::checkResourceLimits` logs it critical but marks it
**nonfatal** (the process continues); filtering it is for signal-to-noise
only, not because scheduling is proven healthy. Revisit if scheduling is
ever actually implicated in a finding.

## Interpreting `<C>`/`<E>` messages

Don't rely on a pre-built catalog — DRP/TEB source has ~450 `logging::
critical`/`logging::error` call sites, most emitting dynamic content a
static catalog can't capture, and message text drifts between releases
(see `psana-daq/reference/release-source.md`). Once you have a line:

1. Grep the **static portion** against the running release's source to
   find the call site and its surrounding context.
2. **Some messages carry their own remedy** — read the full message before
   concluding you need to interpret it further (e.g. "Try XPM TxLink
   reset.").
3. **Misconfiguration vs. hardware fault isn't always obvious from text
   alone** — check whether the failing check validates a user-supplied
   config value (a config bug) vs. a hardware/firmware condition.
4. **Watch for a repeating dump/loop before treating hits as discrete
   events** — a runaway-logging pattern (e.g. a hex-dump loop) can produce
   an outlier line count for one file/process; check for repetition before
   investigating each line as a separate failure.

## Crash signatures

See `reference/crash-signatures.md` for the full corpus derivation. The
definition itself: Python exception (`Traceback (most recent call last)`,
a terminating `<Type>Error:` line) or segfault (`Segmentation fault`,
`(core dumped)`). Bare `task N: Aborted` with neither signature in the
same file means the proximate cause is elsewhere — check sibling logs,
don't gate crash detection on bare `Aborted` alone.

## Component / host catalogs

- **Event builder**: `tebN`. **AMI pipeline**: `ami-mebN`, `ami-global`,
  `ami-client`. **DRP sources**: `drp_bld`, `drp_pva`, `bld_N`, `epics_N`,
  `timing_N`. **Control**: `control`, `control_gui`, `daqstat`, `xpmpva`.
  **Detectors**: `hsd_N`, `epix100_N`, `jungfrau_N`, `wav8_*`, `alvium_*`.
- **DRP nodes**: `drp-srcf-cmp0NN` (recording) / `drp-srcf-mon0NN`
  (monitoring/MEB). **Control host**: `<hutch>-daq`. **IOC hosts**:
  `daq-<hutch>-<detector>-NN`.

## Practical guidance

- Always go through `scripts/daq-logs.py` (or the shell equivalent above)
  so `.log.zst` rotations are included — a plain `grep -l '<C>\|<E>'
  .../<prefix>_*` feeds compressed bytes to `grep` directly and silently
  drops most matches in a typical session.
- Narrow by component or host with `--component`.
- If a component's current log is empty or missing, check whether it only
  exists as a `.log.zst` from an earlier rotation in the same session.
