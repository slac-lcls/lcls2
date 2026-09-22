---
name: psana-daq-logs
description: Read and search raw LCLS-II DAQ log files on disk for the currently running or very recent DAQ session (not historical/archival analysis). Use for "read DAQ log files", "find error messages in DAQ logs", "why did a DAQ component crash", "current/live run log inspection", "tail DAQ logs", "DRP/TEB/MEB log errors".
---

# Skill: psana-daq-logs

# LCLS-II DAQ Live Log Inspection

You are reading raw DAQ log files directly off the filesystem to diagnose the
CURRENTLY RUNNING or very recently completed DAQ session. This skill is
self-contained — it reads only the plain-text/zstd log files at the paths
below. There is no log aggregation database or search index behind this;
every command here is a direct filesystem operation.

**Related skills:** if you arrived here without first checking metrics, load
`psana-daq-monitor` to locate a time window, or `psana-daq` if the user's
report is still vague. Load `psana-configdb` afterward if a log finding
looks configuration-related.

---

## Path convention

    /sdf/home/<first-letter-of-hutch-account>/<hutch>opr/daq/logs/<YYYY>/<MM>/

Example, verified for real:

    /sdf/home/x/xppopr/daq/logs/2026/09/

**This path does NOT exist for every hutch.** Verified by directly testing
every hutch account:

| Status | Hutches |
|---|---|
| Present with data (readable) | xpp, tmo, rix, mfx, ued |
| Directory exists but empty (no year subdirs) | txi, det |
| No directory | xcs, cxi, asc, tst |

Always check with `ls`/`test -d` before assuming the path exists for a given
hutch — do not guess an alternate path if it's absent. Tell the user plainly
if there is no log directory for the hutch they asked about.

---

## Filename grammar

    <DD>_<HH:MM:SS>_<host>:<component>.log[.zst]

Example real filenames from `xpp`'s September 2026 directory:

    15_15:31:17_xpp-daq:ami-client.log
    15_15:31:17_xpp-daq:daqstat.log
    15_15:31:17_drp-srcf-mon008:ami-meb0.log
    15_15:31:17_drp-srcf-mon008:control.log

The `<DD>_<HH:MM:SS>` prefix is **shared across every process/component
started in the same DAQ session** — it is effectively a session ID.

**Do not simply pick the newest prefix — present a session list and ask the user.**
The newest session is often a short test with 0 errors; the session before it may
be the one with 23 errors that the user actually wants to investigate.

### Session selection — present a list and ask

1. Get today's date: `date +%d` → e.g. `18`.
2. Find sessions whose **last-written file's mtime is today** (not by prefix day —
   sessions span midnight; a prefix starting `09_08:19:07` may still be writing on
   the 17th). For each candidate prefix, compute and display:
   - **prefix** (the session ID, format `DD_HH:MM:SS`)
   - **lifetime** = (mtime of newest file) − (timestamp parsed from prefix)
   - **non-RTPRIO error count** = `grep -c '<[EC]>' | grep -v 'Inadequate RTPRIO'`
   - **first error excerpt** (first non-RTPRIO `<C>` or `<E>` line, truncated ~60 chars)
   - **file count**
3. Sort **reverse-lexically** (latest first). This is safe without date parsing:
   the prefix format is fixed-width `DD_HH:MM:SS` (exactly 11 chars, zero-padded;
   all hutches conform). Reverse lexical sort is a stable latest-first ordering.
4. Mark sessions whose prefix day differs from today (spans midnight).
5. Skip filenames not matching `^[0-9]{2}_[0-9]{2}:[0-9]{2}:[0-9]{2}_` — straggler
   files with non-conforming names (e.g. `hsd_mw:*`) exist in some months.

**Example list to show the user:**
```
Sessions with activity today (latest first):
  1. 18_10:33:36   2m    0 errors
  2. 18_10:13:25   4m    0 errors
  3. 18_09:57:56   5m   17 errors  <E> 1 client did not respond to configure
  4. 18_06:56:30  139m   0 errors
  5. 18_06:53:09   3m   23 errors  <E> configure failed to change state
  ...
Which session? (or say 'yesterday' / give a prefix directly)
```

Tell the user they can also request a different day or specify a prefix directly.
Month directories are self-contained (no writes bleed past month end), so only
the day boundary needs handling.

Once you have the session prefix, scope all further greps to
`<dir><prefix>_*` rather than scanning the whole month directory — a single
month directory can hold on the order of 2000+ files (verified: 2778 files in
xpp September 2026 — 2292 `.log` and 486 `.log.zst`).

### Compressed/rotated logs

Rotated logs are **zstd-compressed** (`.log.zst`). You must use `zstdcat`
(not `cat`/`grep` directly) to read them:

    zstdcat foo.log.zst | grep '<E>'

In the verified sample directory, of 2778 total files, 2292 were `.log` and
486 were `.log.zst`.

---

## Header block (every log file starts with one)

Every log file begins with a header block of `#`-prefixed lines containing
operational metadata that is itself useful diagnostic data — not just log content.
As of `lcls2_091826`, some files begin with 1–4 lines of `git describe` stderr
output before the first `#` line (a known regression in the log-header generator
being fixed in this PR). Find the header by scanning for the first `#`-prefixed
line rather than assuming it is line 1.
Verified real example (`drp-srcf-mon008:ami-meb0.log`):

    # SLURM_JOB_ID:69442
    # ID:      ami-meb0
    # PLATFORM:0
    # HOST:    drp-srcf-mon008
    # CMDLINE: monReqServer -P xpp -C drp-srcf-mon008 -M /sdf/group/lcls/ds/daq/prom/xpp -d -n 60 -q 31 -p '0' -u ami-meb0
    # CONDA_PREFIX:/sdf/group/lcls/ds/ana/sw/conda2/inst/envs/daq_20250402_r9
    # CONFIGDB_AUTH:*****
    # TESTRELDIR:/sdf/group/lcls/ds/ana/sw/conda2/rel/xpp/lcls2_090826/install
    # SUBMODULEDIR:/sdf/group/lcls/ds/ana/sw/conda2-v4/rel/lcls2_submodules_09092026

`CMDLINE` reveals the actual runtime tuning flags for that process — here
`-q 31` is the queue depth and `-n 60` a buffer/count parameter, `-M ...` the
Prometheus metrics output directory. This is how you cross-check what
`psana-daq-monitor` shows in Grafana against what was ACTUALLY configured
for that process: Grafana dashboards show current metric values, not the
configured limits that produced them.

Always read the header block before grepping for errors — scan past any
non-`#` preamble lines to find `# SLURM_JOB_ID:`, `# HOST:`, `# CMDLINE:`,
`# TESTRELDIR:` etc. These give the process's host, PID/job, and startup flags.

---

## Log line grammar

    <hutch>-<process>[<pid>]: <L> <message>

where `<L>` is a one-letter level: `<C>` (Critical), `<E>` (Error), `<W>`
(Warning), `<I>` (Info). Verified real examples:

    xpp-teb[1788610]: <C> Inadequate RTPRIO limit: got 0, require 99
    xpp-drp[2352331]: <C> Inadequate RTPRIO limit: got 0, require 99

**Lead any investigation with a Critical/Error grep, excluding the RTPRIO startup noise:**

    grep -E '<[EC]>' <session-prefix>_*.log | grep -v 'Inadequate RTPRIO'

`Inadequate RTPRIO limit: got 0, require 99` fires once per process at startup
and accounts for **1213 of 1382 `<C>` lines** (88%) across one month of xpp
`.log` files — it is benign and appears in every session. Filtering it first
makes the remaining output genuinely high-signal.

Verified counts across one month of xpp `.log` files (2292 uncompressed, 232742 total lines):
1382 `<C>` (1213 RTPRIO, 169 real), 579 `<E>`, 1951 `<W>`, 34291 `<I>`.

---

## Component name catalog

Known component-name patterns seen in real filenames, to help recognize what
a log belongs to:

- **Event builder**: `tebN` (trigger event builder)
- **AMI monitoring pipeline**: `ami-mebN`, `ami-global`, `ami-manager`,
  `ami-node_N`, `ami-prefetch_N`, `ami-client`
- **DRP data-source variants**: `drp_bld`, `drp_pva`, `bld_N`, `epics_N`,
  `epicsArch`, `timing_N`
- **Control-room tooling**: `control`, `control_gui`, `daqstat`, `xpmpva`
- **Detector-specific**: `hsd_N`, `hsdioc_*`, `hsdpvs_*`, `epix100_N`,
  `jungfrau1M_N`/`jungfrau_N`, `wav8_ipm2_N`, `wav8_ipm3_N`,
  `wav8_lodcm_N`, `wav8_user_N`, `alvium_1_N`, `alvium_tt_N`, `zyla_N`

## Host name catalog

- **DRP compute nodes**: `drp-srcf-cmp0NN` (data-recording processes) or
  `drp-srcf-mon0NN` (monitoring/MEB processes)
- **Control host**: `<hutch>-daq`
- **Detector-specific IOC hosts**: `daq-<hutch>-<detector>-NN`, e.g.
  `daq-xpp-hsd-01`

---

## Practical guidance

- Bound greps to the current session's shared filename prefix rather than
  scanning the whole month directory, e.g.:

      grep -l '<C>\|<E>' /sdf/home/x/xppopr/daq/logs/2026/09/15_15:31:17_*

- Narrow further by component or host substring when the user names one,
  e.g. `*teb*.log` or `*drp-srcf-mon008*`.
- Remember rotated `.log.zst` files need `zstdcat`, not `grep` directly —
  `zgrep`-style tooling is not guaranteed to be `zstd`-aware, so pipe through
  `zstdcat` explicitly.
- If a component's current log is empty or missing, check whether it only
  exists as a `.log.zst` from an earlier rotation in the same session.
