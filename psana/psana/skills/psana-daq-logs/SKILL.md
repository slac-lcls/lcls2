---
name: psana-daq-logs
description: Read and search raw LCLS-II DAQ log files on disk for the currently running session or a specific past session you can identify by date/prefix (not a general-purpose archival search tool for arbitrary old data). Use for "read DAQ log files", "find error messages in DAQ logs", "why did a DAQ component crash", "current/live run log inspection", "tail DAQ logs", "DRP/TEB/MEB log errors".
---

# Skill: psana-daq-logs

# LCLS-II DAQ Live Log Inspection

You are reading raw DAQ log files directly off the filesystem to diagnose the
CURRENTLY RUNNING session or a specific past session identified by date/prefix
via the session-listing mechanism below. This skill is
self-contained — it reads only the plain-text/zstd log files at the paths
below. There is no log aggregation database or search index behind this;
every command here is a direct filesystem operation.

**Related skills:** preserve the supplied hutch, time window, launch identities,
release and available evidence. Load `psana-daq-monitor` only when retained
metrics would help; a log-only question needs no metrics preflight. Load
`psana-configdb` for configuration-related findings. For an implicated detector,
use an available matching specialist skill/reference after identifying its launch
and release. Do not assume personal or external skills are installed.

## Path convention

Use a supplied evidence directory first. Otherwise inspect the launcher/config
and operator-home context without importing configuration Python. In
`psdaq/psdaq/slurm/utils.py`, `SbatchManager.get_default_output_root()` selects
`$HOME/daq/logs` for some hutches and `$HOME` for others; `daqmgr --output`
overrides it (`psdaq/psdaq/slurm/main.py`). These are the launcher's HOME and
output settings, not necessarily the investigating agent's HOME.

Logs are placed under `<output-root>/<YYYY>/<MM>/`. Check the configured root
and readable alternatives supported by launch evidence. An absent conventional
path does not prove that no logs exist. Report unavailable paths or incomplete
retention explicitly; do not scan unrelated homes.

## Filename grammar

    <DD>_<HH:MM:SS>_<host>:<component>.log[.zst]

`SbatchManager.__init__` assigns the directory and prefix once using the
launcher's local `datetime.now()`. The prefix is a launch-group hint, not a
DAQ run number or a globally unique session ID. Keep the full directory,
hutch, component/host and header job/command identity: separate launches can
share a prefix, overlap, or restart only some components. Corroborate grouping
with control, detector/DRP and TEB headers before joining their findings.

### Session selection and historical bounds

1. Reuse a supplied session or historical window. Ask only when unresolved
   ambiguity changes the investigation. For a hutch-wide window, include all
   candidate launches intersecting it; do not require one prefix or group the
   report by platform/partition. Keep those values as evidence metadata.
2. Resolve a prefix with the parent **year/month** and the launcher's time zone.
   Record the zone and UTC offset, and convert to UTC for cross-source joins.
   Do not use the agent's current month/year or time zone. At a daylight-saving
   fold, retain both possible instants until offset/timestamps distinguish them.
   A local time in a daylight-saving gap is inconsistent with the stated zone;
   do not normalize it silently. Keep its identity/coverage unresolved.
3. The output path stays fixed for the launch: logs can continue in the old
   month/year directory after midnight or month/year rollover. Search earlier
   launch directories when the requested window may overlap a long-lived
   launch. There is no fixed one-month maximum lifetime in the launcher.
4. Prefer timestamped control transitions, run metadata and XTC BeginRun/EndRun
   evidence for run boundaries. A launch can contain several runs or none.
   Untimestamped component lines can establish ordering/context but cannot be
   assigned exact times merely because metrics have a spike nearby.
5. Prefix time and newest file mtime provide only an **estimated file-activity
   interval**, not DAQ run coverage. Copying, compression, touching, sparse
   logging and missing files can distort mtime. Do not exclude a possibly
   overlapping launch solely because its last log write precedes the window;
   label uncertain overlap/coverage. Reject negative intervals as inconsistent
   evidence rather than silently wrapping the date.
6. If selection is needed, show full dated launch identities, observed or
   estimated bounds with their basis, file count, non-RTPRIO error-line count
   and first excerpt. Sort by resolved full timestamps, not by day-prefix text
   across directories. Mark ambiguous timestamps and overlaps separately.
7. Ignore nonconforming names for prefix grouping, but retain relevant supplied
   files as ungrouped evidence. Once scoped, cache the file list, headers and
   useful excerpts so a narrow follow-up does not rescan whole month directories.

### Compressed/rotated logs and error counts

Read `.log.zst` through a zstd decompressor; never grep compressed bytes. Use
the same reader for counts and excerpts. For one explicitly selected file, this Bash
example filters messages **before** counting and returns `0` successfully for
no matching errors. Run it with `bash -o pipefail`; a read/decompression failure
is unavailable evidence, never a trustworthy zero:

```bash
read_daq_log() {
    case "$1" in
        *.log.zst) zstd -dc -- "$1" ;;
        *.log) cat -- "$1" ;;
        *) printf 'Unsupported log format: %s\n' "$1" >&2; return 2 ;;
    esac
}
count_daq_errors() {
    read_daq_log "$1" | awk '/<[EC]>/ && !/Inadequate RTPRIO/ {n++} END {print n+0}'
}
count_daq_errors "$log_file"
```

Use `zstd -dc` without `-f` here: some `zstdcat` versions pass unrecognized
input through unchanged, concealing corrupt or mislabeled files. Discard stdout
from a failed pipeline even if awk printed `0` before the reader failed.

Apply per file and retain path/count pairs; aggregate only successfully read,
nonduplicate evidence. If both plain and compressed copies represent the same
content, select one or establish rotation/overlap before adding counts. For
excerpts, replace the awk expression with
`'/<[EC]>/ && !/Inadequate RTPRIO/ {print NR ":" $0}'` and cite the file plus
**decompressed** line number. Bound displayed excerpts without truncating the
reader prematurely (which can cause SIGPIPE under `pipefail`).

Keep total `<C>`/`<E>` and RTPRIO counts/excerpts alongside the filtered view.
Repeated lines are not automatically separate incidents; use component,
transition, launch/run context and time evidence before grouping occurrences.

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

Use the decompression-aware reader and filter above for an initial high-signal
view, retaining the unfiltered evidence. In
`psdaq/psdaq/service/Collection.cc::checkResourceLimits`, inadequate RTPRIO is
logged at critical level but marked nonfatal. That means the process may
continue, not that scheduling is healthy. Check subsequent transitions and
scheduling/latency evidence before deprioritizing it; revisit it when scheduling
is implicated. Do not globally erase these lines or count every critical line
as a distinct failure.

## Interpreting `<C>`/`<E>` messages

Do not rely on a pre-built error catalog — DRP/TEB source has roughly 459
`logging::critical`/`logging::error` call sites across `psdaq/drp/*.cc` (339)
and `psdaq/psdaq/eb/src/*.cc` (120), most emitting dynamic (`%s`-forwarded)
content that a static catalog can't usefully capture, and message text drifts
between releases (see "Release-source navigation" in the `psana-daq` router
skill). Once you have a `<C>`/`<E>` line:

1. Grep the **static portion** of the message against the running release's
   source (per the release-source technique) to find the call site and its
   surrounding context/comments.
2. **Some messages carry their own remedy.** For example, `"XPM Remote link
   id register illegal value: 0x%x. Try XPM TxLink reset."`
   (`psdaq/drp/BEBDetector.cc:176`) already tells you what to do. Read the
   full message before concluding you need to interpret it further.
3. **Misconfiguration vs. hardware fault is not always obvious from the
   message text alone.** Check whether the failing check validates a
   user-supplied config value — e.g. `"nDmaBuffers (%u) can't exceed
   evtCounter range (0:%u)"` (`psdaq/drp/DrpBase.cc:144`) is a config bug, not
   a hardware fault. This distinction changes what the user should do next.
4. **Watch for anomalous line counts before treating hits as discrete events.**
   Check for repeated messages or dump loops; compare first/last excerpts,
   launch identity and transition context. A large count can describe one
   persistent condition. Without timestamps, occurrence times remain unknown.

For launch/environment, IPC, output-path or PVA symptoms, read
[references/operational-checks.md](references/operational-checks.md).

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

- Bound reads to selected launch files, then narrow by named component/host.
- Inspect matching control, detector/DRP and TEB evidence for failed transitions.
- Use the reader above for both `.log` and `.log.zst`. Missing or unreadable
  files limit coverage; absence of matches is not evidence of a healthy DAQ.
- A message suggesting a reset or other remedy is a proposal to evaluate,
  not authorization or proof that it will work. This skill remains read-only.
