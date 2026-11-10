# Reading evidence: traps at the keyboard

**Audience: AI assistants debugging or measuring the DAQ.**  Terse by design.  Each entry
is here because a competent reader would plausibly draw the wrong conclusion.  Rules for
*editing* the GPU DRP are in `gpu-drp-rules.md` beside this file; node and driver setup is
on the "GPU node rules" Confluence page.

## Rates, deadtime and latency

- **AN L1ACCEPT RATE IS A DEADTIME MEASUREMENT, NOT A THROUGHPUT MEASUREMENT.**  Measured
  twice on gpu008 on 2026-10-12, four minutes apart, same code and trigger cadence:

      provider   DeadFrac   predicted            measured    agreement
      sockets     0.705     71428*(1-0.705)      19.25 kHz   ~91%
      verbs       0.558     31535 Hz             31.53 kHz   99.97%

  With `L0InpRate{partition=N}` = 71428 Hz (13 ticks, the real trigger cadence),
  **`measured = L0InpRate * (1 - DeadFrac)`** -- in both runs the DAQ took everything
  the XPM offered and neither reached a rate ceiling.

  - **Comparing raw rates across runs compares deadtime.**  The historical "33 kHz" is
    itself deadtime-limited; reaching it needs `DeadFrac` 0.538 against the 0.558
    measured, so a 1.5 kHz shortfall is a 2-point deadtime difference, not a regression.
    **Quote `L0InpRate` and `DeadFrac` beside any rate.**
  - **A provider's cost appears here, not as lost throughput.**  `sockets` runs 8 progress
    and CM threads inside the DRP's cpuset where `verbs` offloads to the HCA; the dead
    fraction fell 70% -> 56% when that changed.
  - **A low rate alone does not show CPU starvation.**  Check `DeadFrac` against
    `L0InpRate` first; the rate is usually explained there.
  - **Trigger period cannot be inferred from L1Accept spacing.**  Vetoed triggers never
    reach the detector, so arrivals are spaced at whole multiples of the cadence and look
    exactly like unanswered triggers.
- **Deadtime and latency are different questions.**  `tstcam1_3`'s thread wrap was called
  harmless from `drp_deadtime 0.0000` while it ran ~1000x its peers' latency.
- **`DRP_evtLatency` on `sockets` cannot resolve a small effect**: uniform 85-94 ms across
  six DRPs, with individual ones swinging 61% between two samples of one run.  Use `verbs`
  (floor ~250x lower, 0.1-0.2 ms) and windowed averages, not instantaneous samples.
- **A latency is wall clock minus the TimingHeader stamp, so it carries a clock offset.**
  The clocks are on different machines under separate NTP; compare a DRP only with itself
  or with DRPs on the same node.  `timing_2`'s `drp_th_latency` stepped ~0.5 ms on
  2026-10-20.
- **`drp_deadtime` can read negative** (-4.8, -11.6 seen on gpu008, 2026-10-15).  Treat
  per-DRP deadtime as unreliable until that is chased; a sick DRP reading 0.362 against
  peers' 0.000 is still a usable discriminator.
- **`readerHist=1` adds ~1.2-1.9 us to every stage's service time.**  Quote
  `DRP_readerTime`/`tigTime`/`reduceTime` from a run without it.
- **With `-W 0` no Reducer runs**, so `DRP_redStarts` and `DRP_redRcvs` are 0 and their
  agreement says nothing.  Nothing is recorded either.

## Prometheus

- **The server is `http://drp-srcf-eb011:9090`; the shell's Squid proxy breaks it.**  Use
  `curl --noproxy '*'` (`requests` needs `trust_env=False`); an un-proxied query returns an
  HTML error page.  `psmetric03` is the old CDS server: live, thousands of targets, none of
  ours.
- **URL-encode the selector** (`--data-urlencode 'query=...'`); a bare `{...}` fails to
  parse.  Filter by `instrument=` or `partition=`, or other hutches' series mix in.
- **A window in the future returns `status: success` with an empty result.**  Check
  `date +%s` against the window before believing an empty answer.
- **Metrics vanish at Unconfigure**: use `query_range`, not point samples, which can also
  land mid-ramp.  A `rate(...[1m])` over a run shorter than the window reads far low (a 12 s
  Enable at 33 kHz read 6.4 kHz); use per-scrape deltas or a window inside the run.
- **Never average a window that contains startup.**  `avg_over_time(...[10m])` across a
  resume made six DRPs look 20-40x worse than the steady 0.7 ms.  Windows beat instants,
  but place the window deliberately.
- **Each DRP exports on its own port, reassigned on every launch.**  A selector returning
  one series does not mean the others are absent; and a port->alias map goes stale at each
  restart (`tstcam1_0` moved onto what had been `timing_2`'s port).  Re-derive it from the
  logs.
- **A gauge pinned at its pool size is saturated, not diverging.**  A queue holding 2040 of
  2048 for minutes is flat at the ceiling.  Check the slope before calling a runaway.

## Logs, processes and nodes

- **First find the `.cnf.py` the DAQ was started with, and read it.**  It decides which
  processes run, on which nodes, with what flags, `-k` kwargs, `cores:` and `env:`, and
  which detectors are commented out.  Each DAQ (hutch or test) has its own.  `daqstat`'s
  log header `CMDLINE` names it relative to the job's working directory, which
  `sacct -j <SLURM_JOB_ID> -o WorkDir%80` gives; if there is no `daqstat`, ask the human.
- **The logs record the cnf's name, not its contents, and cnfs are edited freely.**  If its
  mtime is later than the log's start, what ran is unknown: say so, and ask, rather than
  reasoning from the current file.
- **A DRP's log is `~/daq/logs/<year>/<month>/<DD>_<HH:MM:SS>_<node>:<alias>.log`**, in
  the home of whoever ran the DAQ (hutches: `/sdf/home/<h>/<h>opr/`), and
  includes the full configdb JSON it was given.  For *setup* questions (alias, trigger
  library, buffer sizes) it answers in one file what otherwise takes reconstruction from
  the xtc.  One "undefined symbol" line per Configure is expected: `DrpBase` probes
  `create_producer_<detName>` before `create_producer`.
- **A crashed Python process under `srun` can still show `RUNNING`.**  Read the log.
- **`cores:` in a `.cnf.py` is a request; `Cpus_allowed_list` in `/proc/PID/status` is what
  the process got.**  A job granted 4 CPUs can run in a 3-CPU cpuset when WEKA holds CPUs
  inside the range.  Slurm defines the set to pin within; `sched_getaffinity` reports it.
- **`pgrep -f <pattern>` matches the `srun` wrapper and your own shell first.**  Check
  `/proc/PID/comm`, or use `pgrep -x drp_gpu`.
- **`/proc/PID/maps` is Permission denied even for your own `drp_gpu`** (file
  capabilities).  Read its RPATH with `readelf -d` instead.
- **`top -b -n1 -H` shows every thread at 0.0%.**  Difference `utime + stime` from
  `/proc/PID/task/*/stat` over an interval; the same process had threads at 101% and 93%.
- **An install mid-start names an innocent component** (e.g. `ModuleNotFoundError` in
  `control_gui`).  `/usr/local/bin/drp_gpu` has one copy per node: whoever installs last
  wins, so coordinate.
- **psana's "Problem reading dgram header."** is how it raises `StopIteration` at EOF; it
  says nothing about a missing `EndRun`.

## Detector, configdb and site facts

- ePixUHR3x2 `EvrV2ChannelReg[0]` `RateSel` must be 276 for xpp; 256 is the ASC lab's value
  and the ASC det docs carry it.  Symptom: both panels stop at the same event with
  deadtime 1.0.  Prod and dev xpp configdb BEAM for `epixuhr3x2_{0,1}` hold 276.
- gpu006's `timing_2` runs LINAC=0, so `user.Cu` event codes apply: ePix readout group 4
  is event code 277 (120 Hz), timing's group 2 is 278 (240 Hz).
- In CALIB only the timing DRP is mandatory.  The ePixes need `timing_2`'s CALIB to carry
  BEAM's event codes; CALIB entries come from `calib_cdict()`, `_cfgTypeRef` BEAM
  supplying what they omit.
- A cnf's `env:` entries override the shell's exports (`slurm/utils.py` appends them to
  `--export` after `SUBMODULEDIR`).
- **A DRP's `-o` directory must already contain `<instrument>/`**, or nothing is recorded.
  `/home` is on the node's local disk (checked on gpu008), so data written there stays there.
- `SerNo` (`epixuhr3x2.py`) is `<C1100 f/w version>-<carrier id>-<readout id>` (the
  docstring lists readout before carrier; the code builds carrier first), so every C1100
  f/w update orphans the panel's constants and needs new darks.
- `ctrb.cc` is bit-rotted (32 errors under `-fsyntax-only`), which is why it is not in
  `eb/meson.build`.  Syntax-check it in place: a copy under `/tmp` fails on a relative
  include.

## Transport and event building

- **Do not assume a run used the provider you are testing.**  Check `ep_provider` on the
  log's command line and which subnet the EB links came up on (InfiniBand means `verbs`).
  An unstated provider is whatever `fi_getinfo` returned first.
- **A posted-vs-completed deficit on a Tx CQ is not a loss count.**  It is bound
  `FI_SELECTIVE_COMPLETION`, so successful writes complete silently; the CQ error count is
  what matters.
- **`fi_writedata` success means queued, not delivered.**  A later failure shows only when
  the Tx CQ is drained.
- **A provider's default queue depth is not its ceiling** (the hint is 0).  `fi_cq_open`
  accepts an oversized request on every provider; the refusal comes earlier, in
  `fi_getinfo`.

## Reasoning from evidence

- **Compare posts with receipts by pulse ID, not ring slot.**  A slot recurs every lap;
  per-index counting has produced two confident false retractions.  First establish the
  window has not wrapped.
- **"It didn't crash" is not "it worked."**  Size, extent, damage and shape can all be
  right while the payload is misplaced; read the leading values.
- **A "changed since last time" guard cannot bound output when the value alternates.**  Two
  contributors failing in turn pass a `srcId != prvSrc` test every time.  Bound with a
  count.
- **The absence of a dump is not the absence of a failure.**  Bounded diagnostics must say
  when they go quiet.
