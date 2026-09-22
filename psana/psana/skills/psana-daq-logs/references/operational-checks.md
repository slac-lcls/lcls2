# Focused operational checks

Use only for implicated symptoms, with the selected launch/release. These checks
are source-derived diagnostic leads, not verified remedies. Inspect retained
logs first. Do not submit/cancel jobs, reset devices, remove IPC objects, change
PVs or alter production environments as part of this read-only skill.

## Launch environment and Slurm

For launch failures, correlate control, TEB and detector/DRP headers by host,
job ID, command and launch time. A missing DAQ log may mean failure before the
process started; retain launcher/batch stderr as evidence too.

`psdaq/psdaq/slurm/utils.py::build_sbatch_env` allowlists submission variables,
excluding inherited `SLURM_*` CPU bindings and job IDs. In already captured
`DAQMGR_DEBUG_ENV` dumps, distinguish **BATCH ENV BEFORE SRUN** from **STEP ENV
AFTER SRUN**. Bindings in a new step can be normal; bindings inherited before
`srun` are a lead for CPU-allocation errors. Check the deployed version before
assuming this isolation exists. The debug gate is evaluated while generating
the batch script; setting it in the investigator's shell changes no existing log.
Do not restart jobs to obtain dumps during a read-only investigation.

Use retained scheduler state/exit codes or authorized read-only job inspection
to distinguish allocation failure from a DAQ transition failure. If startup
reports a missing MPI plugin, compare the command's requested plugin with the
execution host's installed capabilities. Do not prescribe a package version or
change Slurm configuration from an old incident. Record missing evidence and
propose a version-appropriate check by the responsible operator.

Debug dumps may contain secrets beyond any built-in redaction. Keep only needed
sanitized fields in a report; never publish whole environments.

## TEB Python trigger and IPC

`psdaq/psdaq/trigger/src/tebPyTrigger.cc::TebPyTrig::configure` uses the trigger's
`pythonScript` with `script_path` (default `.`). Compare the actual TEB command,
working directory and same-release script before treating a Python startup
failure as a detector fault.

IPC names are built from platform/TEB identity in that source. For permission
or connection failures, inspect retained ownership/process evidence on the TEB
execution host. Similar names on different hosts need not collide. Ownership
alone does not prove an object stale; never unlink queues/shared memory during
this investigation. Any proposed cleanup needs separate operator handling and
subsequent Configure/acquisition verification.

## DRP output paths

`psdaq/drp/DrpBase.cc` constructs/checks output paths on the execution host.
Compare the path reported in the error with actual `-o`/instrument arguments
and that host's filesystem evidence. A path visible on the control host does
not prove it is mounted or writable on the DRP host. Propose a path/mount repair
only after locating the failing check; a later successful transition and file
creation would be outcome evidence, not the recommendation itself.

## PVA updates and timing

`psdaq/drp/PvaDetector.cc` defaults PV specs to provider `pva`; a `ca/` prefix
selects CA. Confirm the launched spec and effective provider/environment before
following a discovery diagnosis. Do not apply CA-specific network advice to PVA
without evidence. No network or IOC changes are part of this skill.

That source sets `MissingData` for missing matched PV data and `TimedOut` on
the timeout path. Correlate `drp_event_rate`, `drp_update_rate`,
`drp_empty_count`, `drp_tooOld_count` and `drp_timeout_count` with the same
launch and selected Configure interval. These distinguish missing updates,
timestamp mismatch and timeouts better than aggregate damage alone.

`TebReceiverBase::process` in `psdaq/drp/DrpBase.cc` observes each set damage
bit into `DRP_DamageType`. A histogram sum/count ratio is an average of bit
indices and cannot uniquely identify a mixture. Inspect distributions and
underlying event/log evidence; bit numbers come from
`xtcdata/xtcdata/xtc/Damage.hh`.

For missing triggers, identify whether timing terminates in a DAQ front end or
an external camera-trigger path before choosing checks. Use the relevant
same-release configuration code and an available specialist skill. Detector
register values, firmware/YAML recipes and calibration recovery procedures need
matching hardware/release evidence; a shared symptom alone is insufficient.
