# Actual Jungfrau pedestal workflow: chunk-size workaround

The bug reproduces in the repository's Jungfrau pedestal application on real
data. Raising PS_SMD_N_EVENTS to fit the entire run in one chunk works around
both the step- and event-counter resets in this measured case. Fitting a single
step per chunk is insufficient for the full two-stage workflow.

The generic fallback in psexp/tools.py is 20000, but DataSource's
`_force_mfx_overrides()` sets **1000 for MFX when PS_SMD_N_EVENTS is unset**,
before that fallback is used. Explicit environment values are preserved in both
tested revisions. Therefore 100 is a forced stress setting; 1000 is also the
effective default for this MFX workflow. A separate unset-variable probe of the
unfixed stage-1 step-2 job confirmed effective value 1000 and the same failure.
The successful workaround here is `export PS_SMD_N_EVENTS=10000` inherited by
the serial jobs and MPI ranks. Its sufficiency is established by observing one
actual chunk on this run, not by assuming the setting guarantees a single chunk
for every input/buffer configuration or a larger/live run.

Data: mfx100848724 run 49, detector jungfrau, three physical steps of 1000
events. The documented example run is available locally; Mikhail's latest
private command was not verified. Numerical processing uses supported
`--segind 0` (one 512 x 1024 panel). This is a correctness test, not a full-detector
performance benchmark. Calibration deployment was disabled in every command.

| Code | PS_SMD_N_EVENTS | Stage 1: gates for steps 0/1/2 | Stage 2: selected events | Stage 2 arrays |
| --- | ---: | --- | --- | --- |
| Unfixed | 100 | Steps 0/1 pass; step 2 selects zero and errors | 300 instead of 30 | Different |
| Unfixed | 1000 | Steps 0/1 pass; step 2 selects zero and errors | Correct 30 | Exact match with reference gates |
| Unfixed | 10000 | All three pass | Correct 30 | Exact match |
| Fixed | 100 | All three pass | Correct 30 | Exact match |
| Fixed | 10000 | All three pass | Correct 30 | Reference |

The stage-2 reader built 31, 4 and 1 nonempty input views at settings 100,
1000 and 10000 respectively, including transition-only tails. Unfixed callback
invocation counts equal those counts; fixed runs invoke the callback once.
Stage-1 jobs can stop before run end, so their measured chunk counts differ.

Stage 1 requests the first 10 events separately for each step. With unfixed
small/step-sized chunks, `--stepnum 2` receives no events and the application
raises `UnboundLocalError: local variable 'ievt' referenced before assignment`
in its post-loop logging. This exposes a secondary empty-selection handling
problem; no changes to that application code were made. Step 1 happens to work
for this input layout; it is not correct to claim every nonzero step fails.
In the unfixed RunSmallData.steps(), a resumed open step is yielded first, at
callback-local index 0. A subsequent BeginStep in the same boundary chunk is
then local index 1. This explains the observed step-1 success; local index 2 is
never reached in these small/step-sized chunks, even at physical step 2.

Stage 2 uses `--evskip 2 --nrecs 12 --nrecs1 0`, selecting indices [2:12]
within each physical step: 10 per step, 30 total. Unfixed 100-event chunks
reselect that window in each chunk, processing 100 events per step, 300 total.
This job exits successfully despite using the wrong inputs. Pedestal means
differ at 524088 / 523864 / 523855 pixels for gain steps 0 / 1 / 2. Their maximum
absolute differences are approximately 105.85 / 20.40 / 19.56 ADU. These
differences demonstrate changed statistics in this small sample, not a claim
about production calibration accuracy.

Correct cases have exact selected timestamps, exact gate arrays, and exact
stage-2 pedestal mean, RMS, status and accepted-count arrays. Timestamp truth
comes from direct-file SMD reading, bypassing callback/chunk machinery; the
numeric reference is the actual application on fixed whole-run input. The
numeric check establishes equivalence, not independent validation of the
pedestal algorithm. Every stage-2 case receives copies of the same six correct
reference gate files, so failed stage-1 jobs cannot confound stage-2 behavior.
In particular, unfixed 1000 is not a successful end-to-end workaround.

## Reproduction and evidence

- Unfixed source: `71378658a54dcdd941c9137355a985fa4b637595` in
  `/tmp/jf-callback-ab/unfixed`.
- Fixed production source: `b60fae55728e7d138747db57fad2c84c10543910`.
- The application/calibration Python and native `.pyx` sources are identical
  between these revisions. Both overlays reuse the same local native build.
- Environment: `source setup_env.sh`, Python 3.9.20 ps_20241122; fixed source
  overlay `/tmp/smd-callback-chunk-state/python`; no greenlet dependency.
- `prepare.py` builds the baseline overlay and reads the independent oracle.
- `run_matrix.py` launches 15 serial stage-1 jobs and five stage-2 jobs using
  five MPI ranks (one SMD0, one EB, two BD, one SRV). Each job has a 240 s timeout
  that kills its process group. Jobs run sequentially on sdfiana025.
- `run_stage.py` calls the real CLI, observing callback invocations, timestamps
  passed to detector raw(), nonempty SMD views, and final numeric arrays.
  It does not replace filtering, event iteration, reductions or arithmetic.
- Explicit local data paths and a stub of external logbook file discovery are
  used in every case. Monitoring is disabled. Detector-name lookup is read-only.
- `analyze.py` checks all 20 outcomes, including expected pre-fix failures.
  Raw timestamps retain duplicates for comparison; matching counts alone do
  not constitute a pass. Gate and output arrays are compared elementwise.
  The expected pre-fix outcome branches were written after inspecting the runs
  and are specific to run 49, these chunk sizes and these selection parameters;
  this is an experiment record, not a general reusable regression test.

Per-rank source provenance checks cover two modules: UtilsJungfrauCalib and
smdreader_manager. The uniformly constructed overlay selects the remaining
Python sources from the same checkout, but their resolved paths were not all
recorded. Future runs can expand that instrumentation. The scripts hard-code
Mona's source prefix `/sdf/home/m/monarin/lcls2/psana/psana/` and existing source
overlay `/tmp/smd-callback-chunk-state/python`; another operator must adapt
those paths or provide the same layout. Matrix artifacts are fixed at
`/tmp/jf-callback-ab/matrix`, which must be absent for a fresh complete run.

Run preparation with the fixed overlay first on PYTHONPATH and PS_PARALLEL=none.
Run the matrix in a fresh setup_env.sh environment; its subprocesses select
their own overlays. Use a fresh artifact directory for another complete run.
No production source was edited, so the previous core test results remain
applicable; validation here is the actual workflow matrix and script compile
checks. Two expected stage-1 failures are reported explicitly, not hidden as
successful runs. Some successful MPI logs include a UCX unmatched-receive
warning at shutdown; all five ranks returned and wrote their reports.

Compact results and commands are committed here. Full logs, rank timestamp
reports, gates and numeric snapshots remain under `/tmp/jf-callback-ab/matrix`;
artifact-manifest.json records their SHA256 digests. The preliminary pilot used
a string-valued monitor option and is excluded; the final matrix passes a
Python boolean and is the evidence summarized above.
