Review the real-data Jungfrau pedestal A/B harness, evidence and conclusions at
d5932e447b2ea63d7bdd3c666d2063a4926dc8cc, against base
b60fae55728e7d138747db57fad2c84c10543910. User requested testing unfixed code
with small versus large PS_SMD_N_EVENTS and keeping the independent review loop
active. Round 1 of 3. No commands, edits, posting, delegation or nested loop.
No production code changed; do not audit unrelated psana subsystems.

Read notes/smd-callback-chunk-state/jf-workflow-ab/README.md and scope.md, then
run_stage.py, run_matrix.py, prepare.py, analyze.py. Results.json and commands.json
contain the actual completed matrix. Artifact-manifest.json records raw logs,
rank JSON (exact timestamps and source paths), gates and numeric snapshots under
/tmp/jf-callback-ab/matrix. Raw artifacts outside this checkout are supplied
coordinator evidence, not independently accessible review inputs. Inspect the
production CLI/callback functions as needed: psana/psana/app/jungfrau_dark_proc.py,
detector/UtilsJungfrauCalib.py stage1 callback, detector/UtilsJungfrauCalibMPI.py
filter_callback/MPI reduction, and detector/UtilsCalib.py.

Source A is 71378658a54dcdd941c9137355a985fa4b637595 (before this branch's fixes),
B is b60fae55728e7d138747db57fad2c84c10543910 (reviewed synchronous fix). Both have
identical application/calibration and native .pyx sources, checked by git diff.
Python overlays select the respective source with the same unchanged installed
native extensions. Every rank's resolved_sources was checked against its intended
checkout path. The frozen status predates this review invocation.

Actual input: documented mfx100848724 run49. Direct-file SMD stream0 oracle has
3 steps x1000 unique events. This is the repository example workflow, not claimed
as Mikhail's latest private command. Supported --segind 0 limits calculation to
one 512x1024 panel. Stage1 selects first10 per physical step in separate serial
invocations. Stage2 runs five MPI ranks (SMD0,EB,2BD,SRV), selecting indices [2:12]
per step using nrecs12/evskip2/nrecs1=0. Every stage2 receives copies of the same
six correct reference gates so stage1 failure cannot confound stage2. No deployment,
selection/arithmetic replacement or Slurm submission. Explicit local data path,
read-only detector-name lookup, stubbed external logbook discovery, monitor=False
boolean. Preliminary pilot excluded from evidence.

Validation: analyze.py checks all20 expected outcomes, exit0. Eighteen application
jobs exit0; two pre-fix stage1-step2 jobs exit1 after receiving zero events, with
UnboundLocalError on unassigned ievt in post-loop logging. No timeouts, all five
MPI ranks wrote reports in every stage2 case. Compile and commit hooks passed.
Required core groups previously passed on unchanged fixed production (125+4);
not repeated for this notes-only harness/report. No performance claim.

Observed: fixed-small100 and fixed-large10000 have exact expected timestamps and
identical gate and stage2 mean/RMS/status/accepted-count arrays. Unfixed-large10000
also matches exactly (one measured chunk/callback). Unfixed-small100 yields300
instead of30 stage2 events (100perstep instead of10), changing arrays despite
exit0; 31chunks/callbacks including transition-only tail. Unfixed-step1000 matches
stage2 with reference gates (4chunks/callbacks) BUT fails stage1-step2, so it is
not an end-to-end workaround. Stage1 step1 happens to pass for this layout; the
report does not claim every nonzero step fails.

Check experimental validity, instrumentation effects, input/array comparisons,
overclaims and reproducibility. Timestamps retain duplicate detection. Numeric
equivalence does not independently prove the pedestal algorithm. Some completed
MPI jobs log a UCX unmatched-receive warning at shutdown (disclosed); all ranks
returned. Single-panel, small-statistics limitations are explicit. Do not claim
to run tests or verify inaccessible hashes. Give concise structured verdict,
stable finding IDs and the exact reviewed SHA.
