# Exact Jungfrau stage-1 command comparison

Run Mikhail's command unchanged in three isolated working directories:
`jungfrau_dark_proc -k exp=mfx100848724,run=49 -d jungfrau -o ./work1 --nrecs 50 --nrecs1 50 --stepnum 2`.
Use the full detector, serial mode, unfixed with PS_SMD_N_EVENTS unset, unfixed
with 10000, and fixed with the variable unset. The effective MFX default is 1000.

Unfixed source: 71378658a54dcdd941c9137355a985fa4b637595.
Fixed source: b59c2fd1aad012de195b14b62401bc966e6755c3 in a separate detached
checkout. Use separate Python overlays with the same unchanged native build.
This reproduces his command on our source pair, not his complete installed build.

Protection: fresh directories under /tmp/jf-exact-command; preserve ./work1 to
suppress the central startup log; force readonly detector-name lookup; block
calibration HTTP mutation helpers; no deploy/submission. Use local on-disk file
discovery as in the earlier tests. Observe the actual CLI callback and raw reads,
without replacing filtering or computation. Bound each sequential process to 600s.

Verify the default failure, exact selected/raw event timestamps against the
independent direct-file oracle, full detector shape, and exact numerical equality
of both successful gate outputs. Record effective settings and imported source
paths/hashes. No production edits or new core-suite run are needed. Retain the
implementation/review loop, up to three independent Claude rounds, then push the
reviewed evidence on the already authorized branch.
