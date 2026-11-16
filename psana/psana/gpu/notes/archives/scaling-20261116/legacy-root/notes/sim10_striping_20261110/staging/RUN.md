# Verified-copy stage: complete and accepted

Job61113228 completed successfully on2026-11-10 at23:59:00UTC
(15:59 PST), after34m49s. All job/step exits are0:0. It started at23:24:11UTC,
on nid001029, lcls_g / gpu_premium. One exclusive GPU node, CPU-only staging,
2h limit; step TERM120s before limit. No read-performance benchmark launched.

Reviewed execution commit:1082075619d15ecbb6a82fc54fec5695cba0cb75.
Round2 launch-readiness passed;17 local tests, shell syntax and diff checks
passed. Source sizes and native1-stripe layout parser checked before launch.
All round1 required reads and round2 required reads succeeded. All30 destination
sizes and full SHA256 reads passed, with counts1/2/4/8/16/32 and1MiB stripes.
The coordinator independently audited every receipt against the accepted
manifest and reread current source/output identities and all30 native layouts.
Audit passed; execution snapshot matches the reviewed commit. Final independent
round3 review passed on bffde5e859cde9e4fd5758ea4e993f2fe1e0121d; all required
reads succeeded and no blocking findings remain. Full hashes were performed by the job, not repeated by audit.
Round3 accepted the native integrity/layout evidence with unchanged execution code.
S7 deferred: status update may race on interruption; require complete.json and
successful Slurm accounting, never infer success from stale status alone.

Attempt:
`/pscratch/sd/m/monarin/psana2-gpu/sim10-striping-20261110/attempt01`

- `bundle/`: read-only stage.py, stage.sbatch and accepted transfer-manifest.json,
  extracted from reviewed commit and byte-compared with git show before submit.
- `bundle-provenance.json`: execution commit and all three SHA256 values.
- `datasets/status.json`: progress and ETA after each verified set.
- `datasets/count-01/` through count-02,04,08,16,32: five bigdata files each,
  plus verified.json only after full hash and layout verification.
- `datasets/complete.json`: required final receipt; absent until all sets pass.
- `slurm-61113228.out` and `.err`: native job logs.
- `iris-before.json`, `quota-before.txt`, `submitted-at.txt`: pre-submit checks.
- `slurm-job.txt`, `cpu.txt`, `kernel.txt`, `quota-at-start.txt`,
  `capacity-at-start.txt`: native preflight evidence.

Every destination uses1MiB stripe size,original pool,automatic OST selection.
Each set has335820951264bytes; six sets2014925707584bytes (1.8326TiB).
Source root remains
`/pscratch/sd/m/monarin/psana2-gpu/datasets/mfx101555026-r0010-sim10-100k-jf`.
Five source SHA256 reads precede all copying. Each destination is streamed into
its pre-striped inode, fsynced, given advisory DONTNEED, closed, and independently
hashed through a new buffered read. Local cache residency/remote bytes are not
verified; these are integrity receipts, not cold-read benchmark measurements.

Pre-submit lcls_g remaining: personal2128.28allocation hours (96.5%),
project6823.03hours (61.9%). Scratch quota1.36TiB used /20TiB soft.
Actual completion23:59UTC Nov10 (15:59 PST), within preliminary ETA.
Source verification106.8s; each five-file copy-and-verify set took279-351s.
Native evidence is saved in native/: complete.json, audit.json, accounting-utc.txt,
exact logs and provenance. artifact-index.json pins scratch originals to copies.

No staging work remains. All six datasets are ready for the planned read test.
The read-performance benchmark has not been implemented or launched in this stage.
Final review is saved in review-03.json. S7 is deferred for future harness reuse;
it does not affect this normally completed run. The closure commit changes only
documentation/review records after the reviewed evidence commit.
