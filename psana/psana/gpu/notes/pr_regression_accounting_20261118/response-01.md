# Round 1 response

Review of `1897b6404aa4181050f0288fbd78ef1b5cc8c240`: PASS, no blockers.

- ACCT-01: add offline mocked full-mode orchestration and MPI preflight tests.
  The executable implementation is unchanged; test outcomes recorded separately.
- ACCT-02: accept bounded timeout risk explicitly. The slowest smoke loop was
  21.37 s and measured process 35.68 s. Linear 10x loop plus the observed 14.31 s
  fixed cost predicts 228.01 s, leaving about 72 s within the 300 s step limit.
  The selected 335,571,760,000 bytes need 0.559 GB/s for the 600 s prefix hash
  deadline, plus 50k header seeks. These are estimates, not guaranteed rates.
  A timeout stops the attempt and will be retained, not relabeled. Worst-case
  prefix + placement + controls + six adapter deadlines is about 6,924 s within
  7,200 s allocation walltime; setup/filesystem overhead can consume the margin.
- ACCT-03: preserve raw timestamps in every indexed sample; the coordinator
  report adds per-interface counter and monotonic interval deltas. Controls
  include 0.25 s settle; cells sample immediately after their loop barrier.
  This asymmetry is explicit, and the per-cell coverage floor must independently
  pass. The public HPE driver implementation retrieves a hardware register then
  attaches current realtime to each sysfs read; it does not expose an independent
  firmware-update timestamp. Advancing timestamps do not prove traffic freshness.
- ACCT-04: document that observe mode extends the request AND buffer lifetime.
  It diagnoses pending-at-return versus cancelled-at-return, without claiming
  byte-for-byte reproduction of production object lifetime. Real cells retain
  normal production lifetime and its warning.
- ACCT-05: prelaunch-evidence.json contains login samples and exact installed
  node.py/Git identity. frozen-01-SHA256SUMS was generated from reviewed Git blobs
  and compared with the working source before freezing; it binds the launched
  files to the production commit.

Reference for sysfs formatting/direct register read (primary vendor source):
https://github.com/HewlettPackard/shs-cxi-driver/blob/main/drivers/net/ethernet/hpe/ss1/cass_telem.c
The vendor source is supporting context, not proof of installed driver identity.
