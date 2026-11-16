# Round1 response

Reviewed bf2ecfa62fdd220051370c78398878161dc8948d: PASS with two low findings.

- SR-R1-01 accepted/fixed: manifest note and report explain a2affb7f9 as the
  implementation branch source-record commit, distinct from the measured
  framework, harness, original results review and current packaging commits.
- SR-R1-02 accepted/fixed: add observed per-process RSS high-water and task D2H
  destination-capacity maxima, explicitly not simultaneous node/total pinned
  peaks. Pin a memory summary and retain per-rank host checkpoints.
- Coordinator follow-up: the initial extraction unintentionally retained all
  callback memory samples (19MB). Replace them with exact per-rank extrema and
  sample counts, verified against the raw arrays. The raw sequences stay indexed;
  exact execution bundle, case timings, counts and original receipts unchanged.
- Authored CSV CRLF was fixed before review. The full whitespace diff retains
  one blank EOF line in the byte-exact archived Slurm receipt by design.

Re-review is required because the archive extraction and verifier changed.
