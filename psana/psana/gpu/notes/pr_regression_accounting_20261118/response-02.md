# Round 2 response

Review of `9a1079aa7cfa2f1b626634b7cccf92e31141a628`: PASS, no blockers.

- ACCT-02: the 600 s limit is a bounded diagnostic budget, not an empirically
  qualified completion guarantee. Scaling the entire 1k measured process
  (35.68 s) by ten gives 356.8 s, unlike the previous fixed-cost assumption;
  600 s leaves about 243 s above that crude estimate. The 200-event warmup took
  18 s and lacks the full cache work, so it cannot predict cold 10k duration.
  The failed 300 s step supplies only a lower bound, not its bottleneck. A
  second timeout is accepted as a fail-closed outcome. Four possible 120 s
  stack samples plus stage receipts should locate it; preserve and diagnose
  the failure, never relabel it or widen event/cache/traffic acceptance bounds.
- ACCT-05: frozen-02-SHA256SUMS generated from exact reviewed Git blobs, compared
  with working and frozen source, and retained next to the job evidence.
- ACCT-06: mocked native_worker.main test covers trace selection, exclusive
  file creation/refusal to reuse, cancellation on success/exception, and 1k/
  profile requests remaining untraced. The actual stage writes and MPI behavior
  still require hardware evidence; no unit test claims to exercise MPI.
- ACCT-07: compact collector now retains stage receipts, per-rank trace flags,
  stack-file hashes/sizes and explicit instrumented-timing scope. --failure
  preserves an index, checkpoints and stack tails for failed captures with no
  success receipt; it cannot label them accepted.
- ACCT-08: full suite rerun on exact detached 9a1079aa7 tree, separately from
  the expanded current-tree suite. Validation-04 records both results/hashes.

Production executable sources stay byte-identical to the round-2 reviewed
commit. Changes after it are tests, documentation and coordinator evidence tools.
