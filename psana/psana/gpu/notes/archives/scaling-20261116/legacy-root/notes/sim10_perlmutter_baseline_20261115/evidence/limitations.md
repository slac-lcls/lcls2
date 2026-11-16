# Validation boundaries

Initial unactivated pytest invocation: 27 passed, 15 import errors and one failure;
psana native modules were absent from PYTHONPATH. Repeated under clean_exec/runtime.sh.
First native run: 53 passed; immediate in-place file mutation did not change Lustre
stat identity. The fixture now explicitly changes mtime by two seconds so the
metadata-identity rejection test is deterministic; production reference code unchanged.
Native run after fixture repair: 54 passed, 64 subtests. Final native run after
adding independent-audit corruption tests and timeouts: 63 passed, 64 subtests.
PR298 source and install unchanged.
Neither stat identity nor a historical full checksum proves absence of a later
same-size same-timestamp content edit. No current full payload rehash is claimed.

Ordinary Codex sandbox execution stopped starting processes (ENOENT), while
read-only host diagnostics found both shell and workspace. Scoped host execution
was approved for the task checks; this is not a successful sandbox smoke test.
The scratch-workspaces skill mentioned by the external handoff was not found in
the available Codex/codex-skills/Claude catalogs. The available psana-perlmutter
scratch guidance was used; no substitute skill or hidden approval was invented.

GPU acceptance, full reference completion, actual batch200 memory fit, Slurm
execution, new performance measurements and result review remain pending.

Round-1 repairs: adapter-port.diff records the complete vendor adapter edit:
redirect its calibration source to the reused PR298 tree and require that exact
source SHA before execution. The numerical algorithm is unchanged. The guard
loader now also extracts the original CPU expand() oracle from an unchanged,
provenance-recorded verify.py; the earlier missing-global error is fixed.
The calibration manifest is pinned in the bundle; load_constants hashes every
mapped constant against that pinned manifest before use.

Round 2 passed prelaunch review at 296d69985ba2cfa14e4759b035c886d0c658465f.
The low-priority PM-R2-01 provenance suggestion is addressed by runner-port.diff,
the complete original-package-to-reviewed-runner diff. The exact-commit clean
host-checks06 receipt is now saved here: 90 tests and 64 subtests passed.
No code changed after that reviewed commit; subsequent changes are evidence.


## GPU shape/content revision (bootstrap02)

The preceding entries describe historical bootstrap01 preparation. Job61373306
subsequently passed native device tests/guards but timed out its first ECC step;
see job61373306-preliminary.json. No staging repeat completed. Exact stalled
export/reference operation remains unproven. User now excludes cube export and
CPU cube reference. Current code checks all pixels for finiteness and bin
occupancy on each GPU, with exact dimensions and counts. This cannot prove full
numerical correctness or produce a merged global cube. CPU reference/export
unit tests remain only historical component regressions, not runtime acceptance.
Revised GPU verification and batch200 full-shape execution remain unexecuted.
Host checks gpu-check-host02 passed107 tests/64subtests under the qualified
runtime; follow-up exact-commit evidence is recorded separately for review.
The regenerated runner-port.diff includes the complete no-export revision.
