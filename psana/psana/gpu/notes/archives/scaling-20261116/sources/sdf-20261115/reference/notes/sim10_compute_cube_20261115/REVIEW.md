# Four-stage submission package: independent review complete

**Prepared, not submitted.** Claude's independent second review passed with no
blocking findings on immutable attempt02. This is implementation/prelaunch
acceptance, not CUDA/MPI/full-shape acceptance or a completed performance run.

## Reviewed identity

- Manifest SHA256: `93d5fdcb2b814db9f938aae75b6bf2b47975550e391d38ce740f25db0f3d6c9c`.
- Snapshot: `/sdf/scratch/users/m/monarin/gpu-validation/sim10-compute-cube-20261115/attempt02`.
- Runtime PR298: `67c327a148a08b2aaf972e67e1e02e367c2b298f`.
- Raw verdict: `review-02.json`, SHA256 `6b71ce0cfffc531784eb7714db076a2aa28cda21d55dc469b1650af9674716b9`.
- Receipt used by the launcher: `submission-review.json`.
- No commits or branches created. No core code or frozen native build changed.

Round 1 found a blocking bound-method/integer mistake in admission. The fix
calls the real pinned budget methods and is covered by both method-call tests
and StageTask.prepare with the real budget and a host CUDA mock. Re-review
closed that blocker and the weakref/monitoring/sharing-evidence findings.
Sampled references now read native-verified seed words instead of full payloads;
layout checks remain mandatory. Three actual native events matched bitwise.

## Validation

`checks-02.json` records 65 passing tests, eight host native/setup checks,
14 successful Python AST parses, shell syntax and diff checks. The initial
plain-venv run failed to import native psana; the final run explicitly activates
the pinned source/native overlay. No failing invocation is counted as a pass.
CuPy 14.0.1 exposes ndarray weakref metadata on the host; the actual device
allocation/release probe remains mandatory in every BD setup.

The reviewer explicitly accepts the inherited native compatibility strategy
for preparation only. Its `psana/meson.build` mismatch remains documented.
The allocation must pass the historical GPU regressions, descriptor guards,
full-pixel bitwise acceptance, physical free-memory admission and pilot gates
before the full comparison. No GPU fit or processing rate is yet measured.

## Nonblocking advisories retained

- **IMPL-R2-001 / residual IMPL-R1-002:** sampled references still perform many
  small reads in bin-strided order. The three-event warm parity check is not a
  cold-Weka timing measurement. The 600-second full reference component and
  1,980-second whole-full-case limit are unmeasured operational estimates.
- **IMPL-R2-002:** each ECC case scans about 500,000 record headers at setup;
  the 120-second setup component is also unmeasured. The whole-case limit,
  not separate component timers, is what the launcher enforces.

These are explicitly deferred performance/completion risks, not waived
correctness checks. The requested 24-hour candidate and reviewed budget are
unchanged. Before authorizing a costly run, consider a bounded cold-Weka
reference/index timing diagnostic or a larger justified budget and re-review.
The candidate can stop incomplete on timeout; it does not shorten counts,
skip bins, claim success, or switch storage to rescue the comparison.

## Submission handoff

`submission-command.txt` contains the validated command for this exact frozen
package. It is recorded only, not executed. The job requests one exclusive
four-A100 sdfampere036 allocation, 192 GiB host memory and 24 hours; all stage,
topology, pilot and repeat cases run sequentially on that same node.
The latest read-only availability is in `availability-latest.json`; no node
was reserved. Runtime checks revalidate the allocation and local capacity.
Do not modify the frozen snapshot or substitute a different node, budget or
runtime without preparing and reviewing a new manifest.
