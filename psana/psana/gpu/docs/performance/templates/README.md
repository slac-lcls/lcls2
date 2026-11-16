# New campaign record

Copy [manifest.json](manifest.json) into a new site/workload/date record and
provide a README. Replace placeholders, choose a safe stable campaign ID,
set kind and script roots, and add matching repository-relative catalog paths.
Follow [the common format](../FORMAT.md). Never copy historical approvals.

Select inputs from [current dataset locations](../datasets.md), not a copied
archive's scratch path. For published Sim10 inputs, record `exp=prjdatpsana`,
run 3 or 4, the explicit site-local XTC directory, the published checksums,
the stream subset, the Sim10 adapter, and the selected calibration identity.
Changing only a directory leaves the old experiment/run/filename assumptions
and source hashes invalid. Perlmutter availability must be verified separately.

- Scaling: specify topology matrix, repetitions and pairing; retain allocation boundaries.
- Regression: pin baseline/candidate, equal-work controls, acceptance/equivalence policy and inconclusive outcomes.
- Application: pin callback/application source, numerical policy, stage semantics and reduction/export work.
- Diagnostic: label interventions, instrumentation overhead and limits on throughput interpretation.

`artifact-only-v1` is structural validation, not numerical acceptance. Add and
test a dedicated analysis profile when the campaign requires arithmetic checks.
Freeze the exact runner/configuration per campaign. Use the
[PR regression template and agent instructions](pr-regression.md) for maintained
comparison orchestration. Its run-51 candidate passed the pixel-status check;
durable cross-site dataset qualification and a reviewed native adapter remain
required before execution.
