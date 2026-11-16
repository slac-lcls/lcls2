# Report data and metric definitions

`artifacts/runs.csv` normalizes the four existing source CSVs without merging
allocation cohorts. A row is a recorded pilot, full measurement or anchor.
Only `phase=full` enters the 48-cell summary and heatmaps.

The complete configuration key is `(cohort, media, gpus, bds, workers, streams,
batch, task_mib, bounce_mib)`. The input budget, bulk target and runtime are
constant in this report. Cohort is the Slurm job, not merely the node model.

`repeat` preserves the source value when present; **0 means not provided by
the source CSV**, not a zeroth measured repeat. The task/bounce CSVs do not
provide it. `sequence` uses the source sequence/order when supplied, otherwise
the row's zero-based position in its source CSV. Source `case` and raw path/hash
are retained. No task/bounce paired-repeat claim depends on inferred repeat IDs.

Where a source CSV omits a fixed control (such as one GPU or 16 MiB bounce),
the collector fills it from that campaign's documented settings. Host names
come from the recorded cohort inventory. Resource estimates are derived from
controls, not additional measurements.

- `hz = events / seconds`; `gib_s = payload_bytes / 2**30 / seconds`.
- Every full case has 100,000 global events and 335,820,000,000 payload bytes.
- `mean_hz` and `mean_gib_s` are arithmetic means of individual full-run rates.
- `spread_pct = 100 * (max_hz - min_hz) / mean_hz`; not a confidence interval.
- `events_per_bound_core = mean_hz / bound_physical_cores`, where bound cores
  are `6 + bds * (workers + 3)`. This is not measured utilization or billed cost.
- `gib_s_per_active_gpu = mean_gib_s / gpus`. The storage/GPU allocation reserves
  four GPUs in every cell, regardless of how many are active.
- `bounce_estimate_mib = bds * workers * bounce_mib` assumes one worker buffer;
  it does not measure pinned allocation or total host-memory consumption.
- Relative changes use the control explicitly recorded in `comparisons.json`.
  The first six cohorts use ratios of mean rates. The storage/GPU cohort uses
  the arithmetic mean of same-medium, same-repeat ratios to its one-GPU control.
- Physical traffic ratios are preserved as evidence of the data path, not
  proof of saturation. Small negative idle-adjusted NIC residuals are allowed.
- Missing cells stay missing; no interpolation, zero filling or imputation.

The fixed-total-worker panel repeats two worker-grid cells for comparison and
adds the two-BD point. There are still 48 distinct configurations, not 50.
The third repetition in the storage/GPU campaign does not make its variability
directly comparable to a two-repeat campaign with equal statistical precision.

The source provenance manifest hashes the original compact inputs. Raw-case
hashes are copied from those inputs and are not revalidated against all large
scratch JSONs by this report builder. Separate `validation.json` records the
report source hashes, calculation tests and figure checks.
