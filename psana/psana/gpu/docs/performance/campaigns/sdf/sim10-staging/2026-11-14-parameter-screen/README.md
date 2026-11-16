# SDF Sim10 staging parameter screen

The SDF series covers seven allocations, 48 configurations and 102 full 100k-event staging runs, plus 48 pilots and six separate anchors. Compute nodes were sdfampere017/018/023/036/040; the coordinating login host is not the measured hardware.

The practical staging profile is four BDs sharing one A100 40 GB GPU, four workers/BD, batch 200, one CUDA stream, 1 MiB tasks, 16 MiB bounce, 512 KiB bulk and a 6 GiB input budget/BD. It measured 5.1150 GiB/s in worker job 44770441 and 4.7839 GiB/s in storage job 45028778. These are separate allocation means.

| Storage | GPUs | n | Mean GiB/s | Observed range GiB/s | Mean paired speedup |
|---|---:|---:|---:|---:|---:|
| Weka | 1 | 3 | 4.7839 | 4.5144–5.2989 | 1.0000× |
| Weka | 2 | 3 | 4.6073 | 4.4985–4.6908 | 0.9675× |
| Weka | 4 | 3 | 4.8049 | 4.5663–5.2013 | 1.0115× |
| NVMe | 1 | 3 | 4.6032 | 4.5696–4.6483 | 1.0000× |
| NVMe | 2 | 3 | 4.5973 | 4.4992–4.6835 | 0.9988× |
| NVMe | 4 | 3 | 4.6882 | 4.6593–4.7345 | 1.0185× |

Each cell has three full runs from job 45028778. Speedup is the mean of same-medium, same-repeat ratios against one GPU, not a ratio of the displayed means. Ranges are observed extrema, not confidence intervals. The six separate Weka anchors span 1,451.83–1,667.02 events/s; small differences are not a stable ranking.

This comparison held **four BDs total** fixed while changing GPUs. The Perlmutter scaling campaign instead adds four BDs for every added GPU. Do not interpret their differing curves as a site comparison.

The original report records automated acceptance for all cohorts, independent result review for the first four, and pending result review for the final three. Its heatmap synthesis was not independently reviewed.

[Four-page heatmap PDF](../../../../../../notes/archives/scaling-20261116/sources/sdf-20261115/reference/notes/sim10_parameter_report_20261114/artifacts/heatmaps.pdf) · [Throughput heatmap](../../../../../../notes/archives/scaling-20261116/sources/sdf-20261115/reference/notes/sim10_parameter_report_20261114/artifacts/throughput.svg) · [Matched controls](../../../../../../notes/archives/scaling-20261116/sources/sdf-20261115/reference/notes/sim10_parameter_report_20261114/artifacts/relative_change.svg) · [Repeat spread](../../../../../../notes/archives/scaling-20261116/sources/sdf-20261115/reference/notes/sim10_parameter_report_20261114/artifacts/repeat_spread.svg) · [Resources](../../../../../../notes/archives/scaling-20261116/sources/sdf-20261115/reference/notes/sim10_parameter_report_20261114/artifacts/resources.svg)

All 156 normalized case rows and 48 summary cells are archived. The transfer lacks the original first-four-cohort and task-size source CSVs and some linked historical reports. The archive replay tool regenerates the heatmaps from the retained normalized rows; it does not repeat the original raw-event audit. Original report links outside the transferred subset may require the SDF checkout.

Evidence and scripts: [original report](../../../../../../notes/archives/scaling-20261116/sources/sdf-20261115/reference/notes/sim10_parameter_report_20261114/REPORT.md) · [measurements](../../../../../../notes/archives/scaling-20261116/sources/sdf-20261115/reference/notes/sim10_parameter_report_20261114/artifacts/runs.csv) · [summary](../../../../../../notes/archives/scaling-20261116/sources/sdf-20261115/reference/notes/sim10_parameter_report_20261114/artifacts/summary.csv) · [figure](../../../../../../notes/archives/scaling-20261116/sources/sdf-20261115/reference/notes/sim10_parameter_report_20261114/artifacts/heatmaps.pdf) · [provenance](../../../../../../notes/archives/scaling-20261116/sources/sdf-20261115/reference/notes/sim10_parameter_report_20261114/artifacts/provenance.json) · [validation](../../../../../../notes/archives/scaling-20261116/sources/sdf-20261115/reference/notes/sim10_parameter_report_20261114/validation.json) · [report script](../../../../../../notes/archives/scaling-20261116/sources/sdf-20261115/reference/notes/sim10_parameter_report_20261114/build_report.py) · [report tests](../../../../../../notes/archives/scaling-20261116/sources/sdf-20261115/reference/notes/sim10_parameter_report_20261114/test_report.py)

[Machine-readable record](manifest.json). Paths inside the record are relative to the repository root.
