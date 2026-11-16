# Test scripts and report reproduction

The results branch preserves reviewed experiment scripts under frozen archive roots. Use the source commit and runtime from each campaign manifest, not the branch HEAD, when reproducing an experiment. Creating a new bundle and launching it are separate steps.

| Experiment | Maintained or archived scripts | Purpose |
|---|---|---|
| SDF staging and compute sources | [SDF reference snapshot](../../../notes/archives/scaling-20261116/sources/sdf-20261115/reference/notes) and [later task documents](../../../notes/archives/scaling-20261116/sources/sdf-20261115/latest/notes) | Exact transfer subset, including storage/bounce campaigns, tests and compute preparation |
| SDF heatmaps | [Original builder](../../../notes/archives/scaling-20261116/sources/sdf-20261115/reference/notes/sim10_parameter_report_20261114/build_report.py), [archive replay](replay_sdf_report.py) | Replay retained normalized measurements; the original builder needs additional SDF cohort files |
| C++ stripe screen | [Reader and harness](../../../notes/archives/scaling-20261116/legacy-root/notes/sim10_striping_20261110/reader) | Build, prepare, Slurm launch, raw audit and plotting |
| psana stripe screen | [Harness](../../../notes/archives/scaling-20261116/legacy-root/notes/psana_striping_pr298_20261111/screen-20261115) | Six stripe counts, three blocks, cache and memory checks |
| psana GPU scaling | [Harness](../../../notes/archives/scaling-20261116/legacy-root/notes/psana_gpu_scaling_20261116) | Qualification, 100k screen, NIC/memory gates, audit and paired report |
| Sim10/XPP-cube | [Committed snapshot](../../../notes/archives/scaling-20261116/legacy-root/notes/sim10_perlmutter_baseline_20261115) | Prepared S/E/EC/ECC runner, pixel/bin validation and bounded bootstrap; full results pending |

The original scripts retain environment and scratch paths for their sites. Several preparation scripts pin repository prefixes. They are historical execution sources, not generic portable launchers. Native libraries, data and calibration inputs remain external and are recorded in their provenance files. Historical approvals in imported records do not authorize new submissions.

## Validate the catalog

From the repository root with Python 3.9 or newer:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 psana/psana/gpu/scripts/performance/analysis/check_catalog.py
```

This verifies imported and referenced file hashes, authored links, the complete case/summary populations, site/workload taxonomy, bandwidth arithmetic, authored table values and paired scaling ratios. It reads only archived files and does not query Slurm or launch jobs.

Focused offline contract checks exercise dropped/duplicate cells, incorrect metadata and report values, broken links, and per-cohort population errors:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 psana/psana/gpu/scripts/performance/analysis/test_archive_checks.py -v
```

These checks also need NumPy/Matplotlib for importing the original SDF report module; use a writable shared-scratch `MPLCONFIGDIR`.

## Replay the SDF heatmaps

Use Python with NumPy and Matplotlib installed. On Perlmutter the existing Python 3.11 environment is:
`/pscratch/sd/m/monarin/psana2-gpu/perlmutter-kvikio2610-20261015/attempt01/env/bin/python`.
Choose a new directory on shared scratch for both the result and Matplotlib cache:

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR="$SCRATCH/psana-scaling-mpl-cache" \
  python3 psana/psana/gpu/scripts/performance/analysis/replay_sdf_report.py \
  --output "$SCRATCH/psana-scaling-sdf-replay-NEW" --render
```

The replay checks all 48 cells and all recorded control comparisons against the archived outputs before rendering. It refuses an existing output directory. The original report/figures remain byte-for-byte unchanged. Re-rendered PDFs need not have identical bytes across plotting-library versions.

## Re-audit the Perlmutter scaling run

The archived catalog check is separate from the full native audit. On Perlmutter, with the original raw attempt, data views, runtime and terminal accounting still present, run the exact frozen script:

```bash
PYTHONDONTWRITEBYTECODE=1 /pscratch/sd/m/monarin/psana2-gpu/perlmutter-kvikio2610-20261015/attempt01/env/bin/python \
  /pscratch/sd/m/monarin/psana2-gpu/pr298-striping-20261111/scale-screen01/bundle/report.py \
  /pscratch/sd/m/monarin/psana2-gpu/pr298-striping-20261111/scale-screen01 \
  a493c9cacc29b922e48a82f79d1fa63ed548810b
```

This verifies native receipts and regenerates the attempt's `scaling-summary.json`. It requires the original external evidence and writes to the scratch attempt; it does not submit a job. C++ and stripe-screen reports link their own raw-audit entry points and provenance.
