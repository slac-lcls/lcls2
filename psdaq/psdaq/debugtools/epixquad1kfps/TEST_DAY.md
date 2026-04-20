# ePix Quad 1 kfps Detector-Day Instructions

This note is the operator-facing workflow for collecting and analyzing the
debug-pattern runs used to study the current deployed per-pixel gain-map write
path for `epixquad1kfps`.

The current plan is to keep the deployed write path unchanged and learn from the
results with known trust limits:

- `02_all_modules_anchor_asic0.json` is the best coarse placement test.
- `01_module0_asic_orientation.json` is the next best coarse-orientation test.
- `05_single_asic_row_regions.json` is useful for vertical addressing checks.
- `03_single_asic_quadrants.json` and `04_single_asic_column_regions.json`
  should be treated more cautiously because the current per-pixel write loop
  does not appear to reach bank 2 or 3.

## Files And Diagnoses

| Test JSON | Main Question | Primary Diagnosis |
|---|---|---|
| `tests/02_all_modules_anchor_asic0.json` | Which raw module / coarse ASIC block responds? | `module_location` |
| `tests/01_module0_asic_orientation.json` | After coarse placement, is the 2-point marker flipped or rotated? | `asic_orientation` |
| `tests/03_single_asic_quadrants.json` | Which quadrant inside the winning coarse block responds? | `quadrants` |
| `tests/04_single_asic_column_regions.json` | Which 48-column band responds? | `column_regions` |
| `tests/05_single_asic_row_regions.json` | Which row band responds? | `row_regions` |

## Workflow

The workflow has three layers:

1. Program and collect one DAQ run per pattern.
2. Extract one run at a time from raw data into reusable `.npy` and `.json` summaries.
3. Run one diagnosis at a time from the extracted summaries.

Keep extraction and diagnosis separate. Extraction is expensive because it reads
raw data. Diagnosis is cheap and can be rerun as many times as needed.

## Step 1: Confirm The Pattern Sequence

Dry-run the sequence before detector use:

```bash
cd ~/lcls2
source ./setup_env.sh >/dev/null
PYTHONPATH=psdaq python3 psdaq/psdaq/debugtools/epixquad1kfps/run_pattern_sequence.py \
  --sequence psdaq/psdaq/debugtools/epixquad1kfps/sequences/detector_day_v1.json \
  --dry-run
```

This prints the pattern order and confirms the run driver will attempt one run
per pattern.

## Step 2: Collect Data

For the full detector-day sequence:

```bash
cd ~/lcls2
source ./setup_env.sh >/dev/null
PYTHONPATH=psdaq python3 psdaq/psdaq/debugtools/epixquad1kfps/run_pattern_sequence.py \
  --sequence psdaq/psdaq/debugtools/epixquad1kfps/sequences/detector_day_v1.json \
  --daq-config psdaq/psdaq/cnf/ued.py \
  --duration 2.0 \
  --outdir /tmp/epixquad1kfps_patterns
```

Adjust `--daq-config` to the DAQ config file actually used on the day.

For a smaller first pass, use the priority shortlist:

```bash
cd ~/lcls2
source ./setup_env.sh >/dev/null
PYTHONPATH=psdaq python3 psdaq/psdaq/debugtools/epixquad1kfps/run_pattern_sequence.py \
  --sequence psdaq/psdaq/debugtools/epixquad1kfps/sequences/priority_shortlist.json \
  --daq-config psdaq/psdaq/cnf/ued.py \
  --duration 2.0 \
  --outdir /tmp/epixquad1kfps_patterns
```

## Step 3: Extract Raw Summaries

After the runs are collected, extract one run at a time into reusable run
directories.

Full extraction:

```bash
cd ~/lcls2
source ./setup_env.sh >/dev/null
PYTHONPATH=psdaq python3 psdaq/psdaq/debugtools/epixquad1kfps/validate_pattern_runs.py \
  --sequence psdaq/psdaq/debugtools/epixquad1kfps/sequences/detector_day_v1.json \
  --exp <experiment> \
  --run-start <first_run_number> \
  --detector epixquad1kfps \
  --events 50 \
  --outdir /tmp/epixquad1kfps_validate
```

Pass-0 bookkeeping only:

```bash
cd ~/lcls2
source ./setup_env.sh >/dev/null
PYTHONPATH=psdaq python3 psdaq/psdaq/debugtools/epixquad1kfps/validate_pattern_runs.py \
  --sequence psdaq/psdaq/debugtools/epixquad1kfps/sequences/detector_day_v1.json \
  --exp <experiment> \
  --run-start <first_run_number> \
  --outdir /tmp/epixquad1kfps_validate \
  --pass0-only
```

Outputs from extraction:

- `run_table.json`
- `run_table.csv`
- `run_summaries.json`
- one run directory per analyzed run:
  - `bit14_occupancy.npy`
  - `background_deviation.npy`
  - `dominant_code.npy`
  - `dominant_confidence.npy`
  - `summary.json`

## Step 4: Run Diagnoses In Order

Use the extracted output directory as the input to diagnosis commands.

### 1. `module_location`

This is the first diagnosis to run and the first result to trust.

Question:
- Which raw module `0..3` responds?
- Which coarse `(176,192)` block inside that module responds?

Command:

```bash
cd ~/lcls2
source ./setup_env.sh >/dev/null
PYTHONPATH=psdaq python3 psdaq/psdaq/debugtools/epixquad1kfps/diagnose_pattern_runs.py \
  --diag module_location \
  --input-dir /tmp/epixquad1kfps_validate
```

What to look for first:

- For `02_all_modules_anchor_asic0.json`, identify the winning raw module.
- Check whether the four logical anchors land in four distinct raw modules or
  collapse onto fewer modules.
- Use this as the main coarse placement result.

### 2. `asic_orientation`

This is the next diagnosis to run after coarse placement.

Question:
- After the best module and coarse block are identified, does the 2-point marker
  look like `identity`, `flipud`, `fliplr`, `rot180`, or `unresolved`?

Command:

```bash
cd ~/lcls2
source ./setup_env.sh >/dev/null
PYTHONPATH=psdaq python3 psdaq/psdaq/debugtools/epixquad1kfps/diagnose_pattern_runs.py \
  --diag asic_orientation \
  --input-dir /tmp/epixquad1kfps_validate
```

What to look for next:

- Use this mainly on `01_module0_asic_orientation.json`.
- Compare the four runs for ASIC 0, 1, 2, and 3.
- Ask whether top and bottom ASICs show different transforms.

### 3. `row_regions`

This is the next more trustworthy intra-ASIC diagnosis under the current
deployed write path.

Question:
- Which vertical row band responds inside the winning coarse block?

Command:

```bash
cd ~/lcls2
source ./setup_env.sh >/dev/null
PYTHONPATH=psdaq python3 psdaq/psdaq/debugtools/epixquad1kfps/diagnose_pattern_runs.py \
  --diag row_regions \
  --input-dir /tmp/epixquad1kfps_validate
```

What to look for:

- Use this mainly on `05_single_asic_row_regions.json`.
- Check whether `R0`, `R1`, `R2`, `R3` land in distinct vertical bands.
- This is more informative than the bank-sensitive diagnoses under the current
  write path.

### 4. `quadrants`

This is useful, but interpret it cautiously.

Question:
- Which quadrant-sized region inside the winning coarse block responds?

Command:

```bash
cd ~/lcls2
source ./setup_env.sh >/dev/null
PYTHONPATH=psdaq python3 psdaq/psdaq/debugtools/epixquad1kfps/diagnose_pattern_runs.py \
  --diag quadrants \
  --input-dir /tmp/epixquad1kfps_validate
```

What to look for:

- Use this mainly on `03_single_asic_quadrants.json`.
- Treat right-half results as exploratory rather than conclusive because the
  current write path appears unable to address banks 2 and 3 cleanly.

### 5. `column_regions`

This diagnosis is the most directly relevant to the bank-addressing question,
but it is also the least trustworthy under the current deployed path.

Question:
- Which 48-column band responds inside the winning coarse block?

Command:

```bash
cd ~/lcls2
source ./setup_env.sh >/dev/null
PYTHONPATH=psdaq python3 psdaq/psdaq/debugtools/epixquad1kfps/diagnose_pattern_runs.py \
  --diag column_regions \
  --input-dir /tmp/epixquad1kfps_validate
```

What to look for:

- Use this on `04_single_asic_column_regions.json`.
- Ask whether `C0`, `C1`, `C2`, `C3` map to four distinct 48-column bands.
- Under the current write path, treat any apparent aliasing of `C2`/`C3` onto
  `C0`/`C1` as suggestive, not definitive, because the current software may not
  be reaching all banks.

## Recommended Reading Order Tomorrow

For the current deployed path, read results in this order:

1. `module_location`
   Look first at anchors. This is the main coarse placement result.

2. `asic_orientation`
   Look next at `01_module0_asic_orientation.json` to understand coarse flips
   and rotations.

3. `row_regions`
   Use this for a safer vertical-address sanity check.

4. `quadrants`
   Use this as exploratory evidence only.

5. `column_regions`
   Use this as exploratory evidence for the bank-addressing question, knowing
   the current deployed path may already be biasing the outcome.

## Per-Run And Subset Examples

Diagnose only selected runs:

```bash
cd ~/lcls2
source ./setup_env.sh >/dev/null
PYTHONPATH=psdaq python3 psdaq/psdaq/debugtools/epixquad1kfps/diagnose_pattern_runs.py \
  --diag module_location \
  --run 49,50,51 \
  --input-dir /tmp/epixquad1kfps_validate
```

Diagnose only selected pattern indices:

```bash
cd ~/lcls2
source ./setup_env.sh >/dev/null
PYTHONPATH=psdaq python3 psdaq/psdaq/debugtools/epixquad1kfps/diagnose_pattern_runs.py \
  --diag asic_orientation \
  --patterns 4,5,6,7 \
  --input-dir /tmp/epixquad1kfps_validate
```

## Summary

The intended order is:

1. dry-run the sequence
2. collect one DAQ run per pattern
3. extract raw summaries once
4. diagnose multiple times, starting from coarse placement

Tomorrow, the key result to trust first is coarse module placement from the
anchor runs. Everything finer should be interpreted in light of the known
limitations of the current deployed per-pixel write path.
