# ePix Quad 1 kfps Detector-Day Instructions

This note is the operator-facing workflow for collecting and analyzing the
debug-pattern runs used to study the current deployed per-pixel gain-map write
path for `epixquad1kfps`.

The current plan is to keep the deployed write path unchanged and learn from the
results with known trust limits:

- `02_all_modules_anchor_asic0.json` is the best coarse placement test.
- `01_module0_asic_orientation.json` is the next best coarse-orientation test.
- After `02` and `01` are resolved, use a direct blind-bank probe on one ASIC.
- `05_single_asic_row_regions.json` is useful for vertical addressing checks.
- `03_single_asic_quadrants.json` and `04_single_asic_column_regions.json`
  should be treated more cautiously because the current per-pixel write loop
  does not appear to reach bank 2 or 3.

## Files And Diagnoses

| Test JSON | Main Question | Primary Diagnosis |
|---|---|---|
| `tests/02_all_modules_anchor_asic0.json` | Which raw module / coarse ASIC block responds? | `module_location` |
| `tests/01_module0_asic_orientation.json` | After coarse placement, is the 2-point marker flipped or rotated? | `asic_orientation` |
| `direct/01_blind_bank_probe.json` | If one bank is `178 x 48`, where do bank ids `0..3` land? | visual/raw check first |
| `direct/02_blind_bank_probe_rowband.json` | If one bank is `44 x 192`, where do bank ids `0..3` land? | visual/raw check first |
| `tests/03_single_asic_quadrants.json` | Which quadrant inside the winning coarse block responds? | `quadrants` |
| `tests/04_single_asic_column_regions.json` | Which 48-column band responds? | `column_regions` |
| `tests/05_single_asic_row_regions.json` | Which row band responds? | `row_regions` |

## Workflow

The workflow has three layers:

1. Program and collect one DAQ run per pattern.
2. Extract one run at a time from raw data into reusable `.npy` and `.json` summaries.
3. Run one diagnosis at a time from the extracted summaries.
4. If `02` and `01` converge cleanly, optionally run a direct blind-bank test on one ASIC.

Keep extraction and diagnosis separate. Extraction is expensive because it reads
raw data. Diagnosis is cheap and can be rerun as many times as needed.

## What Each Test Reveals

- `02_all_modules_anchor_asic0.json`
  - gives coarse anchor mapping:
  - logical anchored ASIC -> raw segment/module
  - logical anchored ASIC -> coarse raw ASIC block
  - this strongly constrains, but does not by itself fully prove, the full ASIC enumeration

- `01_module0_asic_orientation.json`
  - gives ASIC-local placement/orientation inside the winning raw block
  - practical outputs:
    - raw ASIC quadrant-like location
    - local transform such as `identity`, `flipud`, `fliplr`, `rot180`, or `unresolved`

- direct full-bank probe
  - gives bank landing geometry for one chosen ASIC
  - practical outputs:
    - whether bank ids `0..3` are all reachable
    - which bank-shape hypothesis fits better: `178x48` or `44x192`
    - whether the banks stack vertically, horizontally, or as row bands

- direct sparse in-bank markers
  - gives final in-bank pixel orientation for one chosen bank
  - practical outputs:
    - bank-local `identity`, `flipud`, `fliplr`, or `rot180`
    - the last missing piece needed to map:
      - `(logical asic, logical bank, logical row, logical col)`
      - to observed `det.raw.raw` coordinates

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

Recommended first-pass order on the detector:

1. Run the priority shortlist.
2. Read `module_location` from `02_all_modules_anchor_asic0.json`.
3. Read `asic_orientation` from `01_module0_asic_orientation.json`.
4. Pick one ASIC with the cleanest coarse location/orientation result.
5. Only then run a direct blind-bank probe on that ASIC.

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

## Step 3b: Optional Direct Blind-Bank Test

Use this only after `02_all_modules_anchor_asic0.json` and
`01_module0_asic_orientation.json` have given a usable coarse answer.

Purpose:

- bypass the current `mrow/mcol`-based write path entirely
- drive explicit `(asic, bank, row, col)` writes
- test bank-shape hypotheses directly on one chosen ASIC

Two direct-write hypotheses are prepared:

- `direct/01_blind_bank_probe.json`
  Interpretation: one logical bank is `178 x 48`, and the JSON `bank` field is
  the hardware bank id.
- `direct/02_blind_bank_probe_rowband.json`
  Interpretation: one logical bank is `44 x 192`, and the JSON `bank` field
  selects a 44-row band inside the ASIC.

Two sparse-marker follow-up files are also prepared:

- `direct/03_bank_markers_178x48.json`
  Two separated marker pixels inside one logical `178 x 48` bank.
- `direct/04_bank_markers_44x192.json`
  Two separated marker pixels inside one logical `44 x 192` bank.

In both files:

- pattern `0` probes logical bank `0`
- pattern `1` probes logical bank `1`
- pattern `2` probes logical bank `2`
- pattern `3` probes logical bank `3`

Run one pattern per DAQ run. The direct path is not yet wrapped by
`run_pattern_sequence.py`; set the environment explicitly before each run.

Example for the `178 x 48` bank hypothesis:

```bash
cd ~/lcls2
source ./setup_env.sh >/dev/null
export EPIXQUAD_DEBUG_DIRECT_WRITE_FILE=psdaq/psdaq/debugtools/epixquad1kfps/direct/01_blind_bank_probe.json
export EPIXQUAD_DEBUG_PATTERN_INDEX=0
```

Then do one normal Configure/BeginRun/EndRun cycle. Repeat with:

- `EPIXQUAD_DEBUG_PATTERN_INDEX=1`
- `EPIXQUAD_DEBUG_PATTERN_INDEX=2`
- `EPIXQUAD_DEBUG_PATTERN_INDEX=3`

Example for the `44 x 192` row-band hypothesis:

```bash
cd ~/lcls2
source ./setup_env.sh >/dev/null
export EPIXQUAD_DEBUG_DIRECT_WRITE_FILE=psdaq/psdaq/debugtools/epixquad1kfps/direct/02_blind_bank_probe_rowband.json
export EPIXQUAD_DEBUG_PATTERN_INDEX=0
```

Then repeat for pattern indices `1`, `2`, and `3`.

What to look for first in raw data:

- do the four runs land in four distinct regions?
- do they stay on the chosen ASIC or jump elsewhere?
- does the changed region look more like `178 x 48` or `44 x 192`?
- do all four bank ids appear reachable?

If one hypothesis gives four clean and distinct regions while the other does
not, use that as the working bank-shape model for the next round of testing.

## Step 3c: Optional Sparse In-Bank Marker Test

Use this only after the full-bank blind-bank test has identified:

- which coordinate hypothesis is more plausible, and
- which logical bank you want to resolve more precisely

Purpose:

- keep the chosen bank hypothesis fixed
- probe only one bank at a time with two separated pixels
- determine the in-bank orientation, not just the bank landing region

This is the step that turns:

- coarse bank landing

into:

- explicit bank-local `(row, col)` orientation inside raw data

Use these files:

- `direct/03_bank_markers_178x48.json`
- `direct/04_bank_markers_44x192.json`

Pattern selection rule:

- choose the sparse-marker file that matches the winning full-bank hypothesis
- choose the same pattern index as the logical bank you want to inspect

Examples:

- if `direct/01_blind_bank_probe.json` pattern `2` gave the cleanest or most
  informative bank landing, then use:
  - `direct/03_bank_markers_178x48.json`
  - `EPIXQUAD_DEBUG_PATTERN_INDEX=2`
- if `direct/02_blind_bank_probe_rowband.json` pattern `1` looked like the
  correct row-band interpretation, then use:
  - `direct/04_bank_markers_44x192.json`
  - `EPIXQUAD_DEBUG_PATTERN_INDEX=1`

Example for the `178 x 48` sparse-marker follow-up:

```bash
cd ~/lcls2
source ./setup_env.sh >/dev/null
export EPIXQUAD_DEBUG_DIRECT_WRITE_FILE=psdaq/psdaq/debugtools/epixquad1kfps/direct/03_bank_markers_178x48.json
export EPIXQUAD_DEBUG_PATTERN_INDEX=0
```

Example for the `44 x 192` sparse-marker follow-up:

```bash
cd ~/lcls2
source ./setup_env.sh >/dev/null
export EPIXQUAD_DEBUG_DIRECT_WRITE_FILE=psdaq/psdaq/debugtools/epixquad1kfps/direct/04_bank_markers_44x192.json
export EPIXQUAD_DEBUG_PATTERN_INDEX=0
```

What these patterns contain:

- one ASIC
- one logical bank
- two marker pixels
- the two pixels are intentionally separated in both row and col so that
  `identity`, `flipud`, `fliplr`, and `rot180` can be distinguished

What to look for:

- do both markers land inside the same raw bank region identified by the full-bank test?
- does the pair preserve the expected up/down ordering?
- does the pair preserve the expected left/right ordering?
- which transform best matches what you see:
  - `identity`
  - `flipud`
  - `fliplr`
  - `rot180`

What this test gives you:

- the final in-bank orientation for the chosen bank
- the last missing piece needed to map:
  - `(logical asic, logical bank, logical row, logical col)`
  - to observed `det.raw.raw` coordinates

## Result Table Template

Use this table to record the detector-day outcome after `02`, `01`, and the
blind-bank runs. The intent is to build up the practical mapping from logical
programming coordinates to observed `det.raw.raw` coordinates.

| Logical ASIC | `02` raw module | `02` coarse raw block | `01` orientation | Blind-bank mode | Bank 0 landing | Bank 1 landing | Bank 2 landing | Bank 3 landing | Working conclusion |
|---|---|---|---|---|---|---|---|---|---|
| `0` |  |  |  |  |  |  |  |  |  |
| `1` |  |  |  |  |  |  |  |  |  |
| `2` |  |  |  |  |  |  |  |  |  |
| `3` |  |  |  |  |  |  |  |  |  |

Suggested values:

- `02 raw module`
  - one of `0`, `1`, `2`, `3`
- `02 coarse raw block`
  - e.g. `upper-left`, `upper-right`, `lower-left`, `lower-right`
- `01 orientation`
  - one of `identity`, `flipud`, `fliplr`, `rot180`, `unresolved`
- `Blind-bank mode`
  - one of `178x48`, `44x192`, `both`, `neither`
- `Bank N landing`
  - short description of the observed region in raw data
  - e.g. `top stripe`, `left narrow band`, `same as bank1`, `not seen`
- `Working conclusion`
  - short statement of the current best mapping for that ASIC

Use this companion table when the chosen blind-bank hypothesis is clear and you
want to record the implied low-level mapping rule.

| Chosen ASIC | Chosen bank mode | Logical bank meaning | Raw region pattern | Still missing |
|---|---|---|---|---|
|  |  |  |  |  |

Examples of `Still missing`:

- `need sparse in-bank orientation`
- `need to confirm bank 2/3 reachability`
- `need second ASIC cross-check`

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

3. direct blind-bank test
   If `02` and `01` are clean enough, test one ASIC with
   `direct/01_blind_bank_probe.json` and/or
   `direct/02_blind_bank_probe_rowband.json`.

4. `row_regions`
   Use this for a safer vertical-address sanity check.

5. `quadrants`
   Use this as exploratory evidence only.

6. `column_regions`
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
