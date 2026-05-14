# ePixQuad1kfps per-pixel gain diagnosis

This note describes how to rerun the FM/FL per-pixel gainbit diagnosis.

The diagnostic compares the raw data gain bits against a binary raw-view gain
map. The map must have shape `(4, 352, 384)`, matching `det.raw.raw(evt)`.

- Map value `0`: expected fixed medium, FM
- Map value `1`: expected fixed low, FL
- Raw gain code: `top2 = (raw >> 14) & 0x3`
- For the current FM/FL tests, observed codes are `FM = 0` and `FL = 1`

## Environment

Run from the lcls2 checkout:

```bash
cd ~/lcls2
source ~/lcls2/setup_env.sh
```

## Generic command

Yes, the experiment, run numbers, and map are all configurable:

```bash
python psdaq/psdaq/debugtools/epixquad1kfps/diagnose_per_pixel_gain.py \
  --exp ued1015980 \
  --runs 2,3,4 \
  --map /path/to/raw_view_gainmap.npy \
  --csv /path/to/output_fp_fn.csv
```

Important options:

```text
--exp       Experiment name, for example ued1015980
--runs      Comma-separated run list, for example 2,3,4
--map       Binary raw-view .npy map with shape (4,352,384); 1=FL, 0=FM
--csv       Optional CSV output path for FP/FN coordinates
--events    Number of valid raw events to check per run; default is 50
--detector  Detector name; default is epixquad1kfps
--xtc-dir   Optional XTC directory passed to psana DataSource
--max-print Maximum FP/FN rows printed per run; default 0 means print all
--fm-code   Override expected FM top2 code; default infers from first event
--fl-code   Override expected FL top2 code; default infers from first event
```

## Existing wrappers

The two wrappers used for the initial diagnosis are:

```bash
bash psdaq/psdaq/debugtools/epixquad1kfps/run_per_pixel_gain_expand2.sh
bash psdaq/psdaq/debugtools/epixquad1kfps/run_per_pixel_gain_expand4.sh
```

They run:

```text
expand2 map -> ued1015980 runs 2,3,4
expand4 map -> ued1015980 runs 5,6,7
```

and write CSV files in:

```text
/sdf/home/m/monarin/tmp/epix-per-pixel-gainmode/roi_ft_ued1015999_r185_t300_v2/
```

## Output interpretation

For each run, the script reports:

- Top2 counts in the first valid event
- Whether the first `--events` gainbit images match the first image
- FP pixel count
- FN pixel count
- FP/FN pixel coordinates

Definitions:

```text
FP: data read back as FL, but the map expects FM
FN: data read back as FM, but the map expects FL
```

Coordinates are reported in ASIC-local bank coordinates:

```text
asic      ASIC index, 0..15
bank      Bank index inside ASIC, 0..3
bank_row  Row inside bank, 0..175
bank_col  Column inside bank, 0..47
```

One ASIC is `176 x 192` pixels. One bank is `176 x 48` pixels.

## Example: rerun expand4 with less terminal output

```bash
python psdaq/psdaq/debugtools/epixquad1kfps/diagnose_per_pixel_gain.py \
  --exp ued1015980 \
  --runs 5,6,7 \
  --map /sdf/home/m/monarin/tmp/epix-per-pixel-gainmode/roi_ft_ued1015999_r185_t300_v2/roiFromAboveThreshold_r185_c0_calib_expand4.npy \
  --csv /sdf/home/m/monarin/tmp/epix-per-pixel-gainmode/roi_ft_ued1015999_r185_t300_v2/expand4_gainbit_fp_fn.csv \
  --max-print 20
```

## Example: use another experiment, runs, and map

```bash
python psdaq/psdaq/debugtools/epixquad1kfps/diagnose_per_pixel_gain.py \
  --exp <experiment> \
  --runs <run1,run2,run3> \
  --map <binary_raw_view_gainmap.npy> \
  --csv <output_fp_fn.csv>
```
