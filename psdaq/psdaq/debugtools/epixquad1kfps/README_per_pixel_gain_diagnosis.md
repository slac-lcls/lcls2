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
--no-value-analysis
            Disable raw14 mean/RMS extraction for FP/FN pixels
--min-ref-pixels
            Minimum agreed FM/FL reference pixels for local noise classification
--no-pedestal-analysis
            Disable FM/FL pedestal residual classification
--fm-ped-index
            Pedestal gain index for FM; default is 1
--fl-ped-index
            Pedestal gain index for FL; default is 2
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
- Raw14 first value, temporal mean, and temporal RMS for FP/FN pixels
- A dark-noise-based `noise_like_gain` estimate for FP/FN pixels
- An FM/FL pedestal-residual-based `pedestal_like_gain` estimate for FP/FN pixels

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

## Value/noise analysis

By default, the diagnostic also extracts raw ADC values with the gain bits
masked off:

```python
raw14 = raw & 0x3FFF
```

For each FP/FN pixel it reports:

```text
raw14_first       Raw14 value in the first valid image
raw14_mean        Mean raw14 value over the inspected images
raw14_rms         Temporal RMS of raw14 over the inspected images
noise_like_gain   FM or FL, based on which agreed reference population has
                  the more similar raw14_rms
```

The reference populations are pixels where the map and data readback agree:

```text
FM reference: map expects FM and top2 reads FM
FL reference: map expects FL and top2 reads FL
```

The script first tries to use reference pixels from the same ASIC and bank as
the mismatch pixel. If there are not enough agreed FM and FL reference pixels,
it falls back to detector-wide references.

This is a dark/noise diagnostic only. In the current ued1015980 data, direct
raw14 values and raw14 mean/RMS did not show a strong or unambiguous FM/FL
separation. FM and FL can have similar dark baselines because the gain mode
mainly changes response slope to deposited charge, not necessarily the
zero-signal ADC value. Charge injection is still the cleaner test for actual
gain response.

## Pedestal residual analysis

By default, the diagnostic also loads pedestal constants from:

```python
det.raw._pedestals()
```

For ePix10ka/ePixQuad, the gain index order is:

```text
0: FH
1: FM
2: FL
3: AHL_H
4: AML_M
5: AHL_L
6: AML_L
```

The script therefore uses pedestal index `1` for FM and index `2` for FL by
default. These can be overridden with `--fm-ped-index` and `--fl-ped-index`.

For each FP/FN pixel, it compares the raw14 values against both pedestal
hypotheses:

```text
fm_residual = raw14 - pedestal_FM[pixel]
fl_residual = raw14 - pedestal_FL[pixel]
```

For the inspected images, the pedestal scores are:

```text
pedestal_score_fm = median(abs(raw14 - pedestal_FM[pixel]))
pedestal_score_fl = median(abs(raw14 - pedestal_FL[pixel]))
```

The lower score is the closer pedestal match. A residual closer to zero means
the pixel is more consistent with that pedestal mode. A large separation between
the FM and FL scores is more meaningful than a small separation.

It writes these fields to the CSV:

```text
ped_fm                  FM pedestal value for that exact pixel
ped_fl                  FL pedestal value for that exact pixel
fm_ped_resid_median     median(raw14 - pedestal_FM[pixel])
fl_ped_resid_median     median(raw14 - pedestal_FL[pixel])
fm_ped_abs_resid_median median(abs(raw14 - pedestal_FM[pixel]))
fl_ped_abs_resid_median median(abs(raw14 - pedestal_FL[pixel]))
pedestal_like_gain      FM if pedestal_score_fm <= pedestal_score_fl, else FL
pedestal_score_fm       same as fm_ped_abs_resid_median
pedestal_score_fl       same as fl_ped_abs_resid_median
```

`pedestal_like_gain` is whichever pedestal gives the smaller median absolute
residual. This is usually a better dark-data test than comparing raw means
directly because it accounts for per-pixel FM/FL pedestal differences.

For the current datasets, the pedestal residual analysis showed:

```text
Dataset   Runs     Kind   Pixels/run   Pedestal-like/run   Median FM score   Median FL score   Median margin   Interpretation
--------  -------  -----  -----------  ------------------  ---------------   ---------------   -------------   ----------------
Expand2   2,3,4    FP     43           40 FM / 3 FL        22.567            54.121            32.683          Strongly FM-like
Expand2   2,3,4    FN     68           0 FM / 68 FL        63.986            63.279            1.084           Weakly FL-like
Expand4   5,6,7    FP     29           27 FM / 2 FL        21.476            52.729            28.278          Strongly FM-like
Expand4   5,6,7    FN     103          1 FM / 102 FL       62.451            61.779            1.132           Weakly FL-like
```

The FP pixels are the clearest case: gainbit reads FL, but the pedestal
residuals are mostly FM-like with a sizable score separation. FN pixels lean
FL-like, but the FM/FL pedestal scores are much closer, so dark data is less
conclusive for FN.

## Visual pedestal-subtracted check

`view_raw_modes.py` can display pedestal-subtracted images and overlay the FP/FN
pixels from a diagnosis CSV.

Example for expand4 run 5:

```bash
source ~/lcls2/setup_env.sh

python ~/lcls2/psdaq/psdaq/debugtools/epixquad1kfps/view_raw_modes.py \
  -e ued1015980 \
  -r 5 \
  --evt-idx 50 \
  --view-raw \
  --show-ped-sub \
  --mask-npy /sdf/home/m/monarin/tmp/epix-per-pixel-gainmode/roi_ft_ued1015999_r185_t300_v2/roiFromAboveThreshold_r185_c0_calib_expand4.npy \
  --fp-fn-csv /sdf/home/m/monarin/tmp/epix-per-pixel-gainmode/per_pixel_gain_diag/20260514/output_fp_fn_ued1015980_run_5_6_7_with_pedestal.csv
```

Useful viewer options:

```text
--show-ped-sub  Add signed raw14 - ped FM and raw14 - ped FL panels
--fm-ped-index  Pedestal index for FM subtraction; default is 1
--fl-ped-index  Pedestal index for FL subtraction; default is 2
--fp-fn-csv     Diagnosis CSV whose FP/FN pixels should be overlaid
```

The CSV may contain multiple runs. The viewer filters the overlay using the run
selected by `-r/--run` and prints the loaded FP/FN counts for that run.

The overlay colors are:

```text
FP: red open squares
FN: cyan open squares
```

The pedestal-subtracted image panels are signed residuals, not absolute values:

```text
Raw14 - ped FM[1]
Raw14 - ped FL[2]
```

For visual interpretation, closeness to zero is the key quantity. For an FP
pixel, the map expects FM but gainbit reads FL; if it is actually FM-like, it
should be closer to zero in the `Raw14 - ped FM` panel than in the
`Raw14 - ped FL` panel. For an FN pixel, the analogous FL-like behavior would
be closer to zero in the `Raw14 - ped FL` panel.

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
