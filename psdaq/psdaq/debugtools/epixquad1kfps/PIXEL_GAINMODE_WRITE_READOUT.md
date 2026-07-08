# ePixQuad1kfps Pixel Gain-Mode Write And Readout Mapping

This note summarizes the gain-mode settings used by
`write_gain_mode_standalone.py` and how to interpret the top two raw ADC bits
reported by `read_xpmmini_rogue_file.py`.

## Terminology

There are three related values that are easy to mix up:

| Name | Where it lives | Meaning |
|---|---|---|
| ASIC pixel value | ePix10ka ASIC pixel config map | Per-pixel mode code written with `WriteMatrixData` or `WritePixelData` |
| `trbit` | ePix10ka ASIC register | Selects high vs medium branch when pixel value is `0xc` or auto branch when pixel value is `0x0` |
| raw gain bit | data word bit 14 | Readout bit that separates low-gain pixels from non-low pixels in these tests |

The readout script reports:

```python
top2 = (raw >> 14) & 0x3
```

With the current default decoder mask, `--bit-mask 0x7fff`, bit 15 is masked
off.  Therefore the diagnostic `top2` values are effectively:

| Raw bit 14 | Reported `top2` |
|---:|---:|
| 0 | 0 |
| 1 | 1 |

## ASIC Configuration Mapping

This follows `epixquad1kfps_config.py` and the standalone writer.

| Config mode | ASIC pixel value | `trbit` | Expected raw bit 14 / `top2` | Effective gain |
|---|---:|---:|---:|---|
| `FH` | `0xc` | `1` | `0` | Fixed high |
| `FM` | `0xc` | `0` | `0` | Fixed medium |
| `FL` | `0x8` | ignored, writer uses `0` | `1` | Fixed low |
| `AHL`, high branch | `0x0` | `1` | `0` | Auto high |
| `AHL`, low branch | `0x0` | `1` | `1` | Auto low |
| `AML`, medium branch | `0x0` | `0` | `0` | Auto medium |
| `AML`, low branch | `0x0` | `0` | `1` | Auto low |

Important implication:

- raw `top2 = 0` does not distinguish `FH` from `FM`
- `FH` vs `FM` is determined by `trbit` when the pixel value is `0xc`
- raw `top2 = 1` means the readout is identifying that pixel as low gain

## Standalone Writer Modes

| Writer mode | Background ASIC pixel value | Background `trbit` | Selected ASIC pixel value | Expected `top2` image |
|---|---:|---:|---:|---|
| `FH` | `0xc` | `1` | none | all or nearly all `0` |
| `FM` | `0xc` | `0` | none | all or nearly all `0` |
| `FL` | `0x8` | `0` | none | all or nearly all `1` |
| `MapFML` | `0xc` | `0` | `0x8` | background `0`, selected FL pixels `1` |
| `MapFHL` | `0xc` | `1` | `0x8` | background `0`, selected FL pixels `1` |

For `MapFML` and `MapFHL`, selected pixels are written after the background
matrix write.  The selected pixels are the only pixels expected to read back
with `top2 = 1`, aside from small detector/readout artifacts.

## Control Rows

The DAQ writes four raw segments plus a separate segment `aux` array.  Rogue
StreamWriter `.dat` files contain the same camera payload plus one or more
record headers before the image words.  The `.dat` helper decodes that payload
directly into the DAQ raw segment order before checking gainbits.

| View | Shape | Contents |
|---|---:|---|
| Rogue/ePixViewer decoded image | `(712, 768)` | Display-oriented image, including ePixViewer-only rows |
| psana raw tiled view | `(704, 768)` | Tiled view of the four DAQ raw segments |
| psana raw detector view | `(4, 352, 384)` | Four DAQ raw segments; this is `det.raw.raw(evt)` |
| psana segment `aux` | `(4, 384)` per segment | Control/calibration rows |

The image extraction used by `read_xpmmini_rogue_file.py` mirrors the row and
column reversal in `psdaq/drp/EpixQuad.cc`.  On the rdsrv421 StreamWriter data,
`1095248` byte image records start after a 20-word header and `1095288` byte
image records start after a 40-word header:

```python
for i in range(176):
    dn_row = 176 + i
    up_row = 176 - i - 1
    for segment, row in (
        (2, up_row), (3, up_row), (2, dn_row), (3, dn_row),
        (0, up_row), (1, up_row), (0, dn_row), (1, dn_row),
    ):
        raw[segment, row] = words[index:index + 384][::-1]
        index += 384
```

The ePixViewer-only decoded rows are `0`, `1`, `354`, `355`, `356`, `357`,
`710`, and `711`.  These rows are not part of the DAQ raw image and are not
counted as configured pixels by the current FP/FN checks.

For gain-map validation, compare expected maps in DAQ raw shape
`(4,352,384)`.  The reader also accepts detector-view tiled `(704,768)` maps
and ePixViewer decoded `(712,768)` maps, but converts them to DAQ raw before
the comparison.

## Write Commands

From the worktree:

```bash
cd /sdf/home/m/monarin/lcls2_worktree/ued-epix10ka-rdsrv421-daq
source setup_env.sh
```

Preview the write plan without touching hardware:

```bash
python psdaq/psdaq/debugtools/epixquad1kfps/write_gain_mode_standalone.py \
  --mode MapFML \
  --dry-run
```

Write full fixed modes:

```bash
python psdaq/psdaq/debugtools/epixquad1kfps/write_gain_mode_standalone.py --mode FH
python psdaq/psdaq/debugtools/epixquad1kfps/write_gain_mode_standalone.py --mode FM
python psdaq/psdaq/debugtools/epixquad1kfps/write_gain_mode_standalone.py --mode FL
```

Write sparse map modes:

```bash
python psdaq/psdaq/debugtools/epixquad1kfps/write_gain_mode_standalone.py --mode MapFML
python psdaq/psdaq/debugtools/epixquad1kfps/write_gain_mode_standalone.py --mode MapFHL
```

Write custom selected pixels in ASIC-local coordinates:

```bash
python psdaq/psdaq/debugtools/epixquad1kfps/write_gain_mode_standalone.py \
  --mode MapFML \
  --no-default-pixels \
  --pixel 0,12,7 \
  --pixel 0,12,55
```

Write custom selected pixels in raw-view coordinates:

```bash
python psdaq/psdaq/debugtools/epixquad1kfps/write_gain_mode_standalone.py \
  --mode MapFHL \
  --no-default-pixels \
  --raw-pixel 0,188,199
```

Optionally save the expected raw-view FL mask:

```bash
python psdaq/psdaq/debugtools/epixquad1kfps/write_gain_mode_standalone.py \
  --mode MapFML \
  --dry-run \
  --save-expected-map /tmp/mapfml_expected_fl_mask.npy
```

## Readout Commands

Decode a Rogue StreamWriter file:

```bash
python psdaq/psdaq/debugtools/epixquad1kfps/read_xpmmini_rogue_file.py \
  data_YYYYMMDD_HHMMSS.dat \
  --data-channel 1 \
  --max-frames 50
```

If the script imports Qt/ePixViewer without a display, run with an offscreen Qt
backend:

```bash
QT_QPA_PLATFORM=offscreen MPLCONFIGDIR=/tmp/epixquad_mpl \
python psdaq/psdaq/debugtools/epixquad1kfps/read_xpmmini_rogue_file.py \
  data_YYYYMMDD_HHMMSS.dat \
  --data-channel 1 \
  --max-frames 50
```

Useful summary lines:

| Output field | Interpretation |
|---|---|
| `gainbit_ones` | Number of pixels in that frame with raw bit 14 set |
| `overall_gainbit_fraction` | Fraction of checked pixels with raw bit 14 set |
| `top2_bit_counts` | Counts of `(raw >> 14) & 0x3`; with `0x7fff` mask, expect only bins `0` and `1` |
| `gainbit_image_same_for_decoded_frames` | Whether the raw bit-14 image is identical across decoded frames |

## False-Positive And False-Negative Checks

`read_xpmmini_rogue_file.py` can compare observed raw bit 14 against an
expected gainbit pattern.

Definitions:

| Term | Meaning |
|---|---|
| false positive | observed gainbit is `1`, expected gainbit is `0` |
| false negative | observed gainbit is `0`, expected gainbit is `1` |

For fixed modes with a uniform expected bit:

```bash
# FH or FM: expected gainbit is 0 everywhere
python psdaq/psdaq/debugtools/epixquad1kfps/read_xpmmini_rogue_file.py \
  data_YYYYMMDD_HHMMSS.dat \
  --data-channel 1 \
  --max-frames 50 \
  --expected-gainbit 0

# FL: expected gainbit is 1 everywhere
python psdaq/psdaq/debugtools/epixquad1kfps/read_xpmmini_rogue_file.py \
  data_YYYYMMDD_HHMMSS.dat \
  --data-channel 1 \
  --max-frames 50 \
  --expected-gainbit 1
```

For map modes, save the expected map when writing the gain mode, then pass that
map to the readout script:

```bash
python psdaq/psdaq/debugtools/epixquad1kfps/write_gain_mode_standalone.py \
  --mode MapFML \
  --save-expected-map /tmp/mapfml_expected_fl_mask.npy

python psdaq/psdaq/debugtools/epixquad1kfps/read_xpmmini_rogue_file.py \
  data_YYYYMMDD_HHMMSS.dat \
  --data-channel 1 \
  --max-frames 50 \
  --expected-gainbit-map /tmp/mapfml_expected_fl_mask.npy
```

The expected map may have any of these shapes:

| Expected map shape | Meaning |
|---|---|
| `(4,352,384)` | psana raw detector view; this is what `write_gain_mode_standalone.py --save-expected-map` writes |
| `(704,768)` | usable tiled psana raw view |
| `(712,768)` | full decoded Rogue/ePixViewer view |

The FP/FN summary reports total occurrences, unique pixels, DAQ raw
`segment,row,col`, and ASIC/bank coordinates.  The `.dat` ePixViewer-only rows
are removed by conversion and are not counted as configured pixels.  Limit the
printed coordinate list with `--fpfn-max-pixels`; use `--fpfn-max-pixels 0` for
counts only.

## Expected Readout Patterns

| Written mode | Expected readout |
|---|---|
| `FH` | `top2` mostly/all `0`; cannot distinguish from `FM` using top bits alone |
| `FM` | `top2` mostly/all `0`; cannot distinguish from `FH` using top bits alone |
| `FL` | `top2` mostly/all `1` |
| `MapFML` | selected FL pixels should be `top2 = 1`; FM background should be `top2 = 0` |
| `MapFHL` | selected FL pixels should be `top2 = 1`; FH background should be `top2 = 0` |

For the FH test file `data_20260626_093130.dat`, the observed result was:

| Quantity | Value |
|---|---:|
| decoded frames | 44 |
| total pixels checked | 24,059,904 |
| total raw bit-14 ones | 150 |
| overall raw bit-14 fraction | 0.000006 |
| `top2 = 0` count | 24,059,754 |
| `top2 = 1` count | 150 |
| unique FP pixels versus expected gainbit `0` | 8 |
| unique usable FP pixels | 4 |
| unique control-row FP pixels | 4 |

That is consistent with an FH or FM fixed-mode readout in the top-bit
diagnostic: almost every pixel reports `top2 = 0`.  It confirms the camera is
not in the earlier auto/map-looking state with about 46% raw bit-14 ones, but
it does not by itself distinguish FH from FM.

The control-row counts in this older FH summary came from the previous
ePixViewer-layout interpretation.  Re-run with the current reader to get DAQ
raw segment coordinates and to exclude the ePixViewer-only rows from the
configured-pixel comparison.

## Practical Checks

To verify the write/read cycle:

1. Write `FH`; read back with `--expected-gainbit 0`.  Usable false positives should be rare or zero.
2. Write `FM`; read back with `--expected-gainbit 0`.  Usable false positives should be rare or zero.
3. Write `FL`; read back with `--expected-gainbit 1`.  Usable false negatives should be rare or zero.
4. Write `MapFML` and save an expected map; read back with `--expected-gainbit-map`.
5. Write `MapFHL` and save an expected map; read back with `--expected-gainbit-map`.

The strongest direct test of the top-bit diagnostic is `FM` versus `FL`, or a
map mode with selected `FL` pixels.  `FH` versus `FM` requires checking the
configured/readback `trbit` or comparing analog response/pedestals, because
both fixed modes are expected to have raw bit 14 clear.
