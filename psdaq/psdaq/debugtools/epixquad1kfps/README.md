# ePix Quad 1 kfps Debug Tools

This directory contains the detector-day tools and pattern definitions used to
locate the mapping from logical ASIC / bank coordinates to observed
`det.raw.raw(evt)` coordinates for `epixquad1kfps`.

The practical goal is:

- identify where each of the 16 ASICs lands in raw detector space
- identify where bank 0 appears for each ASIC
- determine whether each bank uses `identity` or `rot180`
- build the final `(asic, bank, row, col) -> det.raw.raw` mapping

The current mapping notes are summarized here:

- https://confluence.slac.stanford.edu/spaces/LCLSIIData/pages/695780057/ePix10ka+Detector+Layout+and+Raw-to-Bank+Mapping

## Important Setup Note

All debug environment variables must be visible to the running
`epixquad1kfps` DRP process.

In practice, set them through the DRP process environment in the DAQ cnf file
such as `ued.py`. Setting them only in an interactive shell is not sufficient
once the DRP is already running.

## Gain-Mode Mapping

For epix10ka / epixquad, the effective configured gain family comes from:

- pixel config code
- ASIC `trbit`
- raw data gain bit for the auto families

| Mode | pixel code | `trbit` | raw gain bit |
|---|---:|---:|---:|
| `FH` | `12` (`0xc`) | `1` | ignored |
| `FM` | `12` (`0xc`) | `0` | ignored |
| `FL` | `8` (`0x8`) | ignored | ignored |
| `AHL_H` | `0` | `1` | `0` |
| `AML_M` | `0` | `0` | `0` |
| `AHL_L` | `0` | `1` | `1` |
| `AML_L` | `0` | `0` | `1` |

For the detector-day direct-write tests, the useful fixed-mode defaults were:

- background `12` with `trbit = 0` -> `FM`
- selected pixels `8` -> `FL`

That keeps the test patterns simple and independent of the auto-mode data gain
bit split.

## Standalone Gain-Mode Writer

Use `write_gain_mode_standalone.py` to program the test gain modes without the
DAQ or PyDM GUI.  Close any GUI or DAQ process using `/dev/datadev_0` first,
because the script opens the camera register VC.

Preview the register-write plan:

```bash
python psdaq/psdaq/debugtools/epixquad1kfps/write_gain_mode_standalone.py \
  --mode MapFML \
  --dry-run
```

Write one of the full fixed modes:

```bash
python psdaq/psdaq/debugtools/epixquad1kfps/write_gain_mode_standalone.py --mode FH
python psdaq/psdaq/debugtools/epixquad1kfps/write_gain_mode_standalone.py --mode FM
python psdaq/psdaq/debugtools/epixquad1kfps/write_gain_mode_standalone.py --mode FL
```

Write map modes with the built-in sparse selected-pixel list:

```bash
python psdaq/psdaq/debugtools/epixquad1kfps/write_gain_mode_standalone.py --mode MapFML
python psdaq/psdaq/debugtools/epixquad1kfps/write_gain_mode_standalone.py --mode MapFHL
```

For custom selected pixels, use ASIC-local coordinates:

```bash
python psdaq/psdaq/debugtools/epixquad1kfps/write_gain_mode_standalone.py \
  --mode MapFML \
  --no-default-pixels \
  --pixel 0,12,7 \
  --pixel 0,12,55
```

or raw-view segment coordinates:

```bash
python psdaq/psdaq/debugtools/epixquad1kfps/write_gain_mode_standalone.py \
  --mode MapFHL \
  --no-default-pixels \
  --raw-pixel 0,188,199
```

By default the script only writes the ASIC gain registers.  Add
`--load-ued-yaml` if you want it to load the standard UED camera YAML first.
Add `--save-expected-map /tmp/map.npy` to save the expected raw-view FL mask
for later comparison.

## Debugging Modes

Two debug modes are supported.

### 1. JSON Run Sequence

This mode uses:

- `EPIXQUAD_DEBUG_SEQUENCE_FILE=/path/to/sequence.json`
- `EPIXQUAD_DEBUG_PATTERN_INDEX=<int>`

It is intended for stepping through an ordered pattern sequence, usually with
`run_pattern_sequence.py`.

Status:

- supported by the code path
- may currently still be broken
- was not used much on test day

Use this mode when:

- you want one DAQ run per pattern in a predefined sequence
- you want to drive the control-file workflow in `run_pattern_sequence.py`

Main files:

- `sequences/detector_day_v1.json`
- `sequences/priority_shortlist.json`
- `run_pattern_sequence.py`

### 2. Direct JSON / NPY

This is the preferred mode and was the mode used on test day.

It bypasses the higher-level sequence workflow and drives either:

- direct JSON register-write patterns, or
- raw detector masks from `.npy`

#### 2.1 Direct JSON

Use:

- `EPIXQUAD_DEBUG_DIRECT_WRITE_FILE=/path/to/direct.json`
- `EPIXQUAD_DEBUG_PATTERN_INDEX=<int>`

This mode writes explicit `(asic, bank, row, col)` operations through the
direct-write path in `config_expert()`.

##### 2.1.1 ASIC Location By Full Bank-0 Fills

The main file was:

- `direct/03_asic_id_bank0_full.json`

This file contains one pattern per ASIC, each filling logical bank 0 for one of
the 16 ASICs.

It was used to:

- identify where the 16 ASICs land in raw detector space
- show that some banks appear on the right side rather than the left
- show that some banks are effectively rotated by 180 degrees

That was the main coarse ASIC-location probe used on test day.

##### 2.1.2 Sparse Bank 0-3 Markers For ASICs 0-3

The main files were:

- `direct/03_bank_markers_178x48.json` for ASIC 0
- `direct/04_bank_markers_178x48_asic1.json` for ASIC 1
- `direct/07_bank_markers_178x48_asic2.json` for ASIC 2
- `direct/08_bank_markers_178x48_asic3.json` for ASIC 3

These files were used to identify bank orientation after ASIC location was
known.

Important detector-day observation:

- there is a ghost row
- for non-rotated banks it shows up at row `0`
- for banks rotated by `rot180` it shows up at row `176`
- that ghost row behaves as if its gain-mode bit were also set

This ghost row made the orientation analysis harder at first. After ignoring
the ghost rows, the sparse bank markers matched either:

- `identity`
- or `rot180`

with clean agreement.

The combination of:

- `03_asic_id_bank0_full.json`
- sparse bank-marker files for ASICs 0, 1, 2, and 3

gave enough information to construct the correct ASIC / bank / row / col
mapping into `det.raw.raw`, as documented in the Confluence note linked above.

#### 2.2 Direct NPY

Use:

- `EPIXQUAD_DEBUG_MASK_NPY=/path/to/mask.npy`
- `EPIXQUAD_DEBUG_MASK_SELECTED_VALUE=8` (optional)
- `EPIXQUAD_DEBUG_MASK_BACKGROUND_VALUE=12` (optional)
- `EPIXQUAD_DEBUG_MASK_TRBIT=0` (optional)

This mode accepts a raw-view mask with the same shape and orientation as
`det.raw.raw(evt)`:

- shape `(4, 352, 384)`

The mask is converted into direct bank-addressed writes internally.

Use this mode when:

- you already have a raw detector mask in the final detector orientation
- you do not want to hand-author JSON direct-write operations

## Files

Core tools:

- `pattern_loader.py`
- `run_pattern_sequence.py`
- `diagnose_standalone_access.py`
- `validate_pattern_runs.py`
- `diagnose_pattern_runs.py`
- `render_full_bank_layout.py`
- `view_raw_modes.py`

## Standalone Hardware Access Diagnostic

Use `diagnose_standalone_access.py` before starting DAQ when checking a new
test stand.  It reads PGP4 lane status, initializes the C1100 application path,
then tries ePixQuad camera register access on VC1.  It does not start DAQ,
write ASIC pixel maps, or write PROM contents.

Example for the UED rdsrv421 setup:

```bash
cd /sdf/home/m/monarin/lcls2_worktree/ued-epix10ka-rdsrv421-daq
source setup_env.sh
python psdaq/psdaq/debugtools/epixquad1kfps/diagnose_standalone_access.py \
  --dev /dev/datadev_0 \
  --lane 0
```

If the startup PROM calibration check fails, rerun the camera access check in
a fresh process with `--force-prom-bypass`.  A failed `Top.start()` can leave
the camera VC open until the Python process exits, so the diagnostic does not
retry `promWrEn=True` in the same process by default.

```bash
python psdaq/psdaq/debugtools/epixquad1kfps/diagnose_standalone_access.py \
  --dev /dev/datadev_0 \
  --lane 0 \
  --force-prom-bypass
```

To check whether the external XPM timing link is hooked up, run the C1100
startup with standalone timing disabled and skip the camera access:

```bash
python psdaq/psdaq/debugtools/epixquad1kfps/diagnose_standalone_access.py \
  --dev /dev/datadev_0 \
  --lane 0 \
  --standalone-timing false \
  --skip-camera
```

For an external XPM link, expect `ConfigLclsTimingV2()`, `RxLinkUp=True`,
`RxDown=False`, increasing `FidCount`/`sofCount`, and quiet CRC/decode/disparity
error counters.  `ConfigureXpmMini()` means local generated timing, not proof
that the external XPM timing fiber is connected.

## XpmMini Standalone File Capture

To try Larry's suggested standalone path, use the C1100 built-in XpmMini timing
generator and the ePixQuad Rogue `StreamWriter`.  The helper below opens both
the C1100 DevRoot and the ePixQuad camera root, then launches one PyDM GUI with
both servers.  The GUI has separate tabs for `C1100 System`, `C1100 Debug`,
`Camera System`, and `Camera Debug`.

```bash
cd /sdf/home/m/monarin/lcls2_worktree/ued-epix10ka-rdsrv421-daq
source setup_env.sh
python psdaq/psdaq/debugtools/epixquad1kfps/launch_xpmmini_writer_gui.py \
  --dev /dev/datadev_0 \
  --lane 0
```

In the PyDM GUI:

- in `C1100 Debug`, run `DevRoot.StartRun()` to allow XpmMini-triggered traffic
  through the C1100 event builder
- in `Camera System`, load the UED camera YAML:
  `/sdf/group/lcls/ds/ana/sw/conda2-v4/rel/lcls2_submodules_03122026/epix-quad-1kfps/software/yml/ued/epixQuad_ASICs_allAsics_UED_1080Hz_settings.yml`
- for a short debug capture, reduce the camera auto-trigger rate before opening
  the writer: in `Camera Debug` set `Top.SystemRegs.AutoTrigFreqHz` to `10`
  or set `Top.SystemRegs.AutoTrigPer` to `0x989680`
- verify the camera side is armed:
  `Top.SystemRegs.TrigSrcSel=3`, `Top.SystemRegs.AutoTrigEn=True`,
  `Top.SystemRegs.TrigEn=True`, and `Top.RdoutCore.RdoutEn=True`
- go to `Camera System`, or `Camera Debug` -> `Top.StreamWriter`
- set `DataFile` to a writable `.dat` path, or click `AutoName`
- click `Open`
- watch `FrameCount` and `TotalSize`; a full-frame capture should grow by about
  1 MB per image frame, not by only tens of bytes per record
- click `Close`
- in `C1100 Debug`, run `DevRoot.StopRun()`

The ePixQuad writer routes camera VC0 data to file channel `1`, VC2 scope data
to channel `2`, and VC3 monitoring data to channel `3`.  The helper defaults to
`promWrEn=True` only to bypass the startup PROM validation seen on rdsrv421; do
not execute ADC-training or PROM-writing commands unless that is intentional.

For a command-line start of the C1100 run gate, add `--start-run`; the helper
will call `DevRoot.StopRun()` when the GUI exits.

Check the saved file with:

```bash
python psdaq/psdaq/debugtools/epixquad1kfps/read_xpmmini_rogue_file.py \
  /tmp/epixquad_fullframe.data \
  --data-channel 1 \
  --max-frames 5
```

The reader converts the ePixViewer decoded `(712,768)` StreamWriter frames to
DAQ raw `(4,352,384)`, matching `det.raw.raw(evt)`.  ePixViewer-only rows are
not part of the raw frame used for gainbit and FP/FN checks.

If the summary reports only 48-byte records and `decoded_image_frames: 0`, the
file contains timing/event records but no full ePixQuad image frames.

Pattern definitions:

- `tests/*.json`
- `direct/*.json`
- `sequences/*.json`

Reference data:

- `manifest.json`
- `testdata/roiFromAboveThreshold_r50_c0_calib.npy`
