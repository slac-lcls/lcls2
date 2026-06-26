# rdsrv421 ePixQuad1kfps Gain-Mode Test Procedure

This procedure is for the standalone rdsrv421 ePixQuad1kfps test setup using
the C1100 XPM mini timing path and Rogue StreamWriter files.  It is not the
full LCLS DAQ recording path.

For the gain-mode write/readout mapping, expected gainbit values, and control
row handling, see [PIXEL_GAINMODE_WRITE_READOUT.md](PIXEL_GAINMODE_WRITE_READOUT.md).

## Environment

After logging into the test node:

```bash
cd ~/lcls2_worktree/ued-epix10ka-rdsrv421-daq
source setup_env.sh
echo "$SUBMODULEDIR"
```

The procedure below assumes the default device and lane:

```text
/dev/datadev_0, lane 0
```

## 1. Check Standalone Access

Run the standalone diagnostic first.  This checks that the PGP link, C1100
application root, timing receiver, camera top-level registers, ADC status, and
the SACI writer-enable path are usable.

```bash
python psdaq/psdaq/debugtools/epixquad1kfps/diagnose_standalone_access.py \
  --dev /dev/datadev_0 \
  --lane 0 \
  --standalone-timing true \
  --verbose-errors \
  --saci-writer-enable-probe
```

Expected healthy signs:

- Lane 0 is linked.
- `Camera register access succeeded.`
- Camera firmware version/build/GitHash are printed.
- `ADC test done` is `True` and `ADC test failed` is `False`.
- The SACI writer-enable probe succeeds.

The `--saci-writer-enable-probe` option was added because after the camera
firmware update the camera came back in an unhappy state where SACI writes
failed with:

```text
Non zero status message returned on fpga register bus in hardware: 0x2
```

The failure was seen while touching ASIC SACI registers such as
`Top.Epix10kaSaci[0].TestBe` or `Top.Epix10kaSaci[0].CompTH_DAC`.  This means
the SACI memory transaction did not complete.  If this happens, stop here and
do not trust gain-mode writes until the module state is recovered, for example
after a power cycle or help from Larry/Phil.  Once the camera state is stable,
this explicit writer-enable probe can be removed from the routine check.

## 2. Write A Gain Mode

Make the gain-mode write the last ASIC configuration action before recording.
Loading a full camera YAML after this step may rewrite the ASIC matrix or
registers; if that happens, rerun the gain-mode write before capturing data.

Example fixed low-gain write:

```bash
python psdaq/psdaq/debugtools/epixquad1kfps/write_gain_mode_standalone.py \
  --mode FL
```

Other fixed modes:

```bash
python psdaq/psdaq/debugtools/epixquad1kfps/write_gain_mode_standalone.py --mode FH
python psdaq/psdaq/debugtools/epixquad1kfps/write_gain_mode_standalone.py --mode FM
```

Example sparse map-mode write:

```bash
python psdaq/psdaq/debugtools/epixquad1kfps/write_gain_mode_standalone.py \
  --mode MapFML \
  --no-default-pixels \
  --raw-pixel 1,176,0 \
  --raw-pixel 1,176,3 \
  --raw-pixel 0,76,115 \
  --raw-pixel 0,123,115 \
  --raw-pixel 0,250,250 \
  --raw-pixel 1,250,250 \
  --raw-pixel 2,100,250 \
  --raw-pixel 3,100,250 \
  --save-expected-map /tmp/mapfml_minimal_expected.npy
```

## 3. Record Frames With The XPM Mini Writer GUI

Start the GUI:

```bash
python psdaq/psdaq/debugtools/epixquad1kfps/launch_xpmmini_writer_gui.py
```

The launcher initializes the C1100 side, but it does not automatically load the
UED camera YAML.  In the GUI, load:

```text
$SUBMODULEDIR/epix-quad-1kfps/software/yml/ued/epixQuad_ASICs_allAsics_UED_1080Hz_settings.yml
```

If this YAML is loaded after the gain-mode write in step 2, rerun the gain-mode
write before recording.  A GUI restart alone does not intentionally clear the
camera registers, but loading YAML is an explicit camera configuration action.

Recording sequence:

1. In Camera Debug, set the trigger rate to 10 Hz, for example
   `AutoTrigFreqHz = 10`.
2. In C1100 Debug, click `StartRun`.
3. In Camera System, choose the output data file path.
4. In Camera System, click file `Open`.
5. Wait long enough to collect the desired number of frames.
6. In Camera System, click file `Close`.
7. In C1100 Debug, click `StopRun`.

Use a new output filename for each capture so the decode step is unambiguous.

## 4. Decode And Check The Frame Data

For fixed `FL`, the expected raw gainbit is `1`:

```bash
python psdaq/psdaq/debugtools/epixquad1kfps/read_xpmmini_rogue_file.py \
  /path/to/data_YYYYMMDD_HHMMSS.dat \
  --data-channel 1 \
  --max-frames 50 \
  --expected-gainbit 1 \
  --fpfn-max-pixels 128
```

For fixed `FH` or `FM`, the expected raw gainbit is `0`:

```bash
python psdaq/psdaq/debugtools/epixquad1kfps/read_xpmmini_rogue_file.py \
  /path/to/data_YYYYMMDD_HHMMSS.dat \
  --data-channel 1 \
  --max-frames 50 \
  --expected-gainbit 0 \
  --fpfn-max-pixels 128
```

For map modes, pass the expected map saved by the writer:

```bash
python psdaq/psdaq/debugtools/epixquad1kfps/read_xpmmini_rogue_file.py \
  /path/to/data_YYYYMMDD_HHMMSS.dat \
  --data-channel 1 \
  --max-frames 50 \
  --expected-gainbit-map /tmp/mapfml_minimal_expected.npy \
  --fpfn-max-pixels 128
```

The FP/FN report lists whether mismatched pixels are in the usable image area
or in the decoded control rows.  Treat control-row mismatches separately from
configured usable-pixel mismatches.

If Qt/ePixViewer imports require a display, run the decode with an offscreen Qt
backend:

```bash
QT_QPA_PLATFORM=offscreen MPLCONFIGDIR=/tmp/epixquad_mpl \
python psdaq/psdaq/debugtools/epixquad1kfps/read_xpmmini_rogue_file.py \
  /path/to/data_YYYYMMDD_HHMMSS.dat \
  --data-channel 1 \
  --max-frames 50 \
  --expected-gainbit 1 \
  --fpfn-max-pixels 128
```

## Stop Conditions

Stop and report the exact command/output if any of these occur:

- The PGP lane is not linked.
- Camera register access fails.
- ADC test reports failed.
- Any SACI probe or gain write reports hardware bus status `0x2`.
- The StreamWriter file has no image-sized records on channel 1.
