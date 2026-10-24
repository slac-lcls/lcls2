# Detector interface

## What it is

In psana2 you never parse detector data by hand. You ask the run for a detector
by name, `det = run.Detector('epixquad')`, and get an object whose attributes
(`det.raw`, `det.fex`, `det.ttalg`, ...) are interface classes that know how to
decode, and for area detectors calibrate, that detector's data for one event.

The interface classes live in `psana/psana/detector/` (and `psana/psana/hsd/`
for the high-speed digitizer). Which class you get is decided by the
detector type, the DAQ "DRP class" name and the data-format version recorded in
the xtc2 Configure transition.

## Key concepts

### `run.Detector(name, accept_missing=False, **kwargs)`

Defined in `psana/psana/psexp/run.py` (`Run.Detector`).

- For a normal detector it returns a container object with one attribute per
  DRP class found in the data, for example `det.raw` or `det.fex`. Each
  attribute is an instance of the matching interface class. The container also
  carries `det.calibconst` (the calibration constants for this detector, see
  [Calibration constants](calibration.md)) and `det._det_name`.
- If `name` is an EPICS variable or a scan variable, it returns a single
  callable object instead (see below).
- If nothing matches, it raises `KeyError` ("No available detector class
  matched with ..."). With `accept_missing=True` it returns a `MissingDet`
  placeholder (`psana/psana/detector/detector_impl.py`) whose methods return
  `None`.
- Extra `**kwargs` are passed to the interface classes. Area detectors keep
  them and use them, for example, for mask options
  (`psana/psana/tests/test_jungfrau05M_calib.py` passes `status=True`,
  `gain_range_inds=(0,)`).
- Interface objects are reused: `DetectorImpl.__new__` keeps a registry keyed
  by detector name and DRP class name, so calling `run.Detector` again for the
  same detector returns the same objects.

Useful run attributes for discovering names: `run.detnames`, `run.epicsinfo`,
`run.scaninfo`, `run.xtcinfo` (all in `psana/psana/psexp/run.py`). The
`detnames` command-line tool prints the same information
(`detnames exp=...,run=...`; `-e` EPICS aliases, `-s` scan names, `-r` raw data
types, `-i` segment and unique ids; see `psana/psana/app/detnames.py`).

### How the class is chosen

`DgramManager._setup_det_class_table` (`psana/psana/dgrammanager.py`) builds
the class name `<dettype>_<drp class>_<major>_<minor>_<micro>` from the
Configure data and looks it up in `psana/psana/detector/detectors.py`, which
star-imports all detector modules (the naming convention is stated in its first
lines). If no class with that name exists, the detector is skipped silently and
`run.Detector` later raises `KeyError`. A new detector or data version
therefore needs a new class with the right name in one of those modules.

### `DetectorImpl`

`psana/psana/detector/detector_impl.py`. The base class of all interfaces.

- Constructor: `DetectorImpl(det_name, drp_class_name, configinfo, calibconst, env_store=None, var_name=None, **kwargs)`.
- `config(evt)` returns one dict that merges this detector's per-segment
  configuration entries from all Configure dgrams; `evt` is not used.
- `_segments(evt)` returns this detector's data segments for the event, or
  `None` if any expected segment is missing; most `raw`/`calib` methods start
  from it.
- `_add_fields()` creates one method per field in the data description, so a
  field named `x` becomes `det.raw.x(evt)`. Many simple detectors (BLD,
  timetool `raw`, ...) only call this, which means their accessor names come
  from the data, not from the Python code.

### Area detectors

`psana/psana/detector/areadetector.py`:

- `AreaDetector(DetectorImpl)` adds calibration-constant and geometry helpers
  and `image(evt, nda=None, value_for_missing_segments=None, **kwa)`, which maps
  a per-pixel array (by default `self.calib(evt)`) to a 2-D image using the
  detector geometry.
- `AreaDetectorRaw(AreaDetector)` adds `raw(evt, copy=True)` and
  `calib(evt, **kwa)`. The generic `calib` computes
  `(raw - pedestals) * gain_factor * mask`, and falls back to `raw` if there
  are no pedestals and to `raw - pedestals` if there is no gain.
- The constant and geometry accessors are underscore-prefixed:
  `_pedestals()`, `_rms()`, `_gain()`, `_status()`, `_mask(**kwa)`,
  `_calibconstants()`, `_det_geo()`, `_pixel_coords()`, ... They are used in
  the tests and examples even though the names start with `_`.

Concrete detectors subclass these and override `calib` where the generic
formula is not enough, for example `epix10ka_raw_2_0_1` in
`psana/psana/detector/epix10ka.py` (gain-switching ePix10ka) and
`jungfrau_raw_0_1_0` in `psana/psana/detector/jungfrau.py`.

### Other detector kinds (examples)

| Kind | Classes | Methods |
|---|---|---|
| High-speed digitizer | `hsd_hsd_1_2_3`, `hsd_raw_2_0_0`, `hsd_raw_3_0_0` (`psana/psana/hsd/hsd.pyx`) | `waveforms(evt)`, `peaks(evt)`, `peak_times(evt)`, `padded(evt)` |
| wave8 | `wave8_raw_0_0_1`, `wave8_fex_0_0_1` (`psana/psana/detector/wave8.py`) | `raw_all(evt)`, `base_all(evt)`, `integral_all(evt)` |
| Timing system | `ts_ts_1_2_3`, `ts_raw_2_1_0`, ... (`psana/psana/detector/ts.py`) | e.g. `info(evt)`, `sequencer_info(evt)` |
| EPICS / scan variables | `EnvImpl`, `epics_raw_2_0_0`, `scan_raw_2_0_0` (`psana/psana/detector/envstore.py`) | the object is called: `det(evt)` |

For an EPICS PV name or alias, or a scan variable name, `run.Detector` returns
an `EnvImpl` subclass. Calling it with an event returns the value that was
valid at that event (EPICS values arrive in SlowUpdate transitions, so it can
be `None` before the first update).

## Minimal examples

Area detector image, from `psana/psana/tests/test_epix_calib.py`. The xtc2
file ships with the repository, but `image()` needs calibration constants and
geometry, which psana fetches from the calibration database (see
[Calibration constants](calibration.md)):

```python
from psana import DataSource

ds = DataSource(files="psana/psana/tests/test_data/detector/test_epix_calib.xtc2")
myrun = next(ds.runs())
epix = myrun.Detector("epixquad")
for nevt, evt in enumerate(myrun.events()):
    image = epix.raw.image(evt)      # 2-D array, or None if data are missing
```

Calibrated array plus an EPICS variable, adapted from
`psana/psana/tests/ds.py` (`test_standard`). `xpptut15` run 14 is a synthetic
data set that the tests write with the `xtcwriter` and `smdwriter` tools;
[Synthetic test data](../getting-started.md#synthetic-test-data) shows how to
write it into `.tmp`:

```python
from psana import DataSource

xtc_dir = ".tmp"                                       # see Getting started
ds = DataSource(exp="xpptut15", run=14, dir=xtc_dir)
for run in ds.runs():
    det = run.Detector("xppcspad")
    edet = run.Detector("HX2:DVD:GCC:01:PMON")         # EPICS variable
    for evt in run.events():
        calib = det.raw.calib(evt)                     # per-segment array
        value = edet(evt)                              # None until the first EPICS update
```

## Where in the code

- `psana/psana/psexp/run.py`: `Run.Detector`, `detnames`, `epicsinfo`, `scaninfo`, `xtcinfo`.
- `psana/psana/dgrammanager.py`: `_setup_det_class_table` (class lookup).
- `psana/psana/detector/detectors.py`: the list of detector modules.
- `psana/psana/detector/detector_impl.py`: `DetectorImpl`, `MissingDet`.
- `psana/psana/detector/areadetector.py`: `AreaDetector`, `AreaDetectorRaw`.
- `psana/psana/detector/epix10ka.py`, `jungfrau.py`, `epix100.py`, `opal.py`, ...: concrete area detectors.
- `psana/psana/detector/envstore.py`: EPICS and scan variables.
- `psana/psana/hsd/hsd.pyx`: digitizer interface (Cython, not in the generated Python API pages).

API pages: [psana.detector.detector_impl](../api/python/detector_impl.md),
[psana.detector.areadetector](../api/python/areadetector.md),
[psana.psexp.run](../api/python/run.md).
