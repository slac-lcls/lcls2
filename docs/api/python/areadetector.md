# psana.detector.areadetector

Methods common to area (pixel) detectors. `AreaDetector` adds access to
calibration constants and geometry (underscore-prefixed helpers such as
`_pedestals()`, `_gain()`, `_mask()`, `_pixel_coords()`) and `image(evt, ...)`;
`AreaDetectorRaw` adds `raw(evt)` and the generic `calib(evt)`,
`(raw - pedestals) * gain_factor * mask`. Concrete detector classes (ePix,
Jungfrau, Opal, ...) in `psana/psana/detector/` derive from these and override
`calib` where needed. This module is the example area-detector module for the
API reference; `epix10ka.py` and `jungfrau.py` follow the same pattern.

Notes for reading the reference below:

- Unlike the other API pages, this page also lists the methods whose names
  start with one underscore, because user code calls them as
  `det.raw._pedestals()`, `det.raw._mask(**kwa)`,
  `det.raw._calibconstants()` and so on.
- `det.raw._uniqueid` is an attribute, not a method, and is not listed: it is
  set in `DetectorImpl._reset` (`psana/psana/detector/detector_impl.py`) from
  the Configure data. It is the detector type followed by the id of each
  segment, joined with `_` (built in `psana/psana/dgrammanager.py`), and is
  the detector name used to fetch calibration constants (except for
  `exp="xpptut15"`, for which `Run._setup_run_calibconst` in
  `psana/psana/psexp/run.py` uses the fixed name `cspad_detnum1234`; see
  [Calibration constants](../../features/calibration.md)).
- `raw(evt, copy=True)`: `copy` is a parameter, although the generated
  section lists it under "Returns". For a detector with more than one segment
  the segments are stacked into a buffer that the next call reuses (while
  shape and dtype stay the same); `copy=False` returns that buffer instead of
  a copy. For a single segment the segment's array is returned in both cases.

See [Detector interface](../../features/detector-interface.md) and
[Calibration constants](../../features/calibration.md).

::: psana.detector.areadetector
    options:
      filters: ["!^__"]
