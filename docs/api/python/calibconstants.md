# psana.detector.calibconstants

`CalibConstants` wraps the calibration constants of one detector
(`det.calibconst`, a dict `{ctype: (data, metadata)}` fetched from the
calibration database when the run is opened) and provides convenience
accessors (`pedestals()`, `rms()`, `gain()`, `gain_factor()`, `status()`),
geometry (`geo()`, pixel coordinates and indexes) and `image(nda, ...)`. Area
detectors create it on demand (`det.raw._calibconstants()`).

The constructor is `CalibConstants(calibconst, detname, **kwa)`. The "Usage"
block in the module docstring below writes `CalibConstants(calibconst, **kwa)`
and leaves out `detname`, which is required: one instance is kept per
`detname`, and creating it again with the same name returns that instance,
reset to the new `calibconst`.

See [Calibration constants](../../features/calibration.md).

::: psana.detector.calibconstants
