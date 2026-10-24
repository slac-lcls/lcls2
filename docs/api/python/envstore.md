# psana.detector.envstore

`run.Detector(name)` returns an `EnvImpl` subclass when `name` is an EPICS
variable (PV name or alias) or a scan variable. The object is called instead
of having `raw`/`fex` attributes: `det(evt)` returns the value for that event,
`det(step)` the value at the step's BeginStep event, and `det([evt1, evt2])` a
list of values; the value is `None` where the env store has none. Any other
argument type raises `InvalidInputEnvStore`. The subclasses
(`epics_raw_2_0_0`, `scan_raw_2_0_0`, ...) add nothing; they exist because the
class is looked up by its `<dettype>_<drp class>_<version>` name (see
[Detector interface](../../features/detector-interface.md#how-the-class-is-chosen)).

::: psana.detector.envstore
