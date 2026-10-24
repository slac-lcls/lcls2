# psana.psexp.step

`Step` is the object yielded by `run.steps()`, one for each BeginStep
transition. `step.evt` is the BeginStep event, and `step.events()` yields the
L1Accept events until the matching EndStep. The other transitions inside the
step are not yielded; they update the run's env store (EPICS and scan values)
when the run passed one (`esm`). A scan or EPICS detector accepts a `Step`
directly: `run.Detector('motor2')(step)` reads the value at `step.evt` (see
`EnvImpl.__call__` in [psana.detector.envstore](envstore.md)).

See [DataSource, Run and the event loop](../../features/datasource-run-events.md).

::: psana.psexp.step
