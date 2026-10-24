# lcls2

lcls2 is the software repository for data acquisition and data analysis at
LCLS-II (SLAC). It holds four packages that are built together by the root
`meson.build`:

| Package | Language | What it is |
|---|---|---|
| **psana** (`psana/`) | Python, Cython, C++ | psana2, the analysis framework: `DataSource`, runs, events, detector interfaces, calibration, MPI and shared-memory modes. |
| **psdaq** (`psdaq/`) | C++, Python | The DAQ: control (`psdaq/psdaq/control`), data reduction pipeline (`psdaq/drp`), event builders (`psdaq/psdaq/eb`), hardware tools. |
| **xtcdata** (`xtcdata/`) | C++ | The XTC2 data format library (`xtcdata/xtcdata/xtc`) and command-line tools such as `xtcreader`. |
| **psalg** (`psalg/`) | C++, Python | Support libraries used by psana and the DAQ: detector geometry, calibration-constant access, HSD digitizer processing, array allocation, shared-memory transport (`psalg/psalg/shmem`), logging utilities. |

## How these docs are organized

- [Getting started](getting-started.md): build and environment pointers taken from
  `README.md` and the build scripts.
- [Features](features/index.md): one page per major capability (XTC2, the event
  loop, detectors, calibration, MPI, shared memory, psalg, DAQ control, DRP and
  event building). Each page lists where the code lives.
- [Python API](api/python/index.md): reference pages generated from the
  docstrings of the main user-facing modules.
- [C++ API](api/cpp/index.html): Doxygen pages for the public headers of
  xtcdata, psalg, psdaq and psana. The search box of this site does not cover
  them; the C++ pages have their own search box.

The site is versioned with [mike](https://github.com/jimporter/mike): `dev`
follows the `master` branch of
[slac-lcls/lcls2](https://github.com/slac-lcls/lcls2), and release tags
(such as `4.3.4`) get their own version. The version selector is next to the
site title.

## Quick start

With psana installed (see [Getting started](getting-started.md)), this reads
one of the small xtc2 files that ship with the tests
(`psana/psana/tests/test_ts.xtc2`, also used by
`psana/psana/tests/test_ts.py`). Run it from the repository root:

```python
from psana import DataSource

ds = DataSource(files="psana/psana/tests/test_ts.xtc2")
myrun = next(ds.runs())
print(myrun.detnames)                  # {'xppts'}
det = myrun.Detector("xppts")
for nevt, evt in enumerate(myrun.events()):
    print(nevt, evt.timestamp, det.ts.info(evt))
```

For real experiments you normally name the experiment and run instead of a
file, `DataSource(exp=..., run=...)`; psana then finds the run's files itself
(see the `dir` keyword in
[DataSource, Run and the event loop](features/datasource-run-events.md)),
which needs access to LCLS data. Several examples on this site
use `DataSource(exp='xpptut15', run=14, dir=xtc_dir)` instead, as in
`psana/psana/tests/ds.py`: a small synthetic run that you can write yourself
into the directory `xtc_dir` (see
[Synthetic test data](getting-started.md#synthetic-test-data)). Continue with
[DataSource, Run and the event loop](features/datasource-run-events.md).
