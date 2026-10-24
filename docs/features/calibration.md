# Calibration constants

## What it is

Detector calibration constants (pedestals, gains, pixel status, masks, common
mode parameters, geometry) are stored in a calibration database behind a web
service. psana2 fetches all constants for every detector in a run when the run
is opened, and the detector interfaces use them in `calib()` and `image()`.
Command-line tools in `psana/psana/app/` and `psana/psana/pscalib/app/`
produce constants (for example from dark runs) and deploy them to the database.

All of this needs network access to the calibration service and, for most
experiments, credentials. None of it works offline.

## Key concepts

### Where the constants come from

When a run is created, `Run._setup_run_calibconst` (`psana/psana/psexp/run.py`)
loops over the detectors in the run and calls

```python
calib_constants_all_types(det_uniqueid, exp=expt, run=runnum, dbsuffix=...)
```

from `psana/psana/pscalib/calib/MDBWebUtils.py`. `expt` and `runnum` come from
the BeginRun transition in the data; if the data carry no experiment name, no
constants are fetched. `det_uniqueid` is the detector type followed by the detector id of each
segment, joined with `_` (built in `psana/psana/dgrammanager.py`);
`detnames -i` prints it. The
synthetic test experiment `xpptut15` is special-cased: for it the code passes
the fixed name `cspad_detnum1234` instead.

Other modes:

- MPI: only the smd0 rank queries the database; the constants are packed and
  shared with the other ranks through MPI shared memory
  (`MPIDataSource`/`RunParallel` in `psana/psana/psexp/mpi_ds.py`,
  `psana/psana/psexp/calib_xtc.py`).
- Shared memory and DRP: a supervisor process fetches the constants and sends
  them to the other processes over ZMQ (`RunShmem._setup_run_calibconst`,
  `RunDrp._setup_run_calibconst` in `psana/psana/psexp/run.py`).

Related `DataSource` keyword arguments (`psana/psana/psexp/ds_base.py`):
`skip_calib_load` (detector names, or `"all"`), `dbsuffix` (default `""`),
`use_calib_cache` (read constants prepared by the `calib_prefetch` tool from
`/dev/shm/calibconst.pkl`), `fetch_calib_cache_max_retries` (default 60),
`cached_detectors`.

### The web service and credentials

`psana/psana/pscalib/calib/CalibConstants.py` defines the URLs:

- `URL_PRO = 'https://pswww.slac.stanford.edu/ws'`
- `URL_ENV = os.environ.get('LCLS_CALIB_HTTP', URL_PRO)`
- read requests go to `URL_ENV + '/calib_ws/'`.

So `LCLS_CALIB_HTTP` is a base URL and psana appends `/calib_ws/` itself
(this matches the note in `README.md`). Requests are authenticated with a JWT
when the `CALIB_JWT` environment variable is set, otherwise with Kerberos (see
`MDBWebUtils.py`). Database names have the prefix `cdb_` (`DBNAME_PREFIX`).

### What a detector gets

- `det.calibconst` (set by `run.Detector`) is a dict
  `{ctype: (data, metadata)}`, e.g. `det.calibconst['pedestals'][0]` is the
  pedestal array.
- Area detectors wrap it in `CalibConstants`
  (`psana/psana/detector/calibconstants.py`), which knows the constant types
  `pedestals`, `pixel_rms`, `common_mode`, `pixel_gain`, `pixel_offset`,
  `pixel_mask`, `pixel_status`, `status_extra` and `geometry`, and provides
  `pedestals()`, `rms()`, `gain()`, `gain_factor()`, `status()`, `geo()` and
  `image(nda, segnums=None, **kwa)`.
- From a detector object use the underscore accessors of `AreaDetector`
  (`psana/psana/detector/areadetector.py`): `det.raw._pedestals()`,
  `det.raw._gain()`, `det.raw._status()`, `det.raw._mask(**kwa)`,
  `det.raw._calibconstants()`.
- Geometry: if the database has no `geometry` constant, `AreaDetector` falls
  back to a default geometry file from
  `psana/psana/pscalib/geometry/data/geometry-def-*.data`. The geometry is
  handled by `GeometryAccess` (`psana/psana/pscalib/geometry/GeometryAccess.py`,
  e.g. `get_pixel_coords()`, `get_pixel_coord_indexes()`) and
  `img_from_pixel_arrays()` in the same module.

### Querying the database directly

`psana/psana/pscalib/calib/MDBWebUtils.py`:

- `calib_constants(det, exp=None, ctype='pedestals', run=None, time_sec=None, vers=None, dbsuffix='', **kwa)` returns `(data, doc)` for one constant type, or `None` if no document matches the query.
- `calib_constants_all_types(det, exp=None, run=None, time_sec=None, vers=None, dbsuffix='', **kwa)` returns a dict with all types.
- `deploy_constants(data, exp, detname_long, **kwa)` writes constants.

### Command-line tools

Entry points from `[project.scripts]` in the root `pyproject.toml` (run each
with `-h` for its options):

| Purpose | Commands |
|---|---|
| Inspect a run | `detnames`, `datinfo`, `config_dump`, `calibvalidity` |
| Database access | `cdb`, `cdb2`, `calibman` (Qt GUI) |
| Dark processing / pixel status | `det_dark_proc`, `epix_dark_proc`, `jungfrau_dark_proc`, `det_pixel_status` |
| Deploy constants | `epix10ka_deploy_constants`, `jungfrau_deploy_constants`, ... |
| Geometry | `geometry_convert`, `geometry_image` |
| Caching and auth | `calib_prefetch` (writes `/dev/shm/calibconst.pkl`), `jwt` |

`psana/psana/app/det_dark_proc.py` shows, for example,
`det_dark_proc -k exp=tmox49720,run=209 -d epix100 -D`.

## Minimal example

Adapted from `psana/psana/tests/test_jungfrau05M_calib.py`. The xtc2 file ships
with the repository; the constants are fetched from the database for the
experiment and run recorded in the file, so this needs network access:

```python
from psana import DataSource

ds = DataSource(files="psana/psana/tests/test_data/detector/test_jungfrau05M_calib.xtc2")
myrun = next(ds.runs())
det = myrun.Detector("jungfrau", status=True, gain_range_inds=(0,))  # mask options
peds = det.raw._pedestals()        # from det.calibconst['pedestals']
mask = det.raw._mask()             # built from pixel_status with the options above
for evt in myrun.events():
    calib = det.raw.calib(evt)     # pedestal-subtracted, gain-corrected, masked
    img = det.raw.image(evt)       # needs geometry
```

Querying one constant directly, adapted from `issue_2025_03_18` in
`psana/psana/detector/test_issues_2025.py` (needs access to the data and
constants of experiment `ued1006477`). The detector name for the database is
`det.raw._uniqueid`: the detector type followed by the id of each segment,
joined with `_`; it is the same value (`configinfo.uniqueid`) that psana
passes as `det_uniqueid` when it opens a run (except for `exp="xpptut15"`, for
which `Run._setup_run_calibconst` in `psana/psana/psexp/run.py` uses the fixed
name `cspad_detnum1234`).
It is an attribute of every detector interface (set in `DetectorImpl._reset`,
`psana/psana/detector/detector_impl.py`), not a method:

```python
from psana import DataSource
from psana.pscalib.calib.MDBWebUtils import calib_constants

ds = DataSource(exp="ued1006477", run=15)
myrun = next(ds.runs())
det = myrun.Detector("epixquad")
resp = calib_constants(det.raw._uniqueid, exp="ued1006477", ctype="pedestals", run=15)
if resp is not None:          # None if no document matches
    peds, doc = resp
```

## Where in the code

- `psana/psana/psexp/run.py`: `Run._setup_run_calibconst` and the shmem/DRP variants.
- `psana/psana/psexp/mpi_ds.py`, `psana/psana/psexp/calib_xtc.py`: sharing constants between MPI ranks.
- `psana/psana/pscalib/calib/CalibConstants.py`: URLs and environment variables.
- `psana/psana/pscalib/calib/MDBWebUtils.py`: web-service client (`calib_constants`, `calib_constants_all_types`, `deploy_constants`).
- `psana/psana/detector/calibconstants.py`: `CalibConstants`.
- `psana/psana/detector/areadetector.py`: constant, mask and geometry accessors on `det.raw`.
- `psana/psana/detector/detector_cache.py`: `DetectorCacheManager` (pickled pixel-index cache in `/dev/shm`).
- `psana/psana/pscalib/geometry/`: `GeometryAccess` and the default geometry files.
- `psana/psana/app/`, `psana/psana/pscalib/app/`: command-line tools.
- C++ counterparts used by the DAQ side: `psalg/psalg/calib/` (see the [C++ API](../api/cpp/index.html)).

API pages: [psana.detector.calibconstants](../api/python/calibconstants.md),
[psana.detector.areadetector](../api/python/areadetector.md).
