# Python API reference

These pages are generated from the source code by
[mkdocstrings](https://mkdocstrings.github.io/) with the
[griffe](https://mkdocstrings.github.io/griffe/) static analyzer. The code is
parsed, never imported, so the pages build without the compiled extensions
(`psana.dgram`, the Cython modules, ...). Two consequences:

- Objects that only exist in compiled code (`.pyx`, `.cc`) do not appear here,
  for example the `psana.dgram` extension, `psana.hsd` and the algorithm
  modules listed in [psalg algorithms](../../features/psalg.md). For the C++
  side see the [C++ API](../cpp/index.html) (Doxygen).
- The search box of this site covers these pages and the other MkDocs pages,
  but not the C++ symbols. The C++ API pages have their own search box.
- Docstrings are rendered as written. The repository is being documented
  gradually, so some objects only show a signature. Attributes without a
  docstring are hidden (see `docs/hooks/griffe_extensions.py`).

## Curated modules

The pages below cover the user-facing entry points. Each one starts with a
short orientation written for these docs, followed by the generated reference.

| Module | What it contains |
|---|---|
| [`psana.datasource`](datasource.md) | The `DataSource` factory function. |
| [`psana.psexp.ds_base`](ds_base.md) | `DataSourceBase`, the base class of all data sources, and its keyword arguments. |
| [`psana.psexp.serial_ds`](serial_ds.md) | `SerialDataSource` (`exp=` on one process). |
| [`psana.psexp.singlefile_ds`](singlefile_ds.md) | `SingleFileDataSource` (`files=`). |
| [`psana.psexp.run`](run.md) | The `Run` base class and the serial/single-file/shmem/DRP/smalldata run classes. |
| [`psana.psexp.step`](step.md) | `Step`, the object yielded by `run.steps()`. |
| [`psana.event`](event.md) | `Event`, the object yielded by `run.events()`. |
| [`psana.psexp.transitionid`](transitionid.md) | `TransitionId`, the transition codes returned by `evt.service()`. |
| [`psana.dgrammanager`](dgrammanager.md) | `DgramManager`, which reads dgrams from files, shared memory or the DRP. |
| [`psana.psexp.mpi_ds`](mpi_ds.md) | `MPIDataSource` and `RunParallel` (MPI mode). |
| [`psana.psexp.shmem_ds`](shmem_ds.md) | `ShmemDataSource` (shared-memory mode). |
| [`psana.psexp.drp_ds`](drp_ds.md) | `DrpDataSource` (`drp=`). |
| [`psana.psexp.null_ds`](null_ds.md) | `NullDataSource` and `NullRun`, for MPI ranks that read no data. |
| [`psana.smalldata`](smalldata.md) | `SmallData` (HDF5 output from MPI jobs) and its `Server`. |
| [`psana.detector.detector_impl`](detector_impl.md) | `DetectorImpl`, the base class of detector interfaces. |
| [`psana.detector.areadetector`](areadetector.md) | `AreaDetector` / `AreaDetectorRaw` (`raw`, `calib`, `image`). |
| [`psana.detector.calibconstants`](calibconstants.md) | `CalibConstants`, access to calibration constants and geometry. |
| [`psana.detector.envstore`](envstore.md) | `EnvImpl`, the callable detector object for EPICS and scan variables. |
| [`psdaq.control.control`](control.md) | `CollectionManager` and the DAQ control process. |
| [`psdaq.control.DaqControl`](daqcontrol.md) | `DaqControl`, the client API used by DAQ scripts. |
| [`psdaq.control.TimedRun`](timedrun.md) | `TimedRun`, a helper that runs the DAQ for a fixed time. |

Only modules that parse with Python 3 are listed; some files in the repository
are Python 2 only and are deliberately not referenced.
