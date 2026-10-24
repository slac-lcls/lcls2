# Features

Each page describes one part of lcls2: what it is, the main concepts and
classes, a minimal example, and where the code lives. Each page was checked
against the source when it was written; file paths are relative to the
repository root. Where something could not be verified it says so.

Most examples need LCLS data. Some use the small xtc2 files that ship with the
tests (`psana/psana/tests/test_data/`, `psana/psana/tests/*.xtc2`); the others
need access to LCLS data storage, the calibration database, or a running DAQ.

| Page | Main code |
|---|---|
| [XTC2 data format](xtc2.md) | `xtcdata/xtcdata/xtc/`, `psana/src/dgram.cc`, `psana/psana/dgramedit.pyx` |
| [DataSource, Run and the event loop](datasource-run-events.md) | `psana/psana/datasource.py`, `psana/psana/psexp/` |
| [Detector interface](detector-interface.md) | `psana/psana/detector/` |
| [Calibration constants](calibration.md) | `psana/psana/detector/calibconstants.py`, `psana/psana/pscalib/` |
| [MPI parallel analysis](mpi-parallel.md) | `psana/psana/psexp/mpi_ds.py`, `psana/psana/psexp/node.py`, `psana/psana/smalldata.py` |
| [Shared memory and live analysis](shmem-live.md) | `psana/psana/psexp/shmem_ds.py`, `psana/psana/shmem/`, `psalg/psalg/shmem/` |
| [psalg algorithms](psalg.md) | `psalg/psalg/`, `psana/psana/peakFinder/`, `psana/psana/hsd/` |
| [DAQ control](daq-control.md) | `psdaq/psdaq/control/` |
| [DRP and event building](drp-eventbuilding.md) | `psdaq/drp/`, `psdaq/psdaq/eb/`, `psdaq/psdaq/trigger/` |
