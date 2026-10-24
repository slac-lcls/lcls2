# psana.dgrammanager

`DgramManager` reads dgrams from xtc2 files, from a shared-memory server
(`["shmem"]`, through `PyShmemClient` and a background reader thread) or from
the DRP (`["drp"]`). It keeps the Configure dgrams and builds the detector
class tables used by `run.Detector` (`_setup_det_class_table`). The data
sources create it; user code rarely needs it directly. The module also has a
small command-line test (`main`) that prints the contents of an xtc2 file.

See [XTC2 data format](../../features/xtc2.md) and
[Shared memory and live analysis](../../features/shmem-live.md).

::: psana.dgrammanager
