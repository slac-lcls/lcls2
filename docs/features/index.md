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
| [DataSource, Run and the event loop](datasource-run-events.md) | `psana/psana/datasource.py`, `psana/psana/psexp/` |
