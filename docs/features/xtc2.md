# XTC2 data format

## What it is

XTC2 is the file and network format written by the LCLS-II DAQ and read by
psana2. A file (or a shared-memory buffer) is a sequence of **datagrams**
(`Dgram`). Each datagram is one transition (Configure, BeginRun, ...,
L1Accept) and holds a tree of **Xtc** containers. Data are self-describing:
the Configure datagram carries `Names` (field names, types and ranks for every
detector and algorithm), and each event carries `ShapesData` that are decoded
with those names.

The C++ library is `xtcdata` (`xtcdata/xtcdata/xtc/`). psana reads the same
format through the `psana.dgram` extension (`psana/src/dgram.cc`).

## Key concepts

### Datagram

`XtcData::Dgram` (`xtcdata/xtcdata/xtc/Dgram.hh`) derives from `Transition`,
which holds `TimeStamp time` and a packed 32-bit `env`. `Dgram` adds the
top-level `Xtc xtc`; the payload follows it in memory.

- `service()` returns the transition id (from bits of `env`); `isEvent()` is
  true for L1Accept; `readoutGroups()` is `env & 0xffff`.
- `TimeStamp` (`TimeStamp.hh`) has 32-bit seconds and nanoseconds;
  `value()` packs them as `(seconds << 32) | nanoseconds`. This is the value
  psana returns as `evt.timestamp`.

Transition ids (`TransitionId.hh`, `enum Value`):

| Value | Name | Value | Name |
|---|---|---|---|
| 0 | ClearReadout | 7 | EndStep |
| 1 | Reset | 8 | Enable |
| 2 | Configure | 9 | Disable |
| 3 | Unconfigure | 10 | SlowUpdate |
| 4 | BeginRun | 11 | Unused_11 |
| 5 | EndRun | 12 | L1Accept ("Must be 12 to agree with firmware") |
| 6 | BeginStep | | |

The Python mirror is `psana.psexp.TransitionId`
(`psana/psana/psexp/transitionid.py`), which names value 11
`L1Accept_EndOfBatch`.

### Xtc containers

`XtcData::Xtc` (`Xtc.hh`) is a 12-byte header with `Src src`, `Damage damage`,
`TypeId contains` and `uint32_t extent` (header plus payload size in bytes).

- `payload()` points just after the header; `sizeofPayload()` is
  `extent - sizeof(Xtc)`; `next()` points to the following sibling.
- Children are allocated inside a parent with `alloc()`, which grows the
  parent's `extent`.
- `TypeId::Type` is one of `Parent`, `ShapesData`, `Shapes`, `Data`, `Names`.
- `Damage` (`Damage.hh`) records problems such as `Truncated`, `OutOfOrder`,
  `DroppedContribution`, `MissingData`, `TimedOut`.

### Names, ShapesData and NamesId

- `Names` (`ShapesData.hh`) describes one detector segment and algorithm:
  detector name, type and id, an `Alg` (name and version), a segment number,
  and a list of `Name` entries (field name, `DataType`, rank). Constructor:
  `Names(const void* bufEnd, const char* detName, Alg& alg, const char* detType, const char* detId, const NamesId& namesId, unsigned segment=0)`.
- `ShapesData` holds the event data for one `Names`: a `Shapes` child (array
  shapes) and a `Data` child (the values).
- `NamesId(unsigned nodeId, unsigned namesId)` (`NamesId.hh`) is stored in the
  `src` field of both. Readers build a `NamesLookup` (`NamesLookup.hh`) from
  the Configure datagram and use the `NamesId` of each `ShapesData` to find
  its `Names`.
- `DescData` (`DescData.hh`) combines a `ShapesData` with its `NameIndex` to
  read values (`get_value`, `get_array`); `CreateData` writes them.

### Iterating

- `XtcFileIterator(int fd, size_t maxDgramSize)` (`XtcFileIterator.hh`):
  `next()` returns the next `Dgram*` from a file descriptor, or 0 at the end.
  It reuses one internal buffer.
- `XtcIterator(Xtc* root, const void* bufEnd)` (`XtcIterator.hh`): subclass it
  and implement `virtual int process(Xtc* xtc, const void* bufEnd)`;
  `iterate()` calls `process()` for each direct child. It does not recurse by
  itself: implementations call `iterate(xtc, bufEnd)` on `Parent` children.

### Small-data files

For each big-data file `<exp>-r<run>-s<stream>-c<chunk>.xtc2` there is a
small-data file `smalldata/<exp>-r<run>-s<stream>-c<chunk>.smd.xtc2`. It is
produced by `XtcData::Smd::generate` (`xtcdata/xtcdata/xtc/Smd.hh`; the
`smdwriter` tool uses it): all transitions are copied, and each L1Accept is
replaced by a small `smdinfo`/`offsetAlg` record with the fields
`intOffset` and `intDgramSize`, the position and size of that event in the
big-data file. psana reads the small-data files first to build events and
then reads only the needed bytes from the big-data files (see
[MPI parallel analysis](mpi-parallel.md)). The test helper
`psana/psana/tests/setup_input_files.py` shows the naming.

### Command-line tools (`xtcdata/xtcdata/app/`)

Installed by `xtcdata/xtcdata/app/meson.build`:

| Tool | Options (from the source) | What it does |
|---|---|---|
| `xtcreader` | `-f <file>` `[-d] [-n N] [-w nWords] [-s] [-T] [-c config]` | Prints one line per datagram (transition, time, env, payload size, damage, extent); `-d` dumps every Names and ShapesData. |
| `xtcwriter` | `-f <file>` `-n N` `-s seg` `-e period` `-m steps` `-t` `-i nodeId` `-c` | Writes a synthetic run (used by the psana tests). |
| `smdwriter` | `-f <in.xtc2>` `-o <out.smd.xtc2>` `[-n N]` | Writes the small-data file for a big-data file. |
| `amiwriter` | `-f <file>` ... | Writes synthetic data for AMI tests. |
| `xtcupdate` | `-f <file>` `[-n N]` | Demonstrates `XtcUpdateIter` (adds data in memory; writes no file). |

## Minimal examples

Print the datagram headers of a test file (output from
`psana/psana/tests/test_ts.xtc2`, shortened):

```console
$ xtcreader -f psana/psana/tests/test_ts.xtc2 -n 3
event 1,   Configure transition:  time 0x00f53438.0x2e6a3cde,  env 0x02820010, payloadSize 221248 damage 0x0 extent 221260
event 2,    BeginRun transition:  time 0x00f53451.0x2eb309e3,  env 0x04040010, payloadSize 0 damage 0x0 extent 12
event 3,   BeginStep transition:  time 0x00f53451.0x2f40eacd,  env 0x06860010, payloadSize 0 damage 0x0 extent 12
```

Read datagrams from Python with the low-level `psana.dgram` extension (most
analysis code uses `DataSource` instead). The pattern
`dgram.Dgram(file_descriptor=fd)` is the one in
`psana/psana/tests/test_dgraminit.py`; `Dgram(config=config)` reads the next
datagram, decoding it with the Configure datagram:

```python
import os
from psana import dgram
from psana.psexp import TransitionId

fd = os.open("psana/psana/tests/test_ts.xtc2", os.O_RDONLY)
config = dgram.Dgram(file_descriptor=fd)        # first datagram: Configure
print(list(config.software.__dict__))           # detector names, here ['xppts']
while True:
    try:
        d = dgram.Dgram(config=config)
    except StopIteration:
        break
    print(TransitionId.name(d.service()), d.timestamp())
```

In C++, the core loop of `xtcdata/xtcdata/app/xtcreader.cc` is an
`XtcFileIterator` plus an `XtcIterator` subclass (`DebugIter`) that fills a
`NamesLookup` from `Names` and decodes each `ShapesData` with `DescData`.

Writing or editing xtc2 from Python: `DgramEdit`, `AlgDef`, `DetectorDef` in
`psana/psana/dgramedit.pyx` (example: `psana/psana/tests/test_dgramedit.py`).

## Where in the code

- `xtcdata/xtcdata/xtc/`: `Dgram.hh`, `Xtc.hh`, `TransitionId.hh`, `TypeId.hh`, `TimeStamp.hh`, `Damage.hh`, `Src.hh`, `NamesId.hh`, `ShapesData.hh`, `DescData.hh`, `NamesLookup.hh`, `XtcIterator.hh`, `XtcFileIterator.hh`, `ConfigIter.hh`, `DataIter.hh`, `Smd.hh`.
- `xtcdata/xtcdata/app/`: `xtcreader.cc`, `xtcwriter.cc`, `smdwriter.cc`, ...
- `psana/src/dgram.cc`: the `psana.dgram` Python extension.
- `psana/psana/dgramedit.pyx`: writing and editing datagrams from Python.
- `psana/psana/dgrammanager.py`: how psana opens files and reads Configure.
- `psana/psana/psexp/transitionid.py`: Python transition ids.

C++ API pages: [Dgram](../api/cpp/classXtcData_1_1Dgram.html),
[Xtc](../api/cpp/classXtcData_1_1Xtc.html),
[TransitionId](../api/cpp/classXtcData_1_1TransitionId.html),
[Names](../api/cpp/classXtcData_1_1Names.html),
[ShapesData](../api/cpp/classXtcData_1_1ShapesData.html),
[NamesId](../api/cpp/classXtcData_1_1NamesId.html),
[DescData](../api/cpp/classXtcData_1_1DescData.html),
[XtcIterator](../api/cpp/classXtcData_1_1XtcIterator.html),
[XtcFileIterator](../api/cpp/classXtcData_1_1XtcFileIterator.html).
Python: [psana.dgrammanager](../api/python/dgrammanager.md).
