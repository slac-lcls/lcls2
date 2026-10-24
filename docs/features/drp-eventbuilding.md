# DRP and event building

## What it is

The LCLS-II DAQ data path, as implemented in this repository:

1. A **DRP** (data reduction pipeline) process per detector readout
   (`psdaq/drp/`, executable `drp` and variants) receives detector data over
   PGP/DMA, turns it into XTC2 datagrams with a detector-specific `Detector`
   class, and sends a small **trigger contribution** for each event to the
   trigger event builder.
2. The **TEB** (trigger event builder, executable `teb`,
   `psdaq/psdaq/eb/src/teb.cc`) builds the contributions of all DRPs for an
   event, runs a trigger plugin, and sends back a **result**: whether to record
   the event (`persist`/`prescale`) and whether to send it to monitoring
   (`monitor`).
3. Each DRP writes accepted events to its own file and posts monitored events
   to the **MEB** (monitoring event builder, executable `monReqServer`,
   `psdaq/psdaq/monreq/monReqServer.cc`), which builds them and serves them in
   shared memory to online analysis (see
   [Shared memory and live analysis](shmem-live.md)).

All three are controlled by the control process through the transitions
described in [DAQ control](daq-control.md). The code needs DAQ hardware (or the
simulators/emulators in the tree); nothing here runs against offline data.

## Key concepts

### The DRP process

- `drp` (`psdaq/drp/drp.cc`) parses its options (getopt
  `"p:o:l:D:S:C:d:u:k:P:M:W:Q:v"`; e.g. `-p` platform, `-P` instrument, `-C`
  collection host, `-d` DMA device, `-l` lane mask, `-D` detector type, `-u`
  alias, `-o` output directory, `-k` keyword arguments) and runs a
  `PGPDetectorApp`.
- `PGPDetectorApp` (`psdaq/drp/PGPDetectorApp.hh`) derives from
  `CollectionApp` (`psdaq/psdaq/service/Collection.hh`), so it takes part in
  rollcall/alloc/connect and receives the transitions from control.
- The detector class is chosen with `Factory<Detector>` (`psdaq/drp/Detector.hh`)
  from the `-D` name. Names registered in `psdaq/drp/PGPDetectorApp.cc`:
  `fakecam`, `cspad`, `hsd`, `epixquad`, `epixquad1kfps`, `epixhr2x2`,
  `epixhremu`, `epixm320`, `epixUHR`, `epix100`, `jungfrau`, `jungfrauemu`,
  `opal`, `tt`, `tb`, `ts`, `wave8`, `wave8he`, `hrencoder`, `piranha4`, `epixuhr3x2`.
- `DrpBase(Parameters& para, MemPool& pool, Detector& det, ZmqContext& context)`
  (`psdaq/drp/DrpBase.hh`) holds what all DRPs share: the TEB and MEB
  contributors, the TEB result receiver, file writing, and the trigger
  primitive.
- Buffers (`psdaq/drp/drp.hh`): `Parameters` (command-line settings),
  `PGPEvent` (DMA buffers of one event), `Pebble` (one large buffer for
  datagrams) and `MemPool`/`MemPoolCpu`.
- Other DRP executables built by `psdaq/drp/meson.build`: `drp_bld` (beam-line
  data), `drp_pva` (EPICS PV Access), `drp_udpencoder`, `tpr_trigger`, plus
  test tools (`pgpread`, `drp_validate`, ...). `epicsArch`
  (`psdaq/epicsArch/`) records EPICS variables. With CUDA available,
  `psdaq/drpGpu/` builds a GPU variant, `drp_gpu`.
- Python DRP: with `-k drp=python`, `drp` starts the worker
  `python -u -m psdaq.drp.drp_python ...` (`psdaq/psdaq/drp/drp_python.py`),
  and Python code inside it reads the data with
  `DataSource(drp=...)` (`psana/psana/psexp/drp_ds.py`).

### The `Detector` interface

`Drp::Detector` (`psdaq/drp/Detector.hh`) is what a new detector implements.
Key virtual methods (declarations from the header):

```cpp
virtual void connect(const nlohmann::json&, const std::string& collectionId) {};
virtual unsigned configure(const std::string& config_alias, XtcData::Xtc& xtc, const void* bufEnd) = 0;
virtual unsigned beginrun (XtcData::Xtc& xtc, const void* bufEnd, const nlohmann::json& runInfo) {return 0;}
virtual void slowupdate(XtcData::Xtc& xtc, const void* bufEnd) { ... };
virtual void event(XtcData::Dgram& dgram, const void* bufEnd, PGPEvent* event, uint64_t count) = 0;
virtual void shutdown() {};
```

For example, in `psdaq/drp/AreaDetector.cc` (the `fakecam`/`cspad` class)
`configure` adds `Names` to the Configure datagram's `Xtc`, and `event`
writes the event's values with `CreateData` against those names (see
[XTC2 data format](xtc2.md)). Many hardware detectors derive from
`BEBDetector` (`psdaq/drp/BEBDetector.hh`); `XpmDetector`
(`psdaq/drp/XpmDetector.hh`) is another base class used by several of them.

### Event builders

- `Pds::Eb::EventBuilder` and `EbAppBase` (`psdaq/psdaq/eb/src/EventBuilder.hh`,
  `EbAppBase.hh`) are the common event-building code; `EbAppBase(EbParams& prms, const std::string& pfx)`.
  Transport uses libfabric (`EbLfServer`/`EbLfClient` in the same directory).
- DRP side: `TebContributor(const TebCtrbParams&, unsigned numBuffers)` and
  `MebContributor(const MebCtrbParams&)` send to the TEB and MEB.
- `ResultDgram` (`psdaq/psdaq/eb/src/ResultDgram.hh`) is what the TEB returns:
  `persist()`, `prescale()`, `monitor()`, `auxdata()`, `monBufNo()`. The DRP
  writes an L1Accept if `persist() || prescale()` and posts it to the MEB if
  `monitor()` is non-zero (`psdaq/drp/TebReceiver.cc`).
- `teb` (`class Teb : public EbAppBase`, `class TebApp : public CollectionApp`
  in `teb.cc`) and `monReqServer` (`class Meb : public EbAppBase`,
  `class MebApp : public CollectionApp`) are the two event-builder
  executables. The MEB's shared-memory tag is set with `-t`.

### Trigger plugins

- `Pds::Trg::Trigger` (`psdaq/psdaq/trigger/src/Trigger.hh`) runs in the TEB:

  ```cpp
  virtual int  configure(const nlohmann::json& connectMsg, const nlohmann::json& configureMsg, const Pds::Eb::EbParams& prms) = 0;
  virtual void event(const Pds::EbDgram* const* start, const Pds::EbDgram** end, Pds::Eb::ResultDgram& result) = 0;
  virtual void transition(Pds::Eb::ResultDgram& result) {}
  ```

- `Pds::Trg::TriggerPrimitive` (`psdaq/psdaq/trigger/src/TriggerPrimitive.hh`)
  runs in each DRP and produces that DRP's trigger contribution.
- Both are loaded at Configure time from a shared library whose name is the
  `soname` key of the trigger configuration in the configuration database
  (the control process's `-t TRIGGER_CONFIG`). The TEB looks up the symbol
  `create_consumer`; the DRP tries `create_producer_<detName>` and then
  `create_producer` (`psdaq/psdaq/eb/src/teb.cc`, `psdaq/drp/DrpBase.cc`).
- Plugin libraries built by `psdaq/psdaq/trigger/meson.build` include
  `tmoTeb`, `tmoTrigger`, `mfxTripperTeb`, `calibTrigger` and `tstTebPy`; see
  `psdaq/psdaq/trigger/src/tmoTeb.cc` for a small example
  (`extern "C" Pds::Trg::Trigger* create_consumer()`).

## Minimal example

There is no stand-alone example for this part; the realistic "example" is a
platform description. From `psdaq/psdaq/cnf/tmo.cnf` (excerpt), each line
starts one process under `procmgr`:

```python
std_opts = '-P '+hutch+' -C '+collect_host+' -M '+prom_dir
teb_cmd  = task_set+' teb '         +std_opts+' -k '+scripts
meb_cmd  = task_set+' monReqServer '+std_opts
drp_cmd0 = task_set+' drp '    +std_opts0
...
{ host: 'drp-srcf-cmp030', id:'teb0',     flags:'spu', cmd:teb_cmd},
{ host: 'drp-srcf-mon001', id:'meb1',     flags:'spu', cmd:f'{meb_cmd} -t {hutch}_meb1 -d -n 64 -q {ami_workers_per_node}'},
{ host: 'drp-srcf-cmp028', id:'timing_0', flags:'spu', cmd:drp_cmd1+' -l 0x1 -D ts'},
```

Here `-D ts` selects the `TimingSystem` detector class and `-l 0x1` the PGP
lane mask; the `p` and `u` flags make `procmgr` add `-p <platform>` and
`-u <id>`.

## Where in the code

- `psdaq/drp/`: `drp.cc`, `drp.hh`, `PGPDetectorApp.*`, `DrpBase.*`, `Detector.hh`, `BEBDetector.*`, `XpmDetector.*`, `TebReceiver.*`, and one source pair per detector (`Opal.cc`, `Wave8.cc`, `Jungfrau.cc`, ...).
- `psdaq/drpGpu/`: GPU DRP.
- `psdaq/psdaq/eb/src/`: `EventBuilder`, `EbAppBase`, `TebContributor`, `MebContributor`, `ResultDgram`, `teb.cc`.
- `psdaq/psdaq/monreq/monReqServer.cc`: MEB.
- `psdaq/psdaq/trigger/src/`: `Trigger`, `TriggerPrimitive`, plugin examples.
- `psdaq/psdaq/service/Collection.hh`: `CollectionApp` (control protocol).
- `psdaq/psdaq/drp/drp_python.py`, `psana/psana/psexp/drp_ds.py`: Python inside the DRP.

C++ API pages: [Drp::Detector](../api/cpp/classDrp_1_1Detector.html),
[Drp::DrpBase](../api/cpp/classDrp_1_1DrpBase.html),
[Drp::PGPDetectorApp](../api/cpp/classDrp_1_1PGPDetectorApp.html),
[Drp::Factory](../api/cpp/classDrp_1_1Factory.html),
[Drp::MemPool](../api/cpp/classDrp_1_1MemPool.html),
[CollectionApp](../api/cpp/classCollectionApp.html),
[Pds::Eb::EbAppBase](../api/cpp/classPds_1_1Eb_1_1EbAppBase.html),
[Pds::Eb::TebContributor](../api/cpp/classPds_1_1Eb_1_1TebContributor.html),
[Pds::Eb::MebContributor](../api/cpp/classPds_1_1Eb_1_1MebContributor.html),
[Pds::Eb::ResultDgram](../api/cpp/classPds_1_1Eb_1_1ResultDgram.html),
[Pds::Trg::Trigger](../api/cpp/classPds_1_1Trg_1_1Trigger.html),
[Pds::Trg::TriggerPrimitive](../api/cpp/classPds_1_1Trg_1_1TriggerPrimitive.html).
