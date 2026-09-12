# GPU DRP to-dos

Working notes for the `features/gpu` branch.  Each item records enough context to
be picked up cold, because the reasoning behind these decisions is otherwise only
in people's heads.

## Detector configuration

- ~~**`epixuhremu_config.py`.**~~  **Done, and working on drp-srcf-gpu001 on
  2026-09-12.**  From a link-down start, Allocate now logs `epixuhremu: timing link is
  down, calling ConfigLclsTimingV2()`, the fiducial counter starts counting, and
  `xpmdet_connectionInfo()` reads a legal `rxId` where it previously got `0xffffffff`
  and aborted.  Connect and Configure follow through to `Gpu::EpixUHRemu configure`,
  with no devGui clicking.

  The emulator firmware needs LCLS-II timing configured, which the real detectors do
  not, and which is otherwise a devGui click per card — tedious on a multi-datadev
  node.  `ConfigLclsTimingV2()` is guarded on `TimingFrameRx.RxLinkUp`, the live link
  status, because it resets the receive PLL, issues Tx and Rx user resets and sleeps
  three times for a second: calling it unconditionally would add that to every
  Allocate and bounce a link that was working.  `RxDown` is a latch and so is cleared
  afterwards rather than tested.

  It is called **before** `Drp::XpmDetector::connectionInfo()`, not after.  That is not
  cosmetic: with the link down, `xpmdet_connectionInfo()` reads the XPM remote link id
  as `0xffffffff` and raises, so a hook after it never runs — and the link being down
  is the whole case it exists for.  `xpmdet_connectionInfo()`'s own `RxPllReset` retry
  does not recover it, because `ConfigLclsTimingV2()` also clears `UseMiniTpg` and
  issues `TxPhyReset` and the Tx and Rx user resets.  Getting this backwards cost a
  debugging round; the ordering is commented at both ends.

  Because it must precede the barrier-supervisor election inside
  `xpmdet_connectionInfo()`, it runs in every DRP process rather than only the
  supervisor.  That is correct with one process per card, which is the emulator's case.
  Two processes sharing a card could each find the link down and reset it in turn; the
  `RxLinkUp` guard makes that unlikely, not impossible.

  An earlier note here claimed `Drp::XpmDetector` has no Python config hook.  That is
  wrong — it imports `psdaq.configdb.xpmdet_config` at `XpmDetector.cc:37` and looks
  its functions up in that module's dict on every call.  What it lacks is the
  `<detType>_config.py` *selection* machinery, which lives in `BEBDetector::_init()`;
  the module name is hardwired.

  Changing that, or `xpmdet_config.py`, would risk the CPU DRPs for a detector that
  will never run in production, so neither is touched.  Two properties make that
  avoidable:

  - `Gpu::Detector` *wraps* rather than inherits — it holds a `Drp::Detector* m_det`
    built by `_initialize<T>` (`Detector.hh:107`) and delegates `connectionInfo()` to
    it (`Detector.cc:10`).  So overriding `connectionInfo()` on the file-local
    `Gpu::XpmDetector` shim in `EpixUHRemu.cu` is enough, and touches no header.
  - A second, independent Python import from GPU-only code costs nothing.
    `epixuhremu_config` reaches the rogue tree through `xpmdet_config.args['root']`,
    a module global, so it needs no cooperation from `xpmdet_config` at all.

  The GIL is already held at that point: `PGPDetectorApp::connectionInfo` wraps
  `m_det->connectionInfo()` in `PY_ACQUIRE_GIL_GUARD`.

  If an override site were ever unavailable, the fallback is to monkeypatch
  `xpmdet_config.xpmdet_connectionInfo` from an imported module — the C++ resolves
  the function from the module dict per call, not at init, so a replacement takes
  effect.  Action at a distance, and not needed here.

  `Gpu::EpixUHR3x2` needs none of this: it derives from `Drp::EpixUHR3x2`, so it gets
  the `BEBDetector` machinery, and `epixuhr3x2_config.py` already calls
  `ConfigLclsTimingV2()`.

- **EpixUHR3x2 gain encoding.**  `RangeOffset`/`RangeBits` in `EpixUHR3x2.hh` are
  moot as written: the panel delivers data already calibrated to fp16 by firmware,
  so there is no gain range in it.  The accessors return 0 to satisfy the base
  class.  Confirm nothing else wants them.

- **Jungfrau pedestals and gains** are placeholders (0.0/1.0).  The CPU-side
  Jungfrau writes raw data and leaves calibration to analysis, so there is no
  source for real constants yet.  Also unresolved whether the GPU path should
  calibrate Jungfrau at all or pass raw through.

## Slurm and node configuration

- **`gres.conf` publishing.**  `psdaq/psdaq/slurm/gen_gres_conf.py` does the work.
  The workflow, run on the node after every datadev driver load:

      gen_gres_conf.py --expect N --base --output ~/gres.conf
      scp ~/gres.conf psslurmctld001:/etc/slurm/gres.conf
      ssh psslurmctld001 sudo scontrol reconfigure

  `--base` with no argument starts from `/var/spool/slurmd/conf-cache/gres.conf`, the
  copy the controller distributed to this node, and emits the whole file with this
  node's block spliced in.  No editor involved.

  **The hole**: that copy is only as current as the last `scontrol reconfigure`, so
  anything changed on the controller since will be silently reverted when the result
  is copied back.  The script prints the base's size, mtime and sha256 so it can be
  compared against `psslurmctld001:/etc/slurm/gres.conf` before copying; closing the
  hole properly would mean splicing on the controller instead, which needs the node
  name passed in rather than taken from `uname`.

  The script also **comments out**, rather than deletes, any pre-existing
  `NodeName=<this node>` gpu or datadev line outside its own block.  Those are the
  hand written entries it replaces, and leaving them would give the node two records
  for the same `File=` with different `Type=`.

  Note what the existing file already contains: hand written per-node lines with
  `Type=nvidia_h200_nvl`, and a comment recording that "cpo and claus have a guess
  that we should put individual lines with /dev/nvidiaN lines so that slurm can be
  used to select specific gpus to pair with specific datadev fpga boards".  That is
  this task.  Model based type names cannot express the pairing, because every H200
  on the node has the same model name; bus derived names (`dd04`) can.

  With ~25 nodes coming, the copy-and-reconfigure step wants automating.

- **The `g` flag conflicts with a gres request.**  `psdaq/slurm/utils.py:590` emits
  `#SBATCH --gpus-per-task=1 --gpus=1` when `flags` contains `g`, which collides
  with `--gres=gpu:...`: two ways of asking for a GPU in one allocation.  They need
  to be mutually exclusive, with gres winning.  Worse, `generate_as_step()` handles
  no flags at all, so placement differs by launch mode — `#SBATCH` in per-process
  mode, on the `srun` line in step mode.  Must be settled before gres is usable.

- **`cfgDevName=1` everywhere.**  `options datadev cfgDevName=1` in
  `/etc/modprobe.d/datadev.conf` makes the driver name devices by PCI bus number
  (`/dev/datadev_84`) instead of probe order — see `cfgDevName` in
  aes-stream-drivers `common/driver/data_dev_top.c:214`.  Probe order is stable
  today but not guaranteed across a card change, and the generator's type names
  already assume the bus-number form.

- **`GresTypes=gpu,datadev`** in `slurm.conf`.  The distributed copy currently has
  `GresTypes=gpu` only, so datadev entries would be rejected outright.  Each node's
  line also needs a matching `Gres=`: gpu008's says
  `Gres=gpu:nvidia_h200_nvl:6` today, and the generator prints the replacement.

- **Slurm does not notice a vanished GPU.**  With GPU5 off the bus, gpu008 still
  reports `Gres=gpu:nvidia_h200_nvl:6`, `CfgTRES=gres/gpu=6` and `State=IDLE` — not
  drained, no complaint.  `slurmd` validated six `File=` entries when it started,
  before the card went, and nothing rechecks.  So Slurm would schedule six GPU jobs
  onto five GPUs.  This is the argument for running the generator with `--expect`
  after every driver load and at boot, rather than trusting Slurm to self-correct.

## Performance and structure

- **Standalone harness for the graphs.**  Long-standing want: pull the kernels into
  a harness with synthesised input, both as permanent test code and as a profiling
  target.  `_event` is already a template in `ReaderKernels.cuh`, and
  `ReducerAlgo::recordGraph()` and `TriggerPrimitive::event()` already take a
  `cudaStream_t` and plain pointers, so all three are drivable without the
  pipeline.  This would give the **first correctness test of the EpixUHR3x2 fp16
  path and Jungfrau's nested packet walk, neither of which has ever executed**, and
  an `ncu` target free of spin-waits and device-side relaunch.

- **Compressor payload sweep.**  Run `lc`, `pfpl`, `sleek` over 1x, 2x, 4x payloads
  in the harness and measure ratio and throughput.  This is the number that decides
  how many datadevs a GPU can usefully feed: the constraint was always reducer
  throughput, never PCIe.  Measured on gpu008, PCIe locality costs nothing — all
  cards sit at their own PCIe 4.0 x8 ceiling (~102 Gbps, 33035 Hz), whether or not
  they share a switch with their GPU.

- **Recorder and file writing**, including file system bandwidth and
  scatter-gather.  Profile `TebReceiver::_recorder()` and the writer
  before the third-party compressor work: the sink's throughput sets the compression
  ratio the reducers have to achieve.  At 33 kHz the uncompressed calibrated rate is
  ~25 GB/s, above even the 22 GB/s measured with GPUDirect Storage working properly.
  GDS is unavailable (mixed IB/Ethernet is unsupported by WEKA), so cuFile runs in
  compatibility mode — worth asking whether cuFile buys anything over an explicit
  device-to-host copy plus `pwritev` in that mode, and whether scatter-gather writes
  help.

- **Remove `HOST_LAUNCHED_REDUCERS`.**  A temporary switch for seeing whether
  certain reducers worked at all.  It gives reducers two launch paths of which only
  one is ever exercised — the same shape as the `HOST_REARMS_DMA` rot.

- **Third-party compressor support** via shims in `lcls2-dev/subprojects`.  `lc`,
  `pfpl` and `sleek` work; `cusz` and `cuszp` have problems; `eip` was started in
  `~/git/psdaq-reducers_260609` but the upstream code was not ready for use this way
  — a nominally better version needs downloading.

- **Multiple datadevs per GPU: resurrect the multi-Reader event builder.**  The old
  code is in `~/lclsii/daq/obsolete/drpGpu/`, chiefly `Collector.cu_save` and
  `Reader.cu_save`.  In the repo, multi-datadev support was removed by **a856eae8**
  (2026-04-07, "Move to aes-stream-drivers v7; Remove multi-datadev support"), so its
  parent **055d45df** (2026-03-23) is the last commit that has it.  Collector was
  renamed to TrgInpGen later, in 46f9092d (2026-04-30).  Two things to know before
  resurrecting any of it:

  - It did **not** build events by pulse id.  The `_collector` kernel ran one
    thread per panel, each consuming its own `readerQueues[panel]`, then
    `__syncthreads()` and asserted every panel had produced the *same* intermediate
    buffer index — i.e. it assumed the FPGAs deliver in lockstep, and spun forever
    (`while (true)`) on a mismatch.  The pulse id, control, timestamp, env and
    evtCounter comparison across panels was a **host-side diagnostic**, not the
    building mechanism.  Real pulse-id matching would be new work.
  - `MemPoolGpu` was multi-panel then (`panels()`, `hostWrtBufsVec_h()[i]`) and is
    single-panel now (`m_panel`, `panel()`).  That has to be reinstated first.

  Whether this is wanted at all depends on the compressor payload sweep below:
  measured on gpu008, PCIe is not the constraint, so the only reason to feed one GPU
  from two cards is reducer efficiency on a larger payload.  Note bifurcation has
  never worked on these boxes; BIOS was blamed, but the same regression appears
  elsewhere after the RHEL 7 to Rocky 9 upgrade.

- **`HOST_REARMS_DMA` needs an early rearm stage** before it is a real fallback.  It
  currently rearms in `TrgInpGen::_receiver()`, downstream of the trigger kernels,
  so a DMA buffer waits on a dynamically loaded trigger library whose latency is
  unbounded.  See the comment at the macro in `MemPool.hh`.

- **Bulk `gpuSetWriteEn`.**  One ioctl per buffer is ~33k/s at current rates.  A
  masked or ranged form would be a small aes-stream-drivers PR, and it only matters
  for the host-rearm path.

## Calibration and data handling

- **Fetch calibration constants.**  Every detector currently fabricates them:
  `EpixUHRemu`, `EpixUHRsim` and `Jungfrau` fill pedestals with 0.0 and gains with
  1.0 (`@todo: Fetch calibration constants`), and `EpixUHR3x2` needs none because
  its data arrives calibrated from firmware.  Needs a real source and a point in the
  transition sequence to load from it.  Three candidate routes:

  1. reuse Mikhail's code in `lcls2/psana`, which is the source of truth;
  2. resurrect `lcls2/psalg/psalg/calib/`, also Mikhail's, whose headers are still
     there (`CalibPars.hh`, `CalibParsDB*.hh`, `CalibParsStore.hh` and friends);
  3. follow Gabriel's pseudo-code, reproduced verbatim in the appendix at the bottom
     of this file.

  In outline: detector type plus serial number to a "short name"; a metadata query on
  that short name, ordered by run, pointing at the bulk data, needing filtering on
  run number and validity flags; then the bulk fetch.  The appendix notes the calibdb
  schema is documented nowhere else, which is why it is kept here.

- **Calibration mode.**  The DAQ operator selects the **CALIB** alias instead of the
  usual **BEAM** alias.  That selects a different, perhaps derived, set of detector
  register settings, written to the detector through
  `configdb/<detector>_config.py` during Calibrate.  The DRP has to recognise that
  this state is active and record **raw** data as it comes off the detector's fibre,
  rather than calibrated or reduced data, with the reducer bypassed or turned into a
  no-op.  Low rate running is acceptable in this mode.

- **Prescale implementation.**  Record raw data *in addition to* the normal reduced
  data, at low rate, while the normal stream continues at full rate (33 kHz or
  whatever).  The signal is **`keepRaw`**, bit 22 of the datagram's env word —
  `Pds::EbDgram`'s accessor already exists, `psdaq/service/EbDgram.hh:57`,
  `return (env>>22)&1` — asserted at typically 1 Hz.  Each detector's `event()`
  carries the matching `@todo: Deal with prescaled raw for the panel here?`.

  Two things to work out.  The XTC headers have to describe the extra contribution.
  And the buffering: either extend the reducer buffers to hold raw alongside reduced,
  or keep many fewer look-aside buffers and do multiple file writes, scatter-gather
  or similar.  The second trades memory for write complexity, and interacts with the
  recorder item above.

## Runtime behaviour

- **Is the spin-loop the right idea?**  `_waitForDMA` polls the DMA doorbell with a
  `__nanosleep` backoff (8 ns doubling to 256 ns) and gives up so the graph can
  relaunch; `_readerLoop` relaunches itself with `cudaGraphLaunch(...,
  cudaStreamGraphTailLaunch)`.  Worth asking whether that is optimal, and whether to
  spin more or less.  Note the consequence already observed: **a GPU running this
  reads 100% utilisation whether or not data is flowing**, so `nvidia-smi` says
  nothing about real work.  History suggests this was already revisited once —
  f08eacc1 "Relaunch instead of spinning", then 3db054e6 "Revert to separate
  kernels; Time out spin loops".

- **Idle when trigger rates are low**, as the CPU DRP does.  Related to the spin
  loop: at low rates the present arrangement burns a GPU continuously to wait.

## Operations

- **`CpuSpecList` does not reserve whole cores when hyperthreading is on.**  Found on
  drp-srcf-gpu006 on 2026-09-11, and it is a concrete mechanism for the open IT ticket
  about Slurm scheduling onto cores already saturated by WEKA.  The agreement is that
  core 0 is the OS and cores 1-3 are WEKA, expressed as `CpuSpecList=0-3`.  But that
  list is in *CPU* indices, and on gpu006 `cpu0`'s siblings are `0,64`, `cpu1`'s are
  `1,65`, and so on — so CPUs 64-67 are the second thread of those very same physical
  cores and remain schedulable.  Slurm can and will place work on the execution
  resources WEKA is pinning at 100%.

  Two fixes, in preference order:

  - Turn hyperthreading off in the BIOS, which was IT's original instruction and was
    not done on many nodes.  gpu008 has it off.
  - Failing that, extend the reservation to cover the siblings:
    `CpuSpecList=0-3,64-67`.  This needs no BIOS change and is provably right rather
    than relying on the unverified `ThreadsPerCore=1` behaviour.  Note that setting
    `ThreadsPerCore=1` while leaving `CPUs=128` makes the declaration
    self-contradictory (2 sockets x 32 cores x 1 thread = 64), so `CPUs` would have to
    drop to 64 as well.

  `gen_gres_conf`'s `Cores=` output is unaffected either way: it is in core-index
  space, which depends only on sockets x cores-per-socket.

- **datadev driver install at boot via dkms**, so a kernel update does not leave a
  node without its driver, and so the module parameters live in one declared place
  instead of in whoever's copy of `comp_and_load_drivers` ran last.  Wanted in
  `/etc/modprobe.d/datadev.conf`:

  ```
  options datadev cfgDevName=1 cfgMode=2 cfgCont=0 cfgTxCount=4 cfgRxCount=1020 cfgSize=4096
  ```

  `cfgDevName=1` gives the `/dev/datadev_XX` hex bus-number names that `gen_gres_conf`
  keys its `Type=` names off.  **Anything that hardwires `datadev_0` breaks under it.**
  One such was found and fixed on 2026-09-12: `xpmdet_config.py`'s `detect_C1100()`
  opened `/proc/datadev_0` literally, and on a `cfgDevName=1` node the open failed and
  it *returned False* — reporting a C1100 as a KCU1500.  That built the wrong rogue
  tree, whose `refClockRate()` reads 0.0, which is outside every timebase range, so
  `xpmdet_connectionInfo()` went on to program a Si570 the C1100 does not have and
  divided by its zero crystal frequency.  The visible symptom was a `ZeroDivisionError`
  in `_Si570.py`, four steps from the cause; the only clue was one line
  `ERROR:root:Error: File '/proc/datadev_0' not found.` early in the DRP log.  It now
  derives the name from the device it was given, and raises rather than guessing.
  This affects the **CPU** DRPs equally, so grep for other hardwired device names
  before rolling `cfgDevName=1` out more widely.  `cfgMode=2` is `BUFF_STREAM` (`dma_buffer.h:38`),
  i.e. `dma_map_single` with explicit cache synchronisation, rather than the
  `BUFF_COHERENT` default.  The dkms recipe must also pin `DATA_GPU=1` — see below.

  Note that these parameters govern the **CPU-side** DMA buffers only.  The GPU DRP's
  buffers are the ones registered with `gpuAddNvidiaMemory()`, sized by `drp_gpu`, so
  `cfgSize` does not bound them.  Both paths coexist on one card: the GPU DRP is
  restricted to lane 0 and CPU DRPs use any other lane, which is how the ePixUHRemu
  work has been tested — a GPU DRP on lane 0 at ~200 kB per DMA alongside a timing
  CPU DRP on lane 1 happy with 4 kB.  The firmware was confirmed to assert the
  overflow bit when a GPU DMA exceeds its registered buffer.

  `cfgCont=0` deserves its own note, since it differs from the driver's default of 1
  and from current CPU-node practice.  With continuation enabled, an oversized frame
  spans buffers (`AxiStreamDmaV2Write.vhd:325`); with it disabled the write engine
  asserts `overflow` in the `DmaDsc` and sets `dropEn`, discarding the rest of the
  frame.  No DRP reassembles a continued frame — `TrgInpGen.cu`'s
  `dmaDsc->header ^ ~dmaDsc->errorMask()` test rejects any header bit other than SOF,
  and `cont` is bit 3 — so continuation can only produce descriptors the code throws
  out, while `overflow` is a condition it already tests.  This has bitten before.
  Riccardo is being asked whether the CPU nodes' ansible should change to match; the
  GPU sample sets it regardless.

- **An installer that checks the lcls2, driver and firmware builds against the current
  minimum versions.**  This is the right home for consistency checking; the
  alternative is every tool growing its own anomaly detection.  Two traps it should
  cover, both found on drp-srcf-gpu006 on 2026-09-11:

  - The driver only probes the GpuAsyncCore version register when compiled with
    `DATA_GPU` (`gpu_async.c:48`), which `aes-stream-drivers` enables by setting
    `NVIDIA_DRIVERS` (`data_dev/driver/Makefile:88`).  Both builds install as
    `datadev.ko`, so `lsmod` and `modinfo` cannot tell them apart — only
    `GPUAsync Support` in `/proc/datadev_*` (`dma_common.c:1454`) can.  A node can
    look healthy and silently be unable to run `drp_gpu`.
  - Without `DATA_GPU` every card reports `GPU Async En : 0`, which reads as a
    firmware fault and is not one.  Diagnosing firmware requires the right driver
    loaded first.

- ~~Does the ePixUHR3x2 emulator firmware still support GPU DMA?~~  **Resolved
  2026-09-11: yes, it does.**  With the `DATA_GPU` driver loaded, all three cards on
  drp-srcf-gpu006 report `GPU Async En : 1`, `GpuAsyncCore Version : 5` and a
  `DataGPU State` section — `ePixUHR3x2XilinxVariumC1100` on `a1` and `d5`,
  `InterCardTestXilinxVariumC1100` on `84`.  Nothing was reverted; the earlier
  `GPU Async En : 0` was entirely the wrong driver build.  Recorded because the
  reasoning generalises: a firmware capability read through a driver that does not
  probe for it is not evidence about the firmware.

- **Drop CAP_SYS_ADMIN once the registers are mapped.**  `drp_gpu` needs it only for
  the one `cuMemHostRegister(..., CU_MEMHOSTREGISTER_IOMEMORY)` call in
  `MemPool.cc`.  Afterwards the process could remove it from its effective and
  permitted sets, and clear the ambient set, so the capability is not held for the
  lifetime of the run.  Looks straightforward: `capset()` with the bit cleared, plus
  `prctl(PR_CAP_AMBIENT, PR_CAP_AMBIENT_CLEAR_ALL)`, right after the mapping
  succeeds.  Worth confirming nothing later in startup needs it — the mapping is
  early, but `gpuAddNvidiaMemory` and the DMA setup come after.

- **Where the datadev's missing bandwidth goes.**  Cards achieve 102.3 Gbps of the
  126 Gbps a PCIe 4.0 x8 link raw-rates at, i.e. 81%.  Some of that is protocol
  overhead, but the PCIe **MaxPayloadSize** and **MaxReadRequestSize** are worth
  checking, in the BIOS and with `lspci -vv` (`DevCtl: MaxPayload ... MaxReadReq`).
  A small MaxPayloadSize costs a lot of efficiency.  Note "MPS" here means PCIe
  MaxPayloadSize, not CUDA's Multi-Process Service.

- **IT ticket: CUDA and driver mismatch on the sdfada nodes.**  See below.

## Hardware

- **GPU5 on gpu008 drops off the PCIe bus.**  Root cause is *not* the GPU:
  `pciehp` reports `Slot(2002): Link Down` then `Card not present` on switch
  downstream port `d2:01.0`, and the NVIDIA driver removes the device in response
  (Xid 79 raised from `irq/82-pciehp`).  Last occurrence 2026-09-10 09:17:58, on an
  idle machine 14 hours after the DAQ exited, with the card unused — so not load or
  thermal.  Reseat physical slot 2002; if it recurs with no physical cause,
  disabling PCIe hotplug on that port is defensible, since nobody hot-plugs these.
  A `pciehp` behaviour change across the RHEL 7 to Rocky 9 upgrade is a candidate.

- **sdfada CUDA and driver mismatch (S3DF, not a DAQ node).**  IT report all sdfada
  nodes carry the same driver (575) and CUDA (12.9) packages.  Measured differently:
  `sdfada016` has `/usr/local/cuda` -> `/usr/local/cuda-13.2` with `nvcc` reporting
  13.2, while the driver caps at 12.9, so anything built there dies at runtime with
  "CUDA driver version is insufficient for CUDA runtime version" — a major-version
  gap, which minor-version compatibility does not bridge.  `sdfada019` had no
  `/usr/local/cuda*` at all, so the nodes were not identical when checked.  Either
  the 13.2 tree should not be there, or `/usr/local/cuda` should not point at it, or
  the driver should be one that supports 13.x.  **No longer blocking**: gpu008 pairs
  toolkit 13.3 with driver 595, so development moved there.  Worth reporting so the
  next person does not lose an afternoon.

- **`datadev_6` (`a1:00.0`)** runs `XilinxVariumC1100Pgp4_10Gbps` rather than
  `DrpTDetGpuC1100NonBifurcated` and reports `GPU Async En = 0`, and sits on a root
  complex with no GPU.  It cannot be used by the GPU DRP; `gen_gres_conf.py`
  excludes it on firmware rather than by address.


---

## Appendix: retrieving calibration constants from calibdb

Reproduced verbatim from Gabriel's explanation, because the database schema is not
documented anywhere else.  Relevant to the "Fetch calibration constants" item
above, option 3.  Lightly formatted only: the prose and code are unaltered, typos
included.

> Just regarding the calibdb constants loading, I don't think the database schema is
> documented anywhere, but there are 3 parts essentially:
>
> Convert a detector type and a detector serial number into a "short name" which is
> used for constants lookup.
>
> Using the "shortname" query for metadata. This is ordered by run, and has an entry
> pointing to the bulk data. In general, would need to filter on run number,
> validity flags and so on.  This can be done either in the "experiment" database,
> or if that fails, go to the backup "detector" database.
>
> Using the poitner from step 2, retrieve the bulk data.
>
> The rough outline in pseudo-ish code, using rapidjson/cpp-httplib as the examples
> would be along the lines of:

```cpp
// -------------------------- SHORTNAME QUERIES ------------------------------- //
// Transform a serial number and detector type into a "short name" for lookup
httplib::Client cli("https://pswww.slac.stanford.edu");
std::string shortname_endpoint = "/calib_ws/cdb_detnames/" + det_type; // det_type is epixuhr3x2 etc.

if (auto res = cli.Get(shortname_endpoint)) {
  rapidjson::Document docs;
  docs.Parse(res->body.c_str());
  if (!docs.IsArray()) {
    return; // We expect a list of documents returned from this API endpoint
  }

  for (const auto& doc : docs.GetArray()) {
    if (!doc.IsObject()) continue; // Looking for sub-JSON dicts

    std::string doc_ser_no = docs["long"].GetString();
    if (doc_ser_no == det_ser_no) {
      return doc["short"].GetString();
    }
  }
}
```

> Then with the short name you query for the actual constants -- you have to first
> retrieve a reference to the actual object entry. The metadata is stored
> independently of the bulk data:

```cpp
// ----------------- METADATA QUERIES -----------------------------//
httplib::Client cli("https://pswww.slac.stanford.edu");

// Can try first the "experiment" database, and then the "Detector" database.
std::string experiment_endpoint = "/calib_ws/" + db_in_use + "/" + det_short_name;
std::string data_doc_id; // Bulk data pointer
std::string data_type;   // Is it an array, or string etc.
std::string data_dtype;  // Element type
std::size_t data_ndim;
std::size_t data_nelem;
if (auto res = cli.Get(endpoint)) {
  if (!docs.IsArray()) {
    return; // We expect a list of documents returned from this API endpoint
  }

  for (const auto& doc : docs.GetArray()) {
    if (!doc.IsObject()) continue; // Looking for sub-JSON dicts

    std::string doc_constants_type = docs["ctype"].GetString();
    if (doc_constants_type == target_type) { // Target type is the constants you want... E.g. "pedestals"
      data_doc_id = doc["id_data"].GetString();
      data_type = doc["data_type"].GetString();
      data_dtype = doc["data_dtype"].GetString();

      data_ndim = static_cast<std::size_t>(std::atoi(doc["data_ndim"].GetString()));
      data_nelem = static_cast<std::size_t>(std::atoi(doc["data_size"].GetString()));
    }
  }
}
```

> Finally, bulk data retreival:

```cpp
std::string data_endpoint = "/calib_ws/" + db_in_use + "/gridfs/" + data_doc_id; // From the metadata step. db_in_use is eitehr the experimetn or detector database

if (auto res = cli.Get(data_endpoint)) {
  auto* raw_data { reinterpret_cast<const unsigned char*>(res->body.data()) }; // This is it.... just parse now (except XTCAV)
  if (data_type == "ndarray) {
    if (data_dtype == "float32") {
      std::memcpy(constants_buf.data(), raw_data, data_nelem * sizeof(float));
    } else if (data_dtype == "float64") { /* and so on... for all data types... */ }
}
```

> I'm not sure in what state the psalg code is now, but I imagine it must do this as
> well (although database schema may have changed over time.) Regardless,
> procedurally, this is what is done. Even including various checks, its not all that
> much code. The most complex part is the serial number matching at the beginning, as
> in the DRP you may not have the full serial number (unless the code were updated to
> make that accessible).

Note for whoever picks this up: Gabriel's last point is the one to check first.  The DRP
may not have the full serial number to match on, which would need the code made to
expose it.  `Parameters::serNo` exists and is passed to `Names` during configure, so
start there.
