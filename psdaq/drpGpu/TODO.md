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

- **gpu006 is published** as of 2026-09-14, the first node converted.  Slurm now
  advertises `gpu:dda1:1(S:1),gpu:ddd5:1(S:1)`; the `(S:1)` confirms it mapped the
  emitted `Cores=32-63` to socket 1, which is where both GPUs are.  `GresTypes` was left
  alone, since only the GPU is declared.

  **Converting a node drains it, and the drain has to be cleared by hand.**  Expect
  `State=UNKNOWN+DRAIN+INVALID_REG` in passing and then `State=IDLE+DRAIN` with
  `Reason=gres/gpu:ddXX count too low (0 < 1)`.  The two config files are distributed
  together, but the controller applies its view and `slurmd` reloads at slightly
  different moments, so there is a window in which the declared count exceeds the
  detected one.  The reason is stale by the time you read it; Slurm never clears a drain
  by itself:

      sudo scontrol update NodeName=drp-srcf-gpuNNN State=RESUME

  If that same reason returns immediately afterwards, `slurmd` has not reloaded
  `gres.conf` -- `sudo systemctl restart slurmd` on the node, then resume again.  The
  documentation does not say whether `scontrol reconfigure` suffices for that file, and
  on gpu006 it did.  A *persistent* `INVALID_REG`, or that reason surviving a slurmd
  restart, would be the real thing.

  Still to do: gpu001 (one card, one A5000, so `--expect 1` and no `--exclude`), then the
  rest as they are built.  Nothing yet tests that a DAQ process is actually allocated the
  paired GPU -- that wants a run on gpu006 with `nvidia-smi` showing two GPUs in use
  rather than one.

- **`gres.conf` publishing.**  `psdaq/psdaq/slurm/gen_gres_conf.py` derives the
  pairing; publishing is a deliberate paste.  Run on the node after every datadev
  driver load:

      gen_gres_conf --expect N [--exclude CARDS]

  then paste the output into `psslurmctld001:/etc/slurm/gres.conf`, replacing any
  existing lines for that node, and `sudo scontrol reconfigure` there.  `slurm.conf`
  needs the node's `Gres=` line to agree; the tool prints the exact string on stderr.
  Afterwards, `--check` compares the node's distributed copy against its hardware and
  exits non-zero on any difference.

  Stdout is exactly the file content and nothing else; every instruction, count and
  warning goes to stderr.  So `2>/dev/null` yields the block alone and `>/dev/null` the
  instructions alone.  The block itself is two lines per pairing plus one header and the
  cut-here markers, because it repeats per node: at twenty-odd nodes a paragraph of
  preamble each would make the central file unreadable, which is exactly the objection to
  generating it at all.  The reasoning lives on the Confluence page instead.

  It deliberately does **not** write to `gres.conf`.  Earlier versions spliced their
  output in, retiring superseded lines and recording the base file's checksum, which was
  disproportionate: the whole central file is about two dozen lines for nine nodes, so it
  can be read at a glance, and editing it by hand is the recovery path when hardware
  breaks at three in the morning.  A tool that rewrites a shared file has to be
  understood before it can be trusted.  Roughly 200 lines went with that decision, and
  another 50 with the verbose block.

  Note what the existing file already contains: hand written per-node lines with
  `Type=nvidia_h200_nvl`, and a comment recording that "cpo and claus have a guess
  that we should put individual lines with /dev/nvidiaN lines so that slurm can be
  used to select specific gpus to pair with specific datadev fpga boards".  That is
  this task.  Model based type names cannot express the pairing, because every H200
  on the node has the same model name; bus derived names (`dd04`) can.

  That naming is also what makes the 3 a.m. recovery cheap.  `ddXX` is bound to the
  *card*, and which GPU serves it is decided solely by the `File=` on its `Name=gpu`
  line, so moving a card to a spare GPU is one edit to one line plus a reconfigure — no
  `slurm.conf` change, no DAQ configuration change.  Worth provisioning N+1 GPUs per node
  for that reason: had gpu008 had a spare, GPU5's death would have been that one-line fix
  instead of the episode that motivated the degraded-mode machinery.

- ~~**The `g` flag conflicts with a gres request.**~~  Done in `utils.py`.  `get_gres()`
  derives the request from the process's own `-d /dev/datadev_XX`, so
  `#SBATCH --gres=gpu:ddXX:1` replaces the `--gpus-per-task=1 --gpus=1` kludge.  Request
  and device name come from one string and cannot drift apart.  Verified by rendering
  gpu6.py: the two GPU processes get `gpu:dda1:1` and `gpu:ddd5:1`, the CPU DRP and the
  TEB get no GPU request at all.  **Inert until the records are published**, since a
  request for a type that does not exist pends for ever.

  Three decisions worth keeping:

  - The `--gpus` fallback is *not* kept.  An arbitrary GPU is the failure this exists to
    prevent, and asking for none fails immediately in CUDA init rather than running
    slower than it should for reasons nobody can see.
  - The device suffix must be exactly two hex digits, which is what `cfgDevName=1`
    produces and what `gen_gres_conf` derives type names from.  A single digit means
    probe-order naming, so `dd1` would request a type existing nowhere; `get_gres()`
    warns and declines instead.  That is why the CPU `timing_0` on cmp008, which uses
    `-d /dev/datadev_1`, is untouched.
  - `generate_as_step()` is deliberately not changed.  It handles no flags at all, and
    in step mode one allocation covers several processes on a node that want *different*
    gres, so the header would need the union while each `srun` requests its share.
    `as_step` defaults to False and has no known user; `generate()` is the worked
    example if one appears.

- **The datadev is not a gres, and `ConstrainDevices=yes` is why.**  Declaring
  `Name=datadev ... File=/dev/datadev_XX` would let Slurm catch a `.cnf.py` that hands
  two processes the same card -- something that has bitten us more than once.  It is
  deliberately not done: `cgroup.conf` sets `ConstrainDevices=yes`, which per
  `cgroup.conf(5)` constrains "the job's allowed devices based on GRES allocated
  resources".  Naming the datadev would therefore *deny* a job access to any card it was
  not allocated, breaking the arrangement in use on gpu001 and gpu006 -- a GPU DRP on
  lane 0 of a card and a CPU DRP on another lane of the same one.  The second process
  cannot request the same `datadev:ddXX:1`, the count being one.

  So only the GPU is declared, which also means `GresTypes=gpu` suffices and needs no
  change.  Worth revisiting once there is experience of how the GPU allocation behaves:
  the duplicate-`-d` check is worth having, but as a lint over the `.cnf.py`, which costs
  nothing and breaks nothing.

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

- **Nothing coordinates the green context split with the kernels' launch geometry.**
  There are three independent hard-coded SM tables, and they disagree:

  | where | SMs it assumes | context it runs in |
  |---|---|---|
  | `PGPDetector.cc:615` green split | 6 / 40 / remainder | — |
  | `Reader.cu:528`, for `_event` | 6 at `tpSM` 1536, **8** at 2048 | ctx 0, which has **6**, shared with TrgInpGen |
  | `NoOpReducer.cu:158`, for `_reduce` | **20** at 1536, **10** at 2048 | ctx 1, which has **40** |

  `m_green_ctx[0]` goes to both the Reader and TrgInpGen, `[1]` to the Reducer, `[2]`
  (the remainder) to `TebReceiver::_recorder`.  So on an H200, `_event` asks for eight
  SMs' worth of blocks inside a six-SM context it also shares, while `_reduce` uses a
  quarter of its forty.  The two tables even scale in opposite directions as `tpSM`
  rises — 6→8 against 20→10 — which reads like independent tuning at different times
  rather than a plan.  `Reader.cu:523` and `NoOpReducer.cu:167` both already carry a
  to-do saying as much.

  Three further consequences of the numbers being absolute rather than derived:

  - The split is 6 + 40 + remainder regardless of the device, so moving from an A5000
    (64 SMs) to an H200 (132) leaves the remainder context 86 SMs instead of 18.  All
    the extra capacity silently lands on the recorder, which is unlikely to be the
    intended balance.
  - Both `switch (tpSM)` statements `abort()` on anything unrecognised, so a new GPU
    generation stops the DRP with "Unexpected number of threads per MultiProcessor"
    rather than falling back to something sane.
  - Both call `cudaGetDeviceProperties(&prop, 0)` with the device hard-coded, while
    `MemPoolGpu` honours a `gpuId` kwarg.  Harmless under Slurm, which renumbers
    `CUDA_VISIBLE_DEVICES` so the allocated GPU is always index 0, but inconsistent.

  The fix is for one place to own the partitioning and hand each component the SM count
  of the context it was given — `cudaExecutionCtxGetDevResource()` already returns it,
  and `_setupGreenContexts()` logs it.  Each stage then derives blocks and threads from
  that rather than from a table.  Note the Reader and TrgInpGen share a context, so
  whatever owns this has to divide their allocation, not just report it; TrgInpGen's and
  the Reducer's own driver kernels are `<<<1, 1>>>` persistent loops, so they want about
  one SM each, and the bulk belongs to `_event` and `_reduce`.

- ~~**Clear the firmware counters once the timing link is up.**~~  Done in
  `epixuhremu_config.py`; **untested**, wants a run on gpu001 to confirm the counts in
  the log are small.

  After `ConfigLclsTimingV2()` the counters held whatever the link accumulated while
  training, which says nothing about the run about to start: the successful gpu001 run
  logged `FidCount 2347064`, `RxDecErrs 6009085` immediately afterwards.  They were
  already being cleared — `xpmdet_connectionInfo()` calls `ClearRxCounters()` itself —
  but only *after* `dumpTiming()` had logged them, so the reported numbers were noise.

  `tim.ClearRxCounters()` now runs in the hook, but only when `RxLinkUp` confirms the
  link came up.  If it is still down the counts are left alone, because then they are the
  evidence.  Only the `TimingFrameRx` counters are wanted; the `TriggerEventBuffer` ones
  are separate and not of interest.  Note `TimingFrameRx.countReset()` is merely an alias
  for `ClearRxCounters()` (`TimingFrameRx.py:264`), so there is no third thing to call.


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

- ~~**The GPU dkms build failed silently, producing a non-GPU module.**~~  **Fixed
  upstream: `slaclab/aes-stream-drivers` PR #319, merged to `pre-release` 2026-09-14.**
  Root cause was kbuild's two-pass evaluation: `data_dev/driver/Makefile` pulled in
  `Makefile.local` by a bare relative path, which resolves in the top-level pass but not
  in the sub-make whose cwd is `$(KERNELDIR)`, where `ccflags-y` is evaluated.  So
  `NVIDIA_DRIVERS` was empty exactly where `DATA_GPU` was decided, and
  `datadev-gpu-dkms` had **never** produced a GPU-enabled module.  The fix anchors the
  include to the Makefile's own directory, makes `build-nvidia.sh` refuse to skip the
  NVIDIA build unless `ALLOW_NO_NVIDIA=1`, and adds a `POST_BUILD` guard
  (`check-gpu-build.sh`) that greps the built module for `GPUAsync Support : Enabled`
  and fails closed when `Makefile.local` is absent.

  What this means for us now that it is merged:

  - `datadev-gpu.conf` and `dkms-reload.sh` are upstream, so the next driver install
    should take them from `pre-release` rather than from a local branch.
  - No header changed and `DMA_VERSION` is still `0x06`, so the seven headers vendored
    into `psdaq/psdaq/aes-stream-drivers/` remain byte-identical to upstream and there
    is nothing to re-vendor.  (`DmaDest.h` there is ours, not upstream.)
  - Worth rebuilding the driver from `pre-release` on gpu006 and gpu001 to confirm the
    *merged* form still yields `GPUAsync Support : Enabled`, then deleting the
    `pr-require-nvidia-for-gpu-build` branch.  Requires an sdfiana node: DAQ nodes
    cannot reach GitHub.

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

- ~~**Drop CAP_SYS_ADMIN once the registers are mapped.**~~  **Done, and verified on
  drp-srcf-gpu001 on 2026-09-13** across three runs including a Deallocate and a
  Reset:

  ```
  Privilege: uid 1085, euid 1085, CapEff 0x0000000000200000, CAP_SYS_ADMIN yes
  Dropped CAP_SYS_ADMIN; CapEff now 0x0000000000000000
  ```

  `_dropPrivilege()` in `MemPool.cc` runs immediately after the
  `cuMemHostRegister(..., CU_MEMHOSTREGISTER_IOMEMORY)` call, clearing the ambient set
  and then the bit from effective, permitted *and* inheritable, so it cannot be raised
  again.  It re-reads `CapEff` afterwards and warns if the bit survived, rather than
  assuming.

  What made the tight placement safe: the mapping is made once, in `MemPoolGpu`'s
  constructor with a single panel, and is not redone on any transition — Configure and
  Unconfigure allocate ordinary device and host buffers, not I/O memory.  And the
  datadev driver checks no capabilities at all, only ownership by thread group
  (`grep -rn 'capable(\|CAP_SYS' common/driver/ data_dev/driver/src/` is empty), so
  `gpuAddNvidiaMemory()` and the later ioctls do not need it.

  Note it is a real improvement only for the `setpriv` and `setcap` routes.  Under
  setuid root the kernel restores a root process's capabilities across an `exec`, so
  the code warns when `euid` is 0 that dropping the capability is not dropping
  privilege.

- **`pgpread`: switch it to the `HOST_REARMS_DMA` pattern, or retire it.**  It is a
  light-weight, detector-agnostic tool for diagnosing whether data is arriving, and it
  has no performance constraint, so it has no reason to need a privilege.  It is
  currently the only caller of `gpuInitBufferState()` → `gpuMapFpgaMem()` →
  `cuMemHostRegister(..., CU_MEMHOSTREGISTER_IOMEMORY)` in `GpuAsyncLib.cc`, which is
  the tree's second privileged mapping and the reason that path exists at all.  Having
  the CPU rearm the buffers instead would let it run unprivileged and would leave
  `MemPool.cc` as the only place needing `CAP_SYS_ADMIN`.

  Retiring it is the other option: `aes-stream-drivers` now ships `rdmaTest`, which
  appears to cover the same diagnostic ground, and `pgpread` is drifting stale.  What
  argues for keeping it is that colleagues find it an easier sandbox to modify than
  `rdmaTest`.  Someone should confirm `rdmaTest` really is a superset before deleting
  anything.  Either way the status quo is the one option with no upside: a stale tool
  that also keeps a privileged code path alive.

- **Where the datadev's missing bandwidth goes** — largely answered, on 2026-09-14,
  and it was the MaxPayloadSize as this item guessed.  Cards achieve 102.3 Gbps of the
  126 Gbps a PCIe 4.0 x8 link raw-rates at, i.e. 81%.  Mudit or Jeremy found that a
  configuration on gpu008 reached **113 Gbps**, which is 90%.

  The mechanism: PCIe **MaxPayloadSize** is constrained by whichever device in a
  hierarchy needs the lowest value, and Linux's default policy sets it to the smallest
  common value across the tree.  The GPU is the limiter, apparently 256 bytes; the
  datadev can do 1024.  So a root complex carrying both forces 256 on the datadev too.
  Put the datadevs on their own root complex and they run at 1024, which is where the
  extra 11 Gbps comes from.  ("MPS" here is PCIe MaxPayloadSize, not CUDA's
  Multi-Process Service.)

  Chris therefore proposes putting **all GPUs on one root complex and all datadevs on
  the other**.  Two things to settle first.

  **The NVIDIA objection is probably about correctness, not performance.**  NVIDIA
  recommends against this arrangement, reportedly as less likely to work and slower --
  and the measurement contradicts the second half, which may mean it is answering a
  different objection than the one being made.  Mismatched MaxPayloadSize across a
  peer-to-peer path is a validity problem: a TLP carrying 1024 bytes cannot be forwarded
  into a hierarchy whose MPS is 256, which is exactly why the default policy levels it
  down.  That 113 Gbps works could mean the root complex splits oversized TLPs, or that
  the datadev's writes into GPU BAR space are under 256 bytes anyway so the larger MPS
  only helps its host-memory traffic, or that it works by luck and fails rarely and
  data-dependently.  For a DAQ the last is the one that ruins a beamtime months later.

  The cheap test is AER, which counts exactly this failure:

      sudo lspci -vv | grep -E "MaxPayload|MaxReadReq"   # what is actually set
      cat /sys/bus/pci/devices/0000:*/aer_dev_nonfatal   # counters, if AER is enabled
      dmesg | grep -iE "aer|malformed|unsupported request"

  Run the 113 Gbps configuration hard and check those stay static.  Clean AER over a
  long run is decent evidence it is genuinely fine; a slow trickle settles it the other
  way.  A throughput number cannot answer this on its own.  Note gpu006 and gpu008 boot
  with `iommu=off`, so there is no translation layer policing payload sizes either.

  **It would make `gen_gres_conf`'s locality logic vestigial.**  With every GPU on one
  root complex and every card on the other, no pairing is local by construction:
  `shared_depth()` returns 0 or 1 for every pair, all of them print
  `*** DIFFERENT PCIe switch ***`, and the two-pass preference in `pair()` has nothing
  to prefer.  The tool still does the job that matters -- deterministic one-to-one
  pairing, and the one-line spare swap -- but `LOCAL_DEPTH`, the preference pass and the
  warning should then go, because a warning that fires on every pair trains people to
  skim past it.  Reword the generated comment to say the pairing is for determinism
  rather than proximity.

  Separately, Cheolhong proposed measuring with `amd_uncore`/`perf` whether cross-socket
  peer-to-peer traffic bypasses host memory.  That is a different question from the MPS
  one and the two measurements are independent.  Note that the inter-socket hop on these
  AMD EPYC boxes is Infinity Fabric, not PCIe; Ric measured it sustaining a card's full
  rate by running each TDet datadev on gpu008 against GPU0 in turn, every combination at
  33 kHz.

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

## The ePixUHR3x2 Configure failure is a missing clock, not a dead board

Diagnosed 2026-09-14 on drp-srcf-gpu006.  Configure aborts in
`lcls2_pgp_fw_lib/shared/_TimingRx.py:128`, on the first write of
`ConfigLclsTimingBase`:

    self.TimingFrameRx.ModeSelEn.setDisp('UseClkSel')

    rogue.GeneralError: ... Transaction error for block
    Root.ROS[0].FebFpga.App.TimingRx.TimingFrameRx.ClearRxCounters with address
    0x8e080020.  Error Timeout waiting for register transaction 62 message response.

The block is named for `ClearRxCounters` rather than `ModeSelEn` because rogue coalesces
adjacent variables into one block and reports the error against the block; the traceback
is what says which write was attempted.

Two quite different faults give that identical timeout, and the DRP log cannot separate
them, because the only FEB access before it is `Core.SystemDevices.Si5345Pll.Page0.LOL`
and nothing else under `App` is ever touched.  `_initial_power_up()` runs *after* the
timing configuration, so it never gets the chance.  Note also that the hundreds of
`setPollInterval(1)` lines in the log are local rogue tree configuration, **not** bus
traffic: `epixuhr3x2_config.py:196` sets `pollEn=False`, so nothing was ever polled and
their clean record is not evidence of health.

Resolved by `probe_feb.py` (kept in the session directory), which reads `App` registers
outside `TimingRx` with `Core` as a control and nothing written:

- `Core.SystemDevices.Si5345Pll` answers and the PLL is locked, so the FEB has power and
  a working control path.
- `App.BoardCtrl3x2Readout.LTM4664_*` and `App.AxiAds1217Core` answer, so the `App`
  branch is alive and out of reset.
- `App.TimingRx.TimingFrameRx` does not answer.

**So the TimingRx clock domain has no clock.** An AXI-Lite transaction into an unclocked
domain can never complete, which is exactly a timeout and not an error response.  That
makes this a timing problem rather than a detector problem: the board being powered off,
which was the leading hypothesis while Gabriel was away, is ruled out.  The question to
chase is where the FEB's timing clock is meant to come from and why it is absent, not
the `LTM4664` regulators.

Worth keeping the general shape: a register timeout says only that nobody drove a
response, and "unpowered", "held in reset" and "unclocked" are indistinguishable from one
transaction.  Probing a sibling branch and a known-good control separates them cheaply.

## Real-time priority is denied on the DRP nodes

Every GPU DRP log since at least 2026-09-13 opens with

    <C> Inadequate RTPRIO limit: got 0, require 99

so the DRP threads run at normal priority.  Not a correctness problem, and not the cause
of any failure seen so far, but it will bound achievable rate.  Ric raised an IT ticket
for this a few days before 2026-09-14 --
[ECS-11217](https://jira.slac.stanford.edu/browse/ECS-11217) -- since the `RLIMIT_RTPRIO`
ceiling has to be raised in IT's ansible and cannot be set from our side.  Recorded so
that a future rate shortfall is not misattributed.

## Appendix: running the ePixUHR3x2 emulator, from Gabriel

Notes Gabriel sent on Slack on 2026-08-24, kept here because Slack is not a record.
His words, lightly reflowed; the observations under each are mine, from checking the
tree on 2026-09-13.

> There are two minor tweaks needed to work with the emulator - I wasn't sure if this
> should be long term added to configdb or not so for now its just manual:
>
> `epixuhr3x2_config.py:183` — that bool needs to be set to True otherwise it will try
> to initialize asics which don't exist and crash.
>
> `epixuhr3x2.py:81` — this is currently hard-coded to use the CPU path.  There are
> startup problems sometimes and it gets latched onto the wrong data path, so I've had
> this routine that will toggle it back onto CPU.  Presumably that would need to set
> `use_cpu = False` to do GPU development.

Both are still as described.  `emulator: bool = False` at `epixuhr3x2_config.py:183`
feeds `emuMode` and `reset_asic_gt()`; the comment beside it says the emulator does not
have all the registers and `emuMode` prevents erroneous access to them.

Note that setting it True does **not** obviously avoid the failure seen on 2026-09-13,
a register transaction timeout on `FebFpga.App.TimingRx.TimingFrameRx.ClearRxCounters`:
`init_board()` calls `ConfigLclsTimingV2()` gated only on `timebase != "119M"`, not on
`self._emulator`.  Whether `emuMode` prunes the tree enough for that call to succeed is
untested.

`_kick_data_path(use_cpu=True)` is called explicitly at `epixuhr3x2.py:653`, commented
"Force use of CPU data path.  Seems to not determine that sometimes."  It sets
`DataDestination` to 0x0 for CPU, 0x1 for GPU, so **GPU work needs `use_cpu=False`**.
Until that changes, data goes to the CPU no matter what else is configured, and the GPU
path cannot be exercised at all.  This is the hard blocker of the two.

> I programmed DAQ:FEH:XPM:4 with the event codes to run the detector on Seq Engine 5.
> In whatever group setup you use, the following needs to hold:
>
> - Timing's readout group should use event code 278
> - The ePixUHR readout group should use event code 277
> - The run trigger should be set to 276 (but this is already setup in configdb so you
>   may not need to change anything, unless reprogramming the sequencer)

> We didn't setup an IOC for the power supply since its going to be switched soon.  I
> wrote a Python program you can download with pip, or if you prefer you can send the
> serial commands over USB directly from ctl-xpp-cam-03 to turn the detector on and off.
>
> - Query the state: `echo ":OUTput:STATe?" > /dev/ttyUSB0`
> - Turn the power on: `echo ":OUTput:STATe ON" > /dev/ttyUSB0`
> - Turn the power off: `echo ":OUTput:STATe OFF" > /dev/ttyUSB0`
>
> You can read responses to your query from another terminal with `cat /dev/ttyUSB0`.
> This can only be done on ctl-xpp-cam-03 since that is the direct USB connection (no
> Moxa, etc.).
>
> I've left the detector off at the moment.

So the detector was off as of 2026-08-24 and its state since is unknown.  Worth querying
before concluding anything from a register timeout.

**Superseded 2026-09-14:** the FEB is powered and its `App` branch answers, so whatever
`ttyUSB0` controls is not what blocks Configure.  See "The ePixUHR3x2 Configure failure
is a missing clock, not a dead board" above.  The two flags below are still needed, but
they are no longer the first thing in the way.

His own reference run, `~dorlhiac/2026/08/24_15:27:42_drp-srcf-gpu006:epixuhr3x2_0.log`,
used the **CPU** `drp`, `-d /dev/datadev_a1`, `-D epixuhr3x2`, `-W 16`,
`-k pebbleBufCount=1024`, and
`SUBMODULEDIR=/sdf/group/lcls/ds/ana/sw/conda2-v4/rel/lcls2_submodules_07202026`.  That
release is the one to use: the March release the DAQ defaults to has no
`epixuhr-3x2-readout-testing` tree at all, so `enable_epix_uhr3x2` raises on import.
