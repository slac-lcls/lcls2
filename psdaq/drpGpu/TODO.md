# GPU DRP to-dos

Working notes for the `features/gpu` branch.  Each item records enough context to
be picked up cold, because the reasoning behind these decisions is otherwise only
in people's heads.

## The GPU nodes, and which are DRP-capable

As of 2026-09-18.  Card counts are from `lspci | grep -i slac`, **not** from
`/proc/datadev_*`: the latter exists only when the driver is loaded, so a node with cards and
no driver looks cardless.  That mistake was made here first, concluding gpu005 had no FPGA
cards when it has six.  Same trap as trusting `/proc/driver/nvidia/gpus/` to prove a GPU is
usable -- `/proc` reports driver state, `lspci` reports hardware.

CPU topology, which matters because `Cores=` depends on it and the nodes are not uniform:

| node | CPU | sockets x cores x threads | NUMA | `slurm.conf` `CPUs=` |
|---|---|---|---|---|
| gpu001 | Xeon E5-2620 v4 | 2 x 8 x 1 | 2 | 16 |
| gpu003 | Xeon E5-2620 v4 | 2 x 8 x 1 | 2 | 16 |
| gpu005 | Xeon Gold 6444Y | 2 x 16 x 2 | 2 | -- |
| gpu006 | EPYC 9355 | 2 x 32 x **2** | **2** | 128 ✓ |
| gpu007 | EPYC 9355 | 2 x 32 x **2** | **2** | **64 -- wrong, see below** |
| gpu008 | EPYC 9355 | 2 x 32 x 1 | **8** | 64 ✓ |

**gpu008 is the outlier, not gpu006/7, and the BIOS version is why.**  All three report the same
board -- `H14DSG-O-CPU` rev 1.01 in an `AS -5126GS-TNRT` -- so Supermicro's on-arrival
motherboard replacement did not leave gpu008 differing from its siblings.  Note that is all it
shows: if the wrong model was delivered fleet-wide then all three report the wrong thing
identically, and the separate question of whether these should be `AS -5126GS-TNRT2` is not
answered by DMI.  The chassis DMI is only partly programmed anyway
(`product_version 0123456789`, `product_sku To be filled by O.E.M.`).

What differs between them is the firmware:

| node | BIOS | date | hyperthreading | NUMA |
|---|---|---|---|---|
| gpu006 | 1.9 | 2026-01-23 | on | NPS=1 |
| gpu007 | 1.9 | 2026-01-23 | on | NPS=1 |
| gpu008 | **2.0** | **2026-04-01** | **off** | **NPS=4** |

gpu008 took a BIOS update in April that the others did not, and its settings changed with it --
either reset to new defaults or configured deliberately at the same time.

**NPS is "NUMA Per Socket", a BIOS setting rather than a property of the silicon.**  An EPYC
socket is several chiplets around an I/O die, with memory controllers and PCIe roots distributed
across it; that physical arrangement is fixed.  NPS decides how finely the BIOS *describes* it
to the OS.  Measured on the two nodes:

| | gpu006, NPS=1 | gpu008, NPS=4 |
|---|---|---|
| nodes | 2, one per socket | 8, four per socket |
| node0 memory | 386 GB, the whole socket | 96 GB, one quadrant |
| node0 CPUs | `0-31,64-95` | `0-7` |
| a card's `numa_node` | socket granularity | quadrant, e.g. `datadev_85` -> 5 |

So NPS=1 aggregates each socket's memory controllers into one node and interleaves across them;
NPS=4 exposes the quadrants, so the OS can tell that a card is nearer some of its own socket's
memory than the rest.  The hardware is the same either way -- NPS=4 is simply more truthful
about it.

That is not academic for us.  NPS=4 is what makes a card's `numa_node` meaningful, and it is
exactly why the `Cores=` socket-boundary bug surfaced only on gpu008: with NPS=1 a NUMA node
*is* a socket, so the wrong definition and the right one coincide.

### BIOS settings for the November boxes: a recommendation to argue with

Nobody in the group has decided this, and Chris's inclination is to take the defaults until
something pushes otherwise.  We know how to update the BIOS, so all 20+ nodes can be made
uniform; the question is what to make them.  A concrete proposal, with the reasoning, so there
is something to disagree with:

**SMT off** -- settled, and for a concrete reason rather than preference.  `CpuSpecList` does
not reserve whole cores when SMT is on: on gpu006, `cpu0`'s sibling is `cpu64`, so
`CpuSpecList=0-3` leaves `64-67` schedulable and the reservation is half-effective.  That is
the open IT ticket about Slurm landing work on WEKA-saturated cores.  With SMT off it
disappears, and the `CPUs=`/`ThreadsPerCore=` bookkeeping stops being a trap -- as gpu007's
`CPUs=64` against 128 hardware threads has just demonstrated.

**NPS=1** -- recommended, but on weaker grounds, and worth measuring before committing twenty
boxes.

The reasoning starts from a correction.  It is tempting to say NPS does not matter because
datadev traffic goes to the GPU rather than to host memory.  The *payload* does, but the DRP
uses pinned host memory in its hot path: `m_hostWrtBufs` (`MemPool.cc:484`) holds the DMA
descriptors, TimingHeaders and TEB input data, mapped so both CPU and GPU see it.  So there is
per-event host traffic, small but on the critical path, and that is where NPS would bite.

The case for NPS=1 is that **the NPS=4 quadrants are too small to place into**:

| | |
|---|---|
| NPS=4 quadrant | 8 cores, 94 GB |
| one GPU DRP asks for | `cores:4` |
| gpu008 runs | 5-6 DRPs, plus TEB, timing DRP, monitoring |

Two DRPs fill a quadrant, so with six the placement spans quadrants regardless and NPS=4's
finer information buys nothing anyone can act on.  Meanwhile it costs: each quadrant has a
quarter of the socket's local memory bandwidth, so any allocation that does not fit, or any
thread that migrates, takes an Infinity Fabric hop.  NPS=1 interleaves across all four
controllers, giving every allocation the socket's full bandwidth and making the DRP insensitive
to where its pinned buffers land.

Two honest caveats:

- **NPS=4 is what makes a card's `numa_node` meaningful.**  `gen_gres_conf` prints it, and under
  NPS=1 it degrades to socket granularity.  We do not currently act on it, but we would lose
  the ability to.
- **This is reasoning, not measurement.**  The experiment is the same six-DRP configuration at
  both settings, comparing rate.  Expect no difference today, since every card is already
  pinned at its PCIe 4.0 x8 ceiling at 12.788 GB/s -- the question only becomes live at x16
  gen5.  gpu008 is the node to measure on, since it is the one already at NPS=4.

So: **SMT off and NPS=1, uniformly**, with the NPS half offered as a considered guess rather
than a result.  Both `Cores=` fixes
have therefore been exercised: the CPU-versus-core-index fix on gpu006, whose 128 CPUs would
otherwise have produced out-of-range indices, and the NUMA-versus-socket fix on gpu008.
gpu006's published `Cores=32-63` is correct and Slurm confirms it with `(S:1)`.

**gpu007's `slurm.conf` line under-declares its CPUs, and the fix is sitting commented out
directly below it:**

    NodeName=...gpu007 CPUs=64  ... ThreadsPerCore=2 ...      <- active, wrong
    #NodeName=...gpu007 CPUs=128 ... ThreadsPerCore=2 ...     <- commented out, correct

`slurmd -C` detects 128, and 2 x 32 x 2 = 128, so the active line loses half the node:
`CPUEfctv=60` against gpu006's `124` on identical hardware.  It presents as a non-fatal
`error: Node configuration differs from hardware: CPUs=64:128(hw)` at every slurmd start.

**The commented-out line is not a forgotten fix -- Ric wrote it, tried it, and reverted it**
because swapping the comments put Slurm into a bad state, and restoring the old line brought
Slurm back.  So it is still hanging fire rather than waiting to be applied, and an earlier
version of this note wrongly framed it as an oversight to tidy up during the conversion.

Why it probably failed, though this is inference and not established: **gpu007 has jobs running
permanently**, since XPM:13 lives there.  At the time of writing it shows `CPUAlloc=37`,
`State=MIXED`.  Changing a node's CPU count while jobs hold allocations computed under the old
geometry is the kind of transition that goes wrong -- the same class as the drains we have seen
after every gres change, but affecting running work rather than just scheduling.  gpu006
carries the identical parameters with `CPUs=128` and is fine, so 128 is not wrong for this
hardware.

If it is retried, the obvious precautions are to drain the node and let its jobs finish first,
stop the XPM processes deliberately rather than have Slurm evict them, and expect to clear a
drain afterwards.  Worth asking someone who knows Slurm better than we do, rather than
experimenting on a node other people depend on.

| node | datadev cards | GPUs | dkms | notes |
|---|---|---|---|---|
| gpu001 | 1 | 1 A5000 | yes | published `dd02`; no timing while the NEH issue persists |
| gpu003 | 1 | 1 A5000 | **no** | Gabriel's; conversion pending |
| gpu005 | 6 | 1 H100 NVL at `47:00.0` | no | **the first big-box GPU node, so it differs throughout** -- see below |
| gpu006 | 3 | 2 H200 | yes | Mudit's, QSFP work; published `dda1`, `ddd5` |
| gpu007 | 3 | 2 H200 | yes | Matt's stand; hosts XPM:13 on `a1`; rename pending |
| gpu008 | 7 | 6 H200 (one unreliable) | yes | published 5 records; `a1` is InterCardTest |

### gpu005 is the odd one out, for historical reasons

It was the first big-box GPU node, built while we were still learning about GPUs, so its
differences are provenance rather than design:

- **Intel Xeon Gold 6444Y**, 2 sockets x 16 cores x **2 threads**, **2 NUMA nodes** -- where
  gpu006/7/8 are AMD EPYC 9355, 2 x 32 x 1 thread, 8 NUMA nodes.
- **H100 NVL** rather than H200, chosen before we knew better.
- Six datadev cards for one GPU, so five could not be paired if it were ever converted.  That
  ratio is an artefact of the box being early and partly populated, not a configuration to
  plan around.
- The datadev driver is **not loaded**, which is why `/proc/datadev_*` is empty; the cards are
  visible to `lspci`.
- It carries **`nvidia-fs`** (GPUDirect Storage), which no other node has, because
  **Cheolhong has been testing GDS there** -- the 22 GB/s figure quoted under "Recorder and
  file writing" was measured on this node.  Relevant to that work, since GDS is the mechanism
  for writing from GPU memory without a host bounce.
- **It is not fully populated** the way the November nodes will be, being an early box.  So
  its card and GPU counts, and the six-cards-to-one-GPU ratio, are not representative of what
  the rollout has to handle.
- `chan01` and `lorelli` have processes there.

**It is the only remaining node that would exercise the hyperthreaded `Cores=` path.**  The
two `Cores=` bugs needed different topologies to show: CPU-versus-core indices appears only
with `ThreadsPerCore=2`, which only gpu005 has, and NUMA-versus-socket boundaries appears only
with NPS=4, which only the EPYC nodes have.  So if `gen_gres_conf` is ever run here it tests
the fix the EPYC nodes cannot.

## Rules that will bite you, learned the hard way

Each of these has already cost time.  The reasoning is in the findings appendix; these are
the conclusions.

- **Leave `nvidia-powerd` ENABLED** despite `ERROR! UnSupported System`.  It opens the GPUs at
  boot, which is what creates `/dev/nvidia*`, and **slurmd fatals** if a `gres.conf` `File=`
  names a device that does not exist.  The message describes the platform, not a fault.
- **Leave `nvidia-persistenced` DISABLED.**  It holds the GPUs open and blocks the driver's
  automatic recovery from a GPU fault.  Use `disable --now`; a plain `stop` is undone in 100 ms.
- **`nvidia-smi -L` is the only authoritative check that a GPU is usable.**
  `/proc/driver/nvidia/gpus/` and `lspci` list a dead GPU indefinitely once its removal has
  been refused.
- **`/proc/datadev_*` is the only authoritative source about the resident driver** -- its
  `Git Version` and its `Buffer Mode`.  `dkms status` describes the package, `srcversion` the
  sources, `modinfo` the file on disk, and `/sys/module/datadev/parameters/` does not exist.
- **Re-install and re-`setcap` `/usr/local/bin/drp_gpu` after every C++ build**, per node.
  `install` drops file capabilities, and the image check refuses to start rather than running
  stale code.
- **Clear the drain after any reconfigure or node disturbance.**  Slurm never clears one
  itself: `sudo scontrol update NodeName=<node> State=RESUME`.  Read the `Reason` rather than
  skimming the state -- `count too low` is the harmless transient, anything else is real.
- **Read any existing `/etc/modprobe.d/datadev.conf` before converting a node to dkms.**  An
  `insmod`-based node keeps its parameters in a script, so a file may exist that has never been
  in effect and that `modprobe` would silently activate.
- **Converting a node to dkms also drops the NVIDIA module parameters.**  As of 2026-09-18
  `/etc/modprobe.d/nvidia-daq.conf` is in place on gpu001, gpu003, gpu005, gpu006, gpu007 and
  gpu008, so the parameters survive a reboot everywhere; a copy lives in the session directory.
  Nodes whose nvidia module predates the file still read `EnableStreamMemOPs: 0` until their
  next load.
  `comp_and_load_drivers.sh` passes `NVreg_OpenRmEnableUnsupportedGpus=1
  NVreg_EnableStreamMemOPs=1` on its `insmod` line and nothing in the dkms path supplies them,
  so a `modprobe`-loaded node has `EnableStreamMemOPs: 0`.  `drp_gpu` does not care -- its
  kernels write the GpuAsyncCore registers directly -- so the DAQ runs perfectly while
  `rdmaTest` aborts with "Selected GPU lacks stream memory ops".  That asymmetry is what makes
  it easy to miss.  Fix with `/etc/modprobe.d/nvidia-daq.conf`; check with
  `grep EnableStreamMemOPs /proc/driver/nvidia/params`.
- **`fuser` and `lsof` show only your own processes**, so an apparently stale refcount may be
  another user's live service.  `ps -eo user,pid,args` sees what they cannot.
- **A GPU in `Node Reboot Required` state may hang `sudo reboot`** -- use IPMI.


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

- ~~**Pair each datadev card with a GPU, so Slurm allocates the right one.**~~  **Working
  and in use as of 2026-09-17.**  `gen_gres_conf` derives the pairing from PCIe topology and
  `get_gres()` derives each DRP's request from its own `-d` argument, so the two cannot
  drift apart.  Three nodes published; verified end to end with six DRPs on six GPUs at
  33 kHz each.  Further bugs are likely, but this is now a fix-as-found matter rather than
  open work.

  **Rolling it out to new nodes is a separate task** -- currently ad hoc, three coordinated
  edits on the controller per node -- and probably belongs to whoever owns node
  provisioning rather than here.

  The first node, 2026-09-14: gpu006 advertises `gpu:dda1:1(S:1),gpu:ddd5:1(S:1)`; the
  `(S:1)` confirms it mapped the emitted `Cores=32-63` to socket 1, which is where both
  GPUs are.  `GresTypes` was left alone, since only the GPU is declared.

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

  **Verified end to end on 2026-09-14.**  A real DAQ run on gpu006 had the two GPU DRPs
  request `gpu:dda1:1` and `gpu:ddd5:1`, derived by `get_gres()` from their own
  `-d /dev/datadev_XX`, and open `0000:D4:00.0` and `0000:D3:00.0` -- exactly the pairing
  `gen_gres_conf` derives from the PCIe topology.  Each logged `Total GPU devices: 1`,
  confirming `ConstrainDevices=yes` constrains the cgroup to the allocated GPU and that
  `gpuId=0` stays right for every process.  The CPU DRP and the TEB got no GPU request.

  **gpu001 published 2026-09-15**: `Gres=gpu:dd02:1(S:1)`, one card and one A5000, so
  `--expect 1` and no `--exclude`.  Confirmed under load: the DRP requested `gpu:dd02:1`,
  `GresUsed` showed `(IDX:0)`, and it opened `0000:82:00.0`.

  **gpu008 published 2026-09-15**, the largest and the one that exercised the tooling
  properly: seven cards, six GPU-capable, five GPUs.  `--expect 5 --exclude d5`, giving
  `Gres=gpu:dd04:1(S:0),gpu:dd05:1(S:0),gpu:dd53:1(S:0),gpu:dd84:1(S:1),gpu:dd85:1(S:1)`.
  Three notes worth keeping:

  - It was also the dkms *and* `cfgDevName=1` conversion, done as two phases so the driver
    change and the rename could be judged separately.  See "Module parameters are
    invisible in sysfs" for why phase 1 needed its own minimal `cfgMode=2` conf.
  - `--exclude d5` rather than `05` or `85` **costs one local pairing**: `d5` shares a
    switch with GPU `d3`, so including it would give four local pairs instead of three.
    It is still right, because `d5` is the one card `gpu8.py` does not drive, and a free
    non-local hop beats a card that cannot run a GPU DRP at all.
  - The phantom sixth GPU went in the same edit; `slurmd -C` had been reporting
    `gpu:nvidia_h200_nvl:5` against a declared 6 all along.

  Then the rest as they are built.

  **gpu006 was handed to Mudit on 2026-09-15**, for QSFP optical-power readout work.  Its
  gres records were left in place: he is content with the `cfgDevName=1` hex device names
  and does not use Slurm, so nothing there constrains him.  Two things to know if it comes
  back: reflashing card firmware can change what `GPU Async En` reports, and
  `gen_gres_conf` refuses to emit for a card whose firmware has no GpuAsyncCore -- so run
  `--check` before assuming the published records still describe the node.

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

  **One line only when the dead GPU was the last in PCI order.**  Minor numbers are
  assigned over the GPUs actually present, so one falling off the bus shifts every GPU
  above it down by one and invalidates their `File=` too — see "How `/dev/nvidiaN` is
  numbered" below.  Regenerate the node's whole block rather than editing one line,
  unless `--check` says the rest still match.

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

  **Not a to-do: `GresTypes` needs no change.**  Declaring the datadev as a gres would have
  required `GresTypes=gpu,datadev`, but that idea was rejected -- see the reasoning above --
  so the existing `GresTypes=gpu` suffices.  Recorded because the earlier plan said otherwise
  and someone may remember it.

  **Observation, not a to-do: Slurm does not notice a vanished GPU.**  With GPU5 off the bus,
  gpu008 still reported `Gres=gpu:nvidia_h200_nvl:6`, `CfgTRES=gres/gpu=6` and `State=IDLE`
  -- not drained, no complaint.  `slurmd` validates the `File=` entries when it starts and
  nothing rechecks, so it would schedule six GPU jobs onto five GPUs.  Running the generator
  with `--expect` after a driver load or at boot would catch it, but **automating that is
  deliberately not proposed**: it is one more tool to maintain for a failure that announces
  itself as a crashed DRP.  Deal with it when it happens.

## Nothing names the process to comment out when a GPU dies

The degraded procedure's third step is "remove the affected process from the DAQ config", and
on 2026-09-17 it took working out which one.  `gen_gres_conf` says `datadev_85 has no GPU left
to pair with`; `gpu8.py` says `tstcam1_4` and `gpu_cmd%0x85`.  Nothing connects the two, so the
operator translates a bus number into a process name by hand, in the middle of an incident, and
the failure mode for getting it wrong is a job that pends for ever with no explanation.

Both halves already exist.  `SbatchManager.get_gres()` parses `-d /dev/datadev_XX` out of each
process's command, and `scontrol show node <node>` lists the gres actually offered.  So a check
at `daqmgr` start could compare the two and say, precisely:

    tstcam1_4 requests gpu:dd85:1, which this node does not offer.  Comment it out
    of the configuration, or publish a gres record for datadev_85.

That is the "check at daqmgr start" already listed as a known gap on the Confluence page; this
is the concrete case for it.  Worth doing before twenty nodes exist, because the translation
gets harder as the node count grows and it is only ever done under pressure.

Note it belongs at `daqmgr` start rather than in the DRP: by the time `drp_gpu` runs, Slurm has
either given it a GPU or left the job pending for ever, and in the pending case there is no
process to report anything.

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

  **Measured on drp-srcf-gpu001 (RTX A5000) on 2026-09-15**, confirming the split is what
  the code says and giving the constraints the fix has to respect:

      Initial SM resources: 64 SMs
        - Min. SM partition size: 2 SMs
        - SM co-scheduled alignment: 2 SMs
      Final SM resources for context 0: 6 SMs
      Final SM resources for context 1: 40 SMs
      Final SM resources for context 2: 18 SMs

  So 6 + 40 + 18 = 64 exactly, and any derived scheme must land on multiples of 2.  The
  **Measured on drp-srcf-gpu008 (H200 NVL) on 2026-09-15**, and it is *not* the 6 / 40 / 86
  that arithmetic on the hard-coded values predicted:

      Number of multiprocessors: 132
      Initial SM resources: 132 SMs
        - Min. SM partition size: 8 SMs
      Final SM resources for context 0: 8 SMs
      Final SM resources for context 1: 40 SMs
      Final SM resources for context 2: 84 SMs

  It is **8 / 40 / 84**, because the H200's device-level `minSmPartitionSize` is 8 where the
  A5000's is 2, and the clamp at `PGPDetector.cc:617` raised group 0's requested 6 to 8.
  So the one guard that exists did the useful thing here.  Note the per-context
  `minSmPartitionSize` reads 2 after the split, so the device-level value is the one that
  constrains the request.

  That also softens the `_event` complaint above: `Reader.cu` asks for 8 SMs' worth at
  `tpSM` 2048 and context 0 has exactly 8; on the A5000 it asks 6 at 1536 and context 0 had
  6.  So `_event` happens to fit on both -- by coincidence of two independently chosen
  tables, not by design, and nothing would warn if a future device broke the coincidence.
  The real imbalance is that `_reduce` uses 10 of context 1's 40 SMs on an H200 (a quarter),
  and that context 2 -- the recorder -- holds 84 of 132 SMs, 64% of the GPU.

  **And the reason the imbalance is invisible: the DRPs are DMA-bound, not GPU-bound.**
  Re-confirmed on gpu008 on 2026-09-15 with the merged pre-release driver -- 33 kHz on all
  five DRPs, no drops, no overflows.  At `dmaBufSize=387104` that is

      387104 B x 33035 Hz = 12.788 GB/s per DRP,  63.9 GB/s across five

  against 15.75 GB/s raw for one PCIe 4.0 x8 uplink, i.e. 81% of raw and essentially
  payload line rate once TLP and DLLP overhead is taken out.  Each card is pinned at its
  own uplink, so the GPU has spare capacity no matter how badly the SMs are divided.  That
  is why 84 of 132 SMs sitting in the recorder's context costs nothing measurable.

  **This gates the work rather than motivating it.**  Rebalancing the split cannot improve
  a rate that is set by the card's uplink, so it should not be justified on throughput
  until the cards are x16 gen5 -- which is exactly the same condition under which PCIe
  locality stops being free (see the locality note in `gen_gres_conf.py`'s docstring).  The
  two open items have the same trigger and should be revisited together.  Until then the
  argument for touching the partitioning is correctness and comprehensibility -- three
  hard-coded tables that disagree, and an `abort()` on any unrecognised `tpSM` -- not speed.

  Any future rebalancing must be measured against 33035 Hz rather than assumed to improve
  on it.

  Two gaps in the guarding, visible at `PGPDetector.cc:617`: only `group_params[0]` is
  clamped to `minSmPartitionSize`, not `[1]`; and nothing checks that 6 + 40 fits within
  the device, so a GPU with fewer than 46 SMs would fail the split rather than degrade.

  The fix is for one place to own the partitioning and hand each component the SM count
  of the context it was given — `cudaExecutionCtxGetDevResource()` already returns it,
  and `_setupGreenContexts()` logs it.  Each stage then derives blocks and threads from
  that rather than from a table.  Note the Reader and TrgInpGen share a context, so
  whatever owns this has to divide their allocation, not just report it; TrgInpGen's and
  the Reducer's own driver kernels are `<<<1, 1>>>` persistent loops, so they want about
  one SM each, and the bulk belongs to `_event` and `_reduce`.

- ~~**Clear the firmware counters once the timing link is up.**~~  Done in
  `epixuhremu_config.py`, and **confirmed on drp-srcf-gpu008 on 2026-09-15** -- gpu001 lost
  its timing to the NEH outage, so the test happened here instead.  From a link-down start:

      WARNING:root:epixuhremu: timing link is down, calling ConfigLclsTimingV2()
      ConfigLclsTimingV2()
      ...
      WARNING:root:RxRstCount: 0
      WARNING:root:RxDecErrs : 0
      WARNING:root:RxDspErrs : 0

  All three zero, which is the point.  `RxRstCount` is the conclusive one:
  `ConfigLclsTimingV2()` issues `C_RxReset`, so that counter would be non-zero unless
  something cleared it afterwards.

  The run also exposed a flaw in the same file: both `logging.info` calls were invisible.
  A DRP runs Python logging at WARNING -- there were zero `INFO:root:` lines in the whole
  log -- so "timing link is up, leaving it alone" and "timing link is up, Rx counters
  cleared" left no trace, while only the branch that resets the link was visible.  That
  makes "ran and decided to skip" indistinguishable from "never ran", which defeats the
  purpose of a function whose whole job is to record a decision.  Both are now
  `logging.warning`, matching the level the surrounding timing code dumps its counters at.
  One line per Allocate, so there is no noise cost.

  Nothing is deliberately suppressing INFO: the root logger is simply at Python's default
  of WARNING.  `xpmdet_config.py:13` carries the override commented out --
  `#logging.basicConfig(level=logging.INFO)` -- because lowering the *root* level
  unsilences rogue and pyrogue as well, which is unusable.

  **A cleaner fix exists but is not taken yet.**  A named logger carries its own level
  without touching anyone else's, and this was verified to work: `logging.getLogger(name)`
  with `setLevel(logging.INFO)` prints while `pyrogue.*` stays quiet.  It was not used
  because its visibility depends on a handler with a permissive level existing, and the
  only evidence one does is that the log lines carry `basicConfig()`'s default
  `LEVEL:name:` prefix -- *something* in the import chain calls it, but not
  `xpmdet_config`, and it has not been identified.  If that caller ever goes away, Python's
  `lastResort` handler takes over at WARNING and INFO messages vanish silently.  A
  diagnostic that might disappear is worse than one labelled a shade too severely.  Worth
  revisiting together with the wider question of why the DAQ's Python logging is configured
  by accident rather than deliberately -- several `configdb` modules call `basicConfig`,
  and whichever imports first wins.

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


- **Standalone harness for the CUDA graphs.**  Long-standing want: pull the kernels into
  a harness with synthesised input, both as permanent test code and as a profiling
  target.  `_event` is already a template in `ReaderKernels.cuh`, and
  `ReducerAlgo::recordGraph()` and `TriggerPrimitive::event()` already take a
  `cudaStream_t` and plain pointers, so all three are drivable without the
  pipeline.  This would give the **first correctness test of the EpixUHR3x2 fp16
  path and Jungfrau's nested packet walk, neither of which has ever executed**, and
  an `ncu` target free of spin-waits and device-side relaunch.

- **A repeatable way to measure any Reducer's ratio and throughput against payload size.**
  Run each available Reducer -- `lc`, `pfpl`, `sleek` today, others as they arrive -- over 1x,
  2x, 4x payloads in the harness.  Worth building as a reusable recipe rather than a one-off
  measurement: it is the diagnostic that says whether a new Reducer is viable at rate, and
  that question will recur.

  It also decides **how many datadev cards one GPU can serve**, since the constraint was
  always reducer throughput rather than PCIe.  Measured on gpu008, PCIe locality costs nothing — all
  cards sit at their own PCIe 4.0 x8 ceiling (~102 Gbps, 33035 Hz), whether or not
  they share a switch with their GPU.

- **Recorder and file writing**, including file system bandwidth and
  scatter-gather.  Profile `TebReceiver::_recorder()` and the writer
  before the third-party compressor work: the sink's throughput sets the compression
  ratio the reducers have to achieve.  At 33 kHz the uncompressed calibrated rate is
  ~25 GB/s, above the **22 GB/s Cheolhong measured with GPUDirect Storage on gpu005** --
  the only node with `nvidia-fs` installed, and measured there *with* the unsupported
  IB/Ethernet mix rather than on a supported configuration.  So 22 GB/s is a real number from
  real hardware but not an upper bound for a properly supported setup, and the sink is still
  slower than the source either way.

  GDS is nominally unavailable, since WEKA does not support a mixed IB/Ethernet fabric, so
  cuFile falls back to compatibility mode.  Worth asking whether cuFile buys anything over an
  explicit device-to-host copy plus `pwritev` in that mode, and whether scatter-gather writes
  help.  Worth also asking Cheolhong what his 22 GB/s actually exercised, since "GDS on an
  unsupported fabric" could mean the compatibility path rather than true peer-to-peer -- which
  changes whether the number is a floor or a ceiling.

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

- **Bulk `gpuSetWriteEn` in the datadev driver (aes-stream-drivers).**  One ioctl per
  buffer is ~33k/s at current rates.  A
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

  `gen_gres_conf`'s `Cores=` output is unaffected either way: it names whole sockets in
  core-index space, which depends only on sockets x cores-per-socket.

- ~~**datadev driver install at boot via dkms**, so a kernel update does not leave a node
  without its driver, and so the module parameters live in one declared place instead of in
  whoever's copy of `comp_and_load_drivers` ran last.~~  **Done.**  The machinery works and
  all four GPU nodes use it: `dkms-reload.sh` builds, installs, retires the old package and
  verifies the loaded version, and `AUTOINSTALL=yes` rebuilds after a kernel update.
  Upstream in aes-stream-drivers via PRs #319 and #323.

  **Deploying it to each node is a separate task**, and not necessarily ours -- it wants the
  conf file placed and the driver built per node, which is provisioning work.  What follows
  is the content that deployment needs.

  Wanted in `/etc/modprobe.d/datadev.conf`:

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

- **GPU5 on gpu008 keeps failing -- four times by 2026-09-18, now while idle.**  The
  2026-09-17 episodes were GSP heartbeat timeouts raising `Xid 154`, once escalating to
  `Node Reboot Required` on all six GPUs; after the power reset it came back and has since
  dropped out again with no DRP running.  Five of the six GPUs have never failed, so `d4` or
  its PCIe branch is the outlier.  gpu008 is published with five records and runs five DRPs at
  33 kHz, so this is a hardware conversation rather than something to configure around.  The
  earlier analysis below predates the GSP findings:

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
  the driver should be one that supports 13.x.  **Closed as of 2026-09-18, no action
  intended.**  Development moved to gpu008, which pairs toolkit 13.3 with driver 595, and the
  sdfada nodes are being upgraded to Rocky 9, which is expected to resolve the mismatch as a
  side effect -- timing unknown.  IT were asked and responded without a resolution.  We no
  longer care; recorded so that anyone who trips over it recognises it rather than
  investigating afresh.

- **`datadev_6` on drp-srcf-gpu008 (`a1:00.0`)** -- **superseded, kept for context.**  Under
  `cfgDevName=1` this card is now `datadev_a1`, and it was reflashed to InterCardTest firmware
  on 2026-09-17, so it is GPU-capable and must be excluded explicitly with `--exclude a1`.  As
  found, it ran `XilinxVariumC1100Pgp4_10Gbps` rather than
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

## Lower priority, after November's deliverables

- **A generic `recoverLinks` script for operators.**  When a DRP complains about a timing
  link, an operator should be able to run one thing that either brings the link up or says
  "this is a transceiver or optical-path problem, call someone".  Today that took a long
  hunt and ended on a register almost nobody would think to try.

  The escalation ladder is now known, and is the script's body: read and report both
  directions; `C_RxReset`; then the `ConfigLclsTimingV2` set (`TxPhyReset`, `TxUserRst`,
  `RxUserRst`); then `TxPhyPllReset` with `C_RxReset` and `RxDown` cleared after it; and
  only then declare the optical path.

  Two design points decide whether it is worth building:

  - **Firmware independence is the hard part.**  The registers live at different paths per
    board -- `l2si_drp.DrpTDetRoot` for the C1100, `PcieControl.DevKcu1500` for the
    KCU1500, `DevPcie.Hsio.TimingRx` for the epix trees -- so it cannot hard-code a path.
    `root.find(typ=...)` on `TimingPhyMonitor` and `TimingFrameRx` would locate them
    wherever they are, which is how `epixuhr3x2.py` already walks its own tree.
  - **It cannot verify success from the DRP alone.**  The failure that motivated this was
    invisible from the card: every local register read healthy.  Confirming the feedback
    direction means reading the XPM's `RemoteLinkId` for that link over PVA and comparing
    it against `timTxId()`, which is deterministic from the host address.  Without that the
    script can only report that the receive direction works, which is the half that was
    never broken.

  So the script wants PVA access to the XPM, which is a bigger dependency than a recovery
  tool usually carries.  Worth weighing against putting the same check in `control` at
  Configure, where the XPM connection already exists.

- **Let the DRP idle at low power when triggers are absent or slow.**  Rather than polling
  hard for an event that is not coming.  Cheaper than it sounds, because the pattern is
  already there: `Reader.cu` waits with exponential `__nanosleep` backoff in three places
  (`:363`, `:407`, `:448`), doubling 8 ns to a 256 ns ceiling and then returning to yield
  instead of spinning.

  So this is a question about the ceiling, not new machinery.  At 33 kHz the inter-event
  gap is about 30 us, so a 256 ns cap already means roughly 120 wake-ups per event period,
  and proportionally more as the rate falls.  Raising the ceiling -- or adding a second,
  coarser tier once a quiet period is established -- costs added latency only on the first
  event after the quiet spell, which is exactly when latency does not matter.

  Two things to check before doing it: whether the host-side threads also poll (the device
  side is the part with backoff today), and that `__nanosleep`'s guarantees hold at longer
  intervals on the devices in use.  Measure against 33035 Hz, since the point is to change
  power draw and not throughput.

  Shares a mechanism with the `rdmaTest` timeout below -- both are bounded waits -- but not
  a purpose: that one is about saying why nothing arrived, this one about not burning power
  while nothing arrives.


# Appendix: findings

Settled explanations, kept because they were expensive to establish and because the
reasoning behind several decisions above lives here.  Nothing in this appendix is open
work.

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

## slurmd will not start at all if a `File=` device is missing

Learned from drp-srcf-gpu007 on 2026-09-16, which was `DOWN+NOT_RESPONDING` because slurmd
had exited:

    error: Waiting for gres.conf file /dev/nvidia0
    fatal: can't stat gres.conf file /dev/nvidia0: No such file or directory

It waits 19 s for the device to appear and then **fatals**.  So a node whose
`/dev/nvidia*` are absent when slurmd starts does not run slurmd, and the failure presents
as "Not responding" rather than as anything about GPUs.  That is a dependency of every
`gres.conf` carrying `File=`, ours included, and it was not on our radar.

**`nvidia-powerd` is what creates the device nodes, and it is REQUIRED.  Do not disable it.**

They are not part of the driver load: something has to invoke `nvidia-modprobe`, which any
process opening a GPU does implicitly.  `nvidia-powerd` is packaged and enabled by NVIDIA to
initialise the GPUs at boot, and it does exactly that -- it opens them, finds this platform
has no dynamic-boost capability, prints `ERROR! UnSupported System` and exits.  The nodes it
leaves behind are what slurmd needs, and slurmd *fatals* without them.

Confirmed on gpu008 on 2026-09-17, after a power reset with persistenced disabled so nothing
else could be responsible:

    nvidia-powerd   18:19:03   (then exits, "UnSupported System")
    /dev/nvidia0    18:19:03   same second
    slurmd          18:19:36   33 s later

**The hazard is that the error message invites a cleanup.**  Someone reasonably reading
`ERROR! UnSupported System` as noise, and disabling the service to silence it, would take
every GPU node's `/dev/nvidia*` with it -- and the symptom would be
`State=DOWN+NOT_RESPONDING`, pointing at the network rather than at NVIDIA.  That is exactly
how drp-srcf-gpu007 presented on 2026-09-16, where powerd is disabled.

So the choice is between depending on a service whose error message looks like a defect, and
writing and maintaining a small unit that runs `nvidia-modprobe` before slurmd.  The unit is
more honest about intent, but it is more code to keep alive, and the dependency is only
dangerous while it is undocumented -- which this note fixes.  Leave powerd enabled until
something better comes along.

An earlier version of this note called the node creation an "accident".  That was wrong:
powerd opens the GPUs deliberately, and `UnSupported System` describes the platform's lack of
a feature, not a malfunction.

An earlier version of this note credited the udev rule
(`/usr/lib/udev/rules.d/60-nvidia.rules`, `KERNEL=="nvidia", RUN+="/usr/bin/nvidia-modprobe"`)
because the node also appears in the same second as `Finished Wait for udev To Complete
Device Initialization`.  **That was wrong** -- both happen in that second, and gpu007
settles udev *before* nvidia loads yet still gets no nodes, which the udev explanation
cannot account for.  `nvidia-powerd` is enabled on the three working nodes and disabled on
gpu007, which is the whole difference:

| node | `/dev/nvidia0` created | `nvidia-persistenced` |
|---|---|---|
| gpu006 | 15 s after boot | disabled / inactive |
| gpu008 | 27 s after boot | disabled / inactive |
| gpu001 |  9 s after boot | disabled / inactive |
| gpu007 | never | enabled / **failed** |

`nvidia-powerd`: enabled on gpu006, gpu008 and gpu001; **disabled** on gpu007.

So persistenced is not what creates them, and neither is udev: it is `nvidia-powerd`, as
above.  What matters operationally is that **something must open a GPU before slurmd starts**,
and on these nodes that something is powerd.

**`nvidia-persistenced` is the fix, and it is proven.**  Demonstrated on drp-srcf-gpu007 on
2026-09-16, where `nvidia-powerd` is disabled, so there is no ambiguity about the cause:

    boot            17:46:43
    persistenced    17:46:58   active, and stays running
    /dev/nvidia0    17:46:58   +15 s, the same second
    slurmd          17:47:38   40 s of margin

So it creates the nodes deliberately rather than as a side effect, and the ordering is not
marginal.  Getting there needed the broken drop-in removed:
`/etc/systemd/system/nvidia-persistenced.service.d/override.conf` set `--user root` while
the packaged unit keeps `User=nvidia-persistenced`, so it could not chown its own runtime
directory.  With that moved aside and a `daemon-reload`, the packaged unit works unmodified.

**Do not enable it on the DAQ nodes.**  It blocks the NVIDIA driver's automatic recovery
from a GPU fault, which is a much higher cost than the `rmmod` nuisance it was first weighed
against.

Demonstrated on gpu008 on 2026-09-17.  GPU5 (`0000:d4:00.0`) had returned after a reboot, so
six GPUs were published and six DRPs started.  Minutes later `tstcam1_4` died with
`CUDA_ERROR_NO_DEVICE`, and the kernel log said why:

    NVRM: GPU5 _kgspRpcRecvPoll: GSP RM heartbeat timed out
    NVRM: Xid (PCI:0000:d4:00): 154, GPU recovery action changed from 0x0 (None) to
          0x1 (GPU Reset Required)
    NVRM: Attempting to remove device 0000:d4:00.0 with non-zero usage count!

The GPU's onboard GSP processor hung, the driver raised Xid 154 and tried to reset the
device, and **the removal was refused because persistenced held it open**.  That left the GPU
enumerated but unusable: `lspci` and `/proc/driver/nvidia/gpus/` still listed six, while
`nvidia-smi` listed five.

`sudo systemctl stop nvidia-persistenced` alone recovered it -- no `nvidia-smi -r` needed.
The refcount on the `nvidia` module fell from 28 to 3, `Bus Type` went back from `PCI` to
`PCIe`, `current_link_speed` became readable again, and `nvidia-smi` showed all six.  The DAQ
then ran six DRPs at 33034 Hz each, 76.7 GB/s aggregate, with `tstcam1_4` on the recovered
GPU.

So persistenced converts a self-healing transient into a dead GPU needing human
intervention.  **It also casts doubt on the original GPU5 death**, which may equally have
been a recoverable fault held open rather than failing hardware.

Use the narrow alternative instead: a unit running `nvidia-modprobe` before slurmd.  It
creates the device nodes deliberately, holds nothing open, and obstructs neither `rmmod
nvidia` nor the driver's own recovery.  Note the nodes still have to come from *somewhere* --
without persistenced, gpu006, gpu008 and gpu001 get them from `nvidia-powerd` failing as
"UnSupported System", which is the accident described above and not something to rely on.
gpu007 was the outlier twice over: persistenced enabled and failing, and whatever does the
creating on the others not having run.  Its persistenced override is broken independently,
`User=nvidia-persistenced` in the unit against `--user root` in the override, so it cannot
chown its own runtime directory:

    nvidia-persistenced: Failed to change ownership of /var/run/nvidia-persistenced:
                         Operation not permitted

It also lacked the `nvidia-open` package the others have, which makes this look like a
provisioning divergence rather than a fault.  The remedy was to make it match the three
working nodes rather than to fix persistenced.

### gpu007 hosts an XPM, which the rename will break

Not a node of ours, but it will be converted eventually and this is the trap.  gpu007 is an
isolated test stand Matt is making use of.  It has three datadev cards and two H200s, and
`datadev_2` runs **`xpmGenC1100`** firmware -- it is **XPM:13**, a timing source rather than
a DRP card, with `GPU Async En : 0` as expected.  Only gpu007's own two cards are fibred to
it, so it has no external consumers.

**Its driver did not survive a reboot, so it was converted to dkms.**  On 2026-09-16, after
rebooting gpu007, `datadev` was not loaded, `/dev/datadev_*` were absent, `dkms status` had
no datadev package and there was no module in `/lib/modules` -- because its driver came from
`comp_and_load_drivers.sh` via `insmod`, which leaves nothing to load at boot.  XPM:13 went
down with it, and Slurm restarting the XPM job could not help while there was no device to
open.

**Phase 1 applied the same day**, `cfgDevName` left at 0 so the names stay `datadev_0..2`
and `pykcuxpm -d /dev/datadev_2` keeps working: all three cards now report
`7.6.0-29-gb79d0f8-dirty` with `mode=2 cont=0`, dkms has the package for the running kernel,
and `/lib/modules/.../extra/datadev.ko.xz` exists, so it comes back on its own next boot.
That also lets persistenced stay enabled there, since `dkms-reload.sh` only unloads
`datadev` where `comp_and_load_drivers.sh` insists on unloading `nvidia`.

Two traps found on the way, both about parameters living where `insmod` cannot see them:

- gpu007 already had `/etc/modprobe.d/datadev.conf`, written 2026-08-11, which had **never
  been in effect** -- `insmod` does not read `/etc/modprobe.d`, and the script passes
  `cfgMode=2` on its command line.  Converting to `modprobe` would have silently activated
  it.  It happened to contain exactly the built-in defaults
  (`cfgTxCount=1024 cfgRxCount=1024 cfgSize=131072 cfgMode=1 cfgCont=1`), so the only real
  change would have been `cfgMode` reverting from 2 to 1.  Ric replaced it with the GPU DRP
  set instead.  **Read any existing modprobe.d file before converting a node**; do not
  assume the parameters are only in the script.
- A `datadev.conf~` editor backup sat beside it.  Harmless -- modprobe reads only `*.conf`
  -- but worth confirming with `modprobe -c | grep "^options datadev"` that exactly one
  line results, since two would be resolved by file order.

`patches/0001-nvidia-driver-fix-crash.patch`, which `comp_and_load_drivers.sh` applies to
`nvidia-uvm/uvm_hmm.c`, is **obsolete** -- rolled into the NVIDIA open driver.  The dkms
path applies no patches, so nothing is lost by converting; the patch could be dropped from
the repository.

`pykcuxpm` serves it, running as `tmoopr` under Slurm, which is normal for XPM processes.
That is what holds a reference on the datadev module, so `Module datadev is in use` there is
the driver protecting a running service rather than an obstacle.  A reboot clears it and
Slurm restarts the XPM processes.

**When gpu007 is moved to a dkms datadev with `cfgDevName=1`, the rename hits `datadev_2`
too**, and `pykcuxpm`'s device argument breaks -- taking XPM:13 down.  `gen_gres_conf
--exclude` does *not* protect against this: exclusion only affects which cards get gres
records, not what the driver names them.  So the rename check has to cover XPM launch
configuration as well as the DAQ `.cnf.py`.  Deferred deliberately; later rather than sooner.

Two lessons worth keeping:

- **`nvidia-smi` is not a read-only diagnostic.**  It invokes `nvidia-modprobe` and creates
  the device nodes.  Running it while diagnosing gpu007 destroyed the original state.
  `/proc/driver/nvidia/gpus/`, `lsmod` and `lspci` answer the same questions without
  touching anything.
- **`fuser` and `lsof` only see your own processes.**  On gpu007 they reported nothing
  holding the datadev devices, which read as leaked references needing a reboot; the holder
  was `pykcuxpm` running as another user.  `ps -eo user,pid,args` is visible where file
  descriptors are not, so check for a plausible process before concluding a refcount is
  stale.  The same permission boundary had already hidden the journal on that node.
- **`gen_gres_conf --check` belongs in post-boot verification**, not only after a driver
  load.  It would have named this immediately, where the Slurm-side symptom pointed at
  the network.

## How `/dev/nvidiaN` is numbered, and when `gres.conf` goes stale

Measured 2026-09-14, because `gres.conf`'s `File=` is the only thing binding a datadev to
a GPU and a minor number is not a durable identity.

The NVIDIA kernel driver assigns minors sequentially over the GPUs it binds, in PCI probe
order, which is ascending BDF.  Confirmed on three nodes with three populations:

| node   | PCI addresses                        | minors      |
|--------|--------------------------------------|-------------|
| gpu001 | `82:00.0`                            | 0           |
| gpu006 | `d3:00.0`, `d4:00.0`                 | 0, 1        |
| gpu008 | `03`, `54`, `55`, `83`, `d3` (`:00.0`) | 0, 1, 2, 3, 4 |

**So a driver reload with an unchanged PCI population gives the same minors, and
`gres.conf` does not need regenerating for a `dkms-reload`.**  Only a change in GPU
population requires it.  That is already the minimum update frequency; there is nothing
to tune.

**`CUDA_DEVICE_ORDER` is the wrong layer and cannot help.**  It is read by the CUDA
user-space library at `cuInit` and only permutes the device indices *within* a process —
what `cudaSetDevice(0)` means and how `CUDA_VISIBLE_DEVICES` entries map.  It has no path
to the kernel driver's minor assignment, so setting it before `modprobe` or `dkms` is
meaningless.  It is also nearly moot here: Slurm gives each DRP one GPU renumbered to
index 0, and with one device there is no order to choose.  Worth setting only if a
process is ever given more than one GPU.

### The renumbering hazard, and why it is invisible

Because minors count only *present* devices, a GPU falling off the bus shifts every GPU at
a higher PCI address down by one.  gpu008 escaped this by luck: the one that died was
`0000:d4:00.0`, the highest of its six, behind the now-empty downstream port
`0000:d2:01.0` (bus 212).  It took minor 5 and the survivors kept 0-4.  Had `55:00.0` died
instead, `83`, `d3` and `d4` would each have shifted and every `File=` below the failure
would silently have named a different GPU.

The device nodes cannot tell you which happened.  On gpu008 now, `/dev/nvidia0..4` open
and `/dev/nvidia5` returns `ENODEV` — the identical picture a middle-GPU death would
produce.  The nodes are static files created by `nvidia-modprobe` (see
`/usr/lib/udev/rules.d/60-nvidia.rules`), so they neither disappear nor change timestamp
when the mapping moves: on gpu006 `/dev/nvidia0` and `/dev/nvidia1` date from Aug 4 and
survived the Sep 11 driver rebuild.

`/proc/driver/nvidia/gpus/<pci>/information` is the only authoritative statement, and
`gen_gres_conf` reads its `Device Minor` field rather than inferring from names.  This is
what the per-record comment line is for: it preserves both PCI addresses, so a record can
be checked against the hardware later.  Hand-written records that carry no PCI address
cannot be checked at all.  `--check` mechanises the comparison; run it after every driver
load and after anything that touches the GPU population.

If minors ever do prove unstable across reloads, the lever is a stable path in `File=` —
a udev rule creating something like `/dev/nvidia-by-pci/0000:d3:00.0` — not an
environment variable.  Three things to settle before trusting that: `gres.conf(5)` says
nothing about symlinks (enforcement goes through cgroups, which needs `major:minor` from
`stat()`, and `stat()` follows links, so it should work but is undocumented); the nodes
are created on demand by `nvidia-modprobe`, so a rule firing at bind time may find nothing
to link; and it would not remove `gen_gres_conf`, since `Cores=` and the pairing still
come from topology.  Not worth building on present evidence.

## gpu008 advertises a GPU that does not exist

Found 2026-09-14 while checking the above.  Nobody else is inconvenienced by it -- as of
2026-09-15 only we need Slurm and the GPU DRP, and others on these nodes are doing firmware
and orthogonal tests that do not go through gres -- so this is ours to fix at our
convenience rather than urgent:

    gres.conf:22  NodeName=drp-srcf-gpu008 Name=gpu Type=nvidia_h200_nvl File=/dev/nvidia5
    scontrol      Gres=gpu:nvidia_h200_nvl:6  CfgTRES=gres/gpu=6  State=IDLE  (no Reason)
    open("/dev/nvidia5") -> ENODEV

This is the concrete confirmation that **Slurm does not notice a GPU that has fallen off
the bus**: it keeps advertising six, stays `IDLE` with no `Reason`, and will hand the
sixth concurrent GPU job a device that fails in CUDA init at run time.  Fix is to delete
that record and set the node's `slurm.conf` `Gres=` to 5.

Worth converting gpu008 with `gen_gres_conf` rather than just deleting the stale record.
Its five records carry no `Cores=` at all, so nothing places tasks near their GPU, and no
PCI address, so none of them can be verified against the hardware.

Converting it is a bigger job than gpu006 or gpu001 were, and that is the point: it is the
best rehearsal available for the twenty-odd nodes arriving in November.  Specifically, it
is the only GPU node that is **not** dkms-managed -- `dkms status datadev-gpu-dkms` is
empty there, so its driver came from `comp_and_load_drivers.sh` -- and its cards are named
`datadev_0..6`, i.e. probe order, so `cfgDevName=1` has never been set on it.  Both of
those are exactly what a new node will need done, and neither has been exercised
from scratch:

1. Install `/etc/modprobe.d/datadev.conf` with `cfgDevName=1`, which renames all seven
   cards and is the step most likely to surprise something that hardwired a device name.
2. Convert to dkms with `dkms-reload.sh`, retiring the hand-built module.
3. `gen_gres_conf --expect 5` -- note five, not six, and note that two of the seven cards
   report no GPU capability, so `--exclude` will be needed to choose which cards lose out
   rather than letting `/proc` order decide.

No coordination cost: others on the node are doing firmware work that does not go through
Slurm.

## A dead timing feedback link is invisible from the DRP: TxPhyPllReset fixes it

Diagnosed on drp-srcf-gpu008 on 2026-09-15.  The DAQ sat at 100% deadtime; `xpmpva` showed
`RemoteLinkId` on XPM:14 QSFP1-2 as `undef/0` where it should read `TDetSim/gpu008`.  The
**feedback** direction -- DRP to XPM -- was not being received, while XPM to DRP was fine.

**The DRP cannot see this.**  Every register on the failing card was indistinguishable from
four working ones:

    RxLinkUp 1  MmcmLocked 3  TxRstStatus 0x0  RxRstStatus 0x0
    TxClkFreq 185.714 MHz  RxClkFreq 185.714 MHz  Loopback No
    XPM remote link id 0xff0e8d06: XPM:14 link 6 = QSFP1-2

and it read a valid remote link id, so its receive path was genuinely working.  Only the
XPM knows the feedback link is dead, which is why no DRP-side symptom can trigger a
DRP-side remedy.

**The remedy is `TimingPhyMonitor.TxPhyPllReset()`**, followed by epixquad's full sequence:

    TDetTiming.TimingPhyMonitor.TxPhyPllReset()   # then ~1 s
    TDetTiming.TimingFrameRx.C_RxReset()          # then ~2 s
    TDetTiming.TimingFrameRx.RxDown.set(0)

`RemoteLinkId` corrected the instant the PLL reset was issued.

**The fifth link also failed to go down when the datadev driver was reloaded**, while the
other four did (2026-09-15, the phase-1 to phase-2 reload that renamed the devices).  That
is the strongest hint about the mechanism: a driver reload re-probes the card and issues a
user reset, which bounced four links and left this one apparently up.  Whatever state was
wrong survived a driver reload, a card re-probe and a user reset, and yielded only to an
explicit `TxPhyPllReset`.  A GT transmit PLL locked in a bad state fits: it is not in the
user-reset domain, so nothing short of a PLL reset touches it, and a PLL can hold a lock at
the right frequency while producing an eye the far end cannot decode.  Recorded as evidence
rather than conclusion -- the receive path genuinely worked throughout, which a stuck
`RxLinkUp` bit would not explain.

This is a known-flaky bring-up step that the TDet path omits, not a broken component.  The
evidence: `epixquad_config.py:189` and `epixquad1kfps_config.py:822` both call
`TxPhyPllReset()` **unconditionally**, under the comment "To get the timing feedback link
working"; `ConfigLclsTimingV2` in the l2si tree issues `TxPhyReset` and `TxUserRst` but
**never** `TxPhyPllReset`; and `xpmdet_config.py:200` names `TxPllReset` as the remedy in a
message and then aborts rather than doing it.  Cheolhong has been finding XPM firmware bugs
and one may still be outstanding, which would explain why four of five links on identical
hardware and firmware came up fine.

### It recurs, and a firmware fix is coming

**The reset does not survive a driver reload.**  On 2026-09-16, after updating gpu008 to
`7.6.0-27` and restarting the DAQ, link 6 had failed again -- read straight from the XPM
rather than through xpmpva:

    DAQ:NEH:XPM:14:RemoteLinkId0   4211121061   = 0xFB009BA5 = TDetSim/gpu008
    DAQ:NEH:XPM:14:RemoteLinkId1   4211121061
    DAQ:NEH:XPM:14:RemoteLinkId2   4211121061
    DAQ:NEH:XPM:14:RemoteLinkId4   4211121061
    DAQ:NEH:XPM:14:RemoteLinkId6            0   <- datadev_85

That fits the rest of the picture: `datadev_85` is also the only card whose link does *not*
go down when the driver reloads, and the only one that came up with `RxLinkUp 1` while the
other four were down.  Its GT transmit PLL appears to return to the same bad state whenever
the card is re-probed.

Its receive side was degraded too: over the 18.5 h between two Allocates it logged **97
link resets**, 2043 decode and 1967 disparity errors -- about 5 resets and 110 errors per
hour, against zero on the other four.  After the `TxPhyPllReset` it read **zero on all
three**, matching a healthy card exactly, so the reset cleaned up both directions.

### Root cause, from Cheolhong on 2026-09-16

**The TDet firmware clocks the transceiver from an internal clock rather than an external
one, giving more jitter and a poor eye.**  Firmware work is in the pipe and he will work
with Mudit to merge it into the TDet firmware, so the manual `TxPhyPllReset` is an interim
workaround with an end date rather than something to build procedure around.  Do it from
devGui after each driver reload until that lands.

That one cause accounts for every observation, including the ones that defeated a
register-by-register hunt:

- `TxClkFreq` reading a nominal 185.714 MHz while the XPM could not decode the stream.
  Frequency correct, jitter not -- which is why nothing on the DRP side could see the fault.
- Marginal rather than dead, and varying between links: jitter eats margin, so only the
  link with the least of it fails.
- Recurring on every re-probe: the PLL re-locks, sometimes into a worse state.
- The receive side degrading as well.  A single internal reference feeds both the Tx and Rx
  PLLs, so one reset fixing both directions is expected rather than surprising.

An earlier version of this note said a Tx-side firmware fix would not address the
receive-side errors, and flagged that for Cheolhong and Mudit.  **That was wrong**: with a
shared internal reference as the cause, moving to an external clock addresses both.

### Checking it is a one-liner, and worth automating regardless

The closed-loop check proposed below turns out to be trivial, which removes the main
argument against it.  `pvget` on the XPM, compared against `timTxId()`:

    export EPICS_PVA_ADDR_LIST=<xpm> EPICS_PVA_AUTO_ADDR_LIST=YES
    pvget -i DAQ:NEH:XPM:<n>:RemoteLinkId<link>

Zero means that link's contributor is not reaching the XPM.  `timTxId()` is deterministic
from the host's 172.21 address, so the expected value is computable without asking anyone.

One limitation to record: every DRP on a host produces the *same* `TxId`, so all healthy
links from one node read identically.  The check can say "this link's contributor is not
reaching the XPM" but not which process -- which is sufficient, because the link number now
identifies the card via the log line each DRP writes.

Worth doing even after the firmware fix lands: it catches *any* feedback-link failure, and
the failure mode it catches is 100% deadtime with no attribution anywhere.

### Diagnostic order that worked, for next time

1. **Which DRP is on the bad link** -- now a single line in every DRP log, in both the
   absolute and `QSFP%d-%d` notations, so it matches whatever xpmpva shows.
2. **Is the DRP's local PHY healthy** -- the `epixuhremu: RxLinkUp ...` dump.  All nominal
   here, which is the point: it does not exonerate the transmitter.
3. **Is the XPM's digital receive path healthy** -- set that link's `LinkLoopback` briefly.
   It read correctly, so the XPM's GT, decoder and `RemoteLinkId` logic are fine.  Note this
   is an internal loopback and says nothing about its optics.
4. **Is it configuration** -- compare `LinkRxReady`, `LinkRxResetDone`, `LinkTxReady`,
   `LinkTxResetDone`, `LinkIsXpm`, `LinkLoopback`, `LinkGroupMask` against a working link.
5. **`TxPhyPllReset` on the DRP**, with the sequence above.

### Three wrong hypotheses, recorded so they are not repeated

- **`TxPhyReset` would fix it.**  No -- it is the *PLL* reset that is needed, and
  `ConfigLclsTimingV2` already issues `TxPhyReset` without helping this class of fault.
- **Dark fibre / nothing arriving.**  No -- `LinkRxErr` counting and wrapping while
  `LinkRxRcv` stays 0 means the XPM's receiver has signal it cannot decode.  Loss of signal
  does not count errors.
- **Loopback set on the DRP.**  No, it read `No`.  A *near-end* mode was already excluded by
  the card reading a valid `RxId`; only a far-end mode was consistent, and it was not set.

**The trap in all three: `TxClkFreq` reading a nominal 185.714 MHz does not mean the
transmitter is good.**  The DRP measures its own clock frequency, not its eye quality, so a
PLL locked at the right frequency with bad jitter looks perfect from this side and is
undecodable at the far end.  Nothing on the DRP can distinguish those.

### The real fix belongs on the XPM/control side

A DRP-side remedy cannot be triggered by a DRP-side symptom, and issuing `TxPhyPllReset`
unconditionally every Allocate costs ~3 s and bounces four working links to fix a fifth.
The check that would have turned a long hunt into one error message is at Configure, on the
side that has the information: for each link in the partition's `GroupMask`, compare the
XPM's `RemoteLinkId` against the `TxId` the contributor should be sending -- `timTxId()` is
deterministic from the host address, and `xpmdet_connectionInfo` already reports each
contributor's link number as `paddr`.  A mismatch means that contributor's feedback link is
dead, which is a precise, actionable message instead of 100% deadtime with no attribution.

## GPU memory must be GPU-page aligned, and that reopens the packaging question

aes-stream-drivers PR #321 (merged to `pre-release` 2026-09-16) makes the driver **reject**
GPU memory that does not start on a GPU page boundary, where it used to round down
internally and carry the remainder as an offset:

    // Memory must be aligned to GPU page boundary to avoid GpuAsyncCore writing out-of-bounds
    if ((dat.address & GPU_BOUND_MASK) != dat.address) { ... return -EINVAL; }

`MemPool.cc` passed `cudaMalloc`'s pointer straight to `gpuAddNvidiaMemory()`, and
`cudaMalloc` guarantees no such alignment -- upstream's own commit says "APIs such as
cudaMalloc or cuMalloc do not guarantee us alignment".  In practice it returned 512 B
alignment, visible in the logs all along as `dptr 0x...200`, `0x...400`, `0x...600`.  So a
driver update past #321 would have aborted every GPU DRP at startup, with
`gpuAddNvidiaMemory failed` and nothing to suggest why.

**Fixed 2026-09-16** by `_allocAlignedDma()`: over-allocate by `GPU_PAGE_SIZE - 1`, round
the pointer up, and keep what `cudaMalloc` returned in `DetPanel::dmaRawPtrs` for
`cudaFree`.  `dmaBuffers` now holds pointers *into* those allocations, which is why there
are two vectors and why the destroy path must free the raw one.  Costs under one GPU page
per buffer.

Worth noting the *size* was always rounded to 64 KiB at `MemPool.cc:213`, so whoever wrote
this knew of the requirement; what defeated them is that only the address was wrong, and
the old driver hid it.  Both now go through the same `GPU_PAGE_SIZE`.

**Verified on drp-srcf-gpu008 on 2026-09-16**, against `7.6.0-27-g232c8ed`, which is the
first driver that enforces this.  All forty DMA buffers across five DRPs came up on 64 KiB
boundaries, no `Gpu_AddNvidia` rejections, no aborts, and the pairings and typed gres
allocations were unaffected:

    DMA buffer[0] dptr 0x7f0403e10000, size 393216
    DMA buffer[1] dptr 0x7f0403e80000, size 393216
    ...

Compare the same lines on the old driver -- `0x...200`, `0x...400`, `0x...600` -- which it
accepted by rounding down internally.  Worth noting this was the first configuration in
which a mistake in `_allocAlignedDma()` would have been *loud*: the driver returns EINVAL
and `MemPool.cc` calls `abort()`, so there is no subtle middle outcome to misread.

### The goal is the VMM path, via PRs that make the interface usable

Over-allocating is the interim answer, not the intended one.  Upstream took the CUDA VMM
API for its own test app -- `cuMemCreate`/`cuMemMap` behind `vmmCuAlloc()`/`vmmCuFree()`
and a `CudaVMMAlloc` handle -- which asks CUDA for the alignment instead of working around
its absence.  That is where this should end up.

What stops it today is *where the code lives*, and that is the interesting part:

- `include/GpuAsyncLib.h` **declares** `vmmCuAlloc`/`vmmCuFree` and defines `alignValue`.
- The **definitions** are in `data_dev/app/src/GpuAsyncLib.cpp` -- an application source,
  not a header, and not something lcls2 can vendor the way it vendors headers.
- `GPU_BOUND_SHIFT`, the alignment the driver actually enforces, is in
  `common/driver/gpu_async.h`: kernel-side, uses `u64`, not in `include/`.  So lcls2
  duplicates the constant as `GPU_PAGE_SIZE`, across repos, with nothing to keep them in
  step.

**This breaks the assumption that made the current arrangement acceptable.**
aes-stream-drivers has been a headers-only package from lcls2's point of view, which is
precisely why those who maintain such things did not want it as a `SUBMODULEDIR` module --
seven headers copied into `psdaq/psdaq/aes-stream-drivers/` was proportionate.  Needing
*implementations* changes that calculus, and the question of how lcls2 consumes this
package is open again.

So the plan is to feed PRs back until the interface fits, rather than to vendor an app
source or reimplement the helpers:

1. Ask for `GPU_BOUND_SIZE` to be exposed in `GpuAsyncUser.h`, next to the ioctl that
   enforces it.  A userspace caller cannot currently learn the alignment it is required to
   satisfy, which is why the constant is duplicated.
2. Ask for the VMM helpers to land somewhere consumable -- header-inline, or a small
   library the package installs, rather than an app source.
3. Then switch `MemPool.cc` to them and delete `_allocAlignedDma()`.

Also note `drp_gpu` does **not** use `GpuAsyncLib` at all: `MemPool.cc` reaches
`gpuAddNvidiaMemory()` through the vendored `GpuAsyncUser.h`.  Only `pgpread.cc` uses the
local `drpGpu/GpuAsyncLib.{hh,cc}`, which are older copies of the upstream header/source
pair.  They have diverged a long way -- `checkError` changed signature, `DataDev` became
`DataGPU`, `GpuAsyncOffsets` is gone from `include/`, 340 header lines differ -- so porting
pgpread is a real refactor.  It is also unnecessary: nothing else uses those files, they
still build, and they are not in the way.  Leave them until someone needs pgpread itself.

## Every DRP log now names its XPM link

Added 2026-09-15, after `xpmpva` reported a bad link on gpu008 and nothing in five DRP
logs said which process was on it.  The information was always there and always thrown
away: `xpmdet_config.py` read `XpmMessageAligner.RxId` three times and logged it with
`logging.info`, which a DRP filters out (see the logging note above).

The final read -- the one outside the supervisor block, that every process reaches, and
whose value becomes `connect_info['paddr']` -- is now `logging.warning` and decoded:

    XPM remote link id 0xff0e0006: XPM:14 link 6 = QSFP1-2 (10.0.0.100)

Low byte is the link, bits 23:16 the XPM number; `xpmLinkId()` in `psdaq/cas/xpm_utils.py`
already decoded the rest.  **Both link notations are printed on purpose**: `xpmpva` names
ports `QSFP%d-%d` of `port//4` and `port%4` (`xpmpva.py:71`, `:701`), while the register,
the deadtime tables and the illegal-value checks use the absolute number.  Printing one
form only moves the division by four to whoever is reading the log during an incident.

The two earlier reads stay at `info`: they are mid-sequence, before the reset paths have
settled, and three near-identical lines would raise the question of which is authoritative.
The illegal-value paths already log at `warning` and `critical`.

Logging-only, so it is safe for the CPU DRPs that share this file.

## `/usr/local/bin/drp_gpu` is node-local and goes stale silently

`drp_gpu` needs `CAP_SYS_ADMIN` to `cuMemHostRegister` the FPGA registers, and `setcap`
fails on wekafs, so the executable has to live on local disk.  That means a per-node copy
that nothing keeps in step with `$TESTRELDIR/bin/drp_gpu`, and two commands after every C++
rebuild, on every node:

    sudo install -m 0755 $TESTRELDIR/bin/drp_gpu /usr/local/bin/drp_gpu
    sudo setcap cap_sys_admin+ep /usr/local/bin/drp_gpu

The `setcap` is **not optional**: `install` drops file capabilities, so skipping it leaves a
binary that dies at startup for a reason that looks nothing like the cause.

The image check earns its keep.  On drp-srcf-gpu008 on 2026-09-15 the local copy was from
Sep 9 -- predating the `CAP_SYS_ADMIN` drop, the `libdetector` linking, the `dlerror()`
reporting and the null-`m_drp` guard -- and the DRP refused to start:

    <E> Running /usr/local/bin/drp_gpu, which differs from .../install/bin/drp_gpu
    <C> Refusing to start on an image mismatch.  Pass -k imageCheck=warn to continue anyway

Without it the node would have quietly run month-old code.  Note also that `build_all.sh`
does **not** necessarily move `$TESTRELDIR/bin/drp_gpu`: a day of Python-only changes leaves
ninja with nothing to relink, so the timestamps can look stale when they are correct.

**This does not scale to twenty nodes.**  Two manual sudo commands per node per rebuild,
which someone has to remember, is the kind of step that gets skipped on the node nobody was
thinking about.  The answer is to make installing it a deployment step rather than a habit -- ansible, or
whatever installs `drp_gpu` in the first place.

**The capability itself cannot be delegated**, so do not go looking for a way around it.
An earlier version of this note suggested a setuid-root helper that performs the mapping and
passes a descriptor back; that does not work, for the reason Ric had already established.
The privileged call is
`cuMemHostRegister(ptr, size, CU_MEMHOSTREGISTER_IOMEMORY)`, which registers a host pointer
into *the calling process's* CUDA context.  Passing an fd over `SCM_RIGHTS` would let the
parent `mmap` the BAR itself, but the thing that needs `CAP_SYS_ADMIN` is the CUDA
registration, and that is inherently per-process: a mapping made in another process is not
valid in this one.  The same wall stopped the idea of putting the capability on a `.so`,
from the other side -- file capabilities attach to executables, not shared objects.

## `Cores=` must name whole sockets, not the GPU's NUMA node

Found on drp-srcf-gpu008 on 2026-09-15, and fixed in `gen_gres_conf.py`'s `gpu_cores()`
the same day.  Publishing invalidated the node:

    State=IDLE+DRAIN+INVALID_REG
    Reason=gres/gpu GRES core specification 8-15 for node drp-srcf-gpu008 doesn't match
           socket boundaries. (Socket 0 is cores 0-31)

`gpu_cores()` derived the core set from the GPU's sysfs `local_cpulist`, which is its
**NUMA node**.  Slurm requires the set to fall on **socket** boundaries.

**Why it survived two nodes.**  gpu006 (2 sockets x 32 cores) and gpu001 (2 x 8) both run
NPS=1, so a NUMA node *is* a socket there and the two definitions coincide: `32-63` and
`8-15` were simultaneously "the GPU's NUMA node" and "socket 1", and the narrow
interpretation looked correct.  gpu008 is NPS=4 -- eight NUMA nodes over two sockets -- so
a GPU's NUMA node is eight cores of a thirty-two-core socket, and Slurm rejected it.  The
mistake needed a node with more than one NUMA node per socket to become visible.

The fix maps the GPU's local CPUs through `physical_package_id` and emits the containing
socket(s) whole.  Verified on the live nodes rather than argued: gpu001's `--check` stayed
green and both gpu006 and gpu001 still emit identical `Cores=`, so neither needed
re-publishing.

**This revises an earlier note here** which said only that `Cores=` is core indices rather
than CPU indices.  True but incomplete; the full constraint is core indices *on socket
boundaries*.  The NUMA node is still recorded in the comment above each record, so the
finer locality is not lost -- it is simply not something `gres.conf` can express.

Both times a `Cores=` mistake has bitten, it presented as `INVALID_REG` with a precise
reason string.  So `scontrol show node <node> -d | grep -E "State|Reason"` immediately
after `scontrol reconfigure` is the check that earns its place in the procedure; the
reason names the exact rule that was broken.

## `dkms-reload.sh` reported a reload that had not happened

Found on drp-srcf-gpu001 on 2026-09-15, and fixed in `dkms-reload.sh` the same day
(uncommitted in `~/git/aes-stream-drivers`, to go upstream with the `rdmaTest` timeout).

The script printed `==> reloading the module`, then `module matches what the next boot will
load`, then `==> done: datadev-gpu-dkms/7.6.0-17-g77badc8`.  All of that while the module
resident since 2026-09-11 -- `7.6.0-11-g06c52c6` -- was still loaded, with the new one
sitting unused on disk.

The verification was wrong, not merely weak:

    RUNNING=$(cat /sys/module/datadev/srcversion)
    ONDISK=$(modinfo -F srcversion datadev)

`srcversion` hashes the `.c`/`.h` files.  PR #319 changed only the Makefile, the build
scripts and the docs, so it is **byte-identical between the two builds** and the comparison
passes whichever module is resident.  The message was even honest -- "what the next boot
will load" is a statement about `modinfo`, i.e. about disk -- and was read as though it
said "what is running".

The field that discriminates is `GITV`, compiled in per build and printed by
`dma_common.c:1451` as `DMA Driver's Git Version` in `/proc/datadev_*`.  The fix compares
that against the version just installed and fails with the remedy.  Verified against the
broken state: the new check errors, the old one passed.

**So the check to trust is `/proc/datadev_*`'s `Git Version`.**  `dkms status` reports the
*package*; `srcversion` reports the *sources*; neither reports the resident module.  Also
useful: the driver announces itself in the kernel log on every load --
`datadev: aes-stream-drivers <ver>` followed by `datadev: Init` -- so `dmesg -T | grep
datadev` dates the last real reload, and `/dev/datadev_*`'s ctime is the probe time.

Why the reload did not happen is **unexplained**, and the following were checked and
eliminated, so do not spend time on them again:

- No stale duplicate: only one `datadev.ko` exists on that kernel, in `extra/`, and it is
  what `modinfo -n` resolves to.
- Nothing held the module: `refcnt` 0, no entries in `/sys/module/datadev/holders/`, no
  process with the device open.
- Nothing reloads it behind our backs -- the cause the script's own error text suggests.
  There is no `datadev` systemd unit, and nothing matching in `/etc/modules-load.d`,
  `/etc/rc.d/rc.local`, `/etc/rc.local`, `/etc/systemd/system` or `/etc/sysconfig/modules`.
- Not a logging gap: neither `dmesg` nor `/var/log/messages` mentions datadev between the
  Sep 11 load and the manual reload at 14:44, and both record the `Exit.`/`Init` pair for
  loads that did happen.
- A manual `modprobe -r datadev && modprobe datadev` immediately afterwards worked
  cleanly, logging the expected pair and bringing `/proc` to `7.6.0-17-g77badc8`.

The script's own guard should have caught a failed `modprobe -r` and exited 1, and no error
was printed, so on the evidence `lsmod` found the module absent -- which the kernel log
contradicts.  One of those must be wrong and there is no artefact left to say which.  Left
open deliberately rather than guessed at; the new check turns a silent recurrence into a
loud one, which is the part that matters.

Worth noting the shape, because it is the same one PR #319 addressed one layer down: a step
that reports success without testing the thing that matters.  The build is now honest about
what it produced; the reload was not honest about what is running.  Both `rdmaTest`'s
timeout-free wait and this belong to that family.

## `/dev/nvidia*` survive a module reload, but not a boot

Measured on drp-srcf-gpu008 on 2026-09-18, after unloading and reloading nvidia: the device
nodes' timestamps stayed at the *previous boot* (`18:19:03` the day before) while the module
reloaded at `16:25:57`.  They were never removed.

`nvidia-modprobe` creates them as ordinary character special files -- plain inodes with major
195 and a minor number -- so nothing unlinks them when the module unloads.  The nodes become
non-functional and start working again when the driver returns.

So the hazard is narrower than it first appears.  **A module reload does not endanger
slurmd**; only a *boot* does, because `/dev` starts empty and something must create the nodes
before slurmd validates its `gres.conf` `File=` entries.  That is what happened on
drp-srcf-gpu007: nodes absent after a boot, not after a reload.

An earlier warning in this file conflated the two.  Reloading the nvidia module on a node with
published gres records is safe; rebooting one whose `nvidia-powerd` is disabled is not.

## The dkms path silently drops the NVIDIA module parameters

Found on drp-srcf-gpu008 on 2026-09-18.  `comp_and_load_drivers.sh:104` loads nvidia with two
parameters:

    insmod nvidia.ko NVreg_OpenRmEnableUnsupportedGpus=1 NVreg_EnableStreamMemOPs=1

Nothing in the dkms path supplies them -- not `dkms.conf`, not `build-nvidia.sh`, not
`dkms-reload.sh` -- and a boot-time `modprobe` has no command line.  So **converting a node
from `comp_and_load_drivers.sh` to dkms silently loses both.**  Exactly the same trap as
datadev's `cfgMode`, which we did catch; I did not think to check the nvidia side as well.

The symptom is remote from the cause.  `rdmaTest` does its FPGA handshake from the host with
`cuStreamWriteValue32`/`cuStreamWaitValue32`, checks
`CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS_V1` at startup, and aborts:

    WARNING: device does not support CUDA Stream Operations; this code may not run.
    Selected GPU lacks stream memory ops; aborting

Confirmed by reading the attribute directly: 0 on all five GPUs with
`EnableStreamMemOPs: 0` in `/proc/driver/nvidia/params`.  The tool had worked earlier the same
day, before a reboot, because nvidia had then been loaded by `comp_and_load_drivers.sh`.

**`drp_gpu` does not need it.**  Its kernels write the GpuAsyncCore registers directly through
the `CAP_SYS_ADMIN` `IOMEMORY` mapping rather than using stream memory ops, which is confirmed
rather than assumed: the 33 kHz runs on 2026-09-17 after the reboot had the parameter at 0.
So this affects `rdmaTest` and any future host-driven handshake, not the DAQ.

The fix is a modprobe.d file, staged as `nvidia-daq.conf` in the session directory:

    options nvidia NVreg_OpenRmEnableUnsupportedGpus=1 NVreg_EnableStreamMemOPs=1

Named `nvidia-daq.conf`, not `nvidia.conf`, because the packaged
`/usr/lib/modprobe.d/nvidia.conf` already sets three unrelated parameters
(`NVreg_TemporaryFilePath`, `NVreg_EnableS0ixPowerManagement`,
`NVreg_PreserveVideoMemoryAllocations`) and a same-named file in `/etc` would shadow it.
`options` lines from different files are additive; only a repeated *parameter* is resolved by
file order.

**`NVreg_OpenRmEnableUnsupportedGpus=1` is carried over without justification** -- its
necessity here has not been established.  It is included so a converted node matches what the
insmod path provided rather than differing in a way nobody notices.  Worth asking whoever
added it to `comp_and_load_drivers.sh` whether it is still needed.

Raised with Jeremy on PR #323, since the gap is in aes-stream-drivers rather than in lcls2:
the dkms packaging could reasonably ship such a file, or at least document that the insmod
parameters are not carried over.

## Module parameters are invisible in sysfs; /proc is the only source

Found 2026-09-15 while planning the gpu008 conversion.  Every `module_param()` in
`data_dev_top.c` is declared with permission `0`, so `/sys/module/datadev/parameters/`
**does not exist**.  There is no way to ask a loaded module what parameters it was given.
`/proc/datadev_*` is the only source: `Buffer Mode` reports `cfgMode`, and the device
names themselves report `cfgDevName`.

Same shape as the `Git Version` lesson: for anything about the *resident* module -- its
version or its parameters -- `/proc/datadev_*` is authoritative and everything else
(`dkms status`, `srcversion`, `modinfo`, the modprobe.d files) describes disk or sources.

This matters when converting a node whose driver was insmod'd rather than modprobe'd,
because the parameters are on a command line inside a script rather than in
`/etc/modprobe.d`, and a naive reload silently reverts them to the built-in defaults:

    static int cfgMode    = BUFF_COHERENT;   // data_dev_top.c:56
    static int cfgCont    = 1;               // :57
    static int cfgDevName = 0;               // :68

`comp_and_load_drivers.sh:115` passes `cfgMode=2` (BUFF_STREAM), and nothing else.  So on
such a node the *only* non-default parameter is `cfgMode`, and a modprobe.d file must
carry it or DMA buffer allocation quietly changes from streaming with explicit cache
synchronisation to coherent.

## dkms rebuilds per node; nothing is cached on /sdf

Asked 2026-09-14 while planning the move off gpu006.  All of dkms's state is node-local --
`/var/lib/dkms` on `vg_raid-lv_var`, `/usr/src` and `/lib/modules` on `vg_raid-lv_root` --
and it keys its bookkeeping on package/version/kernel/arch there.  So a fresh node has no
record, and `dkms build` compiles from scratch including the `PRE_BUILD` NVIDIA rebuild,
which is the several-minute part.  Identical kernel, CUDA and OS do not help; there is no
cross-node cache.  Only the checkout and the `make dkms` tarball are shared via `/sdf`,
and those are the cheap half.

Fine for two nodes, not for twenty.  dkms 3.2.2 here has no `mkrpm` (dropped in 3.x), but
it does have a native route worth testing before the November boxes arrive:

    dkms mktarball -m datadev-gpu-dkms -v <ver> -k <kernel> --binaries-only
    dkms ldtarball --archive=<tarball>          # on every other node

`--binaries-only` packages the built module, so `ldtarball` installs it without
compiling.  Needs an identical kernel *and* a matching nvidia module version, since the
GPU build resolves nvidia's symbols.  **Untested** -- recorded as the lead, not a recipe.

## rdmaTest is the way to exercise the driver's GPU path without the DAQ

`data_dev/app/bin/rdmaTest` (built with `make cuda`, not `make`) registers CUDA buffers via
`gpuAddNvidiaMemory` and DMAs into them, so it tests the driver's GPU path directly rather
than through a DRP that dies in detector configuration.  Needs `CAP_SYS_ADMIN`, because
`rdmaTest.cu:206` calls `gpuMapHostFpgaMem` -> `cuMemHostRegister(..., IOMEMORY)`, the same
call `drp_gpu` needs it for; simplest is `sudo`.

Two things learned the hard way on 2026-09-15:

- **`-s` must be a multiple of 64 KiB**, which is GPU page/BAR granularity and has nothing
  to do with the driver's `cfgSize`.  The default, `0x100000`, is already valid; passing
  `cfgSize`'s 4096 to match is simply wrong.  The `-h` text does not mention the rule.
- **It hung on drp-srcf-gpu006 with `-l` on `/dev/datadev_84`.**  The hang is the first
  statement of the receive loop, `cuStreamWaitValue32(rxBuffers[0] + 4, 1, GEQ)`, waiting
  for the FPGA to raise the doorbell for event 0.  So no event ever arrived.  Setup had
  armed the free list and cleared the doorbell, so the GPU side was ready; what is missing
  is a data source.  Nothing in `rdmaTest` visibly starts a pattern generator, and the
  firmware is `InterCardTest` -- which may well require a *partner* card, and `84` is the
  only card carrying that firmware on this node.  Left for Mudit and Jeremy, who own it.

  **Answered by Mudit, 2026-09-15**: the flow has to be started from the InterCardTest GUI,
  which `rdmaTest` does not do and does not mention.

      cd /sdf/home/m/mmishra9/project/axi-pcie-devel3/axi-pcie-devel/software/scripts
      python interCardGui.py --dev /dev/datadev_84

  then set `PrbsTx.TxEn` True.  So the data source is a PRBS generator that is off by
  default, and the loopback path is FPGA PRBS -> GPU -> FPGA.  Note the card was
  `/dev/datadev_84` under probe-order naming on gpu006; after `cfgDevName=1` it is
  `/dev/datadev_84` there by coincidence of bus number, so check the name rather than
  assuming.

  Two consequences.  The timeout below should say what to check, not just that nothing
  arrived -- "is a transmitter enabled?" points at this.  And `rdmaTest` arms the GPU side
  and calls `gpuEnableTx`/`gpuEnableRx` but never starts a source, so a first-time user hangs
  with no output; that is a documentation gap at minimum.

  **Both paths now run on drp-srcf-gpu008 (2026-09-18)**, with `a1` reflashed to InterCardTest
  and `NVreg_EnableStreamMemOPs=1` in place.  Receive-only reached 1.18M events and loopback
  1.44M, both with zero invalid events, at 262 kB per event -- so the PRBS generator produces
  quarter-megabyte frames rather than filling the 1 MiB default buffer, and the throughput
  figure reflects the source rather than the buffer size.

  The loopback run also tested the case the timeout is really for, deliberately: `PrbsTx.TxEn`
  was turned off after 1.44M events and the wait reported it.  A source that stops *after* a
  period of apparently normal operation is harder to diagnose than an absent one, because
  nothing distinguishes it from a quiet detector, and previously it was a silent hang.

  Note `rdmaTest` reaches only 3.4-8.6 GB/s where the GPU DRP sustains 12.788 GB/s on the same
  hardware.  Consistent with Jeremy's remark that it was not written with performance in mind,
  and the per-event synchronous pageable copy found in the backtrace below is the obvious
  suspect.  So its numbers should not be quoted as a hardware capability.

### Where the hang actually is, which is not where it looks

Worth knowing before anyone attempts this again.  The obvious reading is that the program
blocks in `cuStreamWaitValue32` on the event-0 doorbell, so a bound belongs around the
following `cuStreamSynchronize`.  Putting one there **does not work**, and produces a
process that spins silently with no output at all.  A `gstack` on drp-srcf-gpu008 on
2026-09-17 showed why:

    #10 cuMemcpyDtoHAsync_v2 ()
    #11 runSimpleLoop (s=...)

The thread is blocked inside the **enqueue** of the header copy, before any polling code is
reached.  `hdr` is an ordinary local, so the destination is *pageable* host memory, and CUDA
documents device-to-host copies into pageable memory as behaving synchronously: the driver
stages through an internal pinned buffer and waits for the stream.  Enqueued while the
stream is still blocked on the doorbell wait, it blocks the host indefinitely.

So the wait has to be **drained before the copy is enqueued**, not after: enqueue the wait,
poll until it clears, then enqueue the copy and synchronise.  With that ordering the stream
is idle when the copy is enqueued and it returns promptly.  Two wrong guesses preceded the
backtrace -- that `cuStreamQuery` was blocking, and before that that the doorbell needed
releasing from a second stream -- and neither survived one `gstack`.

**A per-event host round trip, deferred.**  The same pageable copy means every event pays a
synchronous host round trip for a 16-byte header.  Pinning `hdr` with `cuMemAllocHost` would
make it genuinely asynchronous.  Jeremy notes `rdmaTest` was not written with performance in
mind, so this is a future enhancement rather than a defect -- but the tool does print a
GiB-transferred figure, so it is worth knowing those numbers carry that cost.

- **Give `rdmaTest`'s doorbell wait a timeout, and PR it.**  Agreed 2026-09-15, deferred
  off gpu006.  `cuStreamWaitValue32` on the event-0 doorbell blocks silently and for ever
  when no data source is running, which is the *normal* first-time experience: it turned a
  one-run question into a code read.  It should say "no event in N seconds -- is a data
  source running?" and exit non-zero.

  Note the wait is a *device-side* stream operation, so there is no host-side timeout to
  set: it needs either `cuStreamWaitValue32` in a loop against a host-visible copy of the
  doorbell, or a bounded `cuStreamSynchronize` poll.  Not a one-liner, which is why it was
  not done on the spot.  Applies to the non-loopback path too.

  The push needs an sdfiana node; DAQ nodes cannot reach GitHub.

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
