# GPU DRP to-dos

Working notes for the `features/gpu` branch.  Each item records enough context to
be picked up cold, because the reasoning behind these decisions is otherwise only
in people's heads.

**These files are where the history goes, not the source.**  If a comment in the code
explains what something used to be, when it was found, or why it changed, it belongs
here or in `FINDINGS.md` instead.  See the comment-volume item below.

## Where the rest of this file went

Split 2026-10-07, because 3887 lines held two unrelated projects and the half
needed when reading this code was interleaved with the half needed when
converting a node.

| destination | holds |
|---|---|
| this file | open `drpGpu` work |
| `FINDINGS.md` | settled `drpGpu` explanations, kept because they were expensive to establish |
| `RULES.md` | the code rules that bite while editing |
| Confluence, under **GPU DRP** | the node and fleet material: grub, WEKA, dkms, BIOS, the node table, Slurm, RTPRIO |

Record the day's investigation in HANDOFF, promote to this file only what is still
open, and move an item to `FINDINGS.md` once it is settled.

## Comment volume needs a cleanup pass

The GPU DRP's comments have grown past what a reader can take in, and the pattern is
consistent: commit-message content written into source files.  What a line used to be,
when a problem was found and by whom, where development is heading, numbers restated
in prose that will go stale.  None of that helps someone reading the code now, and it
buries the code that does.

Measured 2026-09-30: `features/gpu..features/gpu-raw-calib` adds **387 comment lines
out of 1079**, 36%.  The worst offenders by comment-block length are `MemPool.hh`
(an 89-line block at the `HOST_REARMS_DMA` macro, and a 19-line one at
`HOST_LAUNCHED_REDUCERS`), `PassthruShim.hh` (31 lines before the class),
`Detector.hh` (a 15-line block on `setPassthru`/`rawSize`), and `PGPDetector.cc`
(a dozen blocks of 5-15 lines through the recorder).

The standing request is **one or two lines per inline comment**.  Worth keeping: a
hardware constraint, a non-obvious ordering requirement, a trap that looks like a bug.
Worth cutting: anything historical, anything a careful reader infers from the next
line, and any number that duplicates a constant.

Not urgent, but it compounds -- each addition is written in the style of what surrounds
it.  Best done as one deliberate pass rather than opportunistically, so the result is
consistent.

## Rules that will bite you: moved out of this file

Split 2026-10-05, because the two halves have different audiences and neither was
findable in a 3800-line file.

- **Code rules -> `psdaq/drpGpu/RULES.md`**, beside the source.  The ten that bite while
  editing this code: cuFile needing a quiescent device, `setPassthru()` only from
  Configure, a GPU `TriggerPrimitive::event()` having to advance `*state` to 2, deciding
  from `pyld.raw` rather than `keepRaw`, `sizeof(*ptr)` as the divisor, and the rest.
  In the repo deliberately: a stale rule about code is worse than an inconvenient one, and
  a clone plus a `grep` finds it.
- **Node and driver rules -> Confluence**, source kept here as
  `confluence-node-rules.storage.xml` in the session directory.  The eleven that bite while
  working on a node: `nvidia-powerd` enabled, `nvidia-persistenced` disabled, `CpuSpecList`
  in abstract IDs, clearing a Slurm drain, reading `modprobe.d` before converting to dkms.
  A page gives a TOC and Expand macros, which suit a standing-at-the-node checklist and are
  what a markdown file cannot offer.

  **Placement, Ric's call 2026-10-05: a child of GPU DRP, _beside_ the datadev driver and
  `gen_gres_conf` pages, not under either.**  The rules span both topics -- some driver,
  some Slurm -- so filing them under one would misplace about half, and a checklist is a
  sibling of a procedure rather than a subsection of it.  The page is titled
  **"GPU node rules"**, which is the title the parent page's cross-link resolves by --
  fragment 5 of `confluence-gpudrp-additions.storage.xml`.  Parent is page id 685820459.
  **Pasted by Ric 2026-10-05.**  Renaming the page means fixing that `ri:content-title`, or
  the link dangles.

  **Owed: hyperlink `RULES.md` from that page once this branch is on `master`.**  It is
  deliberately a plain path today, because the file exists only on `features/gpu-raw-calib`:
  a `master` link would 404 until the merge, a branch link would die when the branch is
  deleted, and a commit permalink would pin the 2026-10-05 text for ever -- stale rules being
  the failure mode that matters.  The page links the repo root instead and says why.  The
  `blob/master/psdaq/drpGpu/RULES.md` form is correct and safe to add after the merge.

The reasoning behind each rule stays in `FINDINGS.md`; the rules themselves
are the conclusions.  **Edit the `.storage.xml` source, not the live page**, or the two
diverge silently -- that has happened once already.

## Event building

- **The TEB posts Results that are never delivered, and the DRP wedges on the
  first of them.  Seen only with `ep_provider=sockets`.**  Six stalls at 13.6 kHz with
  two readout groups at different rates, 2026-10-08 (13:46, 17:14, 18:04), 2026-10-09
  (16:11, 17:57) and 2026-10-11 (13:28), all on gpu006.
  **It affects either epix**: four were `epixuhr3x2_1`, the 17:57 and 13:28 ones were
  `_0`, so an earlier "always `_1`" reading is withdrawn.
  Proven on 10-09 by the `consumed` poison bit; logs in
  `claude-sessions/2026-09-11-gpu-drp/stall-2026-10-09/`.

  **`tcp` does not lose writes, and that is the strongest handle on the cause.**  Same
  node, detectors, rate and code, with only `ep_provider` changed on both the DRP and TEB
  command lines: **11,584,951 batches at 13,684 Hz with `TCtbI_OrphCt` 0 and
  `TCtbI_NPrgCt` flat 0** (2026-10-11 14:44, logs in `tcp-2026-10-11/`).  That is **23.6x
  the `sockets` mean** time-to-first-loss (490833 batches over six stalls) and **7.3x its
  maximum**; under a constant per-batch hazard fitted to those six, surviving it has
  probability ~6e-11.  Treat that as an order of magnitude only -- it rests on n=6 and on
  the hazard being constant, which the UED experience argues against.  `NPrgCt` at zero
  means the matcher never had even one failed pass, not merely no losses.

  **It was a deliberate single-variable test**: `tcp` leaves `rx_attr->size` at 256, the
  same as `sockets`, so **CQ depth did not change** -- which makes a receive-CQ overrun
  the *less* likely explanation, since 256 entries too few at 13.6 kHz should have failed
  identically.  Do not lead with the CQ-depth story.  What is left pointing at `sockets`
  is its own progress/staging path: it is deprecated upstream ("most applications should
  instead use the tcp provider") and `tcp` was rewritten as a native MSG/RDM provider
  around libfabric 1.18.  It also explains why low-rate CPU DAQs never see this without
  needing a rate threshold -- they never enter the regime.

  **Caveat: this proves non-reproduction on `tcp` under these conditions, not a
  mechanism inside `sockets`.**  And it is one run, against a `sockets` distribution with
  a long tail -- it did once survive 1.58M batches.

  **gpu008's clean 33 kHz runs said nothing about this, contrary to how they were long
  read.**  No gpu008 log in `~/daq/logs` contains `ep_provider` at all, and its TEB links
  came up on **172.21.167.165 = `ibp113s0f0`, the InfiniBand interface** -- so those runs
  were `verbs` over IB, not `sockets`, and never contradicted a `sockets`-specific fault.
  They remain good evidence that the recording path is fast enough at 2.5x the rate with
  more instances; they are simply mute on the transport.

  **`sockets` ALONE IS NOT SUFFICIENT: gpu008 ran it clean, 2026-10-12.**  Six DRPs
  (`epixuhremu`) on `ep_provider=sockets` over 10 GbE, at ~19.25 kHz:

      TEB_EvtCt   503,641,618        TEB_BatCt   226,195,659
      TCtbI_OrphCt  0 on all six     TCtbI_NPrgCt  0 on all six
      TCtbI_DefSz   0                TCtbI_MisCt   0, BypCt 0, TEB_SpltCt 0

  **461x the gpu006 `sockets` mean time-to-first-loss (490,833) and 143x its maximum,
  with the trap armed and never firing.**  So the provider is necessary-looking but not
  sufficient, and gpu006 has something further.  The sharpest remaining difference is
  **two readout groups at different rates on gpu006 against one on gpu008** (`gpu6.py` sets
  `groups = platform + ' 3 4'`, `gpu8.py` just `platform`) -- which is also the
  configuration gpu008 had never exercised, the original "what changed" answer from 10-07.
  Secondary differences: real `epixuhr3x2` against `epixuhremu`, 10 GbE against IB
  hardware, and six contributors against three.

  **The duplicate-pulse-ID abort is a symptom of this, not a second bug.**  Read out
  of the 17:57 trace on 10-10: delivery stopped at exactly the stuck Input's Result and
  resumed **2026 events** later, with **2028** batch heads posted-but-never-received over
  a window compared by pulse ID -- the same plateau `TCtbI_DefSz` reports.  The duplicate
  arises because the TEB reuses a Results slot a lap later while the DRP still holds a
  pointer to it, and the DRP then reaches those bytes twice: once via `_defer`'s RoG-skip
  walk out of a preceding batch, once as a batch head in its own right.  The `ent` field
  in the received-ring dump is the position within the batch, which is what distinguishes
  the two routes.  **`_defer` is correct in both runs and wants no change**; the walk is
  required because delivery is per *batch*, not per batch entry.  Two theories died here:
  the remainder path's re-defer is legitimate (`pop_front` precedes it), and only one walk
  ever occurs.

  **PROVEN BY DIRECT MEMORY INSPECTION, 2026-10-11**: the trap below fired on its first
  run, and gdb on the spinning process read the Results slot the lost write belonged in:

      orphaned Input   0002dbbabe83df   L1Accept, RoG 14, pebble idx 1611
      TEB posted       idx 1121, dsts 0x7, entries 1, EOL      (teb0.log)
      slot 1121 holds  0002dbbac09e33   = +2026.6 events, consumed bit 0
      Tx CQ, that link posted 1578435, CQ errors 0, tmo 0
      _noCtxCnt 1576393 vs _batchCount 1576391 -- one-for-one, no credit leak

  The slot holds a write from a **later lap** with `consumed` clear, so it is a fresh
  arrival and not residue: the lap-N write **never reached memory**.  That separates the
  two candidate mechanisms -- a lost *completion* would have left the slot holding the
  awaited pulse ID -- and rules out the receive-side CQ handling, since there was nothing
  to mishandle.  The loss is inbound (TEB -> DRP): the TEB's first `fixup` for this DRP
  came **2043 events later**, i.e. only after it stopped contributing.  Logs and the
  patch are in `claude-sessions/2026-09-11-gpu-drp/stall-2026-10-11/`.

- **A second hang, upstream of the event builder: `TrgInpGen::_receiver` spins on a
  `RingIndexDtoH` that never advances.**  Seen once, 2026-10-11 on the `tcp` run, ~16
  minutes in: **both** epixes froze within 164 batches of each other while `timing_2`
  kept running past 13.1M.  **Not the transport bug** -- the orphaned-Input trap did not
  fire, correctly, because no Input was ever produced for it to orphan.

      TCtbI_DefSz 0, TCtbI_NPrgCt 0, TCtbO_InFlt 0    matcher side clean
      DRP_recordQueue / colQueOcc / dmaWtCtr0 / pblWtCtr0 all 0, dma errors 0
      both GPUs pinned at 100% util, 4491 MiB, doing no work
      threads: TEBreceiver S, Collector S, Recorder S, TigRcvr R (spinning)

  The spin is `while (m_trgInpGenQueue.h->pop(&index))` at `TrgInpGen.cu:423` over
  `RingIndex_DtoH.hh:55`, whose `pop` returns false while `tail == head`.  The TEB's 52
  `missing src 0` fixups all postdate the freeze by minutes, so they are a consequence.
  Both DRPs failing together while the timing DRP survives points at something common --
  a GPU-side ring or doorbell, or timing -- rather than a per-link fault.

  **Not inspected under gdb**: the DAQ was stopped before attaching.  Next time it hangs,
  read `RingIndexDtoH`'s head and tail and whether the GPU-side writer advanced; that
  distinguishes "the kernel stopped writing" from "the host index wrapped wrongly".
  Whether it relates to `tcp` at all is unknown -- it appeared on the first long `tcp`
  run, which is a correlation and nothing more.

- **Could an MMU/page-attribute conflict between libfabric and CUDA explain it?**  Ric's
  hypothesis, 2026-10-11.  **Not for the Results region, which is the one losing writes**:
  `EbCtrbInBase::_region` comes from `Pds::Eb::allocRegion` -> plain `posix_memalign` to
  page alignment (`utilities.cc:22`), and nothing in `drpGpu` ever passes it to
  `cuMemHostRegister`, `cudaHostRegister` or `cudaHostAlloc` -- checked by grep over
  `drpGpu/*.{cc,cu,hh}`.  Only libfabric's `fi_mr_reg` (via `setupMr`) touches those
  pages, so there is no second mapping to collide with.  The GPU DRP's CUDA host
  allocations are elsewhere: the DMA/write buffers, metric words, and `fpgaRegs` --
  the last being the one genuine `CU_MEMHOSTREGISTER_IOMEMORY` mapping
  (`GpuAsyncLib.cc:202`), which is FPGA BAR space and disjoint from the Results region.

  **The `tcp` run has since argued against it.**  Had a mapping-level conflict been the
  cause, changing only the provider should not have fixed it; `tcp` ran 11.6M batches
  clean over the same region, allocated the same way.  Not conclusive -- a different
  provider exercises those pages differently -- but it points away from the mapping and
  towards `sockets`.

  Worth keeping alive for one reason even so.  The pebble is also plain `posix_memalign`
  (`DrpBase.cc:93`) and is both RDMA-registered and read by CUDA kernels, so the
  *Inputs* path does have the overlap the hypothesis describes -- but Inputs are not
  what goes missing.  **The CPU DRPs share `EbCtrbInBase` and have no CUDA at all** --
  whether any CPU DRP has ever shown this stall is therefore a sharp, nearly free
  discriminator, and is not yet known.  Worth asking.

  **An orphaned-Input trap now catches this at the first loss**, 2026-10-11.
  `_orphaned()` declares an Input unpairable once **every** TEB has delivered a batch
  whose head pulse ID is strictly newer than it: events reach the TEB in pulse ID order
  and all three providers in use report `FI_ORDER_RMA_WAW` (measured on gpu006,
  libfabric 2.1), so posting order is arrival order into the Results region, and a TEB
  that is past the Input has delivered everything it ever will for it.  Sound
  independent of readout group, since `_deliver` pairs on pulse ID alone and a batch can
  carry entries for groups this DRP does not share.  **The comparison must use the batch
  head as received**, recorded in `_process` before `_defer`'s RoG walk can advance past
  it -- comparing against `_deferred.front()` would call an Input lost while its Result
  was still in flight, which is the error that cost the 19:23 run.  Replayed against both
  stalls, it fires at the "No progress begins" line (log line 330 of 11131 for 16:11,
  348 of 4728 for 17:57) rather than at the duplicate-pid abort ~2000 events later.
  Exported as `TCtbI_OrphCt`.  Still debugging-shaped in one respect: it aborts, and
  becomes production-ready by replacing that with the sweep below.

  **16 consecutive Results were posted and never arrived, with no error either
  side.**  The TEB posted slots 25..55 contiguously, one event apart, each
  `dsts 0x7`, `entries 1`, EOL -- nothing anomalous near the gap.  The DRP's
  received ring holds 4089 entries spanning `0002b87332bc8b..0002b8733887cf`, of
  which **3577 are older** than the stuck Input `0002b873366931`, so this is not a
  ring-wrap artefact; all 16 pids from the stuck one to +15 events are absent.

  **Poison says non-delivery, not overwrite.**  Slots 25..32 read `STALE` -- read by
  us and never rewritten -- with pids exactly `stuck - k*68`.  Slot 33, where the
  stuck Input belongs, held a pid **+2027 events (0.990 laps)** on: rewritten a lap
  later.  Had the Result arrived, slot 33 would have held the stuck pid with
  `consumed` *clear*.  Nor is it a wrongly-advanced index, which would leave a
  contiguous run of wrong-place pids.  So: **16 lost RDMA writes to one
  destination.**  Suspect the per-destination link, endpoint and MR -- the TEB posts
  each batch per destination, so a write landing for src 1 and 2 but not src 0 is a
  per-link event.  `ep_provider=sockets` was in force on both ends; varying it is what
  the `tcp` run above did.

  **The design invariant, and it holds -- two indices, not one.**  The pebble index
  increments per event entering a DRP and does *not* match across DRPs at different
  rates.  The Results/Inputs batch index derives from the pulse ID and *does* match
  across all contributors, which is why `EbEvent::_add` can overwrite `_immData` per
  contributor (`EbEvent.cc:76`).  Confirmed on 10-09: the stuck Input belongs at slot
  33 and the deferred front, +16 events, is at slot 49.  Because events are built in
  time order, a delayed epix blocks the build, so no newer Result is posted, so the
  common DRP's pebble buffers are not freed either, and it backpressures into
  firmware as deadtime.  **The TEB cannot lap the region past an unprocessed slot**,
  so the lapping seen at these stalls is downstream of the lost writes.

  **Both DRPs saturate at a stall onset; only one gets unstuck.**  10-09:
  `TCtbO_InFlt` reached ~2044 on *both* epixes at 16:12:25; `_0` drained to 0 by
  16:12:51, `_1` stayed pinned until it aborted at 16:13:25.  `TCtbI_DefSz`
  discriminates -- `_0` max 0 throughout, `_1` 2028 from 16:12:26 -- because `_1`
  could not drain without the Result that never came.  (An earlier note here said
  backpressure was working and the queue counters flat zero; that was a *good*
  stretch of the 18:04 run, not an onset.)

  **The TEB's fixup is a consequence.**  10-09: 52 fixups name src 0 (`_1`), one each
  src 1 and src 2; the three multi-bit masks are all one event at stuck+4121 events,
  during teardown.  The first fixup is at **stuck+2027 events**, after the DRP had
  wedged, so the 12 s timeout cannot be what starts this.

  **Two test methods worth keeping.**  Compare posts against receipts **by pulse ID**
  inside a window verified not to have wrapped, never by slot index, which conflates
  laps -- that error produced two false retractions, at 64 and at 4096 entries.  And
  the `consumed` bit distinguishes stale residue from a fresh Result.

  Still worth doing regardless of cause: **name the Input the DRP is stuck on**
  rather than abort on a duplicate pulse ID 2000 events later.  Bound `_deferred`
  well under `_numBuffers` and report the stuck pulse ID.  A fix also wants either
  consumption credit from every receiver or a bound on how far one may lag before the
  TEB is told; gating `TebContributor`'s post on free pebble buffers would at least
  turn the abort into a stall that reports itself.

  Dead ends, not to be revived without new evidence: `rmtAdx`/MR *mis-mapping* (a
  wrong region fails at the first post, not after thousands of laps -- distinct from
  the per-link loss above), the 64-tick batch threshold, SlowUpdates, and our own
  changes.  Detail in HANDOFF.

  Raising `pebbleBufCount` to 32768 remains available as a mitigation (memory
  checked: 23.9 GiB GPU of 140, 12 GiB host per DRP) but **treats a symptom** and
  cannot recover a lost write.  `timing_2` must be raised too --
  `DrpBase.cc:1690`'s `bufErr` requires the common RoG's range >= any secondary's.

- **Thread placement is unmanaged, and `cores:` is guessed.**  Nothing is pinned:
  `DrpBase.cc:1179-1180` hardcodes `core[0] = core[1] = -1` and `pinThread` is a no-op for
  `cpu <= 0` (`utilities.cc:44`).  Ric pinned threads before Slurm and it helped, but
  absolute CPU numbers became unmanageable once Slurm started assigning cores.

  **Slurm does not prevent pinning -- it defines the set to pin within.**  A job's cpuset
  is readable with `sched_getaffinity`, so the DRP can ask the kernel which CPUs it was
  granted and place its own hot threads across *those*, with no absolute numbers in any
  `.cnf.py`, no coordination between instances, and correct behaviour if Slurm moves the
  job to another socket.  `cores:` then becomes the only knob and it means something.
  Make `pinThread` allocation-relative (pin the k-th hot thread to the k-th granted CPU),
  and leave it off behind a kwarg until it is measured.

  **The thread census, from the 2026-10-11 gdb backtrace of one epix `drp_gpu`** -- the
  first hard numbers, and they say `cores: 10` is defensible:

      ours (hot)          4   TigRcvr, TEBreceiver, Collector, Recorder
      libfabric sockets   8   2 progress, 3 cm, 2 conn_listener, 1 pep_listener
      CUDA + cuFile       8
      EPICS/pvAccess     14
      rogue              14
      ZMQ                 6
      Python/prometheus   8
      ~62 total, of which ~58 blocked on futex/poll/recv/select; 4 runnable

  Hot-thread demand is ~7-8; `gpu6.py` grants 10 (`sacct`: `AllocCPUS 10`).  So the count is
  about right and the **placement** is what is unmanaged -- the four hot threads can share
  a core, and can land across the NUMA boundary from the GPU and datadev that
  `gen_gres_conf.py` works to keep local.  Unmeasured as a cause, but `DRP_evtLatency` of
  15-40 ms against 18.8 ms of CQ headroom makes scheduler placement a plausible
  contributor to the spikes.  gpu006 is 2x32 cores, NUMA=2; `gpu6.py` requests 27 of 64.

  **The 10 is gpu006's number, not a per-DRP requirement.**  `gpu8.py` runs six DRPs on a
  61-CPU node at `cores: 4`, where 10 each could not fit.  Measured there 2026-10-12 at
  19.25 kHz, each GPU DRP wanted **~330% of a CPU in its 4** -- six hot threads
  (`TigRcvr` ~102%, an unnamed `drp_gpu` ~93%, `Recorder` ~49%, `Collector` ~48%, another
  unnamed ~33%, `TEBreceiver` ~23%) timesharing four CPUs, with `sockets`' 8 progress and
  CM threads in the same cpuset.

  **That is not a rate ceiling, and the rate question has a different answer -- see
  the deadtime item below.**  `gpu8_save.py`, whose runs reached 33 kHz, used **the same
  `cores: 4`**, so 4 was once sufficient.  The honest statement is that the allocation is
  tight and unmanaged, not that it is the bottleneck.

  **WEKA does NOT overlap the DRPs' cores, checked 2026-10-12.**  `/etc/fstab` mounts
  `drpsrcf/ffb` with `num_cores=3`, and exactly three threads spin: `wekanode` pinned 1:1
  to CPUs **1, 2 and 32**, each at ~100%.  The GPU DRPs ran in `15-18`, `19-22`, `23-26`,
  `33-35` and `36-39`, so the intersection is **empty**.

  **The trap that produced a wrong answer here**: matching `ps` on the process name finds
  **134** WEKA threads across many CPUs, of which only 3 are busy -- the rest are idle and
  their momentary `psr` says nothing about where WEKA runs.  Intersect against the
  *pinned, spinning* threads (`taskset -pc` on a thread with real CPU time), not against
  wherever a sleeping thread was last scheduled.

  So `tstcam1_3`'s 3-CPU cpuset for a `cores: 4` request is **unexplained and not WEKA**.

  **A `cores:` value is a request; what the process gets is what `sched_getaffinity`
  reports.**  That is the argument for having the DRP log both.

  **Check the allocation against what the process needs, and say so.**  The DRP knows how
  many hot threads it is about to start -- it creates them -- and `sched_getaffinity` says
  how many CPUs it was given, so it can compare the two at startup and name the shortfall:
  a `cores: 1` run should log "started N hot threads in 1 CPU; M would be better", not
  silently timeshare.  That is the cheapest possible answer to "we just guess at it": the
  code advises, the config decides.  Derive the number from the threads actually spawned
  for the configured `nReaders`/reducer/recorder, not a constant, so it stays honest as the
  pipeline changes.

  **But threads spawned *underneath* us cannot be counted that way**, and they are the
  majority: libfabric's progress and CM threads (8 for `sockets`), CUDA, cuFile, rogue,
  EPICS and ZMQ all create threads the DRP never sees and whose number depends on the
  provider and the driver version.  A startup estimate must therefore be a floor, and said
  to be one.  Two ways to account for them, both cheap:

  - **Count them at a known-good moment.**  After Configure, walk `/proc/self/task` and log
    the total thread count beside the CPU allocation.  It costs one readdir and makes
    "62 threads in 10 CPUs" visible instead of invisible.  Name the ones we created so the
    remainder is attributable to libraries.
  - **Measure which are actually hot**, since ~58 of those 62 are blocked and irrelevant to
    sizing.  Per-thread `utime + stime` from `/proc/self/task/*/stat` differenced over an
    interval gives the per-thread share of a core; anything above a few percent is a real
    consumer.  Logging the top few once per run would settle `cores:` empirically rather
    than by argument, and would catch a provider that silently starts more threads after an
    upgrade.  `claude-sessions/2026-09-11-gpu-drp/tools/hotthreads.sh` does exactly this
    from outside the process and is the prototype to copy; validated against a program with
    two spinners and two sleepers, it reported exactly the two.  **Limit worth knowing:**
    `utime + stime` attributes CPU to whoever consumed it, so an AUTO progress thread woken
    13.6k times a second shows up correctly -- but a thread *blocked on a lock* held by
    another appears idle while still being on the critical path.  The hot count is a lower
    bound on what matters, not a contention map.

  The DRP advising on `cores:` from its own thread count alone would therefore *understate*
  the need, which is the safe direction for a warning but worth stating in the message.

  **Do the logging below first.**  Too many parameters are set by guess for tuning to be
  productive before the system reports what it actually chose.

- **`getNicIp()` must not fall back to Ethernet when InfiniBand is missing.**  It
  currently prefers the first IB interface and silently drops to Ethernet with only a
  warning (`Collection.cc:218-258`, "No Infiniband interface found - using Ethernet").
  **This has cost real debugging time**: when IB is expected across the DAQ and one node
  has lost its NIC, that node alone comes up on Ethernet, the mixed system cannot
  communicate, and nothing says why.  A warning in one log among dozens is not findable
  when the symptom is "the DAQ will not connect".

  Fail loudly instead.  The fallback is only correct when Ethernet is what was *intended*,
  which `forceEnet=yes` already expresses -- every caller passes it
  (`PGPDetectorApp.cc:570`, `drp/PGPDetectorApp.cc:730`, `BldDetector.cc:1190`,
  `UdpEncoder.cc:1243`, `PvaDetector.cc:1223`), so the deliberate case is already
  distinguishable from the accidental one.  So: no IB and no `forceEnet` should abort with
  a message naming the node and saying that the rest of the DAQ is presumably on IB, not
  warn and carry on.

  Two details found while reading it, 2026-10-12:

  - **The IB branch does not test `IFF_RUNNING` but the Ethernet branch does.**  So a
    *down* IB interface is still selected if it enumerates first, while a down Ethernet
    one is skipped.  gpu008 has `ibp113s0f0` up and `ibp113s0f1` down, so the asymmetry
    is live, not hypothetical -- it happens to pick the working one by enumeration order.
  - **Selection is per node and reported nowhere afterwards.**  The provider and interface
    actually in use had to be recovered by reading peer addresses out of the TEB's log
    (`172.21.167.x` = IB, `172.21.155.x` = Ethernet on gpu008), which is the same argument
    as the logging item below.  Log the chosen interface, its type and its address.

- **An L1Accept rate is a deadtime measurement, not a throughput measurement.**  Measured
  twice on gpu008 on 2026-10-12, four minutes apart, same code and trigger cadence:

      provider   DeadFrac   predicted            measured    agreement
      sockets     0.705     71428*(1-0.705)      19.25 kHz   ~91%
      verbs       0.558     31535 Hz             31.53 kHz   99.97%

  where the XPM reports `L0InpRate{partition=N}` = **71428 Hz** (13 ticks, the real
  trigger cadence) and `DeadFrac` per partition.  **`measured = L0InpRate * (1 -
  DeadFrac)`**, so in both runs the DAQ took everything the XPM offered it and no rate
  ceiling was reached.

  Consequences, because this is easy to misread:

  - **Comparing raw rates across runs compares deadtime.**  The historical "33 kHz" is
    itself a deadtime-limited figure; reaching it needs `DeadFrac` 0.538 against the
    0.558 measured, so a 1.5 kHz shortfall is a 2-point deadtime difference and not a
    regression.  Quote `L0InpRate` and `DeadFrac` beside any rate, or the number means
    little.
  - **The provider's cost shows up here, not as lost throughput.**  `sockets` runs 8
    progress and CM threads inside the DRP's cpuset where `verbs` offloads to the HCA,
    and the dead fraction fell from 70% to 56% when that changed.  That is the measured
    form of the `cores:` pressure above.
  - A CPU-starvation reading of a low rate is therefore not supportable from the rate
    alone -- an earlier version of this file argued one and it was wrong.

- **Log the libfabric configuration once, and log more of it.**  What provider and version
  a run used is the first thing asked when a transport problem is suspected, and today it
  is either absent or buried.  `EbLfClient::connect` (`EbLfClient.cc:58`) and
  `EbLfServer::listen` (`EbLfServer.cc:85`) print version, domain, fabric and provider
  under `-v`, but the client's sits inside the per-link `connect`, so `linksConnect` emits
  one copy **per link** -- N identical lines for N peers -- and the server prints its own
  besides.  Once per process is enough.

  Worth adding, since each has cost time to establish by hand: `rx_attr->size` and
  `tx_attr->size` as **negotiated** (the hint at `EbLfServer.cc:72` asks for 0, so the
  value is whatever the provider chose, and it bounds the CQ); the resulting
  `cq_attr.size`; `tx_attr->msg_order` (the `FI_ORDER_RMA_WAW` guarantee the orphan trap's
  reasoning depends on); `inject_size`; `mr_mode` as negotiated, which the sockets
  provider is known to rewrite (`Endpoint.cc:923`); and `data_progress`/`control_progress`.
  Log it unconditionally rather than under `-v`: it is a few lines once per process, and a
  run whose log lacks it cannot be diagnosed after the fact.  Log the CPU allocation in the
  same place -- the CPUs `sched_getaffinity` reports, their NUMA nodes, and which thread
  went where -- so that `cores:` and thread placement become checkable rather than hoped at.

  **`data_progress` is the sharpest reason to log this.**  We never ask for a progress mode,
  so we take whatever `fi_getinfo` returns first, and that ordering is not contractual:

      provider   infos   AUTO   MANUAL   we take [0]
      sockets      12     12       0     AUTO
      tcp          30     12      18     AUTO
      verbs        26      6      20     AUTO

  **AUTO is what we want as the code stands** -- under MANUAL, reading a CQ is what advances
  the provider, and a poller that reads one completion per call with a 100 ms `pend()`
  timeout would couple all progress to its own call rate.  The clean 11.6M-batch `tcp` run
  was on AUTO.  But if a libfabric update reordered those 30 infos we would silently get
  MANUAL, and the symptom would look like an unexplained throughput collapse.  One logged
  line turns that into a one-line diagnosis.

  MANUAL is worth revisiting later, and only after **batched CQ reads** plus a poller that
  drives progress on a schedule rather than only when it wants a completion.  The prize is
  deleting eight unpinned provider threads from a latency-sensitive process.

  Measured on gpu006, libfabric 2.1, requesting what `Info::initialize()` requests --
  the ceilings differ by provider and the failure mode is not a clamp:

      provider   default   max honored   over-request behaviour
      verbs      384       65536         honored exactly
      sockets    256       256           fi_getinfo returns -FI_ENODATA (whole EP fails)
      tcp        256       65536         honored exactly

  **The 384 is `verbs`' default, not anyone's ceiling** -- which is exactly why it reads
  as a limit: the hint is 0, so 384 is simply what `verbs` hands back, and it appears in
  every log on an IB system.  `verbs` accepts 65536 if asked.  **Only `sockets` has a real
  ceiling**, and it equals its default, so there the queue cannot be deepened at all.

  Note `fi_cq_open` accepts `size = 65536` on **all three**, so "the CQ opened, therefore
  it is deeper" is false -- the refusal happens in `fi_getinfo` on `rx_attr->size`.  That a
  default should have been mistakable for a ceiling for years is the argument for logging
  the negotiated values: it makes this checkable instead of folkloric.

- **Close the three transport error-reporting gaps.**  The Tx side is now covered:
  `EbLfLink::_reapTx()` drains the Tx CQ from `post()`, counts `posted`/`completed`/
  `errors` per link and dumps `fi_cq_err_entry` on `-FI_EAVAIL`.  Because
  `EbLfClient::connect` creates a **separate CQ per link** (`EbLfClient.cc:65`), a Tx
  error is unambiguously attributable to one peer.  Three gaps remain, in the order
  they matter:

  1. **The Rx CQ is shared across all links, so a receive *error* cannot name one.**
     `EbLfServer::_linksConnect` creates one CQ for every peer (`EbLfServer.cc:131`,
     "All links share the same rxCQ"), and `dumpCqError` prints `op_context`, which is
     NULL for a remote write -- the same fact that refuted the credit-leak theory.  So a
     receive-side error says "a TEB", not which one.  Harmless with one TEB, wrong with
     up to `MAX_TEBS` = 4.

     **Recommendation: keep the shared CQ.**  The design is sound -- the sender's
     identity travels in the **immediate data**, not in `op_context`
     (`ImmData::src(data)` at `EbCtrbInBase.cc:494`, `EbAppBase.cc:296`), which is why a
     NULL `op_context` on a remote write is harmless on the success path.  Splitting
     would mean 64 queues on the TEB and polling 64 instead of 1, worse for a
     single-threaded poller.  Instead **decode src from the error entry's `data` field**,
     which `comp_error_dump` already prints (`Endpoint.cc:2082`), so an error can name
     its link without any new queues.  The higher-value change on this list is to
     **batch the CQ reads** (`comp(cqEntry, N)`): today `_poll` takes one completion per
     call.
  2. **Reaping the Tx CQ is now `post()`'s own job: DONE 2026-10-12.**  `_reapTx()` is
     called at the end of both `EbLfCltLink::post` (RMA, `EbLfLink.cc:570`) and
     `EbLfLink::post` (inject/send, `:612`), immediately after `_posted` is incremented.
     It is `protected`, the derived `post()` being its only caller, and there are no
     external call sites left.

     **Why there and not at the call sites.**  `post()` is the only place that knows a
     write has just been queued -- it is where `_posted` is counted -- so putting the two
     together makes it impossible to forget.  The previous arrangement had the call at
     five of six `->post()` sites, and before 10-12 at only one of them; a seventh site
     added later would have been missed again.  It also collapses five calls plus five
     comments into one.

     `EbAppBase.cc`'s post needed no special case in the end: its links are `EbLfSvrLink`,
     accepted with `txcq = nullptr` (`EbLfServer.cc:143,146`), and `_reapTx()` returns 0
     at its first line without one, so the base `post()` simply does nothing there.
     Giving server links a Tx CQ is the prerequisite for reporting a failure on that path,
     and is not done.

     **Checked, because it would be the same class of bug as gap 3:** `Endpoint::writedata`
     also touches the Tx CQ on its `-FI_EAGAIN` path (`Endpoint.cc:1350`), so there are two
     consumers within one `post()`.  It is safe -- that drain goes through
     `check_completion_noctx` -> `comp()` -> `fi_cq_read`, which on an error returns
     `-FI_EAVAIL` and **leaves the error entry queued** for `fi_cq_readerr`, and only
     `_reapTx()` calls that.  So the EAGAIN drain cannot swallow an error entry.

     Cost: a non-blocking `fi_cq_read` of up to 8 entries, `-FI_EAGAIN` on an empty queue,
     error path only on `-FI_EAVAIL`.  Measured in production rather than argued: the same
     per-batch-per-link call ran in gpu008's TEB at ~49000 reads/s for 15000 s with no
     attributable slowdown.  The MEB event post is not hot either -- it runs only when
     `m_mon.enabled()` and `result.monitor()` (`TebReceiver.cc:113-117`).

     **`TebContributor.cc`'s batch post is DONE, 2026-10-12.**  It posts Inputs upward and
     is the exact mirror of the path proven to lose Results, so an Input lost on the way up
     gave the same silent wedge from the other direction.  `_orphanDump` now also prints
     the per-TEB `posted`/`completed`/`errors`/`tmo`, so the dump answers the direction
     question locally instead of requiring the TEB's log -- a nonzero error count there
     means the loss was ours and outbound.  Needed a `TebContributor::links()` accessor,
     `_links` being private.

     Note an Input lost upward draws no Result by design: `teb.cc:622` posts to
     `_receivers(rogs) & ~event->remaining()`, so a contributor still outstanding is
     masked out of the destinations.  It would therefore present as a missing Result,
     not as a missing Input -- which is why this was worth closing even though no Input
     loss has been observed.

     **Cost is already measured in production, not estimated**: this is the same
     `fi_cq_read`-per-batch-per-link call the TEB has run at `teb.cc:961` since 10-11, and
     gpu008's TEB sustained **8162 batches/s x 6 links ~ 49000 reads/s for 15000 s** with
     no attributable slowdown.  A DRP does ~8162/s against one TEB, an order of magnitude
     less.
  3. **`_process` can swallow a non-fatal `pend()` error.**  It logs and returns `rc`;
     `receiver()` escalates only on `-FI_ENOTCONN` or when `rc == rcPrv`
     (`EbCtrbInBase.cc:446-459`), so an error that *alternates* with success is logged
     for ever and never becomes fatal.  Decide deliberately which errors are fatal
     rather than letting repetition be the test.

- **Sweep out a missing Result rather than aborting on it.**  Ric's proposal, 2026-10-10.
  The Results/Inputs matcher has no timeout on a Result that never arrives and does not
  flush an incomplete pair when a newer complete one is available, so one lost write ends
  the run.  Instead: time out the stuck Input, fabricate a Result with `persist` set,
  `monitor` clear and damage set so the Input is still recorded, then log and count it for
  Grafana.  A missing Result should not abort.

  Precedents to copy rather than reinvent.  `EventBuilder::_flush`
  (`EventBuilder.cc:346`) already implements both halves -- an age timeout and retiring an
  older incomplete event once a newer complete one in the same RoG arrives -- and
  `EbAppBase::fixup` (`:499`) sets `Damage::DroppedContribution`, counts it, notifies the
  UI and deliberately does **not** abort (the `abort()` there is commented out in favour of
  going quiet after 50).  `_deliverBypass` (`EbCtrbInBase.cc:793`) already fabricates a
  `ResultDgram` from an Input with `persist(true)` and `setEOL()`, and leaves `monitor`
  clear because no MEB buffer was allocated -- which is also correct for a Result that
  never came.  Damage already reaches Grafana: `TebReceiverBase::process` does
  `damage.increase()` and feeds `DRP_Damage` and the `m_dmgType` histogram.

  **Two constraints.**  `process` aborts unless the pebble index is contiguous
  (`DrpBase.cc:1010`), so a stuck Input cannot just be skipped: delivering a damaged Result
  per swept event is what keeps the index advancing, which makes the fake-up the mechanism
  and not a nicety.  And no-progress is *legitimate* when multiple TEBs present Results out
  of order (`EbCtrbInBase.cc:633,933`), so the timeout must exceed the worst legitimate TEB
  skew and the flush should require a complete, matching-RoG successor -- firing during
  healthy running would fabricate damage silently, which is worse than today's loud abort.
  Choose the timeout deliberately: the TEB's `EB_TMO_MS` is 12 s, far too long here, since
  at 13.6 kHz the 2048-slot Results region laps in ~150 ms, after which the deferred
  pointers are stale.

  `EbCtrbInBase.cc` is shared with every CPU DRP, so this changes recorded data for all of
  them; a kwarg for the timeout and the policy is wanted.  The `tcp` run this was gated
  behind has been taken, so this is now unblocked -- but it still masks the clearest
  signal of the cause, so keep a way to run with the trap armed and the sweep off.
  This is also what makes the orphaned-Input trap production-ready: replace its spin and
  abort with the sweep.

  Not part of this: putting the TEB/MEB event builder in `TebReceiver`.  Considered and
  rejected when it was written -- 64 contributors and multiple RoGs is overkill here, and
  supporting it in three places would add more complexity than it removes.

- **A published MEB buffer index is stranded if no DRP contributes for it.**  Low priority,
  and **not** a blocker for the sweep above -- an earlier reading of this was wrong.  The
  usual case is already safe: `EbEvent::_add` takes `_immData` from any contribution
  carrying a valid value (`EbEvent.cc:75`) and every DRP receives the same `monBufNo`, so
  one surviving contribution carries the index, `_retire` still calls `process()`
  (`EventBuilder.cc:333`), and the buffer returns via `_deleteDatagram`.  The MEB loses a
  payload and damages the event, but its accounting stays intact.

  Only when *no* contribution arrives for a published index does the MEB never get a
  trigger, never build, and leak the index allocated in `_requestDatagram`
  (`monReqServer.cc:263`): no timeout path returns it, and exhaustion surfaces only as
  "No free buffers available" with monitoring quietly stopping.  Retiring an index once a
  later event has passed is the obvious fix and is **not sound as stated** -- the MEB
  spreads requests over TEBs by `idx % _mrqLinks.size()` (`:285`) and each TEB allocates on
  its own schedule, so completion order is not publication order.  An age-based sweep that
  assumes no ordering would be needed.  A related pre-existing leak is already noted in the
  source at `:292`.

- **`nPidJumps` cannot work as conceived, and was never committed.**  The check and its
  metric exist only in `wip-2026-10-12-full-diagnostics.patch`; nothing is owed in the
  code.  Kept here so the idea is not re-proposed.  It tried to spot a trigger the
  detector was given and never answered, by inferring the trigger period from the
  spacing of arriving L1Accepts.  **Deadtime makes that inference impossible, not merely
  inaccurate** -- and the inference is the whole mechanism, so no amount of fixing the
  convergence rule rescues it.

  **Measured on gpu008, 2026-10-12, which settles it.**  The XPM reports
  `L0InpRate{partition="7"}` = **71,428 Hz = 13 ticks** and `DeadFrac` = **0.70**.  So the
  true trigger cadence is 13 ticks, deadtime vetoes ~70% of them, and the L1Accepts that
  arrive are spaced 26, 39, 52... ticks apart -- whole multiples, legitimately.  The
  arithmetic closes: 71428 x (1 - 0.70) = 21.1 kHz against ~19.25 kHz measured.

      latched m_dPid   13     correct, it is the real cadence
      observed gap     26     = 2 x 13, so reported as "1 trigger unanswered"
      reality                 the vetoed trigger was never sent to the detector

  **So the five messages are false in the one case where the period was latched
  correctly.**  Earlier runs latched 64 and 38 against a true 68 and were wrong in the
  other direction, which is what the "minimum, not mode" diagnosis was about -- but that
  was a second bug on top of this one, and fixing it only converts silent nonsense into
  confident false alarms.  A deadtime veto and an unanswered trigger are indistinguishable
  from pulse ID spacing alone: both leave `evtCounter` contiguous and a gap that is a whole
  multiple of the cadence.

  **What would work, if the question is ever worth answering**: compare the XPM's
  `NumL0Inp`/`NumL0Acc` against the DRP's own event count, since the XPM knows what it
  vetoed and the DRP does not.  Those counters are already in Prometheus but the DRP does
  not read them -- it only subscribes to an XPM deadtime PV (`DrpBase.cc:1637-1646`).
  That is a real comparison rather than an inference, and it belongs in a dashboard query,
  not in the DRP's hot path.

  Until then `drp_num_pid_jump` is noise and the five log lines are false alarms.  The
  skipped-trigger hypothesis remains **untested** -- and no longer needed, the lost Result
  writes above having been explained without it.

- **Two MEB sizing questions that prescaling makes reachable**, neither a GPU DRP bug:
  the MEB path copies the raw block into the **pebble** buffer, which the recording
  path deliberately never does (`PGPDetector.cc:467`), so confirm the `pblReach` check
  at `:471` covers a payload copy and not just the descriptors' reach; and
  `MebContributor::post` memcpy's at `MebContributor.cc:207` before testing against
  `_maxEvSize` at `:209`, so an oversized event overruns before the abort reports it.

- **The 2026-10-07..11 debugging changes have been classified and applied, 2026-10-12.**
  What survives is ~348 added lines of the ~1322 that existed, in seven files.  Kept:

  - the `EpixUHR3x2` signal fix (`drp/EpixUHR3x2.{cc,hh}`);
  - the `_defer` readout-group walk, the one substantive bug fix of the set;
  - `_reapTx()` and the per-link Tx counters (`EbLfLink.{cc,hh}`), plus `_noCtxCnt`
    (`EbLfServer.{cc,hh}`), which is what refuted the credit-leak theory;
  - the orphaned-Input trap -- `_orphaned`, `_orphanDump`, `_slotDump`, `TCtbI_OrphCt`,
    `_newestRcvd`/`_rcvdFrom` -- and `TebContributor::links()`, which `_orphanDump` needs
    for the Tx accounting;
  - a `static_assert` in `ResultDgram.hh` guarding `monitor` against `auxdata`, which
    catches a latent bug that predates all of this work: raising `MAX_MRQS` past 6
    silently collides the two fields at bit 8.

  Dropped, and recoverable verbatim from
  `claude-sessions/2026-09-11-gpu-drp/wip-2026-10-12-full-diagnostics.patch` with
  `git apply` if a future stall wants them: the `consumed` poison bit (wire-visible, and
  Ric's instruction was that it must not reach the repo); `nPidJumps` and its two log
  messages (not fixable -- see below); the TEB's per-destination posted-Results ring and
  the DRP's per-entry received ring; and `_spinOrAbort`, whose 60 s spin is a debugging
  policy rather than a default, so both abort sites are plain `abort()` again until the
  sweep above replaces them.

  **A bulk removal of diagnostics will take pre-existing code with it.**  This one deleted
  `EbCtrbInBase.cc`'s own 64-entry `Tbuf`/`_tbDump`, an `abort()` and a `throw` that were
  all there before the session began, because the added and the original machinery shared
  names.  Caught by auditing **every** deleted line against `HEAD` rather than reading the
  diffstat; restored.  Do that audit before believing such a cleanup.

  **Comments in shared code must not name a particular detector.**  `EbCtrbInBase.cc`
  and `teb.cc` serve every DRP, so a comment there explaining the epix's stall reads as
  false to everyone else.  One such comment was written and removed on 10-08; check for
  more before committing.

- **Straighten out the epix-family signal handling.**  `drp/EpixUHR3x2.cc` was fixed on
  2026-10-08; `EpixUHR.cc`, `EpixM320.cc`, `EpixHR2x2.cc` and `Epix100.cc` still share
  all three defects, and `Jungfrau.cc` is the one that mostly gets it right:

  - **`monStreamEnable()` is called from a signal handler**, and it calls into CPython.
    A signal is delivered on whichever thread the kernel picks, which need not hold the
    GIL, so an abort becomes a SIGSEGV that buries the original failure.  It belongs in
    `shutdown()`, whose caller `_disconnect()` does hold the GIL -- not in a destructor,
    which has no such guarantee.
  - **`SIGTERM` is not registered**, which is what Slurm sends to stop a task, so the
    orderly path is unreachable under Slurm -- `SIGINT` only arrives from a terminal.
  - **`REGISTER(SIGKILL)` cannot work** and the failure test is `> 0` where `sigaction`
    returns -1, so no registration error is ever reported.  The print also has `#t`
    inside the string literal, where the preprocessor will not stringify it.

  The handlers chain: `initShutdownSignals` (`PGPDetectorApp.cc:852`) installs
  `Collection.cc`'s handler first, the detector's `sigaction` overrides it, and
  `SA_RESETHAND` plus the explicit restore-and-`raise` hands the second delivery back.
  Confirmed working under `scancel` on 2026-10-08: `Terminated` then
  `Shutting down on TERM signal`, no fault.

- **A double free in EPICS teardown, and a reproducer for it from 2022.**  Every DRP
  takes ~30 s to exit after SIGTERM -- `KillWait`, after which Slurm SIGKILLs it --
  because the shutdown action is `exit(0)` (`PGPDetectorApp.cc:852`), which runs static
  destructors and faults somewhere inside EPICS.  The time is most likely
  `systemd-coredump` writing a dump of a process holding a 1.1 GB pebble; the dump has
  never actually been found, and defeating core dumps on the GPU nodes is outstanding.

  Worth fixing for its own sake: it costs 30 s per stop, and it masks whatever the exit
  path was really doing.  Note Ric has moved many sites to `abort()` over `exit(0)` to
  skip static destruction entirely; the GPU DRP's shutdown action was missed.

  **The reproducer is `~/lclsii/daq/obsolete/drp/pvPutTest.cc` with `p4pImpTest.py`**
  (and `p4pTest.cc` beside them).  The switch between working and double-freeing is
  whether the three `p4p` imports in the `.py` are commented out -- they are commented
  out as the files stand.  Written under RHEL7, so the behaviour may have changed.

## Detector configuration

- **CALIB mode is missing from `EpixUHRsim`, `Jungfrau` and `AreaDetector`.**  None
  overrides `rawSize()`, so the alias aborts in `PassthruShim`, correctly and loudly.  Each
  needs four things, modelled on `EpixUHR3x2` or the simpler `EpixUHRemu`: an unconditional
  `rawSize()` (it is capacity -- see the contract on `Gpu::Detector::rawSize()`), a
  `rawShape()` matching what the CPU DRP already writes, a pass-through per-element policy,
  and a `RawU16Def` replacing today's `{"raw", Name::UINT8, 1}`.

  **`EpixUHRsim` first**: it overrides neither `subframeCount()` nor `firstDataSubframe()`,
  so its payload is one contiguous block and it is close to a transcription of `EpixUHRemu`.
  Its reference buffers have nothing to check in pass-through.

  **`Jungfrau` is the substantial one**, because its data sub-frames are themselves batches
  of 128 UDP packets placed by header `packetnum` rather than by position.  `JungfrauCalib`
  (`Jungfrau.cu:188`) already does that reassembly; factor it out rather than writing a
  second copy, since a subtle divergence between calibrated and raw output would be hard to
  see.  `tdestOfModule()` already hides the non-contiguous tdest mapping.

- **`AreaDetector` (`fakecam`) does not handle L1Accepts and would fault if it did**, which
  blocks CALIB mode there.  Raised by Matt, 2026-09-30.  Its `configure()` Names block is
  `#if 0`'d, `event()` is a stub, and `pedestals_d()`/`gains_d()` return `nullptr` while
  `recordEvent()` launches `PedGainCalib`, which indexes them unconditionally -- a device
  null dereference, latent only because no L1Accept reaches it.  The `#if 0` blocks' "@todo:
  Deal with prescaled raw or calibrated data" predates the raw block; the answer is now what
  `EpixUHRemu` does.  **Matt is fixing this.**

- **EpixUHR3x2 gain encoding.**  `RangeOffset`/`RangeBits` in `EpixUHR3x2.hh` are moot for
  an fp16 payload, which arrives calibrated, but not for the u16 payload `raw=u16` selects
  today: gain is bit 0, the ADC value bits 1-11.  Confirm nothing else wants them.

- **`raw=fp16` is refused, pending the firmware's mode registers.**  Chris's plan, per
  2026-10-05: registers select u16 **or** fp16 for the raw (prescaled) block, not both, and
  the non-raw block becomes "fex" -- calibrated fp16, an ROI, or another extraction.  So
  `rawSize()` and `rawShape()` already serve both modes and `dmaBufSize` is unchanged; the
  abort waits only on the registers and the fex format.  `FINDINGS.md` has the reasoning,
  including why the DMA is not doubled.

  **Ask Chris whether fex shares raw's tdests or has its own.**  That is the open question:
  its own tdests make the tdest *set* differ between marked and unmarked events, so
  `firstDataSubframe()` becomes mode-dependent and `NumSubFrames`/`FirstDataTdest` stop
  being constants.

  Expect the sub-frame scan cache to thrash once fex exists: `Scan` latches one data size
  and one transition size, and `raw + fex` against `fex` alone is two data sizes for one
  slot.  A third slot needs a second layout array, not a third `size_t` -- `m_sub[]` is one
  flat tdest-indexed array that pass two overwrites wholesale.  Costs latency on two events
  per second, not throughput.  **Nothing thrashes today**, which has been misread once.

- **Jungfrau pedestals and gains** are placeholders (0.0/1.0); the CPU-side Jungfrau leaves
  calibration to analysis, so there is no source yet.  Also unresolved whether the GPU path
  should calibrate Jungfrau at all or pass raw through.

## Performance and structure

- **`maxTrSize` bounds a Configure and the buffer it is built in, so it cannot be sized
  from either.**  `BEBDetector::_addJson` (`drp/BEBDetector.cc:233`) builds the JSON into a
  *temporary* buffer sized `maxTrSize`, then copies it into a transition Xtc bounded by the
  same number.  A value that is too small therefore fails at whatever it is raised to
  rather than by a fixed shortfall, which is what made 2026-09-25's two failures read as
  near misses; see `FINDINGS.md`.  It is in the CPU `drp`, shared with every CPU DRP.

  Cheap now that transitions have their own buffers: 2 MiB costs 2 MiB per transition
  buffer, not per reduce buffer.  Worth a `static_assert` or a Configure-time check that
  names the real requirement rather than a kwarg.

- **Nothing coordinates the green context split with the kernels' launch geometry**: three
  hard-coded SM tables that disagree, at `PGPDetector.cc:615` (the 6/40/remainder split),
  `Reader.cu:528` and `NoOpReducer.cu:158` -- the latter two scaling in opposite directions
  as `tpSM` rises.  Both `switch (tpSM)` statements `abort()` on an unrecognised value, so a
  new GPU generation stops the DRP rather than degrading, and both hard-code device 0 while
  `MemPoolGpu` honours `gpuId`.

  The fix is for one place to own the partitioning and hand each stage the SM count of its
  own context -- `cudaExecutionCtxGetDevResource()` already returns it.  The Reader and
  TrgInpGen share a context, so that owner must divide their allocation, not just report it;
  the `<<<1, 1>>>` driver loops want about one SM each and the bulk belongs to `_event` and
  `_reduce`.  Also at `PGPDetector.cc:617`: only `group_params[0]` is clamped to
  `minSmPartitionSize`, and nothing checks that 6 + 40 fits the device.

  **Justify this on correctness, not speed, and do not start before the cards are x16
  gen5.**  The DRPs are DMA-bound -- each card sits at its own PCIe 4.0 x8 ceiling, so the
  GPU has spare capacity however badly the SMs are divided, and the measured 8/40/84 on an
  H200 costs nothing.  `FINDINGS.md` has the measurements.  Same trigger as the PCIe
  locality item in `gen_gres_conf.py`'s docstring; revisit together, and measure any change
  against 33035 Hz.

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

- **Profile the recorder and the writer before the compressor work**, since the sink's
  throughput sets the ratio the reducers have to achieve.  **The sink is the slower end
  either way**: ~25 GB/s uncompressed calibrated at 33 kHz against the 22 GB/s Cheolhong
  measured with GDS on gpu005, the only node with `nvidia-fs`.  Two things to ask: whether
  cuFile buys anything over a device-to-host copy plus `pwritev` in compatibility mode --
  which is what these nodes run, WEKA not supporting a mixed IB/Ethernet fabric -- and what
  his 22 GB/s actually exercised, since GDS on an unsupported fabric may have been the
  compatibility path rather than peer-to-peer.  That decides whether it is a floor or a
  ceiling.  Scatter-gather writes are the other open question.

- **Remove `HOST_LAUNCHED_REDUCERS`.**  A temporary switch for seeing whether
  certain reducers worked at all.  It gives reducers two launch paths of which only
  one is ever exercised — the same shape as the `HOST_REARMS_DMA` rot.

- **Third-party compressor support** via shims in `lcls2-dev/subprojects`.  `lc`,
  `pfpl` and `sleek` work; `cusz` and `cuszp` have problems; `eip` was started in
  `~/git/psdaq-reducers_260609` but the upstream code was not ready for use this way
  — a nominally better version needs downloading.

- **Multiple datadevs per GPU: resurrect the multi-Reader event builder.**  Old code in
  `~/lclsii/daq/obsolete/drpGpu/` (`Collector.cu_save`, `Reader.cu_save`); in the repo,
  **055d45df** is the last commit that has it, removed by **a856eae8**.  Two traps:
  the old `_collector` **did not build by pulse id** -- it assumed the FPGAs deliver in
  lockstep and spun forever on a mismatch, the pulse-id comparison being a host-side
  diagnostic -- so real matching is new work; and `MemPoolGpu` has to be made multi-panel
  again.  Wanted only if the reducer sweep shows a larger payload pays, since PCIe is not
  the constraint.  Bifurcation has never worked on these boxes.

- **`HOST_REARMS_DMA` needs an early rearm stage** before it is a real fallback.  It
  currently rearms in `TrgInpGen::_receiver()`, downstream of the trigger kernels,
  so a DMA buffer waits on a dynamically loaded trigger library whose latency is
  unbounded.  See the comment at the macro in `MemPool.hh`.

- **`pgpread`: switch it to the `HOST_REARMS_DMA` pattern, or retire it.**  It has no
  performance constraint, so it has no reason to need a privilege -- yet it is the only
  caller of `GpuAsyncLib.cc`'s `cuMemHostRegister(..., CU_MEMHOSTREGISTER_IOMEMORY)` path,
  the tree's second privileged mapping and the only reason that path exists.  Host rearm
  would leave `MemPool.cc` as the sole place needing `CAP_SYS_ADMIN`.  Retiring it is the
  alternative, `rdmaTest` seeming to cover the same ground -- but confirm it is really a
  superset first, since colleagues find `pgpread` the easier sandbox.  **The status quo is
  the one option with no upside**: a stale tool keeping a privileged path alive.

- **Bulk `gpuSetWriteEn` in the datadev driver (aes-stream-drivers).**  One ioctl per
  buffer is ~33k/s at current rates.  A
  masked or ranged form would be a small aes-stream-drivers PR, and it only matters
  for the host-rearm path.

## Calibration and data handling

- **Chase the UHR emulator's sequence-number register**, if a per-event liveness check on
  real hardware is ever wanted.  Gabriel believes one exists but is unsure of it; today the
  payload is identical event to event, so a real-panel run has no liveness signal in the
  data at all.  See `FINDINGS.md` for what that does and does not invalidate.

- **Fetch calibration constants.  NEXT UP.**  Every detector fabricates them -- pedestals
  0.0, gains 1.0 -- so the path is proven and the science is not.  `EpixUHR3x2` needs none
  for an fp16 payload, which arrives calibrated.  Three routes: Mikhail's psana code (the
  source of truth), his `lcls2/psalg/psalg/calib/` headers, or Gabriel's pseudo-code,
  reproduced in `FINDINGS.md` because the calibdb schema is documented nowhere else.  Needs
  a point in the transition sequence to load from, too.

  **For ePixUHR3x2 the gain bit alone cannot identify the constants.**  Gabriel,
  2026-09-29: `NRanges = 2` is right, but *which* two states the bit selects depends on the
  configured mode, some of which are fixed rather than auto-ranging.  So the fetch keys on
  the gain configuration as well as the detector.  `epixuhr3x2_config.py` already computes
  `gainMapSelection` and `gainValSelection` at Configure; **neither reaches the GPU today**.
  Ask **Mikhail**, who has done the psana equivalent.

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

## Lower priority, after the January deliverables

- **A generic `recoverLinks` script for operators**, so a timing-link complaint either
  clears or names the optical path.  The escalation ladder is its body and is known: report
  both directions, `C_RxReset`, the `ConfigLclsTimingV2` set, then `TxPhyPllReset` with
  `C_RxReset` and `RxDown` cleared after it.  Two things decide whether it is worth
  building: the registers sit at different tree paths per board, so it must use
  `root.find(typ=...)` rather than hard-code one; and **it cannot confirm success from the
  DRP alone** -- the motivating failure read healthy on every local register, so verifying
  the feedback direction needs the XPM's `RemoteLinkId` over PVA.  That dependency is
  heavy for a recovery tool; weigh it against putting the check in `control` at Configure,
  where the XPM connection already exists.

- **Let the DRP idle at low power when triggers are absent or slow.**  A question about a
  ceiling, not new machinery: `Reader.cu` already backs off exponentially (`:363`, `:407`,
  `:448`), 8 ns doubling to 256 ns.  At 33 kHz that cap is ~120 wake-ups per event period
  and more as the rate falls, so raising it -- or adding a coarser second tier once a quiet
  spell is established -- costs latency only on the first event after the quiet spell.
  Check whether the host threads also poll, and that `__nanosleep` holds at longer
  intervals.  Measure against 33035 Hz: the point is power, not throughput.
