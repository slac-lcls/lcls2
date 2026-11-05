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

## Design rate: 35.7 kHz, i.e. < 28 us per event in every stage

The ePixUHR's designed operating rate is **35.7 kHz (28 us)**.  Today's PCIe bus caps the
system at 33034 Hz (102 Gbps), but the code must be ready for 35.7 kHz.  So each stage --
Reader, TrgInpGen, TEB, Reducer -- must handle an event in **under 28 us**, overheads
included, with margin.  Judge performance work against 28 us, not the 30.3 us of 33 kHz.

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

- **ONE CONTRIBUTOR RUNS ~1.6 ms LATE, AND THE ROLE MOVES BETWEEN DRPs.  SUPERSEDED
  2026-10-15 -- see "The rate regression and the staging-buffer leak" near the end of
  this file.**  Same phenomenon, caught two orders of magnitude larger (60-70 ms) with
  a named mechanism: one DRP's staging pool fills at ~76 buffers/s from uncollected
  reducer completions (`DRP_redStarts - DRP_redRcvs`).  **Test 1 below was run and is
  NEGATIVE** -- `TCtbO_BtAge` spans 25-96 us across all six with no correspondence to
  lateness, so batch accumulation is excluded and `batching`/`maxEntries` is not where
  to look.  The rest is kept for the `EB_arrTime` numbers and the handoff observation.
  Found 2026-10-14 on gpu008, `verbs`, six DRPs; Ric spotted it in `EB_arrTimeN`.

      EB_arrTime<N> (us; ns in the metric -- it is one series per contributor
      index, not a single EB_arrTimeN), 16:55 - 17:10, one sample per minute
      tstcam1_2   1612 1582 1596 1576 1592 1606 1596 |   60   27   12   13   48   10
      tstcam1_5     32   35   49   48    0   20   37 | 1592 1633 1567 1616 1611 1607
      others      7 - 13 throughout                 ^ 17:02, a clean handoff

  **The numbers that make this a bug and not weather:**

  - The late value is **1598 us mean, sd 17 us** over 16 minutes -- a **4% spread**.
    Contention or thermal effects do not hold a figure that still.
  - It is **~114 trigger periods** (13 ticks = 14.0 us at `L0InpRate` 71428) or
    **~50 accepted events** at 31.4 kHz.  A round amount of buffering, not a slow
    device.
  - The handoff at 17:02 was **clean and simultaneous**, and the other four DRPs did
    not move.  So it is not a property of a datadev, a GPU or a NUMA placement -- it
    is a *role* one contributor occupies, and it transfers.
  - **It is anti-correlated with buffer occupancy**: the late DRP holds ~6 host
    buffers where the healthy ones hold ~56 (`drp_dma_in_use` / `drp_pebble_in_use`,
    which are the same number on the GPU path).  So the late DRP runs **emptier and
    later**, which does not fit backpressure.
  - **`EB_arrTime` is lateness relative to the event's first contribution**
    (`EventBuilder.cc:542`: `t0 - event->_t0`), so exactly one source reads 0 by
    construction.  A DRP reading 0 is the *earliest*, not the fastest in absolute
    terms.
  - **It plausibly causes the ~33% `drp_deadtime`** seen on whichever DRP is late,
    rather than being a separate fault: making the TEB wait is what asserts deadtime
    upstream.  `L0AccRate` tracks `L0InpRate * (1 - DeadFrac)` at 100.00% throughout,
    so nothing is lost -- the rate is simply gated by the slowest contributor.

  **The discriminating tests, in order, none yet run:**

  1. **`TCtbO_BtAge` and `TCtbO_BtEnt` per DRP.**  If the late DRP's batch age is
     ~1.6 ms while the others' are short, this is batch accumulation -- a DRP sitting
     one batch behind its peers -- and `batching`/`maxEntries` is where to look.
  2. **The Reader's own `RingIndexDtoH` occupancy**, for the same asymmetry on the
     GPU side of the pipeline.
  3. **Whether the 17:02 handoff coincides with anything in the DRP logs.**  Nothing
     was looked for at the time.

  Not yet known: whether it appears on `sockets` too.  It would have been invisible
  there -- `DRP_evtLatency` was a uniform 85-94 ms on all six, which is ~50x the lag.

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

  **DONE 2026-10-14, and ON BY DEFAULT.**  `Pds::setThreadPinning` and
  `Pds::pinHotThread` in `Affinity.cc`, called from each hot thread after its
  `prctl(PR_SET_NAME)`.  `pinThreads=no` turns it off; `spread` is the only other value.

  **Why on by default** (Ric, 10-14, from Chris's point at that morning's meeting):
  nobody should have to edit every `.cnf.py` out there when a capability appears, and
  flipping a default later costs a **second deployment to N hutches**, which is hard to
  coordinate.  So the knob exists to turn pinning *off* when it is suspected in a
  problem, not to turn it on.

  **VALIDATED ON HARDWARE 2026-10-14, gpu008, on `verbs` over IB -- the transport
  production actually uses.**  Four runs: a `sockets` control (`pinThreads=no`), a
  `sockets` treatment, then `verbs` with the provider and domain kwargs **removed**, so
  `getNicIp`'s automatic path ran, which is what the production `.cnf.py`s do.

      L0InpRate  71,428 Hz    DeadFrac 0.5599    L0AccRate 31,435 Hz
      agreement  100.00% of L0InpRate * (1 - DeadFrac), at every 30 s sample

  31.4 kHz matches the 10-12 `verbs` reference (31.53 kHz at DeadFrac 0.558) and is
  within 2.2 deadtime points of the remembered 33 kHz.  `TCtbI_OrphCt` and
  `TCtbI_NPrgCt` **0 on all six** throughout.  All six DRPs at **0.1-0.2 ms**
  `DRP_evtLatency`.

  - **The `verbs` latency floor is ~250x below `sockets`**: 0.1-0.2 ms against a uniform
    85-94 ms on `sockets`.  That uniformity is why the `sockets` latency comparison was
    worthless -- 85 ms of transport latency masked all per-DRP variation, and individual
    DRPs swung up to **61% between two samples of the same run**, against a ~6% effect.
    Any future placement measurement needs `verbs` and windowed averages, not instant
    samples on `sockets`.
  - **The under-allocated DRP was the one case where pinning cost anything, and the new
    warning is what found it.**  `tstcam1_3` at `cores: 4` gets 3 CPUs (see the WEKA
    item above), so 4 hot threads wrapped onto 3 and its latency ran **2.4 ms average,
    15.7 ms peak** while its unwrapped peers sat at 2-9 us.  The log said
    `will share.  Consider cores: 4`; `cores: 6` -> 5 granted CPUs -> four threads on
    33-36 with one spare, and it joined the others at 0.1-0.2 ms.  So pinning plus an
    adequate allocation beats not pinning, and the warning turns what used to be silent
    timesharing into an explicit instruction.
  - **Pinning's effect on an adequately-allocated DRP is still not measurable** at this
    precision, and that is the honest statement: no regression, no demonstrated gain.
  - **A separate fault surfaced and IS ours to chase** -- see "One contributor runs
    ~1.6 ms late" under Event building.  (An earlier version of this note called it "NOT ours";
    Ric's correction, 10-14: a fixed, quantised lag in a system we wrote is not
    chaos, and he removed that line from the PR.)

  **Wired into all nine processes, Ric 2026-10-14 (he wanted it before the master
  merge).  Each count was read off the `std::thread` sites, not assumed:**

      drp_gpu         TigRcvr, Collector, Recorder, TEBreceiver   4
      drp             PGPreader, Collector, TEBreceiver           3 + nworkers
      drp_pva         Reader, Collector, TEBreceiver              3
      drp_udpencoder  Receiver, Worker, TEBreceiver               3
      drp_bld         Worker, TEBreceiver                         2
      epicsArch       Worker, TEBreceiver                         2
      BldDetectorSlow Worker, TEBreceiver                         2   not built
      teb             Teb                                         1
      monReqServer    Meb                                         1

  `BldDetectorSlow.cc` is commented out of `drp/meson.build:108`, so the compiler never
  sees it; it is carried along with the same four edits as its compiled sibling
  `BldDetector.cc`, and was `-fsyntax-only` checked.

  **Default-on made this an audit, not a choice.**  `pinHotThread` lives in
  `EbCtrbInBase.cc` -> `libcontributor`, which `drp_pva`, `drp_bld` and
  `drp_udpencoder` all link, so with pinning on by default their TEBreceiver threads
  would have pinned whether or not anyone asked -- and without a `setThreadPinning`
  call they could not have been told `pinThreads=no`.  All three are now wired.
  `ctrb` does not link `libcontributor` and is not built by default, so it is moot.

  **A bug default-on exposed in `drp_pva`, worth remembering:** it calls `get_kwargs`
  *inside* its `try` block, after the point the other programs parse kwargs, so a
  `setThreadPinning` placed beside `initExceptionReporting` read an **empty** map and
  silently left pinning enabled with no way to disable it.  Checked across all seven:
  `setThreadPinning` must follow `get_kwargs` and the allowlist, and `drp_pva` was the
  only one where the obvious placement was wrong.

  `nworkers` is `-W` and every config in `runs/eb/data` passes `-W 1`, so the CPU DRP's
  hot set is also 4 in practice, despite `drp.hh:34` defaulting `nworkers` to 10.
  CPU-DRP workers spin only ~1 ms before blocking on a condition variable
  (`spscqueue.hh:109`), so an idle worker is not hot -- which is why the count is
  `cores:` advice and not a pinning requirement.  The GPU DRP has **no** worker threads:
  its `nworkers` sizes CUDA streams and a round-robin index, not a thread pool.

  **Workers are excluded when the pool cannot be spread** (Ric's rule, 10-14).  If
  `nFixedHot + nWorkers` exceeds the granted CPUs, `pinWorkerThread()` becomes a no-op
  and only the fixed threads are pinned; the fixed few gain most from a stable CPU,
  while a pool that outnumbers the allocation would get a fixed round-robin where the
  scheduler can at least migrate.  This matters for configs that exist today:
  **`psdaq/psdaq/cnf/epixHrEmu.cnf:112` runs `-W 26` with no `cores:` key**, and
  `slurm/utils.py:357` defaults an absent `cores:` to **1**, so that DRP would otherwise
  pin 29 threads round-robin onto 1 CPU.  Verified on the real `drp`: `-W 26` in a
  4-CPU cpuset warns `29 hot threads (26 of them workers) exceed 4 granted CPUs ...
  Consider cores: 29` and pins only the 3 fixed ones.

  - **The old `core[]` path is GONE from the production programs** (Ric, 10-14, having
    spotted that `Meb::run()` called both mechanisms back to back).  What it did: the
    `core[]` call ran *first* with the default `-1`, which `pinThread` treats as a
    no-op, so `pinHotThread` then pinned from the granted set and the MEB thread was
    pinned correctly -- but if anyone had passed `-1 41`, `pinThread` would have pinned
    to 41 and `pinHotThread` would have **silently overridden** it.  Measured both ways
    with `tools/orderprove.cc` before deciding.  Removed: the `pinThread(core[])` calls
    in `Meb::run`, `Teb::run` and `EbCtrbInBase::receiver`, the `-1`/`-2` getopt entries
    and usage text in `teb`/`monReqServer`, their `CORE_0`/`CORE_1`, `DrpBase`'s
    hardcoded `-1`s, and the now-dead "Thread core numbers" log lines.
    - **`ctrb`'s pinning is removed too, so `eb.hh`'s `core[2]` fields are gone**
      (Ric, 10-14).  `core[]`, `CORE_0` and `CORE_1` no longer appear anywhere in
      `psdaq`, leaving exactly one pinning mechanism.  `ctrb` had no `prctl` naming and
      no kwargs, so there was nothing to hook the new mechanism onto; its
      `EbCtrbInBase` thread already would not pin, `setThreadPinning` never being
      called there.
    - **`ctrb.cc` is bit-rotted independently of this work, which is why it is absent
      from `eb/meson.build`.**  `-fsyntax-only` in place: **32 errors at HEAD, 23
      after** -- its calls do not match current `EbCtrbInBase` signatures
      (`EbCtrbInBase(prms, exporter)` against a 1-arg ctor, `configure()` against
      `configure(unsigned)`).  Diffing the error *sets* shows **no error introduced**
      and four classes removed.  Trap worth noting: syntax-checking a copy under
      `/tmp` fails instantly on the relative `#include "MebContributor.hh"` and
      reports a misleading "1 error", so check it in place.
    - **A standalone run names its CPUs with `taskset -c <list>`, not `-1`/`-2`**, and
      that is now in `teb`'s usage text so the removal is discoverable.  It reproduces
      the old behaviour exactly: `taskset -c 18,19` lands the threads on 18 and 19.
      Note the `.cnf` precedents already paired `-1 18 -2 19` *with* a `taskset`
      (`psdaq/cnf/lab3-ric.cnf:37`), so the cpuset was always the outer constraint and
      `-1`/`-2` only chose within it -- which is what `pinHotThread` now does.
      **Standalone caveat: with no cpuset the granted set is the whole machine**, so
      pinning picks the lowest-numbered CPUs unless a `taskset` narrows it.
    - **The removal re-taught the 10-12 lesson.**  Deleting the pin blocks took the
      `int rc` declaration that each function's *later* `while` loop depends on, in
      three files; the compiler caught it, and the fix is a plain `int rc = 0;`.  Every
      deleted line was then audited against HEAD, which is how the `ctrb` getopt
      inconsistency was found.
  - **`Eb::pinThread` MOVED to `Pds::pinThread` in `Affinity.cc`** (Ric's call: it should
    not have been copied, and `Eb::pinThread` should go away).  All five callers --
    `teb.cc`, `ctrb.cc`, `EbCtrbInBase.cc`, `monReqServer.cc`, `eblf_pingpong.cc` -- now
    call `Pds::`.  This also removes the reason the copy existed: `eb/meson.build:6`
    links `service_lib`, so `service` could not have called *into* `Pds::Eb`.
  - **NO relative/absolute or abstract/machine id hazard, and the reason is structural:
    we only ever hand back ids the kernel just gave us.**  Ric's question, 10-14 -- could
    a numbering clash pin a thread onto a WEKA CPU?  Measured on gpu008, not reasoned:
    - `sched_getaffinity` returns **absolute** ids inside a cpuset.  `taskset -c
      37,41,52` reads back `[37, 41, 52]`, not `[0, 1, 2]`; a relative scheme would give
      the latter.  Last night's real Slurm allocations likewise reported 9-15, not 0-2.
    - **WEKA's cores cannot be selected because the kernel has already removed them.**
      `weka-drpsrcf/cpuset.cpus.effective` is `1-2,32`, and those exact three ids are
      the *only* ones absent from our affinity mask.  `cpuAllocation()` therefore never
      offers a WEKA CPU.
    - SMT is off on these nodes (`Thread(s) per core: 1`), so abstract and machine ids
      coincide -- the condition under which `CpuSpecList` and the cgroup are comparable.
    - Round-trip proven end to end: `taskset -c 37,41,52 ./pintest spread 3` lands on
      37, 41 and 52, and the harness's "is this CPU in the granted set?" assertion stays
      silent.
    The property to preserve: **never compute a CPU number, only index the granted
    list.**  An absolute number from a config or a `cores:` count would reintroduce
    exactly the hazard asked about.
  - **SMT is ON on 22 of 116 `drp-srcf-*` nodes**, so this is not hypothetical:
    `cmp057`-`cmp076` (a contiguous block of 20), `gpu007` and `mon014`, all
    `ThreadsPerCore=2` with 128 CPUs.  Surveyed from the controller with
    `scontrol show nodes`, which touches no node.
    - **Safety holds there anyway, and `CpuSpecList` is NOT what provides it.**  On
      `gpu007`, `CpuSpecList=2-5,64-65` misses four of WEKA's six cores
      (`1-2,32,65-66,96`), yet all six are absent from a process's affinity mask: the
      **cgroup** does the exclusion and the kernel composes it with Slurm's allocation.
      So an accidental non-clash between `CpuSpecList` and WEKA does not expose us.
      The SMT `cmp` nodes have `CpuSpecList=0-3`, reserving CPUs 0-3 while their
      siblings 64-67 stay schedulable -- the trap already recorded in these notes.
    - **What SMT did break was the spread, not the safety**, and that is now fixed.
      `_spreadOrder()` reorders the granted set so one CPU per physical core comes
      first and the siblings follow, using `coreOfCpu()` (the first id in
      `thread_siblings_list`).  Plain ascending order put 4 threads on 2 cores for a
      granted set of `6,7,70,71`, reporting "4 distinct CPUs" while two pairs shared
      execution units.
    - **Greppable for the nodes nobody can check** (Ric's ask, 10-14): an SMT-paired
      allocation logs at startup
      `SMT is on: the N granted CPUs are only M physical cores, so any hot thread past
      the first M shares a core`.  Emitted once from `setThreadPinning`, so it is
      deterministic rather than from whichever thread raced first.  Verified on
      `gpu007` with the real `drp`; **silent on SMT-off nodes**, where the order is
      also returned unchanged.
    - **Known limit, left deliberately: the worker fit test counts CPUs, not cores.**
      4 hot threads in 4 SMT-paired CPUs (2 cores) still pins the workers, since
      4 <= 4.  Siblings are real scheduling slots, so this is better than not pinning,
      and the SMT warning says what happened -- whereas testing against cores would
      silently stop pinning workers on all 22 SMT nodes, a larger behaviour change
      than it is worth.  Revisit if a measurement says sibling sharing hurts.
  - **`pinThread`'s guard is now `cpu >= 0`, was `cpu > 0`.**  The old form reserved CPU
    0 for the OS, which is now an external policy: Slurm's `CPUSpecList=0-3` on the DRP
    nodes already excludes it, so a granted set never contains CPU 0 under Slurm and
    reserving it again in our code only made CPU 0 unpinnable standalone.  Ric, 10-14:
    the allowable-CPU policy is governed externally, and a standalone run is the
    runner's look-out.  `pinHotThread` therefore uses `pinThread` rather than repeating
    it.  (An earlier note here called this an imminent bug; it was latent, since
    `CPUSpecList` keeps CPU 0 out of every Slurm allocation.  The 61-CPU set it was
    observed in was a login shell, where only WEKA's cgroup cores 1, 2 and 32 are
    excluded.)
  - **The CPU index comes from an atomic counter, not from the caller.**  The threads are
    created in several libraries, so a shared numbering would have to be threaded through
    all of them; only distinctness matters.  Ric, 10-14: leave it, fix it if it bites.
  - The fit decision is made once in `setThreadPinning`, from counts the *app* knows,
    rather than per thread -- `EbCtrbInBase.cc` is shared by all four processes and
    cannot know either count.
  - Verified against the kernel with `tools/pintest.cc`, which reads
    `pthread_getaffinity_np` back rather than trusting the log: 4 of 4 pinned to distinct
    granted CPUs by default, 0 of 4 under `pinThreads=no`, 3 of 29 with the worker
    exclusion, all pinned when the pool fits, a thread on CPU 0 when granted, and
    `abort()` on a bad value.  Also driven through the real `drp` binary for all three
    modes.
  - On a node whose granted set skips WEKA's cores, indexing the granted set avoids them
    for free -- gpu008's allocation omits 1 and 2, so `spread` picked 0,3,4,5.

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

  **`tstcam1_3`'s 3 CPUs for a `cores: 4` request IS WEKA after all -- SOLVED
  2026-10-14, and the "not WEKA" conclusion above was wrong.**  WEKA holds CPU **32**,
  which is the **first CPU of NUMA node 1**.  `tstcam1_3` is the first job placed on
  node 1 (its GPU and datadev are there, `gen_gres_conf.py` keeping them local), Slurm
  hands it `32-35`, and the kernel subtracts WEKA's 32, leaving `33,34,35`.  The
  allocation order makes it visible:

      teb0      9-11    timing_2  12-14   cam0  15-18
      cam1     19-22    cam2      23-26   cam3  33-35  <- jumps to node 1, loses 32
      cam5     36-39

  **Why the 10-12 check missed it, which is the lesson**: it intersected WEKA's cores
  with the DRP *ranges* and correctly found nothing -- the intersection is empty
  **because the kernel had already removed 32** from what `tstcam1_3` was granted.  The
  subtraction happens upstream of the comparison, so comparing the end states can never
  see it.  Compare what Slurm *allocated* (`sacct`, or the contiguous-from-the-previous
  -job pattern above) against what `sched_getaffinity` *reports*.

  Consequence: the first job on each NUMA node loses a CPU wherever WEKA pins one there,
  so a `cores: N` request for it should be `N+1`.  WEKA's node-0 cores (1, 2) sit below
  the first DRP allocation, which is why only the node-1 boundary shows the effect.

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

- **`getNicIp()`: DONE 2026-10-13.**  All three parts -- abort rather than fall back,
  require `IFF_RUNNING` on both branches, and log the selection.  `Collection.cc:218-285`.

  It used to prefer the first IB interface and drop to Ethernet with only a warning.
  **That cost real debugging time**: when IB is expected across the DAQ and one node has
  lost its NIC, that node alone comes up on Ethernet, the mixed system cannot communicate,
  and nothing says why -- a warning in one log among dozens is not findable when the
  symptom is "the DAQ will not connect".  The fallback is only right when another
  interface was *intended*, which `ep_domain` already expresses, so the accidental case
  can abort.  It names the node and the interface it rejected: **"Provide
  `ep_domain=<name>` to use that interface deliberately"**.

  **`ep_domain` takes an interface NAME, not an IP** -- it reaches the
  `getNicIp(const std::string&)` overload, which compares against `ifa->ifa_name`
  (`Collection.cc:202`) and returns that interface's address.  `cnf/rix.py:41` has the
  existing precedent, `ep_domain=eno1`.  So the message can name exactly what to paste.
  It advises `ep_domain` rather than `forceEnet=yes` because `forceEnet` only re-expresses
  the same guess one level up -- the first running Ethernet interface -- whereas the node
  that needs help is the one where that guess is wrong.

  **RoCE is why this matters, and it is the one case the type test cannot settle.**
  gpu005's two ConnectX cards report `ibstat` **Link layer: Ethernet**, and their netdevs
  (`enp153s0f0np0`, `enp153s0f1np1`) carry `hatype` **1 = ARPHRD_ETHER**, not 32.  An IB
  card in Ethernet/RoCE mode is therefore indistinguishable from a plain NIC at the
  `getifaddrs` level, so no amount of care in this function finds it; both ports are also
  Down today, so gpu005's only running interface is the 1 GbE `enp3s0f0`.  Asking for
  Infiniband there now aborts and names `enp3s0f0` -- which is the right outcome: it says
  the node cannot give what was asked for, instead of silently running the DAQ over the
  management NIC.  A RoCE deployment selects its port by name with `ep_domain`.

- **RoCE: how to ask for it, and what is NOT yet in place.**  Measured 2026-10-13 on
  gpu005 (RoCE-mode ConnectX) and gpu008 (IB-mode).  **A RoCE run is expressible today**,
  but only via `ep_domain`, and one piece of plumbing is missing.

  **The two "domain" words are different things, which is the thing to get straight.**
  `ep_domain` is **our** kwarg and takes a **netdev name** (`getNicIp` matches
  `ifa->ifa_name`, `Collection.cc:202`); libfabric's *domain* is an HCA
  (`mlx5_0`).  We never name a libfabric domain -- `Fabric::initialize` passes the
  **address** as `fi_getinfo`'s `node` (`Endpoint.cc:922`), so **the chosen IP is what
  selects the HCA**.  Hence a RoCE deployment is specified as
  **`ep_provider=verbs,ep_domain=<roce-netdev>`**: the netdev gives the address, the
  address resolves to `mlx5_x`, and `verbs` runs RoCE over it.

  **Confirmed on the hardware:** with its IB port up, gpu008 offers
  `provider: verbs, domain: mlx5_0`.  gpu005 offers **no `verbs` at all** -- only
  `tcp`/`tcp;ofi_rxm` on `enp3s0f0` -- because both RoCE ports are Down and have no IP.
  The RDMA stack itself is fine and RoCE-capable (`mlx5_0`/`mlx5_1` present,
  `gid_attrs/types/0` = `IB/RoCE v1`), so `verbs` should appear once a port is up and
  addressed.  **Untested end to end: no RoCE port has been brought up yet**, so "verbs
  appears and connects over RoCE" is an expectation, not a measurement.

  **The automatic path cannot find RoCE**, since `getNicIp(forceEnet)` decides from
  `hatype`.  So on a RoCE node `ep_domain` is the only correct answer.

  **Decision, Ric 2026-10-13: leave the kwargs alone for now.**  Not urgent, and the hard
  requirement on any future change is that **the existing `.cnf.py` files keep working** --
  the default and whatever they specify today must behave exactly as they do now.
  Augmenting the choice set for RoCE is allowed when it is wanted; changing the meaning of
  what is already written is not.

  If it is taken up, the better selector is **RDMA capability rather than `hatype`**: for
  each candidate netdev ask whether `/sys/class/infiniband/*/device` resolves to the same
  PCI device, which is true for RoCE *and* IB and false for a plain NIC.  That makes "give
  me the RDMA interface" automatic in both modes.  It belongs behind a new, explicitly
  requested value rather than silently redefining the present default.

- **The IB NIC is contended between inter-DAQ traffic and WEKA, and the NUMA map is
  hostile.**  Ric's point, 2026-10-13: both uses must be optimised together or one
  produces deadtime in the other.  Measured on gpu008, and the numbers are specific:

      HCA  mlx5_0 (0000:71:00.0)          NUMA 0     the ONLY HCA; mlx5_1 is its 2nd port
      datadevs 04,05,53:00.0              NUMA 0
      datadevs 84,85,a1,d5:00.0           NUMA 1     four of seven
      GPUs 03,54,55:00.0                  NUMA 0
      GPUs 83,d3:00.0                     NUMA 1     two of five
      WEKA's 3 pinned cores: 1, 2 (NUMA 0) and 32 (NUMA 1)
      gpu008 is 2 sockets x 32 cores, NUMA node0 = CPU 0-31, node1 = 32-63

  **WEKA and the DAQ share the interface, not merely the fabric**: `/etc/fstab` mounts
  `drpsrcf/ffb` with **`net=ibp113s0f0`**, which is exactly the netdev `getNicIp` selects
  for IB.  So recorded data going out to WEKA and inter-DAQ Results/Inputs traffic cross
  the same port and the same HCA.

  **The structural problem is that there is one HCA and it is on NUMA 0**, while most of
  the datadevs and two GPUs are on NUMA 1.  A DRP serving a NUMA-1 card therefore cannot
  be local to its datadev, its GPU *and* the HCA at once -- the best available placement
  is local to the datadev and GPU, with the eb traffic crossing the socket link.  That is
  a reason to measure rather than assume which locality matters more, and it is the
  concrete form of the thread-placement item above: `sched_getaffinity` gives the granted
  CPUs, and `/sys/bus/pci/devices/*/numa_node` gives each device's node, so the DRP can
  log the three-way relationship instead of leaving it to chance.

  **And the NIC's placement is not a free choice, so this cannot be fixed by moving it.**
  Chris, 2026-10-13: every regular slot in these chassis is spoken for by a GPU or an FPGA,
  so the mlx card goes in one of the two AIOM slots -- which fixes its NUMA node and
  guarantees some DRPs are remote from it.  Treat the asymmetry as a constraint to schedule
  around, not a layout bug.

  **A competing partitioning is on the table: Cheolhong wants all GPUs on one NUMA node and
  all FPGAs on the other**, so that MPS can be 256 for the GPUs' gen5 x16 and 512 for the
  FPGAs' gen4 x8, reaching 113 Gbps.  That is a different optimum from "each DRP local to
  its own datadev and GPU", since it makes *every* DRP cross the socket link for one of the
  two, and it interacts with the single HCA's node as well.  **Three claims on the same
  topology that have not been evaluated together**: datadev/GPU locality per DRP, the
  GPU/FPGA split for MPS and link rate, and the HCA shared with WEKA.  Settle it with
  measurements rather than by argument -- the deadtime identity gives a single number to
  compare against.

  **What to measure, before any tuning**: WEKA's throughput and the DAQ's deadtime as a
  pair, since the point is that neither may be optimised alone.  `DeadFrac` beside
  `L0InpRate` is the DAQ-side number (see the deadtime item); WEKA's own counters are the
  other half.  Also worth knowing whether WEKA's `num_cores=3` and our `cores:` can be
  kept on opposite sockets -- today WEKA has 1 and 2 on NUMA 0 and 32 on NUMA 1, so it
  straddles, and its NUMA-0 cores are the ones nearest the HCA it shares with us.

  **The `IFF_RUNNING` test was on the Ethernet branch only**, so a *down* IB interface
  was selected if it enumerated first.  It is now a single test ahead of the type check.
  gpu008 has `ibp113s0f0` up and `ibp113s0f1` down, so this was live, not hypothetical --
  it happened to pick the working one by enumeration order.

  The loop now takes the first running interface of the *wanted* type and **breaks**.  It
  used to collect an IB name and an Ethernet name in one pass and choose afterwards, which
  the old fallback needed -- with the fallback gone only one of the two could ever select
  anything.  The Ethernet name is still noted when Infiniband was wanted, solely so the
  abort message can name what it declined to use silently.

  **Trap, checked rather than assumed:** `/sys/class/net/*/flags` does **not** show
  `IFF_RUNNING` -- it reads `0x1003` for gpu008's up *and* down IB interfaces alike, which
  makes the change look like it would abort on a healthy node.  `getifaddrs` does report
  it (`0x11043` running, `0x1003` not), and that is what the code reads.  Check with
  `tools/ifflags.c` in the session directory, not with sysfs.

  Verified on gpu008 against the real function (`tools/nictest.cc`), five cases: IB chosen
  (`ibp113s0f0`, 172.21.167.165); `forceEnet` chosen (`enp73s0f0`, 172.21.155.165) -- both
  addresses independently matching what had previously been recovered by hand out of the
  TEB's log; Ethernet-only abort with the new message; down-Ethernet-only abort; and
  loopback-only abort.  The last three in a `unshare -rn` namespace.

  Selection was also **reported nowhere**, which is why the interface in use had to be
  recovered from peer addresses in the TEB's log.  Both overloads now log type, name and
  address at `LOG_INFO`, which is the default without `-v` in both the DRP
  (`drpGpu/PGPDetectorApp.cc:781`) and the TEB (`teb.cc:1446`), so it lands in every log.

- **An L1Accept rate is a deadtime measurement -> MOVED to `RULES.md`, 2026-10-15**, under
  "Reading measurements", with the two other measurement traps.  Nothing is owed; it was
  only ever a rule about not misreading a rate, and it sat here looking like a task.

- **Log the libfabric configuration: DONE 2026-10-13.**  `Pds::Eb::logFabricInfo()`
  (`eb/src/utilities.cc`) is called from both `linksConnect()`s, guards itself with a
  `static bool` so it reports once however many links connect, and logs unconditionally at
  `LOG_INFO`.  It replaced two `-v` prints, one of which sat inside the per-link
  `connect()` and so emitted N identical copies for N peers.

  It reports version, provider and its version, domain and fabric, then the values that
  had each cost time to establish by hand: `rx_attr->size`/`tx_attr->size` as
  **negotiated** (the hint asks for 0, and `rx_size` bounds the Rx CQ), `inject_size`,
  `cq_data_size`, `data_progress`/`control_progress`, `mr_mode`, and `msg_order`.  So a
  log now carries the provider, the interface *and* the HCA, which on 10-12 had to be
  reconstructed from peer addresses in the TEB's log.

  **The trap, and it is worth knowing for anything else formatting libfabric values**:
  `fi_tostr()` returns a pointer to a single static buffer, so two or more calls in one
  statement all render the last one evaluated.  The first version of this printed
  `mr_mode FI_PROGRESS_AUTO`, which looks plausible and is wrong.  One `fi_tostr()` per
  logging call.  **Caught by running it; compiling could not have.**

  Still open, and deliberately left with the thread-placement item rather than here: log
  the CPU allocation in the same place -- the CPUs `sched_getaffinity` reports, their NUMA
  nodes, and which thread went where -- so `cores:` and placement become checkable.

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
  MANUAL, and the symptom would look like an unexplained throughput collapse -- which is
  what the logging above now makes a one-line diagnosis instead of a hunt.

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

- **`_process` can swallow a non-fatal `pend()` error -- the one transport
  error-reporting gap still open.**  It logs and returns `rc`; `receiver()` escalates only
  on `-FI_ENOTCONN` or when `rc == rcPrv` (`EbCtrbInBase.cc:446-459`), so an error that
  *alternates* with success is logged for ever and never becomes fatal.  **Decide
  deliberately which errors are fatal** rather than letting repetition be the test.  Needs
  no run; it is a decision plus a few lines.

  **The other two of the original three are settled** (2026-10-12/13) and kept below for
  the reasoning: the Tx side is covered by `EbLfLink::_reapTx()` draining the Tx CQ from
  `post()`, counting `posted`/`completed`/`errors` per link and dumping `fi_cq_err_entry`
  on `-FI_EAVAIL`, with attribution genuine because `EbLfClient::connect` creates a
  **separate CQ per link** (`EbLfClient.cc:65`).  And the shared Rx CQ is **a decision to
  keep it, not a gap**.

  1. **SETTLED -- KEEP THE SHARED Rx CQ.**  It is shared across all links, so a receive
     *error* cannot name one.
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
  3. **`_process` swallowing a non-fatal `pend()` error** -- this is the open item, stated
     at the head of this entry.

- **Sweep out a missing Result rather than aborting on it.  DEFERRED, 2026-10-13: leave a
  missing Result blocking so that it stays easy to see.**  Ric's call -- the abort is the
  clearest possible signal that the fault recurred, and masking it before the cause is
  known trades diagnosis for uptime.  Revisit if it becomes an operational problem.  The
  design below stands as written; nothing about it has been invalidated.

  Original proposal, 2026-10-10.
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

- **`MebContributor::post` overruns its target buffer before the abort can report it.**
  Not a GPU DRP bug and not ours: `MebContributor.cc:207` does
  `memcpy(buffer, ddg, sz)` into a slot of `_maxEvSize`, and only *then* does `:209` test
  `sz > _maxEvSize` and abort.  So an oversized L1Accept has already written past the slot
  by the time it is diagnosed, and the critical message describes damage already done.
  Swap the two.  Re-verified 2026-10-15 against the source.

  **The companion question about the pebble is CLOSED by Gabriel's `58686d7f`**, which
  made the MEB path copy only the payload regions -- respecting the prescale layout
  `[Dgram][raw shapes][raw][reduced shapes][reduced]` -- instead of a blanket
  `sizeofPayload()` from `buffer + cpSize`.  `PGPDetector.cc`'s `pblReach` check now
  computes `hdrSplit + rawBytes + prescaleBufsDscr()` explicitly and its comment records
  that the raw block is never written on the host, so there is nothing left to confirm.

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

- **Epix-family signal handling: DONE 2026-10-13.**  `EpixUHR.cc`, `EpixM320.cc`,
  `EpixHR2x2.cc` and `Epix100.cc` now match the `EpixUHR3x2` fix, and `Jungfrau.cc` has
  the macro's two defects fixed plus `SIGQUIT`.  All four epixes were structurally
  identical -- same handler, same `REGISTER` block, same empty `shutdown()` -- so the fix
  transferred verbatim; Jungfrau needed only the macro, its handler calling
  `cleanup()`/`sls` rather than CPython.  `monStreamEnable` is now driven by
  `m_monStreamOn`, since `enable()` disables the stream and `disable()` re-enables it
  every cycle.  **Compile-tested only: none of these five detectors has been run.**

  The three defects, for the record:

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

- **The EPICS teardown double free is EXPLAINED, 2026-10-13: two builds of EPICS 7.0.7
  are loaded into one process.**  Not an EPICS design problem and nothing to raise
  upstream -- a conda-environment layout issue with a local fix, demonstrated on the
  reproducer but **not yet on the tree**.  **Both copies confirmed mapped into a live
  `drp_gpu` on 2026-10-14**; see the maps check below.

  The conda environment carries the same upstream EPICS version twice:

      epics/lib/linux-x86_64/libCom.so.3.22.0                1906200 B   what we link
      site-packages/epicscorelibs/lib/libCom.so.7.0.7.99.0    549032 B   what p4p loads

  Both report `EPICS R7.0.7` / `EPICS 7.0.7.0` and **both define
  `fileDescriptorManager`**.  So a process that links one and imports anything bringing
  the other ends up with two `fdManager` objects and two static destructors over the same
  symbol.  At `_dl_fini` the second frees what the first already freed:

      #6  free ()
      #7  fdManager::~fdManager (this=0x... <fileDescriptorManager>) at ../fdmgr/fdManager.cpp:73
      #8  __cxa_finalize ()
      #9  __do_global_dtors_aux () from .../epics/lib/linux-x86_64/libCom.so.3.22.0
      #10 _dl_fini ()

  **WHAT BRINGS IN THE SECOND COPY IS `p4p`, AND THAT NAMES THE EXPOSED SET**
  (established 2026-10-15, from lcls2 issue #134).  Seven config modules import it --
  `epix100_config`, `hsd_connect`, `hsd_config`, `ts_connect`, `ts_config`, `xpmmini`,
  `tb_config` -- so the set is **any DRP whose detector config touches `p4p`**, not
  every DRP as this entry used to say.  The CPU `drp` links the same
  `libCom.so.3.22.0` and embeds the same `libpython3.9` as `drp_gpu` (`ldd`), so
  nothing about the mechanism is GPU- or hutch-specific.

  **It explains a report that looked like three separate faults.**  Issue #134 filed
  this as an MFX `epix100` shutdown bug, with occurrences on `epix100_0_0`,
  `epix100_1_0` and `timing_1`.  All three import `p4p`:
  `configdb/epix100_config.py:191` and `configdb/ts_config.py:5`, the latter being what
  a `TimingSystem` detector goes through.  **The common factor is `p4p`, not the
  device** -- and "only on the TERM path" is not a property of the shutdown code but of
  static destructors running only at `_dl_fini`.

  **Not measured: whether an MFX DRP maps both copies.**  The `/proc/<pid>/maps` check
  was done on a GPU DRP here.  For MFX it is inference from identical binaries, an
  identical environment and a `p4p`-importing config module.  One
  `grep libCom /proc/<pid>/maps` on a live MFX DRP settles it and needs their `sudo`.

  **The fix that works: link the copy `p4p` already loads.**  Same source, only the
  `-I`/`-L` changed, plus an explicit `-lpthread` (that build does not carry
  `pthread_atfork` itself):

      minimal.cc    linking epics/           double free, SIGABRT
      minimal.cc    linking epicscorelibs    exit 0
      pvPutTest.cc  linking epics/           double free, SIGABRT
      pvPutTest.cc  linking epicscorelibs    exit 0

  `epicscorelibs` ships what would be needed: headers plus `libCom`, `libca`, `libpvData`,
  `libpvAccess`, `libpvAccessCA`, `libdbCore`, `libdbRecStd`, `libqsrv`.  In
  `meson_options.txt` terms it is `epics_base` pointing at
  `site-packages/epicscorelibs` rather than `epics/`.

  **What is NOT established, and is the work:**

  - **Twelve `meson.build` files reference EPICS** (recounted 2026-10-15; this said ten),
    and only two standalone programs have been tried.  A real change means rebuilding everything that links it -- `epicsArch`,
    `PvaDetector`, `xpm`, `hsd`, `epicstools`, `app` -- and showing nothing regresses.
  - ~~**The two builds may not be interchangeable**~~ **NO SYMBOL IS MISSING, measured
    2026-10-14.**  `nm -D --undefined-only` over all 18 installed targets that link EPICS
    (13 executables + 5 libraries) gives 859 undefined symbols, of which **88** are
    satisfied by `epics/` today and **0** are absent from `epicscorelibs` -- which exports
    8054 against `epics/`'s 8027.  So the 3.5x size difference is not missing
    functionality.  **This is a link-time check only**: the include layouts differ (below),
    so header compatibility is untested, and behaviour is untested.
  - **`-Depics_base=<epicscorelibs>` cannot work as-is -- the layouts are different
    shapes.**  This is a build-system port, not a reconfigure:

        epics/         lib/linux-x86_64/libCom.so.3.22.0  include/{,compiler/gcc,os/Linux}/
        epicscorelibs  lib/libCom.so.7.0.7.99.0           include/  flattened, no subdirs

    `meson.build:68` composes `lib/<epics_host_arch>`, and `epicstools/meson.build:9-11`
    add `include/compiler/gcc` and `include/os/Linux` as include directories.  Neither
    subdir exists under `epicscorelibs`; the headers themselves are all present, just
    flat (`compilerSpecific.h`, `osdSock.h`, `epicsAtomicOSD.h` sit directly in
    `include/`).  A symlink shim presenting the epics-base shape would let this be tested
    without touching any `meson.build`, if it is ever wanted.
  - ~~**Whether a live DRP really loads both**~~ **CONFIRMED 2026-10-14 on a live
    `drp_gpu`** (gpu001, Unallocated): `grep libCom /proc/<pid>/maps` shows five mappings
    of `epicscorelibs/lib/libCom.so.7.0.7.99.0` and five of
    `epics/lib/linux-x86_64/libCom.so.3.22.0`, distinct inodes, each with its own writable
    `rw-p` segment -- so each copy has its own `fileDescriptorManager`.  The reproducer is
    no longer standing in for anything.

  **Then give Valerio instructions**, he heading conda environment development (Ric,
  10-13).  The ask is for an environment where a program linking EPICS and a Python module
  importing EPICS resolve to **one** set of shared objects -- either by not shipping the
  second copy, or by having `epics/` be a link to `epicscorelibs`, or by building
  `epicscorelibs` against the environment's EPICS rather than vendoring its own.  Which of
  those is practical is his call.

  **ORDERING DECIDED 2026-10-15: ask him WITHOUT rebuilding the tree first.**  This entry
  used to say the tree rebuild came first, "the demonstration being what makes the request
  concrete".  The demonstration is already concrete -- named cause, live-process
  confirmation, a 9-line reproducer, and 0 missing symbols -- and the rebuild is a *port*
  that this same entry calls deliberately not ours.  Doing it first would mean doing his
  job to prove he should do his.  Route: **a note on issue #134, then a heads-up to him**;
  the issue is the durable record Seshu's tooling and other hutches will find, the message
  is what gets a human to act.  Drafts of both are in
  `~/claude-sessions/2026-09-11-gpu-drp/` (`issue-134-comment.md`,
  `valerio-message.md`).

  **DONE 2026-10-15: both posted and sent, and Valerio has replied that he is on it.**
  So the environment fix is with its owner and **nothing here is owed** -- do not start
  the `epicscorelibs` port.  What may still come back to us is a question about what the
  DAQ needs from the environment, or a request to test a candidate environment; the
  symbol comparison and the reproducer are the things to hand over if asked.

  **Scope of our help, Ric 2026-10-14: the environment and the build are Valerio's to
  sort out; help where we cheaply can, but do not bend over backwards.**  So what we
  hand over is the diagnosis, the live-process confirmation and the symbol comparison --
  all of which are done.  **Porting `meson.build` to the `epicscorelibs` layout is
  deliberately NOT ours** and was stopped before it started: it would change what every
  EPICS target links against, which is the wrong change to land right before a master
  merge and a stability demonstration.

  **Consequence for the 30 s exit, which is the lesser half.**  Every DRP takes ~30 s to
  exit after SIGTERM -- `KillWait`, after which Slurm SIGKILLs it -- because the shutdown
  action is `exit(0)` (`PGPDetectorApp.cc:852`), which runs the static destructors that
  fault.  `abort()` instead skips a teardown that is broken in a *library* for reasons
  outside our control, which is a better argument than speed; Ric has moved many sites
  that way already and the GPU DRP's was missed.  The time itself is most likely
  `systemd-coredump` writing a dump of a process holding a 1.1 GB pebble, but **that dump
  has never been found**, so the attribution is unverified; defeating core dumps on the
  GPU nodes is outstanding.

  **The reproducer, and what it took to reduce it.**  Ric's original is
  `~/lclsii/daq/obsolete/drp/pvPutTest.cc` with `p4pImpTest.py` (and `p4pTest.cc` beside
  them), pared down from the DRP in 2022 under RHEL7; it still reproduces.  The switch is
  whether the three `p4p` imports in the `.py` are commented out -- as shipped, they are.
  Built and bisected 2026-10-13 in
  `claude-sessions/2026-09-11-gpu-drp/doublefree/`, which also holds a 9-line
  `minimal.cc`: link `libCom`, `PyImport_ImportModule("p4p.nt")`, return.  **No EPICS call
  is made at all.**  What the bisection settled:

      imports commented out (as shipped)      exit 0
      any ONE of the three p4p imports        crashes -- not specific to p4p.server
      PyGILState_Ensure() removed             still crashes
      Py_Finalize() removed                   still crashes
      all Python removed                      exit 0
      minimal.cc linking libCom               crashes
      minimal.cc NOT linking libCom           exit 0

  So the GIL and `Py_Finalize` are both red herrings, which is worth knowing because the
  reproducer's shape invites blaming them.  The trigger is only: link EPICS directly
  **and** import a module that carries its own.

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

- **Reducer-specific metrics.**  The general path is done: an algorithm sets the
  `retCode` that `recordGraph()` hands it, `_reducerLoop` carries it in `ReducerTuple`, and
  the recorder counts non-zero codes as `DRP_redErrors`, logging the first ten.  Still
  open, Ric's second idea: a virtual `ReducerAlgo::setupMetrics(exporter, labels)`,
  default no-op, called from `Reducer::setupMetrics()`, for numbers only the algorithm
  knows (compression ratio, say).

- **`maxTrSize`: the buffer-sizing problem IS fixed.  What is left is only the error
  message.**  Two separate things wore the same name, which is why this item has read
  confusingly:

  1. **`maxTrSize` governing every reduce buffer -- FIXED.**  `ReducerAlgo::payloadSize()`
     used to be silently overridden by a `totalSize < maxTrSize` floor, so a
     Configure-shaped hole was carved out of **all `nbuffers()`** to serve something that
     happens once per run.  Transitions now have their own allocation --
     `createTransitionBuffers(m_para.maxTrSize, pebble.nTrBuffers())`, `Reducer.cu:107`,
     separate from `createReduceBuffers(payloadSize, ...)` at `:101` -- and the floor is
     gone (no `totalSize` comparison remains anywhere in `drpGpu`).  This is the one worth
     remembering as done.
  2. **The error message when `maxTrSize` is too small -- STILL OPEN, and it is only a
     message.**  `BEBDetector::_addJson` (`drp/BEBDetector.cc:233`) builds the JSON into a
     *temporary* buffer sized `maxTrSize`, then copies it into a transition Xtc bounded by
     the same number, and the check at `:264` reports **the extent it reached** rather than
     what was needed.  So raising the value moves the failure instead of fixing it: the two
     2026-09-25 failures read as 392-byte near misses at 256 kiB *and* at 512 kiB, while
     the real config was 902026 B.  The value itself is settled -- 2 MiB in the GPU DRP
     (`drpGpu/PGPDetectorApp.cc:858`), 8 MiB in the CPU `drp` (`drp/drp.cc:170`).

  So the remaining work is a Configure-time check that names the real requirement.  It is
  in the CPU `drp` and shared with every CPU DRP, which is what makes it a cheap CPU-side
  diagnosis win -- not a sizing bug.

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

- **Nsight on the live pipeline: deferred.**  With the 12.x tools neither Ric nor NVIDIA
  could profile the device-launched graphs and tail launches.  13.3 is installed and may
  do better; untested, and finding out may take a while.  `RmProfilingAdminOnly: 1` on
  gpu008, so the GPU counters need root.  `nsys` on sim mode is the cheap first probe.
  Recipe: Confluence "GPU DRP" page (PSDMInternal 685820459), "Running nsys" -- Cheolhong's
  `nsys launch` + start/stop timeline of the live DRP; add `--cuda-graph-trace=node`.
  Whether that sees inside the tail-launched graphs is unrecorded.

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

- **Port or retire `EipReducer`, `CuSzReducer`, `CuSZpReducer`.**  Unbuilt, and still on
  the removed `hasGraph()`/`reduce()` interface; each needs a `recordGraph()` that sets
  state 2 and writes its size below the payload, as `NoOpReducer` does.  Ric to ask Matt
  about EIP.

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

- **Port `drp/EpixUHR3x2` into `drpGpu/` if borrowing it costs more.**  The GPU class
  relies on it for configuration and for the event `Names` (hence `setRawAlgVersion()`).
  Ric, 2026-10-22: if further complications arise, a GPU-owned version would be simpler,
  and symmetric with `EpixUHRemu`.

- **psana's `epixuhr3x2_raw_1_0_0` is on Gabriel's `features/uhr_cpu_v5`**, along with
  the CPU DRP's move to raw 1.0.0 and the v5.0.0 single-panel layout.  Until that is on
  master, GPU-recorded 3x2 data has no `det.raw` from a master build.

- **Use the f/w header on tdest 4.**  Its per-ASIC OutputFP16 bit could select the u16 or
  fp16 policy instead of the `raw` kwarg; its maximum value is a cheap sanity check.

## Runtime behaviour

- **`drp_deadtime` aborts the DRP when its XPM PV is slow to connect.**  `_pvGetVecElem`
  (`drp/DrpBase.cc`) calls `abort()` on the first scrape if `DeadFLnk` isn't ready, after
  Configure has succeeded; shared with the CPU DRP.  Hit once, 2026-10-21 16:53:07,
  `tstcam1_5` only.  Ric's idea: skip the sample and retry on later scrapes.  Trap:
  `PV::m_ready` is latched from `getComplete(5)`, whose promise allows one call only, so a
  retry must key off `connected()` plus a first `updated()`, not call it again.

- **`drp_dma_in_use_max` was the wrong constant, and is now `pool.pgpEvents.size()`.**
  Found 2026-10-14, settled 2026-10-15 by Ric's own trace.  The gauge read 56 against
  a `_max` of 8 on a GPU DRP, making `psalg/psalg/daqPipes/daqPipes.py:610` compute
  `%DMA` as **700%**.  Cosmetic -- daqPipes is not in regular use -- but the constant
  was simply wrong.

  **THERE ARE TWO TIERS OF "DMA BUFFER" AND THE NAMING IS GENUINELY AMBIGUOUS**
  (Ric, 10-14).  The f/w tier *is* DMA: 8 buffers, `dmaBufCount`, shared between the
  f/w and the GPU.  They are **very short lived** -- the f/w fills one, the `Reader`
  copies it out, and it is recycled -- and their contents land in the host staging
  buffers (`hostWrtBufs` plus `calibBuffers`), which live longer than a DMA buffer but
  not as long as a pebble.  Ric has called these "intermediate buffers"; **"staging
  buffers"** is the better term and is what the code comments now use.

      m_dmaCount / m_nDmaBuffers   8      f/w; nDmaBuffers() = nextPowerOf2()
      m_nbuffers                   2048   staging, = pebbleBufCount

  **The counters track the staging tier, not the f/w tier** -- Ric, 10-15, correcting
  a misreading of mine.  `allocateDma()` counts when a f/w buffer's *arrival is
  detected*, so its count matches f/w allocations that are never visible to the host;
  `Reader::freeDma`'s `// Count freed 'DMA' (host) buffers` is accurate, those being
  `hostWrtBufs`.  The decisive line is `TrgInpGen.cu:413`, which masks `evtCounter`
  with **`nbuffers()-1`**, not `nDmaBuffers()-1` as all six CPU call sites do.

  **The CPU path was always right**: `timing_2`'s `_max` of 1024 is correct, its
  `allocateDma()`/`freeDma()` counting buffers out of and back into the datadev
  driver.  So this is not a CPU-side bug and `pebbleBufCount` is not implicated.

  **Why `pgpEvents.size()` and not `nbuffers()`**: it is the in-flight slot count,
  which is the quantity the alloc/free pair brackets on *both* paths -- it evaluates
  to `nDmaBuffers` on the CPU path (`DrpBase.cc:171`) and to `nbuffers` on the GPU one
  (`MemPool.cc:446`), with no virtual and no GPU-specific branch in shared code.

  **The f/w tier's in-use count is deliberately NOT exported** (Ric's call, and the
  code agrees -- `MemPoolGpu::_freeDma` is `{ return 0; }`, the host taking no part in
  f/w buffer reuse, which is the `writeEnable`/freelist handshake).  Getting it would
  mean a device-to-host copy per event for a number that stays in [0,8] and only
  matters when pinned at 8.  A saturation flag on the existing metrics copy would be
  the cheap version if anyone ever wants it.

  **Three dead ends of mine, recorded so nobody walks them again:**

  1. "Nothing in `drpGpu` increments the counters" -- **wrong**, `Reader.cu:706`
     calls `freeDma`.  I grepped for `m_dmaAllocs` instead of for the methods.
  2. "56 > 8 is impossible, so the gauge is broken" -- **wrong**, it is 56 of the
     2048-deep staging tier, nowhere near its rail.
  3. "The GPU path never allocates, only frees" -- **wrong**, `TrgInpGen.cu:516`
     calls `allocateDma()`.


- **Is the spin-loop the right idea?**  `_waitForDMA` polls the DMA doorbell with a
  `__nanosleep` backoff (8 ns doubling to 256 ns) and gives up so the graph can
  relaunch; `_readerLoop` relaunches itself with `cudaGraphLaunch(...,
  cudaStreamGraphTailLaunch)`.  Worth asking whether that is optimal, and whether to
  spin more or less.  Note the consequence already observed: **a GPU running this
  reads 100% utilisation whether or not data is flowing**, so `nvidia-smi` says
  nothing about real work.  History suggests this was already revisited once —
  f08eacc1 "Relaunch instead of spinning", then 3db054e6 "Revert to separate
  kernels; Time out spin loops".

  **Measured 2026-10-16 on gpu008 (H200), `readerHist=1`:** an idle pass -- spin out,
  three kernel launches, tail-relaunch -- takes **12.2-12.3 us on every DRP**, tightly
  in one log2 bin, so relaunching does not differ between DRPs and does not explain
  the asymmetry.  The spin itself is only ~0.5 us of that; the rest is launch cost.
  What it does cost: a DMA arriving just after the spin gives up waits up to ~12 us
  to be seen, against a 30.3 us per-event budget at 33 kHz.  And `pop` blocks the
  same way -- a failed `pebbleQueue` pop also costs a relaunch, which may be what the
  laggard's slow pops are (open item in the rate-asymmetry section).  Spinning longer
  before returning would cut that latency at the price of a longer-held SM; the
  histograms can measure the trade directly.

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

## The rate asymmetry: what is still open after 2026-10-16

Two causes found and fixed; see the 2026-10-16 entry in `FINDINGS.md`.  What remains:

- **A DRP whose DMA ring fills becomes ~3 us/event slower and stays saturated.**  Measured
  2026-10-16 from the per-second `readerHist` slices, same DRP, same run: `tstcam1_1` ran
  21.0 us with its backlog low and 23.8 us in its 492 seconds with the ring full;
  `tstcam1_0` 20.9 us in 491 low or partly backlogged seconds (backlog up to 15 costs
  nothing) and 23.4 us in its 19 full ones.  Peers that never filled stayed at 20.7-21.1.
  The extra time is in `pop` (+~1 us) and `_event` (+~2 us); no failed pops or pushes, and
  every Reader runs on the same physical SMs.  Normal is ~21 + ~8 overhead = 29 us, under
  the 30.3 us budget at 33 kHz; full is ~32 us, over it, so a DRP that fills can never
  drain.  It is re-chosen at every Enable, without a Configure, and a DRP returns to
  normal once its ring drains.  **What makes a full ring cost 3 us is not known.**
  Candidates: the FPGA restarting a DMA the moment a buffer is freed, so the inbound
  write overlaps the Reader's reads of the same region; contention in L2 or HBM.
  The 10.2 kHz negative control held: no ring filled, every DRP 20.2-20.7 us in every
  second.  Load alone adds only ~0.5 us (33 kHz healthy DRPs run 20.7-21.3); a full ring
  adds ~3.  **Next:** find the mechanism.  Headroom of ~2 us would let even a full DRP
  keep up.
- **Only node-0 DRPs have ever been the laggard** -- `tstcam1_0`, `_1`, `_2`, never `_3` or
  `_5`, over ten runs.  Node 0 also carries the IB NIC (`mlx5_0`, `71:00.0`), WEKA, and
  `tstcam1_1`'s cross-root DMA (`05` -> `55`) landing on switch 51 beside `tstcam1_2`'s GPU.
  Speculative; nothing measured yet ties any of these to a slowdown.
- **The Reader's margin against the design rate is ~3.7 us** (2026-10-20): 16.2 us of service
  on hardware + 8.1 us of measured overhead is ~24.3 us of the 28 us budget.  A third of it is
  overhead, which is what batching attacks.
- **Host accesses left in TrgInpGen and the Reducer** (read 2026-10-20, not measured):
  - **The Reducer's per-event index is pinned host memory** (`m_indices`, `Reducer.cu:68`),
    though nothing on the host reads it.  Every thread of every reduce block dereferences it,
    so each event costs ~one PCIe round trip per block (NoOpReducer: 320 blocks).  Likely the
    largest per-event host cost left; moving it to `cudaMalloc` looks like a one-line change.
  - TrgInpGen's per-event index and state are already device memory; its kernels are
    `<<<1,1>>>` and touch the host only through the rings fixed above.
  - `TmoTebPrimitive::_event` writes 2 words into the pinned `hostWrtBufs` per event: posted
    writes, which are the data the host needs.
- **Process events in batches per stage** (Ric, 2026-10-20), to amortize the per-event
  inter-stage cost: one ring push/pop, relaunch and launch per batch instead of per event.
  The host already batches at the collector (`TrgInpGen::_receiver()` hands `nRet` events
  on); the GPU stages do not.  For the Reader, a batch is "every DMA already landed", which
  the backlog walk in `_waitForDMA` already counts, so it costs no latency when idle.  Same
  family as kernel fusion and the persistent kernel; weigh them together.  The cost it
  divides is measured: 8.1 us per event of relaunch and launch overhead on the H200 (6.3 on
  the A5000), from simulator runs with the ring full (2026-10-20).  The simulator
  (`-d /dev/null -D epixuhrsim -k sim_l1_delay=0`) is Reader-limited, so it is the test bed:
  46.1 kHz on gpu008 GPU 0, 38.7 kHz on gpu001.
- **More threads for `_event`** (Ric), for headroom.  Its grid already fills the Reader's
  partition (256 blocks x 64 threads = 8 SMs x 2048 on an H200, ~12 elements per thread), so
  more threads means a larger Reader context, e.g. 16 SMs from the recorder's 76.  `_event`
  is ~12 of the ~21 us, so this is the biggest lever if the kernel scales.  Keep the grid
  sizing in `Reader::_recordGraph()` and the SM count in `_setupGreenContexts()` in step.
- **Ring index caching and `RingIndexDtoD` in device memory: measured 2026-10-20.**  Each
  ring's GPU side keeps its own index in device memory and reads the host's index only when
  its cached copy says empty or full; `RingIndexDtoD` lives entirely in device memory and the
  metric copies it out at scrape.  At 33 kHz the Reader's service fell from ~21 to 16.2 us
  (pop 5.7 -> 2.7, push 3.4 -> 1.7) on all five DRPs.  No ring stayed full, so the full-ring
  penalty is unmeasured; one brief fill showed +0.4 us in that second, which proves little.
  Not worth forcing: even the old ~3 us would drain at 35.7 kHz.  Revisit if a ring stays full.
- **The remaining ~1.3 ms "laggard" looks like the PCIe ceiling, not software** (2026-10-20,
  inferred).  With all five, `tstcam1_0` (whose card also carries `timing_2`) lagged and
  asserted deadtime while its Reader backlog stayed ~0; without it, all four tstcams did
  the same against `timing_2`, at the same 33 kHz, Readers unchanged at 16.2 us.
  Every TDet datadev has a timing fiber, so `timing_2` can move off `datadev_04` (e.g. to
  the spare 0x85 card); untested whether the shared card is what makes `tstcam1_0` lag.
  TrgInpGen and Reducer are still unmeasured against 28 us.
- **Node memory on gpu008.**  417 GB of `nfs_inode_cache` and 43 GB of `dentry` leave 3-8 GB
  free per node.  Our allocations are now bound and immune, but other processes are not.
  Ric's call whether to raise it with whoever runs the node, or try `drop_caches`.
- **`MAX_BUFFERS 8` in `GpuAsyncOffsets.h:49` is a stale comment.**  It says "must match
  firmware value"; the firmware does 1024 and only `pgpread.cc` uses it.  Fix the comment.
- **`drp_deadtime` reads negative on some DRPs** (-4.8, -11.6 seen 10-15).  Not understood.
- **The A5000 split (6/2/40/16) has not been run on gpu001.**  Only runnability matters there.
