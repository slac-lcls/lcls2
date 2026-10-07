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
