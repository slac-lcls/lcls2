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

- **`AreaDetector` (A.K.A. `fakecam`) does not handle L1Accepts, and would fault if it did.**
  Raised by Matt, 2026-09-30.  Three things are inert or wrong in `AreaDetector.cu`:

  - `configure()`'s Names block is inside `#if 0`, so `m_namesLookup` is never populated for
    `EventNamesIndex` and nothing describes the event payload;
  - `event()` is an empty stub that only logs;
  - `pedestals_d()` and `gains_d()` both return `nullptr`, yet `recordEvent()` launches
    `PedGainCalib`, whose `pedGainCalibrate()` indexes `pedArray`/`gainArray` unconditionally
    (`ReaderKernels.cuh`).  That is a null dereference on the device, latent only because no
    L1Accept reaches it today.

  Both `#if 0` blocks carry the same "@todo: Deal with prescaled raw or calibrated data for each
  panel here?" comment, which dates from before the raw block existed, so the answer is now
  known: it does what `EpixUHRemu` does.

  It should also get **CALIB mode** once it works.  See the item below for what that takes.

### CALIB mode is still missing from most detectors

`ePixUHR3x2` and `EpixUHRemu` have it as of 2026-09-30.  **`Jungfrau`, `EpixUHRsim` and
`AreaDetector` do not**: none of them overrides `rawSize()`, so a CALIB alias against any of them
aborts in `PassthruShim` -- correctly and loudly, with "it has no raw mode, so it cannot serve a
CALIB configuration," but it does mean the alias is not yet universal.

Four things each, all modelled on `EpixUHR3x2` or the simpler `EpixUHRemu`:

1. `rawSize()` returning the frame's u16 byte count **unconditionally** -- it is capacity, and
   prescaling needs it in BEAM too, so it must not test `passthru()`.  See the contract on
   `Gpu::Detector::rawSize()`;
2. `rawShape()`, whatever shape offline expects of that detector -- which means **the shape the
   CPU DRP already writes**, not whatever is convenient on the device;
3. a pass-through per-element policy, like `EpixUHR3x2Calib` or `EpixUHRemuCalib`;
4. a `RawU16Def` for `configure()` -- all three currently declare `{"raw", Name::UINT8, 1}`, a
   flat byte array that would misdescribe u16 pixels.

**Do these after prescaling, not before, and in this order.**  Prescaling changes the relationship
between a pass-through policy and the raw block: it has the Detector copying raw into the raw
region *while* the selected Reducer writes reduced data, where CALIB writes raw instead of
reduced.  Three policies written against today's shape risk being written twice, and the
duplication would land in the per-element device code that is hardest to verify.

1. **`EpixUHRsim`** -- start here.  It overrides neither `subframeCount()` nor
   `firstDataSubframe()`, so its payload is one contiguous block and it is close to a
   transcription of `EpixUHRemu`.  Its reference buffers, which exist to verify the calibration,
   have nothing to check in pass-through: no calibration runs, so a raw policy ignores them.
   Cheap, and it exercises the CALIB path on a second detector.
2. **`Jungfrau`** -- the substantial one; see below.  The reassembly refactor is the real work.
3. **`AreaDetector`** -- blocked until it handles L1Accepts at all.  See the item above: its Names
   block is `#if 0`'d, `event()` is a stub, and its `nullptr` pedestals would be dereferenced on
   the device.  CALIB mode means nothing there until that is fixed.

**`Jungfrau` is the substantial one**, and not a copy of either existing policy.  Its payload is
a batch whose data sub-frames are *themselves* batches: each module's sub-frame holds
`PacketNum` = 128 UDP packets, each a `JungfrauData::Header` plus `PixelPerPacket` = 4096 u16
pixels, and the header's `packetnum` -- not the packet's position in the batch -- says where
those pixels belong.  `JungfrauCalib` (`Jungfrau.cu:188`) already does that reassembly, building
an inverse slot-to-packet map in shared memory per module, zeroing slots no packet claimed,
zeroing an absent module's whole frame, and dropping a `packetnum` outside the frame.  A
pass-through policy has to reproduce all of it, writing u16 to `pyld.raw` instead of float to
`pyld.out`.  Worth factoring the reassembly out so the two policies share it rather than
duplicating that logic, since getting it subtly different between calibrated and raw output
would be a hard bug to see.

Its tdest mapping is also not a contiguous run -- module 0 is at tdest 2, later modules at
`module + 3`, skipping 3 -- but `tdestOfModule()` already encapsulates that.

`EpixUHRsim` should be close to `EpixUHRemu` -- it overrides neither `subframeCount()` nor
`firstDataSubframe()`, so its payload is one contiguous block too.  Note that its reference
buffers, which exist to verify the calibration, have nothing to check in pass-through: no
calibration runs, so a raw policy there ignores them.  `AreaDetector` needs the item above fixed
first.

- **EpixUHR3x2 gain encoding.**  `RangeOffset`/`RangeBits` in `EpixUHR3x2.hh` are
  moot for an fp16 payload, which arrives already calibrated, but **not** for the u16
  payload the firmware presents today and which `raw=u16` now selects by default: the
  gain bit is bit 0 and the ADC value bits 1-11.  Confirm nothing else wants them.

- **`raw=fp16` is refused, pending the firmware's mode registers.**

  **The plan, corrected by Chris on 2026-10-05.**  An earlier reading of it here was
  wrong and is worth stating plainly, because several notes below were written against
  it.  What the updated firmware will actually do:

  - new **registers select u16 or fp16 mode** for the panel;
  - the **"raw"** block -- "raw" being the label for the *prescaled* data -- is then u16
    **or** fp16 according to that register, **not both**;
  - the **non-raw** block becomes **"fex"** data: fp16 calibrated, or an ROI, or some
    other signal extraction.

  **So the DMA buffer does not need to be twice the full raw size.**  u16 and fp16 are
  both 2 B/pixel, so the raw block is 387072 B in either mode and `dmaBufSize` is
  unchanged by the mode register.  What the marked-event DMA has to accommodate is
  `32 + raw + fex`, where fex is *at most* a full fp16 frame and may be much smaller --
  against the 774176 B that "u16 alongside fp16" would have required.

  The wrong premise was "u16 *and* fp16 together on an asserted keepRaw, roughly
  doubling the DMA".  Nothing is doubled, and the "separate sub-frames vs interleaved
  `{u16, __half}` elements" question is **moot**: a mode register means one format at a
  time, so there is no pairing to lay out.

  **Consequences for this code:**

  - `rawSize()` stays `NPixels * 2` and needs no mode awareness -- it is already right
    for both.
  - `rawShape()` stays as it is, but **`RawU16Def` does not generalise**: the Xtc type
    system has **no fp16** (`ShapesData.hh:56`: `UINT8, UINT16, ..., FLOAT, DOUBLE`), so
    an fp16 raw block cannot be described as fp16.  It must be declared `UINT16` and
    reinterpreted offline, or widened to `FLOAT` on the device at 2x the bytes on disk.
    **That is a schema question for Chris and psana, and it is new** -- the old note said
    "no schema question arises", which was true only while raw was always u16.
    `FINDINGS.md` carries the worked proposal, which landed as `FLOAT16` = 13.
  - the per-element work is still undecided, so the `raw=fp16` abort stays.  But what it
    is waiting for has changed: not a layout, just the mode register and the fex format.

  `EpixUHR3x2::EpixUHR3x2()` aborts on `raw=fp16` with a message naming the reason, and
  `m_u16` defaults to true.  `EpixUHR3x2Beam` is **kept and still instantiated** by
  `recordEvent()`'s unreachable `else`, so it keeps compiling as the surrounding code
  moves and the revisit starts from working code rather than from rot.

  Things that will matter at the revisit, found while preparing for it:

  - **The sub-frame scan cache will still thrash, for a different reason.**
    `EvtBatcherSubFrames::Scan` latches exactly *two* sizes -- `bytes` for the data
    layout and `noDataBytes` for transitions -- and `Reader.cu:389` rescans whenever a
    payload matches neither.  Under Chris's plan a marked event carries `raw + fex` and
    an unmarked one carries `fex` alone, so there are **still two distinct L1Accept
    sizes** competing for the single `bytes` slot: every marked event evicts the
    ordinary scan and the next ordinary event evicts it back.  The *magnitude* of the
    difference changed -- it is the raw block, not a doubling -- but the thrash does not
    depend on the magnitude, only on there being two sizes.

    The one case that would avoid it is a fex block sized so that `raw + fex` equals
    some other recurring size, which is not worth engineering.

    **This is a prediction about the planned firmware, not a defect in prescaling as
    implemented.**  Nothing thrashes today: one L1Accept size and one transition size fit
    the two slots exactly.  It was misread once as a known problem with the committed
    code.  Note also that the fix is not a third `size_t` -- `m_sub[]` is a single flat
    tdest-indexed array that pass two overwrites wholesale, so a third slot needs a second
    layout array beside it.  The cost is small either way: two short walks per second
    against ~33 kHz of events, so it is latency on two events, not throughput.
  - **`firstDataSubframe()` may still become mode-dependent**, if raw and fex arrive on
    different tdests.  Then the tdest set differs between marked and unmarked events, so
    `count` and the per-tdest offsets change rather than just the total size, and
    `NumSubFrames` and `FirstDataTdest` stop being single constants.  **Ask Chris**
    whether fex replaces raw on the same tdests or occupies its own -- that is now the
    open layout question, and it decides whether this item is real.
  - ~~**Interleaved elements would break the sub-frame element counts.**~~  **Moot**: a
    mode register means one format at a time, so there are no `{u16, __half}` pairs and
    `cnt = sub.size / sizeof(*src)` keeps meaning a pixel count.  `_copyRaw`'s strided
    copy per ASIC stays a copy rather than becoming a gather.

- **Jungfrau pedestals and gains** are placeholders (0.0/1.0).  The CPU-side
  Jungfrau writes raw data and leaves calibration to analysis, so there is no
  source for real constants yet.  Also unresolved whether the GPU path should
  calibrate Jungfrau at all or pass raw through.

## Performance and structure

- **`maxTrSize` does two unrelated jobs and wants to be two numbers.**  It bounds a
  transition's Xtc -- `Drp::Detector`'s `m_xtcbuf(para->maxTrSize)`, reached through
  `trXtcBufEnd()` -- *and* it floors the reduce buffers' payload, at `Reducer.cu`:

      if (totalSize < m_para.maxTrSize)  payloadSize = m_para.maxTrSize - headerSize;

  So a value chosen to fit the largest Configure is multiplied by `nbuffers()` of GPU
  memory, and a value chosen to bound the payload may be too small for a Configure.  The
  two pull in opposite directions.

  Worse, `BEBDetector::_addJson` (`drp/BEBDetector.cc:233`) builds the JSON config into a
  *temporary* buffer **also** sized `maxTrSize` and then copies its payload into the
  transition Xtc.  So one number bounds both the config and the thing it must fit inside,
  and a value that is too small fails at whatever it is raised to rather than by a fixed
  shortfall.

  Found on 2026-09-25: ePixUHR3x2's Configure on drp-srcf-gpu006 aborted with

      Xtc.hh:111: Insufficient space for 524 bytes (... extent 262012)    # 256 kiB
      Xtc.hh:111: Insufficient space for 760 bytes (... extent 524036)    # 512 kiB

  The extent grew to fill whatever it was given, which is the tell.  **`epixuhr3x2_0`'s
  configuration is 902026 bytes of JSON**, so neither limit was close; the "overflowed by
  392 bytes" reading of the first failure was where it stopped, not what it needed.  Raised
  to 2 MiB, which costs ~4 GiB of GPU memory at `nbuffers = 2048`, all of it unused in
  pass-through mode since `PassthruShim::payloadSize()` is 0.  The CPU DRP's 8 MiB would
  cost ~16 GiB.

  Making either a kwarg would help, but the coupling is the real problem -- raising one for a
  detector's sake silently enlarges every reduce buffer.

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

### The UHR emulator's payload is constant, so it cannot prove per-event freshness

Gabriel, 2026-10-05: the data the UHR emulator currently emits **is the same from event to
event**.  He believes a register exists that would make it emit sequence numbers instead but
is not sure of it or how to set it.

That matters for what a run can demonstrate.  On `EpixUHRemu` the frame counter ramping
0,1,2,... with all deltas 1 was the proof that each raw block is freshly copied -- it is
what validated runs 269 and 277.  On the real ePixUHR3x2 the payload is event-invariant, so
`xtc-rawdump.py` reports `element[0]` constant and identical `first12` and nonzero counts on
every event (runs 65 and 66).  **That is expected, not a fault** -- but it means a real-panel
run has **no per-event liveness signal in the data**: one stale frame repeated would look
identical.  Liveness has to come from elsewhere -- the pulse ids and timestamps, which do
advance, and the `keepRaw`/extent agreement.

Worth chasing that register if a per-event check on real hardware is ever wanted.

- **Fetch calibration constants.**  Every detector currently fabricates them:
  `EpixUHRemu`, `EpixUHRsim`, `Jungfrau` and now `EpixUHR3x2` in its u16 mode fill
  pedestals with 0.0 and gains with 1.0 (`@todo: Fetch calibration constants`), which
  makes the calibrated values numerically equal to the raw **ADC field** -- not to the raw
  word, where a gain bit below the data shifts the value (`FINDINGS.md`).  The path is
  proven, the science is not.  `EpixUHR3x2` needs none for an fp16 payload, which
  arrives calibrated from firmware.  Needs a real source and a point in the transition
  sequence to load from it.  Three candidate routes:

  1. reuse Mikhail's code in `lcls2/psana`, which is the source of truth;
  2. resurrect `lcls2/psalg/psalg/calib/`, also Mikhail's, whose headers are still
     there (`CalibPars.hh`, `CalibParsDB*.hh`, `CalibParsStore.hh` and friends);
  3. follow Gabriel's pseudo-code, reproduced verbatim in `FINDINGS.md`.

  In outline: detector type plus serial number to a "short name"; a metadata query on
  that short name, ordered by run, pointing at the bulk data, needing filtering on
  run number and validity flags; then the bulk fetch.  The calibdb schema is
  documented nowhere else, which is why `FINDINGS.md` keeps it.

  **For ePixUHR3x2 the gain bit is not enough to identify the constants.**  Gabriel,
  2026-09-29: the single bit "always selects between only two states", so
  `NRanges = 2` is right, but *which* two depends on the configured mode -- "your bit
  status could mean your pixel is in high or low gain 1, or high or low gain 2 ... You
  need the configuration ... to be able to complete the picture."  Some modes are fixed
  rather than auto-ranging.

  So the fetch has to be keyed on the gain configuration, not just on the detector.
  The good news is that the information is already computed in the DRP:
  `configdb/epixuhr3x2_config.py` fills **`gainMapSelection`** (per-pixel, from
  `cfg["expert"]["pixelBitMaps"][...]` when `user.Gain.UsePixelMap` is set) and
  **`gainValSelection`** (uniform, from `user.Gain.SetGainValue`) at Configure.  Neither
  reaches the GPU today.  Whoever does this should ask **Mikhail**, who has done the
  equivalent on the psana side, and check Confluence for a TID write-up -- Gabriel does
  not recall one.

- **Calibration mode.**  The DAQ operator selects the **CALIB** alias instead of the
  usual **BEAM** alias.  That selects a different, perhaps derived, set of detector
  register settings, written to the detector through
  `configdb/<detector>_config.py` during Calibrate.  The DRP has to recognise that
  this state is active and record **raw** data as it comes off the detector's fibre,
  rather than calibrated or reduced data, with the reducer bypassed or turned into a
  no-op.  Low rate running is acceptable in this mode.

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

## Lower priority, after the January deliverables

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

  Shares a mechanism with the `rdmaTest` timeout (Confluence) -- both are bounded waits -- but not
  a purpose: that one is about saying why nothing arrived, this one about not burning power
  while nothing arrives.
