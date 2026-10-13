# GPU DRP findings

Settled explanations, kept because they were expensive to establish and because the
reasoning behind several decisions in `TODO.md` lives here.  Nothing here is open
work.

## CALIB mode records raw data end to end, validated 2026-10-01

Run 269 on drp-srcf-gpu001 with `EpixUHRemu` at 10 Hz, the first execution of the CALIB path and
of `libcalibTrigger_gpu.so`.  36 L1Accepts of 48 events total, in
`/home/claus/data/tst/tstx00817/xtc/tstx00817-r0269-s001-c000.xtc2` (the detector is `s001`;
`s000` is the timing DRP), read back with `xtcreader -f <file> -d`:

| check | result |
|---|---|
| declared type | `Type 1 Rank 1` = `UINT16`, rank 1 -- the `RawU16Def` and flat `rawShape()`, not a byte array.  **Rank 1 was the shape at the time; it is rank 2 `[NumAsics][AsicPixels]` now -- see the emulator-shape rule in `RULES.md`.  A re-run reads `Rank 2`, with the same element count and extent** |
| element count | 193536 = 387072 / 2, so the Names entry matches the raw block as u16 |
| `payloadSize` / `extent` | 387128 / **387140** = 387072 raw + 68 of descriptors, uniform on all 36 |
| damage | `0x0` on every event |
| data | emulator frame counter ramps **0 to 35 with no gaps**; last element constant |
| shutdown | Disable, EndStep, EndRun all clean |

The gapless ramp is the load-bearing check: it says the trigger primitive's kernel completed the
`0 -> 1 -> 2 -> 0` state cycle on every event.  A missing state advance stalls on the first one,
so extent and damage alone would not have distinguished a working kernel from a stalled graph
that happened to record one buffer.

Compare BEAM on the same detector: `2048 * (80 + 0 + 774144)` and extent 774212.  CALIB gives
`2048 * (80 + 387072 + 0)` and 387140.  Both descriptor overheads are 68, which is the figure to
use for `EpixUHRemu`; ePixUHR3x2's stage-1 result was 56 for a rank-2 shape, so the overhead
follows the Names shape and is not a constant to carry between detectors.

## epixuhremu_config.py configures the emulator's timing. Done

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

## Give transitions their own buffer, so a Configure cannot dominate the L1A buffers

**Done, and validated on drp-srcf-gpu001 with `epixuhremu` and `NoOpReducer` on 2026-09-29.**
The log now reads `2048 * (80 + 0 + 774144) B` for the reduce buffers, where 774144 is exactly
`NoOpReducer::payloadSize()` -- `NPixels * sizeof(float)` -- so the floor is gone and
`payloadSize()` means what it says.  Transitions got `128 * 2097152 B` of their own, and the
allocation fell from 4.00 to 1.73 GiB: **2.27 GiB back** at `nbuffers = 2048`.  `xtcreader`
confirms the file: Configure extent 7348, L1Accept extent 774212 = 774144 + 68 of Dgram and Xtc
descriptors, and SlowUpdates present.

The count is `pebble.nTrBuffers()`, 128, taken from the CPU pool rather than invented: every
transition except SlowUpdate is synchronous, so no new one can be emitted until the one in
progress is acknowledged, and SlowUpdate at 1 Hz is the only one that can accumulate.  128 is
therefore about two minutes' worth.  Caveat from Ric: the SlowUpdate rate has occasionally been
raised to 10 Hz, which would make it twelve seconds, so there may be missing protection
somewhere for that case.  Not chased.

The slot is derived from the pointer the CPU already recorded --
`(dgram - pebble.trBuffer()) / pebble.trBufSize()` -- so there is one allocator and no second
lifetime to manage.

What follows is the original reasoning, kept because the numbers still justify the shape.

Ric's proposal, 2026-09-25, and the numbers argue for it strongly.  **The special case already
exists; it just does not pay its way.**  What is already true:

- transitions already take a **separate branch** in the recorder,
  `buffer -= sizeof(Dgram)` rather than the L1A path's arithmetic
  (`PGPDetector.cc`, the `else { // Transitions` arm);
- Configure already has a **dedicated buffer index**, `m_configureIndex`, and is re-copied to
  the GPU at BeginRun from a separate *host*-side `m_configureBuffer`;
- `ReducerAlgo::payloadSize()` is already **silently overridden** by the `maxTrSize` floor in
  `Reducer.cu`, whose only purpose is to make every L1A buffer big enough for a transition.

So the present design carves a Configure-shaped hole out of **all `nbuffers()`** to serve
something that happens once per run.  The shape of the change:

    createReduceBuffers(payloadSize, headerSize, rawBytes);  // payload = the algo's ask
    createTransitionBuffers(maxTrSize, nTrBuffers);          // a handful, not nbuffers()

then delete the `if (totalSize < m_para.maxTrSize)` floor and point the existing transition
branch at the new allocation.

**What it saves**, with `raw` = 387072 B for ePixUHR3x2 and a real reducer asking for an fp32
frame:

| | `nbuffers` = 2048 | `nbuffers` = 32768 (1 s of latency at 33 kHz) |
|---|---|---|
| today, payload floored at 2 MiB | 4.74 GiB | **75.81 GiB** |
| transitions separated | 2.22 GiB | **35.44 GiB** |

At the buffer count 1 s of latency actually wants, it **halves** GPU memory -- 40 GiB back on a
140 GiB card -- and it decouples the two, so a detector with a larger Configure costs one buffer
instead of 32768.  It is also paying now, not just later: the 2 MiB set on 2026-09-25 costs
~4 GiB that pass-through never touches, `PassthruShim::payloadSize()` being 0.

Arguably this is *less* complex than what is there: it **removes** a coupling rather than adding
a mechanism, and `payloadSize()` starts meaning what it says.  The care needed is that
transitions and L1As then index different allocations, so anything computing
`&reduceBuffers_d()[index * stride]` must know which kind it holds -- but that code already
branches on `isEvent()`.  Size it for a few transitions rather than one: Configure, BeginRun,
BeginStep and Enable can be in flight together, and the recorder holds Configure to re-write it
at BeginRun.  Four would be ~8 MiB against 40 GiB saved.

## The GPU FileWriter honours the directIO kwarg now. Done

- ~~**The GPU `FileWriter` ignores the `directIO` kwarg.**~~  **Done 2026-09-30.**
  `TebReceiver::setup()` hardcoded `constexpr auto dio{true}`, so `-k directIO=no` was accepted
  and silently disregarded.  Noticed on 2026-09-29, when `directIO=no` was added because
  recording crashed the *timing* CPU DRP: the CPU DRP picked the change up on restart and the
  GPU DRP did not care either way, which is what exposed it.  Now reads the kwarg through a
  `getDioFlag()` mirroring `drp/TebReceiver.cc:18`, same `"yes"` default, and `open()` logs the
  state at debug level -- its absence from the log is part of why this went unnoticed.

  **cuFile supports both states**, so the kwarg is worth having rather than removing.
  `O_DIRECT` was mandatory until CUDA 12.2 / GDS 1.7.x, and the note in `cufile.h` still says
  so -- *"the file needs to be opened in O_DIRECT mode to support GPUDirect Storage"* -- but
  that text is stale against the 13.3 runtime on these nodes.  Per NVIDIA's troubleshooting
  guide, *"Starting with CUDA toolkit 12.2 (GDS version 1.7.x) files can also be opened with
  non-O_DIRECT mode. Even in such a case, whenever the library software deems fit, it will
  follow the GDS enabled O_DIRECT path"*, and the API guide adds that this holds *"in compat
  mode and also with nvidia-fs.ko installed"* -- so it is not only a compat-mode concession.
  In compatibility mode, which is what these nodes run, `cuFileWrite` is `pwrite` underneath
  and the flag buys nothing either way.

  Keeping it selectable matters because the CPU and GPU DRPs need not write to the same file
  system, and a file system that cannot do aligned direct I/O needs it off.  **Untested at
  `directIO=no` on the GPU side**, though: the writer's buffer and offset arithmetic were
  written under the direct-I/O assumption, and while non-`O_DIRECT` is strictly more
  permissive, the first run with it off is a test rather than a formality.

## The firmware counters are cleared once the link is up. Done

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

## The UHR emulator's payload is constant, so it cannot prove per-event freshness

Gabriel, 2026-10-05: the data the UHR emulator emits **is the same from event to event**.
He believes a register exists that would make it emit sequence numbers instead but is not
sure of it or how to set it.

That decides what a run can demonstrate.  On `EpixUHRemu` the frame counter ramping
0,1,2,... with all deltas 1 was the proof that each raw block is freshly copied -- it is
what validated runs 269 and 277.  On the real ePixUHR3x2 the payload is event-invariant, so
`xtc-rawdump.py` reports `element[0]` constant and identical `first12` and nonzero counts on
every event (runs 65 and 66).  **That is expected, not a fault** -- but it means a real-panel
run has **no per-event liveness signal in the data**: one stale frame repeated would look
identical.  Liveness has to come from the pulse ids and timestamps, which do advance, and
from the `keepRaw`/extent agreement.

## The reduced payload is half the raw one, and that is correct

Noticed on run 60, 2026-10-05, and it reads as corruption until the bit layout is recalled.
A prescaled event's two arrays do **not** hold the same numbers:

    'raw'  (shape: 6 32256):  0 48 96 100 52 4 512 560 ...   u16
    'noOp' (shape: 774144):   0.0 24.0 48.0 50.0 ...         f32

Exactly a factor of two, because `pedGainCalibrate()` calibrates the *extracted ADC field*,
not the raw word.  For `EpixUHR3x2` the gain bit is **bit 0** and the ADC value **bits 1-11**,
so `data = (raw >> 1) & 0x7ff` and the gain bit is shifted out.  With fabricated pedestal 0
and gain 1, `calib == raw >> 1`.

So raw and reduced are only comparable after extracting the same field.  They would be equal
only if the ADC value started at bit 0.  The halving is also independent confirmation of
Gabriel's layout: an earlier note had gain in bit 11, which would have made the two arrays
roughly equal instead.

## Prescaling records raw beside reduced on marked events. Validated

- ~~**Prescale implementation.**~~  **VALIDATED on hardware 2026-10-02, run 273 on gpu001
  with `EpixUHRemu` and `NoOpReducer`.**  Raw data recorded *in addition to* the reduced data
  on the events the timing system marks with **`keepRaw`**, bit 22 of the env word
  (`psdaq/service/EbDgram.hh:57`), so offline can reproduce the reduction and verify it.

  What run 273 establishes, after runs 270-272 each failed one layer deeper:

  | check | result |
  |---|---|
  | `xtcreader` and `xtcreader -d` | **both rc=0**, walking all 181 records to EndRun |
  | prescaled events | 16 of 158 L1Accepts, extent **1161340**, two ShapesData |
  | ordinary events | 142, extent **774212**, one ShapesData |
  | `keepRaw` bit vs extent | **0 mismatches in 158 events** |
  | child walk | two children, `remaining: 0` -- no gap, no overrun |
  | damage | `0x0` on all 181 records |
  | raw payload | 193536 u16, 66.1% nonzero, **byte-identical to the validated CALIB run 269** |
  | reduced payload | present on **every** event, prescaled included |
  | guards | neither the split check nor the pebble-reach check fired |

  The raw comparison against run 269 is the strongest single check: both go through the same
  `_copyRaw()`, so identical first-12 values, nonzero count and maximum say the prescale path
  delivers exactly what the already-validated CALIB path does.

  Both questions this item used to pose are settled.  **The buffering** is a small pool of
  prescale buffers (`NPrescaleBuffers = 32`), each `[hdr][raw][reduced]`, into which a marked
  event's whole datagram is assembled; the reduce buffers stay `[hdr][reduced]` and the
  reduced payload is copied across device to device.  Reserving raw in all `nbuffers()`
  reduce buffers was implemented first and **rejected**: it costs 0.74 GiB at
  `nbuffers=2048` and 11.81 GiB at 32768, where the pool costs 0.03 GiB.  The copy is
  ~0.02 us against a 26.67 us event budget at 37.5 kHz.  **The XTC description** is two
  containers: the Detector's raw array under `EventNamesIndex` and the Reducer's under
  `ReducerNamesIndex`, the raw one appended only on marked events.  That follows the HSD,
  whose `psalg/psalg/digitizer/Hsd.hh:43` says outright that *"if raw or fex data is
  missing, then the associated header is also missing"*, so a per-event extent is
  established practice.

  **One Names block with two arrays would give contiguous payloads and a single memcpy --
  considered and REJECTED, with the measurements, because this keeps being rediscovered.**
  `Shapes` is one Xtc holding an *array* of `Shape`, so a block declaring N arrays costs
  `sizeof(Dgram) + 3*sizeof(Xtc) + N*sizeof(Shape)` rather than N times the whole descriptor
  set: a second array adds only `sizeof(Shape)` = 20 B, and all the descriptors stay ahead of
  all the payloads.  Measured at `EpixUHRemu` sizes, one block puts raw at [100, 387172) and
  reduced at [387172, 1161316) -- contiguous, header 100 B, and the extent is 36 B *smaller*
  than the two-block form.  (An accident worth knowing: `MaxRank*sizeof(uint32_t)` = 20 =
  `sizeof(Shape)`, which is why the reserve formula above happens to be right for one array.)

  Rejected because the cost lands on the wrong events.  One block means the raw array is
  always declared, so every event must fill it -- `CreateData`'s destructor *aborts*, not
  warns, on an unfilled entry -- so ordinary events carry a zero-length raw array plus its
  `Shape`: **+20 B on every event**, ~57 GB/day at 33 kHz, to avoid two memcpys costing ~1 us
  once per 37500 events.  It would also couple a dlopened Reducer to the Detector's Names
  declaration at Configure, and change the on-disk schema psana reads.  Two blocks also match
  the HSD precedent above: a missing array means a missing header, rather than a present
  header faking absence with a zero shape.

  Nine things worth not relearning:

  - **`rawSize()` is capacity, not presence.**  It is read once per Configure to size every
    buffer, so it cannot consult `keepRaw()` -- there is no event yet -- and must not consult
    `passthru()` either, since prescaled raw arrives in BEAM.
  - **A prescale buffer's header reserve is not a reduce buffer's.**  `headerSize` in
    `Reducer.cu` is `sizeof(Dgram) + 3*sizeof(Xtc) + MaxRank*4` = **80 B**, which is one
    Dgram plus *one* ShapesData.  A prescaled datagram describes two arrays, and each extra
    one costs `3*sizeof(Xtc) + MaxRank*4` = **56 B**.  Deriving the pool's reserve from
    `reduceBufsReserved()` therefore made every prescale buffer 56 B too short, and run
    270 aborted in `Xtc::alloc` on the first prescaled event.  `createRawBuffers()` now
    takes the reserve from the caller.  The 56 B is confirmed twice over: it predicts
    run 269's CALIB extent of 387140 and run 270's abort extent of 387196 exactly.
  - **Descriptors are INTERLEAVED with payloads, not gathered ahead of them.**  The costly
    one, and it survived the size fix above: `CreateData` writes each array's Shapes and
    Data Xtcs immediately before that array's bytes, so a two-array datagram is
    `[Dgram][raw descr][raw][reduced descr][reduced]`.  Treating the reserve as one block
    ahead of both payloads put the raw bytes 56 B past where their own descriptors pointed;
    run 271 then recorded 150 events with correct extents, zero damage and a perfect
    `keepRaw` correlation, and **`xtcreader -d` still aborted** on the first prescaled event
    with `corrupt xtc with too small extent`.  Every size check passed because every size
    was right.  `MemPool` now publishes `prescaleRawOffset()` and `prescaleRedOffset()` as
    the single statement of the layout, the host header is copied in two pieces around the
    raw block, and a guard compares the described offset against the buffer offset --
    because nothing else did.
  - **The host's header is ALREADY laid out the way the device wants it, so the two memcpys
    are not a contortion.**  `set_array_shape()` advances the Xtc over the raw payload, so
    the pebble buffer holds `[Dgram][raw descr][raw-sized hole][reduced descr]` -- the two
    descriptor blocks are separated on the host exactly as they are on the device, at the
    same offsets.  Both copies therefore use *identical* source and destination offsets and
    simply skip the hole; it is two copies rather than one only because a single copy would
    overwrite the device's raw block with the pebble's garbage.  Copying `dgram + hdrSplit`
    instead of `dgram + hdrSplit + rawBytes` reads the hole, which is uninitialised: run 272
    recorded 68 B of zeros where the reduced descriptors belong, with the extent still
    correct at 1161340 and every size guard silent.
  - **What must fit in the pebble is the header's REACH, not its size.**  Those descriptors
    straddle a raw-payload-sized hole, so a prescaled event reaches
    `hdrSplit + rawBytes + dscr` = 387208 B into a 393216 B pebble -- 6008 B of slack, by
    luck rather than design.  The old guard compared the 136 B *sum* against the buffer and
    passed vacuously.  It now checks the reach, because at twice this raw size the pebble
    silently overflows; `pebbleBufSize` is the kwarg that fixes it.  `EpixUHR3x2` has the
    same 387208 B reach, not twice it -- see the corrected item below.
  - **"It didn't crash" is not "it worked".**  Two runs in a row looked healthy in the log
    and were wrong in the file.  Read the data back with `xtcreader -d`, which walks the
    Xtc tree and so finds what a size check cannot.  Note `-d`'s output is lost on abort
    unless you line-buffer it: `stdbuf -oL`.
  - **A host/device counter pair needs atomics, not just pinned memory.**  The claim kernel
    compares a monotonic device ticket against a release count the recorder advances.  Read
    plainly, that is not merely stale-prone but unbounded: a release count the device never
    sees advance trips the overflow test after `NPrescaleBuffers` prescaled events whatever the
    true occupancy, so the belt-and-braces abort becomes a guaranteed one about 32 s into a
    run at 1 Hz.  `cuda::std::atomic` in pinned memory with release/acquire, as
    `RingIndex_HtoD.hh` does it.  Pinned sysmem is visible to both sides, but nothing
    invalidates a cached load without the ordering.
  - **Container order is forced.**  `CreateData::set_array_shape()` grows the Xtc extent in
    call order via `_shapesdata.data().alloc()`, while the bytes sit in buffer order with raw
    first.  So the raw container must be appended *before* the Reducer's.  Reversing them
    would present as corrupt data, not as a layout error.
  - **The Reducer capacity check had to change.**  It read
    `reduceBufsRaw() ?: reduceBufsSize()`, conflating "has a raw block" with "is in
    pass-through".  Once raw is reserved in BEAM, that picks the raw size as the limit and a
    reducer writing its full payload aborts falsely.  It now tests `passthru()`.

  Still to do: the other detectors -- see the CALIB-mode detector gaps in `TODO.md`, which share the
  per-detector work.  Two things run 273 did **not** exercise, so neither is proven:

  - **The overflow abort never fired**, and cannot at these rates: the pebble pool bounds
    in-flight prescaled events far below 32.  So `_claimRawSlot`'s occupancy test and the
    atomics behind it are exercised only in the non-overflow direction.
    **Superseded 2026-10-05**: runs 59-65 ran at 100% keepRaw, 1850 consecutive prescaled
    events at 120 Hz, and it still never fired and no slot leaked.  The pebble bound holds at
    ~120x the design rate, so the non-overflow direction is now very well exercised.
  - ~~**`EpixUHR3x2` will not fit the default pebble.**~~  **Wrong, and it was my arithmetic.**
    `rawSize()` is `NPixels * sizeof(uint16_t)` = 193536 * 2 = **387072 B**, not 774144 --
    that figure is `NoOpReducer`'s **f32 output** size.  So the reach is
    `80 + 387072 + 56` = **387208 B** into 393216, the same 6008 B of slack as `EpixUHRemu`,
    and for the same reason: the emulator emulates this detector, so both have 193536 u16
    pixels.  Confirmed on gpu006 run 59 -- the guard stayed silent and `pebbleBufSize` was
    never set.  **No action needed; this was never a blocker.**

  Also open, and deliberately deferred: the slot claim would be cleaner on the host, but that
  needs the graph to stop being recorded once per Configure.  **Revisit with letting the GPU
  idle at low trigger rates**, which has to change the launch model anyway.

## keepRawRate is not programmed in Cu mode, so every event is marked

Found 2026-10-05 on drp-srcf-gpu006, runs 59-65.  Every L1Accept came back prescaled --
1850 of 1850 on run 59, 1614 of 1614 on run 60, 964 of 964 on run 65 -- against a configDB
`keepRawRate` of **1.0 Hz** for every readout group.

**The DRP is not at fault, and that was settled by a control rather than by reading the
code**: the timing DRP on cmp040 is the stock CPU `drp`, no GPU code in it, and its own file
shows `keepRaw=1` on **3228 of 3228** events at 240 Hz.  Both detectors see L0Raw asserted on
100% of events at their own trigger rates.  Worth keeping as the technique: when a bit looks
wrong, find a consumer of the same bit that shares none of your code.

**`keepRaw` was being decoded correctly**, which is the other thing a 100% reading could mean.
`(env>>22)&1` is `TransitionBase::keepRaw()` (`xtcdata/xtc/Dgram.hh:26`) and bit 22 of `env` is
b6 `L0Raw` of `L1Dgram::reserved()`.  Decoding the whole reserved byte across the timing run
gives 32 distinct values: `L0Tag` cycling 0-31 with ~101 events each, `L0Accept` set,
`L0Reject` clear.  The field is live and b6 is pinned high inside it -- so not a stuck decode.

### Two units, and only one path converts between them

- configDB `user.{Cu,SC}.groupN.keepRawRate` is a **rate in Hz**.
- the PV `DAQ:FEH:XPM:<master>:PART:<group>:L0RawUpdate` is a **divisor in 929 kHz timing
  frames** (Matt, 2026-10-05): the number of frames between raw updates, 1 meaning every frame.

So 1 Hz is a divisor of ~928571, not 1.0.  The conversion is `int(TPGSEC/keepRawRate)` at
`psdaq/psdaq/configdb/ts_config.py:82` -- **in the SC branch only**.  The Cu branch
(`:66-72`) programs `L0Select`, `L0Select_EventCode` and `DstSelect` and never puts
`L0RawUpdate` into `pvdict` at all, so with `user.LINAC == 0` the field is read into the
recorded config and never reaches the XPM.  The SC branch additionally `raise`s when the key
is missing; Cu has no equivalent.

That is the bug.  Whether the fix is to move the conversion above the branch or to duplicate
it is for Matt -- it is his file, and `L0RawUpdate`'s Cu semantics are his to confirm.

### Why intermediate values looked like no change at all

A divisor only thins when the update rate falls below the trigger rate:

| `L0RawUpdate` | update rate | effect at 120-240 Hz triggers |
|---|---|---|
| 1 | 928571 Hz | marks every event |
| 1000 | 929 Hz | **still marks every event** |
| ~3900 / ~7700 | 240 / 120 Hz | threshold where thinning first shows |
| 928571 | 1.0 Hz | ~1 marked event per second |

So 1000 and 1 are indistinguishable in the data, which is what made a correct hand-written PV
look like it had been ignored.  Compute the resulting rate before concluding a write failed.

### Finding the master XPM, which is where the PVs live

Run 65 failed to thin because the PVs were written on **XPM 5, which is not the master**.
Three ways to tell, cheapest first:

1. **The `.cnf.py`**: `control`'s `-x` flag names it -- `-x 4` in `gpu6.py`.
   `control.py:715` reads it into `xpm_master`, `:486` builds `pv_base + ':XPM:%d'`.
2. **The control log**, every Configure: `<I> master XPM is 4` (`control.py:1604`).
3. **`DAQ:FEH:XPM:<n>:PART:<group>:Master`** -- the hardware's own view, 1 on the master
   and 0 elsewhere.  **Check the timestamp**: XPMs 0 and 2 still read 1 from an earlier
   session, so a bare 1 is not sufficient.

`ts_connect.py:99-113` actively demotes the downstream XPMs for the master's groups at
Configure, so XPM 5's `PART:{2,4}:Master` went to 0 in the same instant XPM 4's stayed 1.
Writing a downstream XPM's `L0RawUpdate` therefore cannot do anything.

### Run 66 confirms the PV path works, so the bug is only ts_config.py

With **928571 on `DAQ:FEH:XPM:4:PART:4:L0RawUpdate`**, run 66 is the first fully correct
prescaled run on real hardware:

| check | result |
|---|---|
| `xtcreader` | rc=0, 1401 records to EndRun |
| L1Accepts | 1382 at 120.01 Hz |
| unmarked | **1370 at extent 774212** -- reduce only |
| marked | **12 at extent 1161340** -- raw + reduced |
| `keepRaw`/extent mismatches | **0 of 1382** |
| damage | `0x0` on all 1401 |
| marked rate | **0.992 Hz**, gaps of exactly 122 events / 1.008 s, no jitter |
| raw payload | 193536 u16, 99.9% nonzero, at the right offset |

The extent splitting into two values is what no earlier run could show, so **the unmarked
reduce-only path and the per-event mode switch are both exercised for the first time**.
Eleven identical gaps is a periodic divisor, not statistical thinning.

So the PV and everything downstream of it are sound, and the defect is only the missing
`L0RawUpdate` in the Cu branch.  **For Matt:** hoist the conversion above the Cu/SC
branch, or duplicate it into Cu?  His file, and the Cu semantics are his to confirm.

## psana could not open our files: EventNamesIndex was declared twice.  Fixed

**Found and fixed 2026-10-06.  It predated that day's work** -- the pre-FLOAT16 build
failed identically, so nothing about the type addition was involved.  `xtcreader` is
perfectly happy with these files; **psana aborted before the first event**:

    NamesIter.cc: Found duplicate namesId 0xa
    terminate called after throwing an instance of 'char const*'

`NamesIter::process()` throws on a repeated `NamesId`, where `xtcreader` just prints both.
So **every file the GPU DRP has ever recorded is unreadable by psana**, including run 66.
This is worth knowing before anyone promises end-to-end operation.

**Cause.**  `Gpu::EpixUHR3x2::configure()` calls `m_det->configure()` first -- the CPU
`Drp::EpixUHR3x2`, which at `EpixUHR3x2.cc:216` already declares `Names` at
`NamesId(nodeId, EventNamesIndex)` with `Alg("raw", 0, 1, 0)`.  The GPU detector then
declares **its own** `Names` at **the same** `namesId`, with `Alg("raw", 0, 0, 0)`.  Hence
`0xa` twice, which `xtcreader -d` shows as the same detName and alg at two versions:

    namesid: 0xa  Alg: raw, Version: 0x000100     <- the CPU base class
    namesid: 0xa  Alg: raw, Version: 0x000000     <- Gpu::EpixUHR3x2

The two descriptions genuinely differ: the base class adds `epixUHR3x2RawDef`, ours adds
`RawU16Def`.  So this is not a duplicate declaration of the same thing, it is **two
different schemas claiming one id**, and the reader that wins is whichever lands first.

**Only `EpixUHR3x2` is affected, and the reason is structural.**  `EpixUHRemu` and
`EpixUHRsim` also call `m_det->configure()`, so the shape looks identical -- but their
`m_det` is an `XpmDetector`, which declares no `Names` at all, while the 3x2's is a
`BEBDetector` that declares the panel's event `Names` from the config.  So the collision
needs a CPU base class that describes events, and only the 3x2 has one.  Runs 269 and 273
were recorded by `EpixUHRemu` and so should open; **unverified** -- those files expired
from gpu001's `/tmp`.

**The fix: adopt the base class's `NameIndex` instead of declaring a second `Names`.**
The two declarations were *identical* in everything offline can see -- same detName,
detType, detId, segment, and both a single `{"raw", UINT16, 2}` at index 0, since
`RawU16Def` and `Drp::EpixUHR3x2RawDef` agree.  Only the Alg version differed, 0.0.0
against the base's 0.1.0.  So the second block carried no information and
`Gpu::EpixUHR3x2::configure()` now does

    m_namesLookup[namesId] = m_det->namesLookup()[namesId];

`rawEvent()` needs the entry in *this* Detector's lookup for `CreateData`, not another
block in the Xtc.  `NameIndex`'s assignment operator deep-copies (`malloc` + `memcpy` of
the whole `Names` extent, `NameIndex.hh:40`), so the copy outlives `m_det` regardless.
`RawU16Def` is now unused here and is gone; `EpixUHRemu` keeps its own, which it needs.
Recorded Alg version becomes 0.1.0, matching what the CPU DRP writes.

**Only `EpixUHR3x2` was affected, and the reason is structural.**  `EpixUHRemu` and
`EpixUHRsim` also call `m_det->configure()`, so the shape looks identical -- but their
`m_det` is an `XpmDetector`, which declares no `Names` at all, while the 3x2's is a
`BEBDetector` that declares the panel's event `Names` from the config
(`BEBDetector::configure()` always calls `_configure()`, in both modes).  So the
collision needs a CPU base class that describes events, and only the 3x2 has one.

**Validated by run 74 on gpu006**, `drp_gpu` md5 `c6270e78584ea`.  `xtcreader -d` shows
**one** `namesid: 0xa` block where run 66 showed two, at the base class's
`Version: 0x000100`, and **psana opens the file and reads the arrays** -- the first time
any GPU DRP output has been readable by psana:

| check | run 74 |
|---|---|
| `xtcreader` | rc=0, 1315 L1Accepts |
| unmarked / marked | 1303 @ 774212, 12 @ 1161340 |
| `keepRaw`/extent mismatches | **0 of 1315** |
| damage | `0x0` on all 1334 records |
| psana `detnames` | `epixuhr3x2`, `epixuhr3x2hw` |
| psana events walked | **1315, all of them** |
| psana arrays | 12 (the marked events), `uint16`, 99.9% nonzero |
| marked spacing | events 6, 125, 244, ... 1314: gaps of **119** (one 118) |
| marked rate | **1.009 Hz** at 120 Hz triggers |

psana's `raw()` returns shape **`(1, 336, 576)`** = 193536 pixels, not the `6 32256` the
file declares: `raw_v01` reshapes and descrambles per `epixuhr3x2.py`, which is its job.
The leading values differ from `xtcreader`'s for the same reason -- same payload,
different presentation.  **"Problem reading dgram header." at the end is normal**: it is
how `dgram.cc:865` raises `StopIteration` at end of file, not an error -- and it says
nothing about whether the file has an EndRun.  Run 74 does; check the file, not that
message.

**Run 74 is also the first correctly prescaled run driven entirely by configDB**, with no
manual PV write: `keepRawRate` of 1.0 Hz for groups 2 and 4 programmed
`L0RawUpdate = 910000`, read back on both before RUNNING, giving 1.009 Hz measured against
the 1.0204 Hz nominal -- the TPGSEC 2% and nothing else.  Run 66 needed the PV written by
hand; this one did not.

## keepRawRate is programmed in Cu mode now.  Fixed, and the units bite

Matt was ambivalent about who should fix it (meeting, 2026-10-06), so we did.  The
conversion is **hoisted out of the SC branch** in `ts_config.py` and now runs for both
LINAC modes, reading Cu's flat `groupN_keepRawRate` or SC's nested `groupN.keepRawRate`.
A Cu config predating the entry **warns** rather than raising -- Cu never programmed this,
so such a config is no worse off than before, and `logging` had to be imported for that
warning to work at all (Matt's 2024 `46fcce78` used it without the import, which is
plausibly why someone later replaced the warning with a `raise`).

**The trap, found live on run 73: the config field is a RATE, the PV is a DIVISOR, and
they differ by ~10^6.**  `user.Cu.group{2,4}_keepRawRate` had been set to **928571.0** --
the divisor written by hand for run 66 -- so `int(910000/928571.0)` truncated to **0** and
the XPM was programmed to never insert raw data.  Every structural check passed; the PV
simply read 0.

So `_rawUpdateDivisor()` now **refuses** a rate it cannot represent instead of writing 0:

- `rawRate <= 0`, or a rate so high the divisor truncates below 1, names the units
  mistake explicitly and gives the maximum, `TPGSEC` Hz;
- a rate so low the divisor exceeds **20 bits** (`l0RawUpdate`'s width in
  `pyxpm/xpm/_XpmApp.py`) reports the minimum, 0.8678 Hz.

Exercised over 1.0 / 0.992 / 10.0 / 910000 Hz valid and 928571 / 0 / -1 / 0.8 / 2e6
rejected.  `ts_config_store.py`'s help string now says "(Hz), NOT the L0RawUpdate
divisor".

**Note `TPGSEC` is 910000 but the fiducial rate is 1.3 GHz/1400 = 928571.4 Hz**, so every
`keepRawRate` lands ~2% high -- 1.0 Hz asks for 1.0204 Hz.  `tsdef.py:81` already labels a
910000 divisor as "1.02Hz", so this is deliberate and long-standing in Matt's design.
**Left alone on Ric's instruction, 2026-10-06**: the 91 factors into a great deal else.

## The fp16 plan: one format at a time, so nothing is doubled

Chris, 2026-10-05, correcting an earlier reading here that several notes had been written
against.  What the updated firmware will do:

- new **registers select u16 or fp16 mode** for the panel;
- the **"raw"** block -- "raw" being the label for the *prescaled* data -- is then u16
  **or** fp16 according to that register, **not both**;
- the **non-raw** block becomes **"fex"**: fp16 calibrated, an ROI, or another extraction.

**The wrong premise was "u16 *and* fp16 together on an asserted keepRaw, roughly doubling
the DMA".**  Nothing is doubled.  Both formats are 2 B/pixel, so the raw block is 387072 B
either way and `dmaBufSize` is unchanged by the mode register; a marked event carries
`32 + raw + fex`, where fex is at most a full fp16 frame, against the 774176 B that "u16
alongside fp16" would have needed.  The "separate sub-frames vs interleaved `{u16, __half}`
elements" question is **moot** -- one format at a time means no pairing to lay out, so
`cnt = sub.size / sizeof(*src)` keeps meaning a pixel count and `_copyRaw`'s strided copy
per ASIC stays a copy rather than becoming a gather.

So `rawSize()` stays `NPixels * 2` and `rawShape()` stays as it is, both already right for
either mode.  `RawU16Def` is the part that does not generalise, which is what prompted the
`FLOAT16` proposal below.

`EpixUHR3x2::EpixUHR3x2()` aborts on `raw=fp16` naming the reason, and `m_u16` defaults to
true.  `EpixUHR3x2Beam` is **kept and still instantiated** by `recordEvent()`'s unreachable
`else`, so it keeps compiling as the surrounding code moves and the revisit starts from
working code rather than from rot.

## Proposal for Chris: add FLOAT16 to the Xtc type system

Needs Chris's buy-in before anything is written -- `xtcdata` is shared with psana and
every CPU DRP, so it is not ours to change unilaterally.  Recorded here as a worked
proposal rather than a decision.

**The question that prompted it was whether x86 gcc even has fp16, and if not, whether to
leave the block opaque and reinterpret it in psana.**  It does, so the opaque route is not
forced.  Checked with the conda-forge gcc 13.3.0 this tree builds with:

| check | result |
|---|---|
| `_Float16` accepted under `-std=c++17` and `c++20` | yes |
| `sizeof` / `alignof` | 2 / 2 |
| arithmetic | `1.5 * 2.25 = 3.375` |
| bit layout vs CUDA `__half` | **identical**: 1.5 -> `0x3e00`, pi -> `0x4248` |
| `np.float16` in the daq env | itemsize 2, `NPY_HALF` = 23, numpy 1.26.4 |

Two caveats to state when proposing it.  Without `-mavx512fp16` gcc emits libgcc calls
(`__extendhfsf2`, `__truncsfhf2`) for fp16 *arithmetic*; with it, native `vmulsh`.  That
does not matter here, because the DRP only stores and copies fp16 and never computes on
it -- psana computes, and numpy handles that.  And **`__fp16` is not available on x86
gcc**, only `_Float16`; the storage-only spelling is an ARM thing.

**The change is three mechanical sites:**

1. `ShapesData.hh` -- **append** `FLOAT16` to `Name::DataType`.  Appending is not a style
   preference: the enum value is written into the file, so inserting it would reinterpret
   every existing dataset.
2. `ShapesData.cc` -- `sizeof(_Float16)` into `element_sizes[]` and `"FLOAT16"` into
   `str_type()`.  The comment there already requires the two to track the header.
3. `psana/src/dgram.cc` -- one more `case` yielding `NPY_HALF`.

**Why this is better than declaring the block `UINT16` and reinterpreting offline.**
`dgram.cc`'s array switch ends in `default: throw std::runtime_error("dgram.cc:
Unsupported array type")`.  So "opaque u16" does not give psana something to reinterpret
-- it gives every psana user a `uint16` array of nonsense, silently, with nothing in the
file recording that the bytes are floats.  A new enum value instead fails **loudly** on an
old psana, which is the right behaviour for a file it genuinely cannot read.  It also
beats widening to `FLOAT`, which doubles the bytes on disk for data the firmware already
produced as fp16.

## FileWriter spun on an unrecoverable write error.  Fixed at 7b045c66

Run 64, 2026-10-05: `/tmp` on gpu006 filled (20G, 100%), and every `cuFileWrite` then
returned `EIO`.  The log carries dozens of identical lines, same buffer and same count:

    <E> Write error: buffer 0x7fae2e000000, count 32517856: Input/output error
    <E> File writing failed: rc -1

`FileWriter::writeEvent()` logged and **returned without clearing `m_count` or restoring
`m_writing`**, so the same buffer was retried on the next event, for ever.  The CPU
`drp/FileWriter.cc` uses `logging::critical` at the equivalent point, which aborts; the GPU
copy had downgraded it to `error` and so could not make progress.

Now aborts once, as `FileWriterAsync` in the same file already did.  Two things went with it:
`_flush()` ignored `_write()`'s return and then cleared `m_count`, silently truncating the
file's tail, and `%m` was unreliable -- `cufile.h` says data path errors come back as standard
error codes, so `rc` is `-errno` and cuFile need not set `errno` itself.  `strerror(-rc)` now
names the condition, so a full disk reads `No space left on device (-28)`.

**The abort branch itself is untested**: reaching it means filling a filesystem, which was
not worth doing deliberately on a shared node.

Sizing, for whoever runs this next: at 100% keepRaw an `EpixUHR3x2` event costs ~1.16 MB
(387072 B raw + 774144 B reduced + descriptors), so 20 GB of `/tmp` is about 17000 events --
under three minutes at 120 Hz.  Six runs exhausted it.

## CAP_SYS_ADMIN is dropped once the registers are mapped. Done

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

---

## Appendix: retrieving calibration constants from calibdb

Reproduced verbatim from Gabriel's explanation, because the database schema is not
documented anywhere else.  Relevant to `TODO.md`'s "Fetch calibration constants"
item, option 3.  Lightly formatted only: the prose and code are unaltered, typos
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

## The green context split, measured: 6/40/18 on an A5000 and 8/40/84 on an H200

Both on 2026-09-15, and they are the constraints any derived scheme has to respect.
`m_green_ctx[0]` goes to the Reader *and* TrgInpGen, `[1]` to the Reducer, `[2]` -- the
remainder -- to `TebReceiver::_recorder`.

    drp-srcf-gpu001, RTX A5000        drp-srcf-gpu008, H200 NVL
      Initial SM resources: 64          Number of multiprocessors: 132
      Min. SM partition size: 2         Min. SM partition size: 8
      context 0:  6 SMs                 context 0:  8 SMs
      context 1: 40 SMs                 context 1: 40 SMs
      context 2: 18 SMs                 context 2: 84 SMs

The A5000 is 6 + 40 + 18 = 64 exactly, so a derived scheme must land on multiples of 2.
**The H200 is 8/40/84, not the 6/40/86 that arithmetic on the hard-coded values predicts**,
because its device-level `minSmPartitionSize` is 8 where the A5000's is 2 and the clamp at
`PGPDetector.cc:617` raised group 0's requested 6 to 8.  The one guard that exists did the
useful thing.  Note the *per-context* `minSmPartitionSize` reads 2 after the split, so it
is the device-level value that constrains the request.

That softens the `_event` complaint: it asks for 8 SMs' worth at `tpSM` 2048 and context 0
has exactly 8; on the A5000 it asked 6 and had 6.  **It fits on both by coincidence of two
independently chosen tables, and nothing would warn if a future device broke it.**  The
real imbalance is `_reduce` using 10 of context 1's 40 on an H200, and the recorder holding
84 of 132 SMs -- 64% of the GPU.

**And why that imbalance is invisible: the DRPs are DMA-bound, not GPU-bound.**
Re-confirmed on gpu008 with the merged pre-release driver, 33 kHz on all five DRPs, no
drops or overflows.  At `dmaBufSize=387104`:

    387104 B x 33035 Hz = 12.788 GB/s per DRP,  63.9 GB/s across five

against 15.75 GB/s raw for one PCIe 4.0 x8 uplink -- 81% of raw, essentially payload line
rate once TLP and DLLP overhead is out.  Each card is pinned at its own uplink, so the GPU
has spare capacity no matter how the SMs are divided.  Rebalancing therefore cannot improve
a rate the uplink sets, which is why the open item is gated on x16 gen5.

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

## Pass-through records correct data, 2026-09-28; the stall is in the FEB

**Stage 1 is validated.**  414 L1Accepts recorded on drp-srcf-gpu006 against the hardware
emulator, and the file is what offline needs:

| check | result |
|---|---|
| declared type | `Type 1 Rank 2` = `UINT16`, rank 2 -- matches `Drp::EpixUHR3x2`'s |
| `payloadSize` | 387128 = 387072 raw + 56 descriptors |
| `extent - payloadSize` | **12** -- the header abuts the payload, no gap |
| damage | `0x0` on every event |
| pixels | 193536 u16, **99.7% non-zero**, all six ASICs ~16050 of 16128 |
| shim | `DRP_redStarts == DRP_redRcvs`, so every event completed |

Read it with `xtcreader -f <file> -d`, or decode the last 387072 bytes of an L1A payload as u16.

**Bit layout, from Gabriel 2026-09-28: gain is bit 0, ADC is bits 1-11, bits 12-15 are zero.**
An earlier reading here had gain in bit 11, which was wrong.  Pass-through copies verbatim so it
does not care, but stage 3 will.

### Two bugs fixed to get there

- **The shim hung the DRPs.**  With `hasGraph()` false and `HOST_LAUNCHED_REDUCERS` undefined,
  `Reducer` skips `configure()` and `setup()`, and `startup()` launches nothing because the
  worker-thread branch is `#else`-compiled out.  Nothing posts a completion and the recorder
  blocks on `receive()` for ever.  Fixed by following the graph path: `hasGraph()` true, a
  `<<<1,1>>>` kernel that moves no data, sets the size and advances the state.
- **The size slot collided with the raw block.**  `_reducerLoop` read `((size_t*)data)[-1]`,
  which lies *inside* the raw block, so the size and the last four u16 pixels overwrote each
  other.  Both now use `((size_t*)(data - rawSize))[-1]`; real reducers are unchanged since
  `rawSize` is 0 for them.

### The remaining blocker is the FEB's backpressure, not the DRP

After a few hundred events the DAQ goes to 100% deadtime and Disable will not complete.  **It is
not the GPU DRP**, and the evidence is conclusive:

| run | consumer | events |
|---|---|---|
| GPU, `dmaBufCount=8` | `PassthruShim` | 414 |
| GPU, `dmaBufCount=32` | `PassthruShim` | 651 |
| **CPU DRP** | stock `drp`, none of this code | **365** |

So it is not a fixed-length acquisition (the count varies), not GPU-specific (the CPU path does
it too), and not a DRP buffer-return failure -- `/proc/datadev_a1` showed `Buffers In User: 0`
with `Buffers In Hw: 1020`, i.e. software held nothing and the driver was not starved, and
`RX Frame Count` equalled the events processed, so the DRP consumed everything it was given.

What the FEB shows while stalled, from both ePix devGuis:

    L1AcceptCount = 794        triggers the FEB accepted
    RX Frame Count = 653       frames that reached the datadev
    XpmPause = True, FifoPause = True

**The FEB accepted 141 more triggers than it could push out, asserted backpressure, and never
released it.**  That the count varies run to run fits a FIFO filling on timing rather than a
counted burst.  Ric suspects a high-water mark whose release condition never becomes true, and
that `L0Delay` -- currently 0 for all partitions and readout groups -- is the adjustment, though
whether 0 is the conservative end wants confirming with Matt.

Useful that it reproduces with the stock CPU DRP: the GPU work need not enter that discussion.

## Pass-through reached Paused on gpu006, 2026-09-25, and what it took

First run of the pass-through work (`features/gpu-raw-calib`) against the hardware emulator.
It reached **Paused**, which validates the whole transition path with the raw region in it:

    EpixUHR3x2: pass-through mode -- recording raw u16, uncalibrated and unreduced
    PassthruShim: recording 387072 B of raw data per event, unreduced
    Reduce buffers: ... size 2048 * (80 + 387072 + 2097072) B
    PGPReader / Collector / TebRcvr / Recorder  all saw Configure

**Still unproven: no L1Accept has passed through the pass-through kernel.**  So the per-element
copy, the recorder's contiguous-region arithmetic and the file layout are all untested.  Three
things gate that: the trigger setup below, an output path (WEKA is not mounted on gpu006 --
`/cds/data/drpsrcf` is a bare empty mountpoint there, Gabriel has an IT ticket for the IB link;
`-o /home/claus/data` is the workaround), and then the XTC comparison against the CPU DRP.

**Four of my bugs, none of which compiling would have caught:**

1. `raw` was missing from the kwarg allowlist in `PGPDetectorApp.cc`, so the DRP died at startup
   with `Unrecognized kwarg 'raw=1'`.  The allowlist working as intended.
2. `maxTrSize` was 256 kiB, too small for an ePixUHR3x2 Configure.
3. Raising it to 512 kiB failed identically, because the extent **grows to fill whatever it is
   given**:

       Xtc.hh:111: Insufficient space for 524 bytes (... extent 262012)    # 256 kiB
       Xtc.hh:111: Insufficient space for 760 bytes (... extent 524036)    # 512 kiB

   Both read as near misses -- "overflowed by 392 bytes" -- and neither was: `epixuhr3x2_0`'s
   config is **902026 bytes of JSON**, so each number is where it stopped, not what it needed.
   `BEBDetector::_addJson` builds the JSON into a temporary buffer *also* sized `maxTrSize`,
   which is why the shortfall is never fixed.
4. Settled at 2 MiB.

The lesson for the plan's verification section: it checked that the code compiled and installed,
so a runtime-only failure like a kwarg allowlist was invisible until the thing actually ran.

**The ASIC-ordering question is answered, and the stale table is gone.**  Gabriel confirms the
CPU DRP's output order is correct, so tdest 3+k is ASIC k and no remapping is needed.  The
`AsicForDataSubFrame = {1,3,5,0,2,4}` table and its comment -- *"Anything writing XTC must
descramble with this"* -- were **wrong**, and were also never referenced by any code: I wrote
them in `e210c54e` when the GPU EpixUHR3x2 was first added, on an assumption that did not hold.
Removed 2026-09-25.

Keeping identity in the pass-through kernel was what made the CPU comparison meaningful; had the
code been "corrected" to match that comment, the comparison would have failed for the wrong
reason and the table would have looked vindicated.  A confident comment with no code depending
on it is worth distrusting.

**The trigger setup is what blocks L1Accepts.**  `xpmpva` shows the sequence is not currently
loaded.  Gabriel's notes in the appendix below specify it, and happily for the same XPM
`gpu6.py` already uses (`groupca DAQ:FEH 4`):

- Timing's readout group: event code **278**
- The ePixUHR readout group: event code **277**
- The run trigger: **276**, already in configdb, so possibly nothing to change there
- He programmed `DAQ:FEH:XPM:4` on **Seq Engine 5**

So what is missing is the sequencer programming, not the XPM choice.  Without it the DAQ goes
into deadtime immediately.  There is an `xpm-seq` skill for LCLS-II sequence programming if
reprogramming from those three codes is preferable to waiting.

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

## Unconfigure hung in cuFile, because un-pinning needs an idle device

Found on 2026-09-29 on drp-srcf-gpu001, running `epixuhremu` with `NoOpReducer`.  With
recording **enabled**, the GPU DRP never acknowledged Unconfigure: the control level
complained, and while `TebRcvr saw Unconfigure` appeared in the log, `Recorder saw
Unconfigure` never did.  The process stayed alive with the recorder thread apparently busy.
With recording off, Allocated/Running could be cycled repeatedly at 1, 10 and 100 Hz with no
trouble.

`gdb -p <pid> -batch -ex 'thread apply all bt'` is what settled it, and it named the frame
outright:

    #9  cuMemHostUnregister ()                   from libcuda.so.1
    #14 cuFileBufDeregister ()                   from libcufile.so.0
    #15 Drp::Gpu::FileWriter::close              FileWriter.cc
    #16 Drp::TebReceiverBase::closeFiles         DrpBase.cc:975
    #17 Drp::Gpu::TebReceiver::_recorder         PGPDetector.cc

So the recorder was not stuck on Unconfigure at all: it was still inside **EndRun**, whose
`closeFiles()` never returned, and Unconfigure sat unprocessed behind it in the queue.  That
is also why recording mattered -- `closeFiles()` does nothing unless `m_writing` is true, and
`FileWriter::close()` only reaches the deregister when `m_fd > 0`.

**The cause: `cuFileBufRegister` was being undone mid-cycle, while the Reader graphs were
still running.**  Those graphs relaunch themselves (`Reader.cu`,
`cudaStreamGraphTailLaunch`) until `terminate` is set, so between `Reader::startup()` and
`PGPDrp::unconfigure()` the device is never idle.  `cuFileBufDeregister` reaches
`cuMemHostUnregister()`, which has to quiesce the device's mappings, and it cannot while work
keeps re-queueing itself.

**The fix** moves the buffer registration to the `FileWriter`'s ctor and dtor, which is what
Ric's first implementation did before an unrelated problem pushed it into `open()`.  Both ends
are quiet there: the ctor runs during Configure before the graphs launch, the dtor at the next
Configure after `m_terminate` is set.  `TebReceiver::setup()` also needed an explicit
`m_fileWriter.reset()` before its `make_unique`, or the new writer's registration would
briefly coexist with the old one's.

Validated by Ric the same evening: two cycles with recording on, reaching Allocated from
Running cleanly.

Three things worth keeping from how this went wrong:

- **It was latent, not a regression.**  The register/deregister pair dates to `705a8264`
  (2025-07-01); the self-relaunching graph loop was written *later*, and the FileWriter was
  never retested against it.  Nothing on the `features/gpu-raw-calib` branch touched
  `FileWriter.cc`, so this belongs on `features/gpu` too.
- **One record-enabled cycle triggers it.**  Earlier runs looked like a race that needed
  three cycles, but the first two had recording off and so never opened a file.  A
  deterministic one-shot failure, not a race.
- **Per-stream synchronization is not the answer.**  Two attempts went that way first -- the
  reasoning being that `writeEvent()` queues `cudaMemcpyAsync` and `close()` never waited --
  and the hang was unchanged.  The constraint is device-wide.  A `cudaStreamSynchronize` was
  added to `FileWriter::_write()` anyway and kept, because reading the buffer while copies are
  in flight was genuinely unsound: `cuFileWrite` could see bytes that had not landed, so
  mid-run flushes could write stale data.  It is a real fix for a different bug.

The line number in the backtrace is a reliable version check when retesting this, since the
deregister moved: `FileWriter.cc:230` is the original, `:224` the reordered-`close()`
intermediate, and neither once the call lives in the dtor.

### `cufile.json` is not being read from the run directory

Noticed while investigating the above.  `~/lclsii/daq/runs/eb/data/gpu001/cufile.json` has no
effect: cuFile looks at `$CUFILE_ENV_PATH_JSON`, which is unset, then `/etc/cufile.json`,
which on gpu001 symlinks through `/etc/alternatives` to
`/usr/local/cuda-13.3/gds/cufile.json`.  Both files happen to set `allow_compat_mode: true`,
so behaviour today is the same either way and compat mode is in force as expected -- but any
*other* setting in the run-directory copy has never taken effect.  Point
`CUFILE_ENV_PATH_JSON` at it if it is meant to be authoritative.

## `nPidJumps` cannot work: a deadtime veto and an unanswered trigger are indistinguishable

The idea was to spot a trigger the detector was given and never answered, by inferring the
trigger period from the spacing of arriving L1Accepts.  **Deadtime makes that inference
impossible, not merely inaccurate** -- and the inference is the whole mechanism, so no
amount of fixing the convergence rule rescues it.  Never committed: the check and its
metric exist only in `wip-2026-10-12-full-diagnostics.patch`, whose base is `c58f0185`.

**Measured on gpu008, 2026-10-12, which settles it.**  The XPM reports
`L0InpRate{partition="7"}` = **71,428 Hz = 13 ticks** and `DeadFrac` = **0.70**.  So the
true trigger cadence is 13 ticks, deadtime vetoes ~70% of them, and the L1Accepts that
arrive are spaced 26, 39, 52... ticks apart -- whole multiples, legitimately.  The
arithmetic closes: 71428 x (1 - 0.70) = 21.1 kHz against ~19.25 kHz measured.

    latched m_dPid   13     correct, it is the real cadence
    observed gap     26     = 2 x 13, so reported as "1 trigger unanswered"
    reality                 the vetoed trigger was never sent to the detector

**So the five messages were false in the one run where the period latched correctly.**
Earlier runs latched 64 and 38 against a true 68 and were wrong in the other direction,
which is what the "minimum, not mode" diagnosis was about -- but that was a second bug on
top of this one, and fixing it would only convert silent nonsense into confident false
alarms.  Both a veto and an unanswered trigger leave `evtCounter` contiguous and a gap
that is a whole multiple of the cadence.

**What would work, if the question is ever worth answering**: compare the XPM's
`NumL0Inp`/`NumL0Acc` against the DRP's own event count, since the XPM knows what it
vetoed and the DRP does not.  Those counters are already in Prometheus but the DRP does
not read them -- it only subscribes to an XPM deadtime PV (`DrpBase.cc:1637-1646`).  That
is a real comparison rather than an inference, and it belongs in a dashboard query, not in
the DRP's hot path.

The skipped-trigger hypothesis it was built to test remains **untested**, and is no longer
needed: the lost Result writes it was meant to explain were explained without it.
