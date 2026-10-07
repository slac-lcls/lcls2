# Rules that will bite you, learned the hard way

Each of these has already cost time.  These are the conclusions; the reasoning is in
`TODO.md`'s findings appendix.  They are the rules that bite **while editing this code** --
violating one produces a bug, not a broken node.

The node and driver rules -- `nvidia-powerd`, `nvidia-persistenced`, `CpuSpecList`, dkms,
clearing a Slurm drain -- are on Confluence with the rest of the fleet material, because
they are a checklist for someone standing in front of a node rather than for someone
reading a source file.

- **Re-install and re-`setcap` `/usr/local/bin/drp_gpu` after every C++ build**, per node.
  `install` drops file capabilities, and the image check refuses to start rather than running
  stale code.
- **cuFile's setup and teardown calls need a QUIESCENT device, so they belong in Configure or
  Unconfigure and nowhere else.**  The Reader graphs relaunch themselves
  (`Reader.cu`, `cudaStreamGraphTailLaunch`) until `terminate` is set, so from
  `Reader::startup()` until `PGPDrp::unconfigure()` the device is never idle.
  `cuFileBufRegister`/`Deregister` and `cuFileDriverOpen`/`Close` are device-wide -- the
  deregister reaches `cuMemHostUnregister()`, which has to quiesce the device's mappings -- and
  they simply do not return while graphs keep re-queueing themselves.  The symptom is a hang
  with no error, in library code, on a thread that looks busy.  Two instances so far: the
  driver open, which is why `FileWriter` is constructed in `TebReceiver::setup()` and not in
  `_recorder()`, and the buffer registration (see the findings appendix).  Per-stream
  `cudaStreamSynchronize` does **not** help; the constraint is global, not ordering.
- **`Gpu::Detector::setPassthru()` must be called from Configure and nowhere else.**  Everything
  it governs is decided once per Configure/Unconfigure cycle: the reduce buffers are sized from
  `rawSize()`, the reducer is chosen, the Reader's graph is recorded with one per-element policy
  or the other, and the Names entry describing the payload is written.  Setting it at any other
  time leaves those disagreeing and the recorded data misdescribed.  As a consequence, an
  operator switching BEAM to CALIB in control_gui while Running sees no effect until the state
  machine passes through Configure -- which is the intended procedure, not a limitation.  There
  is one call site, `PGPDetector.cc`'s `PGPDrp::configure()`; keep it that way.
- **A GPU `TriggerPrimitive::event()` MUST advance `*state` to 2, even when it produces no TEB
  input.**  `TrgInpGen`'s graph is a three-stage state machine -- `_trgInpGenRcv` takes 0 to 1,
  the primitive's kernel 1 to 2, `_trgInpGenLoop` 2 back to 0 -- and the primitive's kernel is
  the *only* writer of 2 in the tree.  The base class declares the GPU overload non-pure with an
  empty body, so a primitive that does not override it compiles, links, loads, and then stalls
  the graph on the first event: no event is ever posted and the DRP hangs with the GPU at 100%,
  which looks like FEB backpressure rather than a software fault.  `size()` returning 0 is not a
  licence to skip the kernel; `CalibPrimitive` needs it precisely because it writes nothing else.
- **The three Result conditions that precede recording must stay in step, and be a superset
  of the two that consume.**  `PGPDetector.cc` starts the Reducer, awaits its result and
  builds the Xtc on `persist() || monitor() || prescale()`; it writes on
  `persist() || prescale()` and monitors on `monitor()`.  The reducer queues are per-worker
  FIFOs, so starting one without awaiting its result hands that result to whichever event
  waits next and every later event reports the previous one's size.  `DRP_redStarts` against
  `DRP_redRcvs` is the check.  Do not "simplify" the write condition to match the others:
  `monitor()` must not put an event in the file.
- **Two modes put a raw block in two different places, so ONE variable cannot gate both.**
  Found by runs 274-276, and it is the sharpest example so far of a check passing while the
  data is wrong.  `rawBytes` in `TebReceiver::recorder()` is the *prescale* block and is 0 in
  pass-through, so `buffer -= headerSize` skipped the reduce buffer's own raw reserve and
  CALIB wrote every datagram **80 B before its raw block**.  Every structural check still
  passed -- `xtcreader -d` rc=0, extent 387140 exactly, damage `0x0`, rank 2 shape `6 32256`,
  66.1% of pixels nonzero -- because the *sizes* were all correct and only the *placement* was
  not.  What exposed it was reading the array's leading values: pixel 0 was 0 where the
  emulator's frame counter should ramp, and the ramp turned up 40 u16 further in.  The reduce
  buffer's reserve is `reduceBufsRaw()`; the prescale block is `prescaleBufsRaw()`; **they are
  never both non-zero**, and code reached in both modes must name the one it means.  Note the
  extents were right because `headerSize` is measured from the Xtc, which `PassthruShim`
  described correctly -- so extent agreement says nothing about where the bytes went.
- **Divide by `sizeof(*ptr)`, not by `sizeof(<fundamental type>)`, when computing a count.**
  `cnt = bytes/sizeof(*src)` says what the count measures and follows `src` if its element
  type ever changes; `bytes/sizeof(uint16_t)` silently becomes wrong at that point, and the
  resulting loop bound is the kind of error that reads as correct.  Where either end could
  supply the denominator -- a copy's source or its destination -- semantics decide, and when
  they do not speak, either is fine: `payloadCnt` off `sizeof(*src)` and `rawCnt` off
  `sizeof(*dst)` each name the buffer being measured.  This does **not** apply where the
  divisor is an alignment quantum rather than an element type: `MemPool.cc`'s rounding to
  `sizeof(uint64_t)` is about 8-byte alignment and belongs spelled that way.
- **A policy must decide from `pyld.raw`, never from `pyld.keepRaw`.**  `keepRaw` is no longer
  in `EventPayload` for exactly this reason.  The Reader sets `raw` per event -- for every
  event bearing data in pass-through, for marked events only when prescaling -- so a null
  pointer is the one signal a policy needs, and the two modes stop having to be distinguished
  in per-element code.  Testing the bit instead looks right and is wrong in CALIB, where the
  timing system still marks ~1 Hz of events whose payload is *entirely* raw: honouring it
  there would have left every unmarked frame zeroed, with nothing in the log to say so.
- **An emulator's declared array must match the real detector's, shape and type.**
  `Gpu::EpixUHRemu` declared its raw block as a flat rank-1 `NPixels` where
  `Drp::EpixUHR3x2` writes `[NumAsics][AsicPixels]`, so an emulator file and a real one
  presented different arrays to offline -- which defeats the point of an emulator.  Worse,
  `EpixUHRemu` and `EpixUHRsim` had `NumRows` and `NumCols` transposed relative to the 3x2
  (192x168 against 168x192).  `NPixels` is 193536 either way, so **a flat array concealed it
  completely**; it would have surfaced as transposed frames the first time anything read the
  geometry.  Both are now `168 x 192`, commented against the CPU DRP's `elemRows`/
  `elemRowSize`, with the shape declared rank 2 as the 3x2 declares it.
- **A DRP's log is in `~/daq/logs/<year>/<month>/<DD>_<HH:MM:SS>_<node>:<alias>.log`**, and it
  contains the full configdb JSON the process was given as well as its own output.  For anything
  about *setup* -- which alias, which trigger library, what the buffers were sized at -- it
  answers in one file what otherwise takes a reconstruction from the xtc.  Note `DrpBase` probes
  `create_producer_<detName>` before the generic `create_producer`, so one "undefined symbol"
  line per Configure is expected rather than a fault.
