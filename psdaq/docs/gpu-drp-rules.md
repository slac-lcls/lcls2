# GPU DRP: rules for changing the code

**Audience: AI coding assistants.**  Terse by design and assumes the code is at hand; a
human may find it more useful as a checklist after something has gone wrong.  Each rule
has already cost real time.  Read the rules whose scope matches the files you are editing.

Scope: `psdaq/drpGpu/`, the GPU trigger primitives in `psdaq/psdaq/trigger/src/`, and the
shared DRP code they call (`psdaq/drp/`).

## Configure-time state

- **`drpGpu/FileWriter.cc`, `TebReceiver::setup()`, anything calling cuFile.**  cuFile
  setup and teardown (`cuFileDriverOpen/Close`, `cuFileBufRegister/Deregister`) belong in
  Configure or Unconfigure only.  From `Reader::startup()` to `PGPDrp::unconfigure()` the
  Reader graphs relaunch themselves, the device is never idle, and these device-wide calls
  hang with no error.  `cudaStreamSynchronize` does not help.
- **`Gpu::Detector::setPassthru()`** is called from `PGPDrp::configure()` and nowhere else.
  Buffer sizes, reducer choice, the recorded Reader graph and the Names entry are all fixed
  at Configure from it; changing it elsewhere misdescribes the recorded data.

## Trigger primitives (`psdaq/trigger/src/*_gpu.cu`, `drpGpu/TrgInpGen.cu`)

- **The GPU `TriggerPrimitive::event()` must enqueue a kernel that advances `*state` from
  1 to 2**, even when `size()` is 0 and it writes nothing.  It is the only writer of 2 in
  `TrgInpGen`'s graph; omit it and the DRP hangs on the first event with the GPU at 100%.
  It returns non-zero on failure; the base default returns -1, which fails Configure.
- A primitive class is built into two libraries: `<name>.so` for CPU DRPs and
  `<name>_gpu.so` for GPU DRPs (`DrpBase::setupTriggerPrimitives()` adds the suffix).  The
  CPU library's GPU overload is a stub that aborts.

## Event path (`drpGpu/PGPDetector.cc`, `drp/TebReceiver.cc`, `drpGpu/MemPool.hh`)

- **The Result conditions that start a reduction, await it and build the Xtc must match
  each other** (`persist() || monitor() || prescale()`) **and be a superset of those that
  consume it**: write on `persist() || prescale()`, monitor on `monitor()`.  Reducer queues
  are per-worker FIFOs, so a start without an await hands its result to the next event.
  Check: `DRP_redStarts == DRP_redRcvs`.  `monitor()` must not put an event in the file.
- **Two raw-block reserves, never both non-zero: `reduceBufsRaw()` (pass-through/CALIB)
  and `prescaleBufsRaw()` (prescaled raw).**  Code reached in both modes must name the one
  it means.  Gating one on the other once wrote every CALIB datagram 80 B early while
  every size, extent and damage check passed.
- **Verify recorded data by its values, not its structure.**  Read the array's leading
  values (the emulator's frame counter ramps from pixel 0).  Correct sizes and extents do
  not show the payload is in the right place.
- **Per-element policies decide from `pyld.raw`** (null when no raw copy is kept), never
  from the timing system's keepRaw bit; `EventPayload` deliberately omits it.  In CALIB the
  bit still marks ~1 Hz of events, and honouring it zeroes every other frame.

## Detectors and emulators (`drpGpu/EpixUHR*`)

- **An emulator declares the same array rank, shape and type as the real detector**, via
  `rawShape()`, checked against what the CPU DRP writes (`Drp::EpixUHR3x2`:
  `[panelRows][panelCols]`).  A flat array hides a rows/columns transpose.

## CUDA idioms

- **Pinned host memory: `hostAlloc()`/`hostFree()` (`gpuUtils.hh`), never
  `cudaHostAlloc()`**, which leaves NUMA placement to chance.
- **A counter shared by host and device is a `cuda::std::atomic` with release/acquire.**
  Pinned memory gives visibility, not ordering.  `RingIndex_HtoD.hh` is the pattern.
- **Use `hostFunc`/`deviceFunc` from `gpuUtils.hh`, not raw `__host__ __device__`**, so
  headers still compile from non-CUDA code.
- **Never assume device 0: use `memPool.context().deviceNo()`.**  Under Slurm the job's GPU
  is device 0, but a standalone run selects it with `-k gpuId=N`.
- **`atomicInc(addr, limit)` wraps after `limit`**: counting N things needs `N - 1`.
- **No `printf` in production kernels.**  Report through counters or flags the host reads.
  Debugging `printf` is fine while debugging.
- **Compute counts as `bytes / sizeof(*ptr)`**, not `sizeof(uint16_t)`, so the count
  follows the pointer's type.  Alignment rounding (`sizeof(uint64_t)` in `MemPool.cc`) is
  the exception.

## Shared code

- **Comments in shared event-builder code (`psdaq/eb/`) must not name a detector.**
- **psdaq Python on the DAQ's path must not import `psana`**: it pulls in `mpi4py` and
  OpenMPI noise.  Shared code goes in `psalg` (e.g. `psalg/psalg/calibdb/`).

## Building and installing

- `ninja -C builddir install`; a bare `ninja` does not install, and `$TESTRELDIR` is
  `install/`.  A fresh builddir needs `env -u LDFLAGS meson setup ...` (conda's `LDFLAGS`
  break `nvcc`).
- **Never install while a DAQ running from that tree is starting or running**: `install/`
  is its live Python path and library directory, and a partial tree fails misleadingly.
- **`/usr/local/bin/drp_gpu` is node-local**: after every build it must be re-copied and
  re-`setcap`'d on each node (needs `sudo`; NFS cannot hold capabilities).  The image check
  refuses a stale copy.  Compare md5s against `install/bin/drp_gpu`, not `builddir`'s.
- A changed `drp_gpu` md5 does not prove the code changed (the link is not reproducible);
  an unchanged one proves identity.

Traps in reading measurements and logs are in `reading-evidence.md` beside this file.
Node and driver setup (Slurm, `nvidia-persistenced`, dkms) is on the "GPU node rules"
Confluence page, not here.
