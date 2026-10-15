#pragma once

#include "gpuUtils.hh"

#include <cstddef>
#include <vector>
#include <thread>
#include <atomic>

#include <cuda_runtime.h>
#include <cuda/std/atomic>
#include <nvtx3/nvtx3.hpp>

#include "drp/drp.hh"

// NVTX provides the ability to annotate code and structures for the purpose of
// making traces in the Nsight Systems profiler more easily identifiable.  It
// nominally adds very little overhead but the documentation warns against
// instrumenting code that takes less than 1 us to run.  The NVTX_DISABLE macro
// is used by NVTX header files to disable NVTX calls in the codebase.
//
// Left enabled: the ranges are what give an nsys trace of this application any
// application-level structure, and without a collector attached NVTX resolves to
// little more than a check, which is why it is meant to be left compiled in.  This
// cannot be a runtime kwarg -- NVTX_DISABLE is read by the NVTX headers, so
// defining it compiles the ranges out altogether -- but it can still be turned off
// for a build with -DNVTX_DISABLE if one of them ever proves too costly.  Note
// that nsys needs --cuda-graph-trace=node to show anything inside a CUDA graph,
// which is nearly everything here.
//#define NVTX_DISABLE

// If the HOST_REARMS_DMA macro is defined, the GPU DRP can be run without
// privileges.  The CPU rearms the DMA buffers for writing as early as possible,
// but necessarily later than when the GPU can rearm them.  This will impact
// performance so this definition is normally commented out.
//
// For the GPU to rearm the DMA buffers, its kernels must be able to write the
// GpuAsyncCore FPGA registers, which means cuMemHostRegister() has to map them
// with CU_MEMHOSTREGISTER_IOMEMORY (see MemPool.cc).  That mapping is privileged:
// without the privilege it fails with CUDA_ERROR_NOT_PERMITTED.
//
// The privilege needed is CAP_SYS_ADMIN.  Measured on drp-srcf-gpu008 (Rocky 9):
// CAP_SYS_ADMIN alone suffices, and neither CAP_SYS_RAWIO nor CAP_IPC_LOCK is
// needed.
//
// Beware of older recipes: CAP_SYS_RAWIO used to be enough.  The same test
// program, with 'setcap cap_sys_rawio+ep' and run from a local file system, made
// this call succeed under RHEL 7 and gets CUDA_ERROR_NOT_PERMITTED under Rocky 9.
// The check is not in the aes-stream-drivers datadev driver, so it lies either in
// the NVIDIA driver or in the kernel's hardening of PFN lookup for VM_PFNMAP
// mappings, which tightened between those two kernels.
//
// Two ways of granting it are known to work, both leaving the process running as
// an ordinary user rather than as root.
//
// Preferred, an ambient capability.  Nothing is stored on the file system, so the
// executable can stay where it is built, and because the privilege is inherited
// from the parent rather than gained at exec, the process does not enter
// secure-execution mode and LD_LIBRARY_PATH keeps working:
//
//   sudo -E setpriv --reuid=$(id -u) --regid=$(id -g) --init-groups
//        --securebits=+no_setuid_fixup
//        --inh-caps=-all,+sys_admin --ambient-caps=-all,+sys_admin
//        $TESTRELDIR/bin/drp_gpu <args>
//
// (all one command; no line continuations here because a backslash at the end of
// a // comment makes it a multi-line comment, which -Wcomment objects to)
//
// --securebits=+no_setuid_fixup matters: without it the kernel clears the
// capabilities when the uid drops from 0.
//
// Or a file capability, which needs the executable on a local file system:
//
//   sudo setcap cap_sys_admin+ep /home/<user>/install/bin/drp_gpu
//
// This works, including the fact that a file capability does put the process into
// secure-execution mode, where the loader ignores LD_LIBRARY_PATH: the absolute
// RPATHs meson bakes in are sufficient on their own.  The costs are copying the
// executable to a local file system after every build, and that setcap fails
// outright on wekafs, which supports user.* extended attributes but not
// security.capability.
//
// The remaining alternative is root ownership and setuid, which grants a great
// deal more than is needed and also has to be redone after every build:
//
//   sudo chown root $TESTRELDIR/bin/drp_gpu; sudo chmod u+s $TESTRELDIR/bin/drp_gpu
//
// HOST_REARMS_DMA works, but read this before relying on it.
//
// It first needed a driver fix.  GpuAsyncCore ownership is claimed by whichever
// thread calls gpuAddNvidiaMemory() -- MemPoolGpu's constructor -- but the datadev
// driver's ownership checks compared against current->pid, the thread id, rather
// than current->tgid.  Every gpuSetWriteEn() from another thread, which is where
// the rearming happens, failed with EBUSY:
//
//   datadev 0000:04:00.0: Gpu_SetWriteEn: Called by non-owner PID (3417898)
//
// even though that is a thread of the owning process.  It defeated the initial
// arming in Reader::startup() too, so it failed immediately rather than under load.
// Fixed in aes-stream-drivers by PR #318, which compares against current->tgid in
// Gpu_AddNvidia(), Gpu_RemNvidia(), Gpu_SetWriteEn(), Gpu_EnableTx() and
// Gpu_EnableRx(), and is merged to its pre-release branch.  This path needs a
// driver built from that or later.
//
// With that fix it reaches the same 33 kHz as the GPU-rearm path on
// drp-srcf-gpu008, though the rate is measurably less steady.
//
// The remaining concern is structural rather than immediate.  A DMA buffer is
// finished with as soon as Gpu::_event() has copied its headers out and calibrated
// its payload, which is where the GPU rearms it.  But TrgInpGen::_receiver() runs
// downstream of the trigger kernels, so rearming there holds the buffer for however
// long the dynamically loaded trigger library takes.  That is user code, perhaps
// doing pattern recognition or machine learning, so its latency is unbounded: a
// dmaBufCount that suffices for a trivial trigger need not suffice for a real one.
// Decoupling the two means moving the rearm to a stage immediately after the Reader
// and before the trigger -- a device-to-host queue, per RingQueue_DtoH.hh, feeding
// a thread whose only job is the ioctl.  A bulk form of gpuSetWriteEn() would help
// as well, since one ioctl per buffer is ~33k/s at these rates.
//
//#define HOST_REARMS_DMA                 // Commented out => need CAP_SYS_ADMIN

// The HOST_LAUNCHED_REDUCERS macro is used to determine when the Reducer GPU
// code is launched.  Without this macro defined, Reducers constructed using a
// CUDA graph are launched at startup time using the Reducer::startup() method.
// Reducers launched this way remain present on the GPU for the duration of a
// Configure/Unconfigure cycle.  The idea behind this is to amortise the launch
// overhead and to pay it at a non-critical time.
// Reducers composed of raw kernels are handled as if the macro were defined.
// With it defined, Reducers, whether constructed using a graph or composed of
// raw kernels are launched upon reception of a TEB result (in
// TebReceiver::complete()).  The Reducer runs for as long as it takes for it to
// process one event and then exits.  This means that the launch overhead is
// paid in the real-time loop.
//
// Retiring this macro, and ReducerAlgo::hasGraph() with it, is a to-do.  Whoever
// does it should grep for both names first: every ReducerAlgo, PassthruShim
// included, assumes the graph path, and the #ifdef blocks in Reducer.cu are what
// make hasGraph() == false a silent hang rather than a compile error.  Nothing in
// those files points back here.
//#define HOST_LAUNCHED_REDUCERS


namespace Drp {
  namespace Gpu {

// Where the prescale slot a prescaled event claimed sits in that event's
// host-visible header buffer: written by the device, read by the recorder.  Here
// rather than in ReaderKernels.cuh because host code needs it too, and it describes
// the buffer layout this file owns.  NoRawSlot means the event claimed none.
hostFunc deviceFunc
inline size_t      rawSlotIndex(size_t hdrBufsCnt) { return hdrBufsCnt - 2; }
constexpr unsigned NoRawSlot{0xffffffff};

// @todo: Move to a common header file or use std::pair/std::tuple
template <class T>
struct Ptr
{
  T* h = nullptr;                       // A host pointer
  T* d = nullptr;                       // A device pointer
};

// DmaDsc structure from:
//   https://github.com/slaclab/surf/blob/main/axi/dma/rtl/v2/AxiStreamDmaV2Write.vhd
struct __attribute__((packed)) DmaDsc
{
  uint32_t header;
  uint32_t size;

  inline uint32_t result()    const { return  header        & 0x03; }
  inline uint32_t overflow()  const { return (header >>  2) & 0x01; }
  inline uint32_t cont()      const { return (header >>  3) & 0x01; }
  inline uint32_t lastUser()  const { return (header >> 16) & 0xFF; }
  inline uint32_t firstUser() const { return (header >> 24) & 0xFF; }

  // firstUser = 0x02 is SOF: not an error
  inline uint32_t errorMask() const { return 0xfdffffff; }
};

static_assert(sizeof(DmaDsc) == 8, "DmaDsc must be 64-bits (8-bytes)");

/**
 * Wraps a data_dev device so it can be automatically freed
 */
class DataDev
{
public:
  DataDev(const char* path);
  ~DataDev()
  {
    close(fd_);
  }

  int fd() const { return fd_; }

protected:
  int fd_;
};

struct DetPanel
{
  DataDev               datadev;
  CoreRegisters         coreRegs;       // GpuAsyncCore registers wrapper
  std::vector<uint8_t*> dmaBuffers;     // Host vector of dmaCount dptrs, GPU page aligned
  std::vector<uint8_t*> dmaRawPtrs;     // What cudaMalloc returned, for cudaFree
  uint8_t**             dmaBuffers_d;   // Device array of dmaCount dptrs
  std::string           name;

  DetPanel(std::string& device) : datadev(device.c_str()), name(device) {}
};

class MemPoolGpu : public Drp::MemPool
{
public:
  MemPoolGpu(Parameters& para);
  virtual ~MemPoolGpu();
  int initialize(Parameters& para);
public:   // Virtuals
  int fd() const override { return m_panel->datadev.fd(); }
  int setMaskBytes(uint8_t laneMask, unsigned virtChan) override;
private:  // Virtuals
  ssize_t _freeDma(unsigned count, uint32_t* indices) override { return 0; /* Nothing to do */ }
public:
  const CudaContext& context() const { return m_context; }
  const std::shared_ptr<DetPanel> panel() const { return m_panel; }
  void createHostBuffers(size_t size);
  void destroyHostBuffers();
  void createCalibBuffers(unsigned nElements);
  void destroyCalibBuffers();
  void createReduceBuffers(size_t nBytes, size_t reserved, size_t rawBytes = 0);
  void destroyReduceBuffers();
  void createTransitionBuffers(size_t nBytes, unsigned nBufs);
  void destroyTransitionBuffers();
  void createPrescaleBuffers(size_t nBytes, unsigned nBufs, size_t hdrBytes, size_t dscrBytes);
  void destroyPrescaleBuffers();
  using vecpu32_t = std::vector<uint32_t*>;
  const auto& hostWrtBufs()      const { return m_hostWrtBufs; }
  const auto& calibBuffers_d ()  const { return m_calibBuffers_d; }
  const auto& reduceBuffers_d()  const { return m_reduceBuffers_d; }
  size_t hostWrtBufsSize()       const { return m_hostWrtBufsSize; }
  size_t calibBufsSize()         const { return m_calibBufsSize; }
  size_t reduceBufsSize()        const { return m_reduceBufsSize; }
  // Bytes reserved ahead of the reduced payload for the datagram header alone.
  // The recorder grows backwards into this from wherever its block starts, so
  // this bounds the header, not the header plus raw.
  size_t reduceBufsReserved()    const { return m_reduceBufsRsvd; }
  // Bytes reserved between the header reserve and the reduced payload, for a
  // block of raw data.  Zero unless the Detector asked for one.  The raw block
  // begins at reduceBuffers_d()[idx*stride] - reduceBufsRaw().
  size_t reduceBufsRaw()         const { return m_reduceBufsRaw; }
  // The buffer-to-buffer stride: every region a reduce buffer comprises.  Use
  // this to index reduceBuffers_d() -- `&reduceBuffers_d()[idx * stride]` -- in
  // preference to adding the parts up, which silently goes wrong when a region
  // is added.
  size_t reduceBufsStride()      const
    { return m_reduceBufsRsvd + m_reduceBufsRaw + m_reduceBufsSize; }
  // Transitions are written through their own buffers, so that the largest
  // transition's size is not multiplied by nbuffers() of reduce buffer.  A
  // transition's whole datagram is copied from the host, so unlike a reduce buffer
  // there is no reserve to grow backwards into: the Dgram starts at the buffer.
  const auto& transitionBuffers_d() const { return m_trBuffers_d; }
  size_t      trBufsSize()          const { return m_trBufsSize; }
  unsigned    trBufCnt()            const { return m_trBufCnt; }
  // Prescaled events assemble their whole datagram -- header, raw and a copy of the
  // reduced payload -- into one of these, so that it is contiguous for a single
  // write.  Far fewer than nbuffers(), since only the ~1 Hz the timing system marks
  // needs one; see createPrescaleBuffers().  Each is laid out as
  //
  //   [ Dgram ][ raw descr ][ raw ][ reduced descr ][ reduced copy ]
  //                          ^ rawOffset()           ^ redOffset()
  //
  // The descriptors are INTERLEAVED with the payloads, not gathered ahead of them:
  // CreateData writes each array's Shapes and Data Xtcs immediately before that
  // array's bytes.  So a payload's offset here must equal the offset its own
  // descriptors give it, and the two offsets below are the single statement of that.
  // Treating the reserve as one block ahead of both payloads leaves a hole the width
  // of the second descriptor block, which reads back as corrupt data.
  const auto& prescaleBuffers_d()    const { return m_prescaleBuffers_d; }
  size_t      prescaleBufsReserved() const { return m_prescaleBufsRsvd; }
  size_t      prescaleBufsRaw()      const { return m_prescaleBufsRaw; }
  size_t      prescaleBufsDscr()     const { return m_prescaleBufsDscr; }
  size_t      prescaleBufsSize()     const { return m_prescaleBufsSize; }
  unsigned    prescaleBufCnt()       const { return m_prescaleBufCnt; }
  // Where each payload sits within a prescale buffer.  m_prescaleBufsRsvd is the Dgram
  // plus ONE ShapesData's descriptors; m_prescaleBufsDscr is a second block of them.
  size_t      prescaleRawOffset()    const { return m_prescaleBufsRsvd; }
  size_t      prescaleRedOffset()    const
    { return m_prescaleBufsRsvd + m_prescaleBufsRaw + m_prescaleBufsDscr; }
  size_t      prescaleBufsStride()   const
    { return prescaleRedOffset() + m_prescaleBufsSize; }
  // The device claims a slot with an atomicAdd on the ticket, the recorder advances
  // released once it has written one, and the kernel compares the two to tell whether
  // it has outrun the pool; a kernel cannot abort, so it only raises overflow.  The
  // latter two cross the host/device boundary, hence atomics in pinned memory as
  // RingIndexHtoD's head and tail are.  The ticket stays plain, being device-only.
  unsigned*                    rawTicket_d() const { return m_rawTicket_d; }
  cuda::std::atomic<unsigned>* rawOverflow() const { return m_rawOverflow; }
  cuda::std::atomic<unsigned>* rawReleased() const { return m_rawReleased; }
public:
  int64_t nPgpInUser () const { return dmaGetRxBuffinUserCount  (fd()); }
  int64_t nPgpInHw   () const { return dmaGetRxBuffinHwCount    (fd()); }
  int64_t nPgpInPreHw() const { return dmaGetRxBuffinPreHwQCount(fd()); }
  int64_t nPgpInRx   () const { return dmaGetRxBuffinSwQCount   (fd()); }
private:
  int  _gpuMapFpgaMem(int fd, CUdeviceptr& buffer, uint64_t offset, size_t size, int write);
  void _gpuUnmapFpgaMem(CUdeviceptr& buffer);
private:
  CudaContext                  m_context;
  std::shared_ptr<DetPanel>    m_panel;
  bool                         m_setMaskBytesDone;
  size_t                       m_hostWrtBufsSize;
  uint32_t*                    m_hostWrtBufs;        // [nBuffers * nElements]
  size_t                       m_calibBufsSize;
  float*                       m_calibBuffers_d;     // [nBuffers * nElements]
  size_t                       m_reduceBufsSize;
  size_t                       m_reduceBufsRsvd;
  size_t                       m_reduceBufsRaw;
  uint8_t*                     m_reduceBuffers_d;    // [nBuffers * nBytes]
  size_t                       m_trBufsSize;         // Bytes per transition buffer
  unsigned                     m_trBufCnt;           // How many there are
  uint8_t*                     m_trBuffers_d;        // [m_trBufCnt * m_trBufsSize]
  size_t                       m_prescaleBufsRsvd;   // Dgram + the raw array's descriptors
  size_t                       m_prescaleBufsRaw;    // The raw block
  size_t                       m_prescaleBufsDscr;   // The reduced array's descriptors
  size_t                       m_prescaleBufsSize;   // Room for the reduced payload copy
  unsigned                     m_prescaleBufCnt;     // How many there are
  uint8_t*                     m_prescaleBuffers_d;  // [m_prescaleBufCnt * prescaleBufsStride()]
  unsigned*                    m_rawTicket_d;        // Monotonic claim counter, device
  cuda::std::atomic<unsigned>* m_rawOverflow;        // Raised when a claim outran the pool
  cuda::std::atomic<unsigned>* m_rawReleased;        // Slots the recorder has finished with
};

  } // Gpu
} // Drp
