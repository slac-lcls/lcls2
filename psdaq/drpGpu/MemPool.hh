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

// NVTX ranges give nsys traces their structure and cost little more than a check
// with no collector attached.  -DNVTX_DISABLE compiles them out.  nsys needs
// --cuda-graph-trace=node to show anything inside a CUDA graph.
//#define NVTX_DISABLE

// Define HOST_REARMS_DMA to run without CAP_SYS_ADMIN, which the GPU needs to write
// the GpuAsyncCore registers.  The CPU then rearms DMA buffers, later and only after
// the trigger.  Needs aes-stream-drivers PR #318; see the "GPU DRP" Confluence page.
//#define HOST_REARMS_DMA                 // Commented out => need CAP_SYS_ADMIN

namespace Drp {
  namespace Gpu {

// Which of an event's host-visible header buffers holds its prescale slot claim,
// written by the device and read by the recorder.  NoRawSlot: no claim.
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
  // The datagram header's reserve; the recorder grows the header backwards into it.
  size_t reduceBufsReserved()    const { return m_reduceBufsRsvd; }
  // The raw block's reserve, 0 unless the Detector asked for one.  It begins at
  // reduceBuffers_d()[idx*stride] - reduceBufsRaw().
  size_t reduceBufsRaw()         const { return m_reduceBufsRaw; }
  // Index reduceBuffers_d() with this, not a sum of the parts.
  size_t reduceBufsStride()      const
    { return m_reduceBufsRsvd + m_reduceBufsRaw + m_reduceBufsSize; }
  // Transitions have their own buffers, so the largest isn't multiplied by nbuffers().
  // The whole datagram comes from the host, so the Dgram starts at the buffer.
  const auto& transitionBuffers_d() const { return m_trBuffers_d; }
  size_t      trBufsSize()          const { return m_trBufsSize; }
  unsigned    trBufCnt()            const { return m_trBufCnt; }
  // A prescaled event's whole datagram, contiguous for a single write; there are far
  // fewer of these than nbuffers() (createPrescaleBuffers()).  Laid out as
  //
  //   [ Dgram ][ raw descr ][ raw ][ reduced descr ][ reduced copy ]
  //                          ^ rawOffset()           ^ redOffset()
  //
  // Descriptors are interleaved: CreateData writes each array's Xtcs just before its
  // bytes.  One reserve ahead of both payloads leaves a hole that reads as corrupt data.
  const auto& prescaleBuffers_d()    const { return m_prescaleBuffers_d; }
  size_t      prescaleBufsReserved() const { return m_prescaleBufsRsvd; }
  size_t      prescaleBufsRaw()      const { return m_prescaleBufsRaw; }
  size_t      prescaleBufsDscr()     const { return m_prescaleBufsDscr; }
  size_t      prescaleBufsSize()     const { return m_prescaleBufsSize; }
  unsigned    prescaleBufCnt()       const { return m_prescaleBufCnt; }
  size_t      prescaleRawOffset()    const { return m_prescaleBufsRsvd; }
  size_t      prescaleRedOffset()    const
    { return m_prescaleBufsRsvd + m_prescaleBufsRaw + m_prescaleBufsDscr; }
  size_t      prescaleBufsStride()   const
    { return prescaleRedOffset() + m_prescaleBufsSize; }
  // The device claims a slot by atomicAdd on the ticket and, having no way to abort,
  // raises overflow when the ticket outruns released, which the recorder advances.
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
