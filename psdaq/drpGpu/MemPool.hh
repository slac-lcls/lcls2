#pragma once

#include "gpuUtils.hh"

#include <cstddef>
#include <vector>
#include <thread>
#include <atomic>

#include <cuda_runtime.h>
#include <nvtx3/nvtx3.hpp>

#include "drp/drp.hh"

// NVTX provides the ability to annotate code and structures for the purpose of
// making traces in the Nsight Systems profiler more easily identifiable.  It
// nominally adds very little overhead but the documentation warns against
// instrumenting code that takes less than 1 us to run.  The NVTX_DISABLE macro
// can be defined to remove the NVTX calls from the codebase.
//#define NVTX_DISABLE

// If the HOST_REARMS_DMA  macro is defined, the GPU DRP can be run without
// privileges.  The CPU rearms the DMA buffers for writing as early as possible,
// but necessarily later than when the GPU can rearm them.  This will impact
// performance so this definition is normally commented out.  In order to have
// the GPU rearm the DMA buffers, the process must run with an as yet to be
// determined privilege (cap_sys_rawio, perhaps?), but I've not had success yet
// doing that.  Instead, set the executable up with root ownership and suid:
//
// sudo chown root $TESTRELDIR/bin/drp_gpu; sudo chmod u+s $TESTRELDIR/bin/drp_gpu
//
//#define HOST_REARMS_DMA                 // Commented out => need sudo

namespace Drp {
  namespace Gpu {

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
  Ptr<void>             fpgaRegs;
  std::vector<uint8_t*> dmaBuffers;     // Host vector of dmaCount dptrs
  uint8_t**             dmaBuffers_d;   // Device array of dmaCount dptrs
  std::string           name;
  CoreRegisters         coreRegs;

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
  void createReduceBuffers(size_t nBytes, size_t reserved);
  void destroyReduceBuffers();
  using vecpu32_t = std::vector<uint32_t*>;
  const auto& hostWrtBufs_h() const { return m_hostWrtBufs_h; }
  const auto& hostWrtBufs_d() const { return m_hostWrtBufs_d; }
  const auto& calibBuffers_d ()  const { return m_calibBuffers_d; }
  const auto& reduceBuffers_d()  const { return m_reduceBuffers_d; }
  size_t hostWrtBufsSize()       const { return m_hostWrtBufsSize; }
  size_t calibBufsSize()         const { return m_calibBufsSize; }
  size_t reduceBufsSize()        const { return m_reduceBufsSize; }
  size_t reduceBufsReserved()    const { return m_reduceBufsRsvd; }
public:
  int64_t nPgpInUser () const { return dmaGetRxBuffinUserCount  (fd()); }
  int64_t nPgpInHw   () const { return dmaGetRxBuffinHwCount    (fd()); }
  int64_t nPgpInPreHw() const { return dmaGetRxBuffinPreHwQCount(fd()); }
  int64_t nPgpInRx   () const { return dmaGetRxBuffinSwQCount   (fd()); }
private:
  int  _gpuMapFpgaMem(int fd, CUdeviceptr& buffer, uint64_t offset, size_t size, int write);
  void _gpuUnmapFpgaMem(CUdeviceptr& buffer);
private:
  CudaContext               m_context;
  std::shared_ptr<DetPanel> m_panel;
  bool                      m_setMaskBytesDone;
  size_t                    m_hostWrtBufsSize;
  uint32_t*                 m_hostWrtBufs_h;    // [nBuffers * nElements]
  uint32_t*                 m_hostWrtBufs_d;    // [nBuffers * nElements]
  size_t                    m_calibBufsSize;
  float*                    m_calibBuffers_d;   // [nBuffers * nElements]
  size_t                    m_reduceBufsSize;
  size_t                    m_reduceBufsRsvd;
  uint8_t*                  m_reduceBuffers_d;  // [nBuffers * nBytes]
};

  } // Gpu
} // Drp
