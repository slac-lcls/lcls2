#pragma once

#include "GpuAsyncLib.hh"

#include "GpuAsyncOffsets.h"

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

// DmaDsc structure from:
//   https://github.com/slaclab/surf/blob/main/axi/dma/rtl/v2/AxiStreamDmaV2Write.vhd
struct DmaDsc
{
  uint32_t error;
  uint32_t size;
  uint32_t _rsvd[6];
};

struct DetPanel
{
  DataGPU        gpu;
  GpuDmaBuffer_t dmaBuffers[MAX_BUFFERS];
  GpuDmaBuffer_t swFpgaRegs;
  CUdeviceptr    hwWriteStart;
  DetPanel(std::string& device) : gpu(device.c_str()) {}
};

// @todo: Move to a common header file or use std::pair/std::tuple
template <class T>
struct Ptr
{
  T* h = nullptr;                       // A host pointer
  T* d = nullptr;                       // A device pointer
};

class MemPoolGpu : public Drp::MemPool
{
public:
  MemPoolGpu(Parameters& para);
  virtual ~MemPoolGpu();
  int initialize(Parameters& para);
public:   // Virtuals
  int fd(unsigned unit=0) const override;
  int setMaskBytes(uint8_t laneMask, unsigned virtChan) override;
private:  // Virtuals
  ssize_t _freeDma(unsigned count, uint32_t* indices) override { return 0; /* Nothing to do */ }
public:
  const CudaContext& context() const { return m_context; }
  const std::vector<DetPanel>& panels() const { return m_panels; }
  void createHostBuffers(unsigned panel, size_t size);
  void destroyHostBuffers(unsigned panel);
  void createCalibBuffers(unsigned nPanels, unsigned nElements);
  void destroyCalibBuffers();
  void createReduceBuffers(size_t nBytes, size_t reserved);
  void destroyReduceBuffers();
  using vecpu32_t = std::vector<uint32_t*>;
  const auto& hostWrtBufsVec_h() const { return m_hostWrtBufsVec_h; }
  const auto& hostWrtBufsVec_d() const { return m_hostWrtBufsVec_d; }
  const auto& hostWrtBufs_d()    const { return m_hostWrtBufs_d; }
  const auto& calibBuffers_d ()  const { return m_calibBuffers_d; }
  const auto& reduceBuffers_d()  const { return m_reduceBuffers_d; }
  size_t hostWrtBufsSize()       const { return m_hostWrtBufsSize; }
  size_t calibBufsSize()         const { return m_calibBufsSize; }
  size_t reduceBufsSize()        const { return m_reduceBufsSize; }
  size_t reduceBufsReserved()    const { return m_reduceBufsRsvd; }
  // @todo: Right place for these?
  int64_t nPgpInUser (unsigned unit) const { return dmaGetRxBuffinUserCount  (fd(unit)); }
  int64_t nPgpInHw   (unsigned unit) const { return dmaGetRxBuffinHwCount    (fd(unit)); }
  int64_t nPgpInPreHw(unsigned unit) const { return dmaGetRxBuffinPreHwQCount(fd(unit)); }
  int64_t nPgpInRx   (unsigned unit) const { return dmaGetRxBuffinSwQCount   (fd(unit)); }
private:
  int  _gpuMapFpgaMem(int fd, CUdeviceptr& buffer, uint64_t offset, size_t size, int write);
  void _gpuUnmapFpgaMem(CUdeviceptr& buffer);
private:
  CudaContext             m_context;
  std::vector<DetPanel>   m_panels;
  unsigned                m_setMaskBytesDone;
  size_t                  m_hostWrtBufsSize;
  std::vector<uint32_t*>  m_hostWrtBufsVec_h; // [nPanels][nBuffers * nElements]
  std::vector<uint32_t*>  m_hostWrtBufsVec_d; // [nPanels][nBuffers * nElements]
  uint32_t**              m_hostWrtBufs_d;    // [nPanels][nBuffers * nElements]
  size_t                  m_calibBufsSize;
  float*                  m_calibBuffers_d;   // [nBuffers * nPanels * nElements]
  size_t                  m_reduceBufsSize;
  size_t                  m_reduceBufsRsvd;
  uint8_t*                m_reduceBuffers_d;  // [nBuffers * nBytes]
};

  } // Gpu
} // Drp
