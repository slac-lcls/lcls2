#pragma once

#include "GpuAsyncLib.hh"

#include "GpuAsyncOffsets.h"

#include <cstddef>
#include <vector>
#include <thread>
#include <atomic>

#include <cuda_runtime.h>
#include <cuda/atomic>

#include "drp/drp.hh"

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
  int fd() const override;
  int setMaskBytes(uint8_t laneMask, unsigned virtChan) override;
private:  // Virtuals
  void _freeDma(unsigned count, uint32_t* indices) override { /* Nothing to do */ }
public:
  const CudaContext& context() const { return m_context; }
  const std::vector<DetPanel>& panels() const { return m_panels; }
  void createHostBuffers(unsigned panel, unsigned nBuffers, size_t size);
  void destroyHostBuffers(unsigned panel);
  void createCalibBuffers(unsigned nBuffers, unsigned nPanels, unsigned nWords);
  void destroyCalibBuffers();
  void createReduceBuffers(unsigned nBuffers, unsigned nWords);
  void destroyReduceBuffers();
  using vecpu32_t = std::vector<uint32_t*>;
  const auto& hostBuffers_h() const { return m_hostWriteBufs_h; }
  const auto& hostBuffers_d() const { return m_hostWriteBufs_d; }
  const auto& hostPnlBufs_d() const { return m_hostPnlWrBufs_d; }
  const auto& calibBuffers () const { return m_calibBuffers; }
  const auto& reduceBuffers() const { return m_dataBuffers; }
private:
  int  _gpuMapFpgaMem(int fd, CUdeviceptr& buffer, uint64_t offset, size_t size, int write);
  void _gpuUnmapFpgaMem(CUdeviceptr& buffer);
private:
  CudaContext             m_context;
  std::vector<DetPanel>   m_panels;
  unsigned                m_setMaskBytesDone;
  std::vector<vecpu32_t>  m_hostWriteBufs_h; // [nPanels][nBuffers][nWords]
  std::vector<uint32_t**> m_hostWriteBufs_d; // [nPanels][nBuffers][nWords]
  uint32_t***             m_hostPnlWrBufs_d; // [nPanels][nBuffers][nWords]
  float*                  m_calibBuffers;    // [nBuffers][nPanels][nWords]
  float*                  m_dataBuffers;     // [nBuffers][nWords]
};

  } // Gpu
} // Drp
