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
  const auto& calibBuffers_h ()  const { return m_calibBufsVec_h; }
  const auto& calibBuffers_d ()  const { return m_calibBuffers_d; }
  const auto& reduceBuffers_h()  const { return m_reduceBufsVec_h; }
  const auto& reduceBuffers_d()  const { return m_reduceBuffers_d; }
  size_t hostWrtBufsSize()       const { return m_hostWrtBufsSize; }
  size_t calibBufSize()          const { return m_calibBufSize; }
  size_t reduceBufSize()         const { return m_reduceBufSize; }
  size_t reduceBufReserved()     const { return m_reduceBufRsvd; }
private:
  int  _gpuMapFpgaMem(int fd, CUdeviceptr& buffer, uint64_t offset, size_t size, int write);
  void _gpuUnmapFpgaMem(CUdeviceptr& buffer);
private:
  CudaContext             m_context;
  std::vector<DetPanel>   m_panels;
  unsigned                m_setMaskBytesDone;
  size_t                  m_hostWrtBufsSize;
  std::vector<vecpu32_t>  m_hostWrtBufsVec_h; // [nPanels][nBuffers][nElements]
  std::vector<uint32_t**> m_hostWrtBufsVec_d; // [nPanels][nBuffers][nElements]
  uint32_t***             m_hostWrtBufs_d;    // [nPanels][nBuffers][nElements]
  size_t                  m_calibBufSize;
  std::vector<float*>     m_calibBufsVec_h;   // [nBuffers][nPanels * nElements]
  float**                 m_calibBuffers_d;   // [nBuffers][nPanels * nElements]
  size_t                  m_reduceBufSize;
  size_t                  m_reduceBufRsvd;
  std::vector<uint8_t*>   m_reduceBufsVec_h;  // [nBuffers][nBytes]
  uint8_t**               m_reduceBuffers_d;  // [nBuffers][nBytes]
};

  } // Gpu
} // Drp
