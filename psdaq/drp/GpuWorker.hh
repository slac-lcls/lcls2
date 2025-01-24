#pragma once

#include "GpuAsyncLib.hh"

#include "GpuAsyncOffsets.h"

#include <cstddef>
#include <vector>
#include <thread>
#include <atomic>

#include <cuda_runtime.h>
//#include <cuda/atomic>

#include "drp.hh"


template <typename T> class SPSCQueue;

namespace Pds {
  class TimingHeader;
}

namespace Drp {

class Detector;

struct GpuMetrics
{
  GpuMetrics() :
    m_nevents     (0),
    m_nDmaRet     (0),
    m_dmaBytes    (0),
    m_dmaSize     (0),
    m_nDmaErrors  (0),
    m_nNoComRoG   (0),
    m_nMissingRoGs(0),
    m_nTmgHdrError(0),
    m_nPgpJumps   (0)
  {
  }
  std::atomic<uint64_t> m_nevents;
  std::atomic<uint64_t> m_nDmaRet;
  std::atomic<uint64_t> m_dmaBytes;
  std::atomic<uint64_t> m_dmaSize;
  std::atomic<uint64_t> m_nDmaErrors;
  std::atomic<uint64_t> m_nNoComRoG;
  std::atomic<uint64_t> m_nMissingRoGs;
  std::atomic<uint64_t> m_nTmgHdrError;
  std::atomic<uint64_t> m_nPgpJumps;
};

struct Batch
{
  uint32_t start;
  uint32_t size;
};

struct DetSeg
{
  int            fd;
  GpuDmaBuffer_t dmaBuffers[MAX_BUFFERS];
  GpuDmaBuffer_t swFpgaRegs;
  CUdeviceptr    hwWriteStart;
  GpuDmaBuffer_t swVersionRegs;
};

class MemPoolGpu : public MemPool
{
public:
  MemPoolGpu(Parameters& para);
  virtual ~MemPoolGpu();
  const CudaContext& context() const { return m_context; }
  int initialize(Parameters& para);
  const std::vector<DetSeg>& segs() const { return m_segs; }
public:
  virtual int fd() const override { return m_segs[0].fd; } // @todo: Only the first one for the moment
  virtual int setMaskBytes(uint8_t laneMask, unsigned virtChan) override;
  unsigned count() const { return MAX_BUFFERS; }
private:
  int  _gpuMapFpgaMem(int fd, CUdeviceptr& buffer, uint64_t offset, size_t size, int write);
  void _gpuUnmapFpgaMem(CUdeviceptr& buffer);
  virtual void _freeDma(unsigned count, uint32_t* indices) override { /* Nothing to do */ }
private:
  CudaContext         m_context;
  std::vector<DetSeg> m_segs;
  unsigned            m_setMaskBytesDone;
};

class GpuWorker
{
public:
  GpuWorker(unsigned id, const Parameters&, MemPoolGpu&);
  ~GpuWorker(); // = default;
  void start(Detector* det, GpuMetrics& metrics);
  void stop();
  void freeDma(unsigned index);
  void handleBrokenEvent(const PGPEvent&);
  void resetEventCounter();
  SPSCQueue<unsigned>& dmaQueue() { return m_dmaQueue; }
  Pds::TimingHeader* timingHeader(unsigned index) const;
  unsigned lastEvtCtr() const { return m_lastEvtCtr; }
  DmaTgt_t dmaTarget() const;
  void dmaTarget(DmaTgt_t dest);
public:
  MemPool& pool()   const { return m_pool; }
  unsigned worker() const { return m_worker; }
private:
  int     _setupCudaGraphs(const DetSeg& seg, int instance);
  CUgraph _recordGraph(CUstream& stream, CUdeviceptr hwWritePtr, CUdeviceptr hwWriteStart, uint32_t* hostWriteBuf); //, int* bufRdy);
  void    _reader(Detector&, GpuMetrics&);
private:
  MemPoolGpu&              m_pool;
  const DetSeg&            m_seg;
  volatile int*            m_terminate;
  //cuda::atomic<int>*       m_terminate;
  //volatile int*            m_bufRdy[MAX_BUFFERS];
  std::vector<CUstream>    m_streams;
  std::vector<CUgraph>     m_graphs;
  std::vector<CUgraphExec> m_graphExecs;
  std::thread              m_thread;
  std::vector<uint32_t*>   m_hostWriteBufs;
  SPSCQueue<unsigned>      m_dmaQueue;
  ///std::atomic<uint32_t>    m_batchLast;
  ///std::atomic<uint32_t>    m_batchStart;
  ///std::atomic<uint32_t>    m_batchSize;
  uint32_t                 m_lastEvtCtr;
  unsigned                 m_worker;
  const Parameters&        m_para;
};

};
