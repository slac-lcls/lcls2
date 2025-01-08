#pragma once

#include "GpuAsyncLib.hh"

#include "GpuAsyncOffsets.h"

#include <cstddef>
#include <vector>
#include <thread>
#include <atomic>

#include <cuda_runtime.h>

#include "drp.hh"


template <typename T> class SPSCQueue;

namespace Drp {

class Detector;

struct Batch
{
  uint32_t start;
  uint32_t size;
};

struct DetSeg
{
  int         fd;
  CUdeviceptr dmaBuffers[MAX_BUFFERS];
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
  bool                m_setMaskBytesDone;
};

class GpuWorker
{
public:
  GpuWorker(unsigned id, const Parameters&, MemPoolGpu&);
  ~GpuWorker() = default;
  void start(SPSCQueue<Batch>& workerQueue, Detector* det);
  void stop();
  void freeDma(PGPEvent* event);
  void handleBrokenEvent(const PGPEvent&);
  void resetEventCounter();
  void dmaIndex(uint32_t pgpIndex) { m_batchStart = pgpIndex + 1; }
  unsigned lastEvtCtr() const { return m_lastEvtCtr; }
  DmaTgt_t dmaTarget() const;
  void dmaTarget(DmaTgt_t dest);
public:
  MemPool& pool()     const { return m_pool; }
  uint64_t dmaBytes() const { return m_dmaBytes; }
  uint64_t dmaSize()  const { return m_dmaSize; }
  unsigned worker()   const { return m_worker; }
private:
  void _reader(unsigned index, SPSCQueue<Batch>& collectorGpuQueue, Detector& det);
private:
  MemPoolGpu&              m_pool;
  std::vector<CUstream>    m_streams;
  std::vector<std::thread> m_threads;
  std::atomic<uint32_t>    m_batchLast;
  std::atomic<uint32_t>    m_batchStart;
  std::atomic<uint32_t>    m_batchSize;
  unsigned                 m_dmaIndex;
  uint32_t                 m_lastEvtCtr;
  unsigned                 m_worker;
  const Parameters&        m_para;
  uint64_t                 m_dmaBytes;
  uint64_t                 m_dmaSize;
  uint64_t                 m_nDmaErrors;
  uint64_t                 m_nNoComRoG;
  uint64_t                 m_nMissingRoGs;
  uint64_t                 m_nTmgHdrError;
};

};
