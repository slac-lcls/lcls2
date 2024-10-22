#pragma once

#include "AreaDetector.hh"              // Detector  implementation
#include "GpuWorker_impl.hh"            // GpuWorker implementation

#include "drp.hh"

namespace Drp {

class Parameters;
class MemPool;

class AreaDetectorGpu : public AreaDetector
{
public:
  AreaDetectorGpu(Parameters& para, MemPool& pool);
public:
  virtual GpuWorker* gpuWorker() override { return &m_worker; }
  // __device__ void event(const TimingHeader&, PGPEvent*);
  // __device__ void slowUpdate(const TimingHeader&);
private:
  GpuWorker_impl m_worker;
};

}
