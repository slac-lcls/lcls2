#pragma once

#include "AreaDetector.hh"              // Detector  implementation
#include "GpuWorker_impl.hh"            // GpuWorker implementation

#include "drp.hh"

namespace Drp {

class Parameters;
class MemPool;

class AreaDetectorGpu : public GpuWorker_impl
{
public:
  AreaDetectorGpu(Parameters& para, MemPool& pool);
public:
  // __device__ void event(const TimingHeader&, PGPEvent*);
  // __device__ void slowUpdate(const TimingHeader&);
public:
  AreaDetector m_det;
};

}
