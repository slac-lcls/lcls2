#include "AreaDetectorGpu.hh"

#include "drp.hh"

using namespace Drp;

AreaDetectorGpu::AreaDetectorGpu(Parameters& para, MemPool& pool) :
  AreaDetector(&para, &pool),
  m_worker(para, pool, *this)
{
}

//__device__ void event(const TimingHeader&, PGPEvent*) {}
//__device__ void slowUpdate(const TimingHeader&) {}

// The class factory

extern "C" Detector* createDetectorGpu(Parameters& para, MemPool& pool)
{
  return new AreaDetectorGpu(para, pool);
}
