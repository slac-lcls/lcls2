#include "AreaDetectorGpu.hh"

#include "drp.hh"

using namespace Drp;

AreaDetectorGpu::AreaDetectorGpu(Parameters& para, MemPool& pool) :
  GpuWorker_impl(para, pool, m_det),
  m_det(&para, &pool)
{
}

//__device__ void event(const TimingHeader&, PGPEvent*) {}
//__device__ void slowUpdate(const TimingHeader&) {}

// The class factory

extern "C" GpuWorker* createDetectorGpu(Parameters& para, MemPool& pool)
{
  return new AreaDetectorGpu(para, pool);
}
