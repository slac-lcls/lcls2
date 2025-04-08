#pragma once

#include "Detector.hh"

#include "drp/AreaDetector.hh"          // Detector implementation
#include "drp/drp.hh"

namespace Drp {
  namespace Gpu {

class AreaDetector : public Gpu::Detector
{
public:
  AreaDetector(Parameters& para, MemPoolGpu& pool);
  virtual ~AreaDetector() override;
public:
  Gpu::Detector* gpuDetector() override { return this; }
public:
  void recordGraph(cudaStream_t&                      stream,
                   const unsigned&                    index,
                   const unsigned                     panel,
                   uint16_t const* const __restrict__ data) override;
  // @todo: To be implemented or moved
  //void recordReduceGraph(cudaStream_t&             stream,
  //                       const unsigned&           index,
  //                       float* const __restrict__ calibBuffers,
  //                       float* const __restrict__ dataBuffers) override;
private:
  unsigned m_nPixels;
};

  } // Gpu
} // Drp
