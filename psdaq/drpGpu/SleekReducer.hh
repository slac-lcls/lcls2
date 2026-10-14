#pragma once

#include "ReducerAlgo.hh"

#include <SingleCompressorLossy.hh>

namespace Drp {
  namespace Gpu {

class SleekReducer : public ReducerAlgo
{
public:
  SleekReducer(const Parameters& para, const MemPoolGpu& pool, Detector& det);
  virtual ~SleekReducer() {}

  bool   hasGraph()    const override { return true; }
  size_t payloadSize() const override { return m_compressor.maxSize(); }
  void   recordGraph(cudaStream_t       stream,
                     unsigned*    const state,
                     unsigned*    const index,
                     float const* const calibBuffers,
                     size_t       const calibBufsCnt,
                     uint8_t*     const dataBuffers,
                     size_t       const dataBufsCnt) override;
  void     reduce   (cudaGraphExec_t,
                     cudaStream_t,
                     unsigned  index,
                     size_t*   dataSize,
                     unsigned* retCode) override;
  unsigned configure(XtcData::Xtc&, const void* bufEnd) override;
  void     event    (XtcData::Xtc&, const void* bufEnd, unsigned dataSize) override;
private:
  SLEEK::SingleCompressorLossy m_compressor;
};

  } // Gpu
} // Drp
