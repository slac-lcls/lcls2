#pragma once

#include "ReducerAlgo.hh"


namespace Drp {
  namespace Gpu {

class NoOpReducer : public ReducerAlgo
{
public:
  NoOpReducer(const Parameters& para, const MemPoolGpu& pool, Detector& det);
  virtual ~NoOpReducer() {}

  void recordGraph(cudaStream_t&      stream,
                   const unsigned&    index,
                   float const* const calibBuffers,
                   const size_t       calibBufsCnt,
                   uint8_t    * const dataBuffers,
                   const size_t       dataBufsCnt,
                   unsigned*          extent) override;
  unsigned configure(XtcData::Xtc&, const void* bufEnd) override;
  void     event    (XtcData::Xtc&, const void* bufEnd, unsigned dataSize) override;
};

  } // Gpu
} // Drp
