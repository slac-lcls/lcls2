#pragma once

#include "ReducerAlgo.hh"


namespace Drp {
  namespace Gpu {

class NoOpReducer : public ReducerAlgo
{
public:
  NoOpReducer(const Parameters& para, const MemPoolGpu& pool);
  virtual ~NoOpReducer() {}

  virtual void recordGraph(cudaStream_t&   stream,
                           const unsigned& index,
                           float**   const calibBuffer,
                           uint8_t** const dataBuffer,
                           unsigned*       extent) override;
#if 0 // @todo: Revisit
  virtual unsigned configure(XtcData::Xtc&, const void* bufEnd, XtcData::ConfigIter&) override;
  virtual void     event    (XtcData::Xtc&, const void* bufEnd) override;
#endif
private:
  size_t _calibSize;                    // Bytes
};

  } // Gpu
} // Drp
