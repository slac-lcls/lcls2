#pragma once

#include "ReducerAlgo.hh"

#include <cuSZp.h>

namespace Drp {
  namespace Gpu {

class CuSZpReducer : public ReducerAlgo
{
public:
  CuSZpReducer(const Parameters& para, const MemPoolGpu& pool, Detector& det);
  virtual ~CuSZpReducer() {}

  bool   hasGraph() const override { return false; }
  size_t payloadSize() const override { return m_pool.calibBufsSize(); }
  void   recordGraph(cudaStream_t       stream,
                     const unsigned&    index,
                     float const* const calibBuffer,
                     const size_t       calibBufsCnt,
                     uint8_t    * const dataBuffer,
                     const size_t       dataBufsCnt) override;
  void     reduce   (cudaGraphExec_t, cudaStream_t, unsigned index, size_t* dataSize) override;
  unsigned configure(XtcData::Xtc&, const void* bufEnd) override;
  void     event    (XtcData::Xtc&, const void* bufEnd, unsigned dataSize) override;
private:
  double m_errorBound;
};

  } // Gpu
} // Drp
