#pragma once

#include "ReducerAlgo.hh"

#include <lc/lc-compressor-QUANT_ABS_0_f32-BIT_4-RZE_1.hh>

namespace Drp {
  namespace Gpu {

class LcReducer : public ReducerAlgo
{
public:
  LcReducer(const Parameters& para, const MemPoolGpu& pool, Detector& det);
  virtual ~LcReducer() {}

  bool   hasGraph() const override { return true; }
  size_t payloadSize() const override { return m_compressor.maxSize(); }
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
  LC_framework::LC_Compressor m_compressor;
};

  } // Gpu
} // Drp
