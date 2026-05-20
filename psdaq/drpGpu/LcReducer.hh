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

  bool   hasGraph()    const override { return true; }
  size_t payloadSize() const override { return m_compressor.maxSize(); }
  void   recordGraph(cudaStream_t       stream,
                     unsigned*    const state_d,
                     unsigned*    const index_d,
                     float const* const calibBuffers_d,
                     size_t       const calibBufsCnt,
                     uint8_t*     const dataBuffers_d,
                     size_t       const dataBufsCnt) override;
  void     reduce   (cudaGraphExec_t,
                     cudaStream_t,
                     unsigned  index,
                     size_t*   dataSize,
                     unsigned* retCode) override;
  unsigned configure(XtcData::Xtc&, const void* bufEnd) override;
  void     event    (XtcData::Xtc&, const void* bufEnd, unsigned dataSize) override;
private:
  LC_framework::Compressor m_compressor;
};

  } // Gpu
} // Drp
