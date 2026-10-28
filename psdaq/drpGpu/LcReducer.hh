#pragma once

#include "ReducerAlgo.hh"

#include <lc-compressor-QUANT_ABS_0_f32-BIT_4-RZE_1.hh>

namespace Drp {
  namespace Gpu {

class LcReducer : public ReducerAlgo
{
public:
  LcReducer(const Parameters& para, const MemPoolGpu& pool, Detector& det);
  virtual ~LcReducer() {}

  size_t payloadSize() const override { return m_compressor.maxSize(); }
  void   recordGraph(cudaStream_t       stream,
                     unsigned*    const state_d,
                     unsigned*    const index_d,
                     float const* const calibBuffers_d,
                     size_t       const calibBufsCnt,
                     uint8_t*     const dataBuffers_d,
                     size_t       const dataBufsCnt,
                     unsigned*    const retCode) override;
  int      configure(const nlohmann::json& configureMsg,
                     const nlohmann::json& connectMsg,
                     size_t                collectionId) override { return 0; }
  unsigned configure(XtcData::Xtc&, const void* bufEnd) override;
  void     event    (XtcData::Xtc&, const void* bufEnd, unsigned dataSize) override;
private:
  LC_framework::Compressor m_compressor;
};

  } // Gpu
} // Drp
