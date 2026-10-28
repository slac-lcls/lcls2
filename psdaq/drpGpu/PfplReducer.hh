#pragma once

#include "ReducerAlgo.hh"

#include <f32_abs_comp_gpu.hh>

namespace Drp {
  namespace Gpu {

class PfplReducer : public ReducerAlgo
{
public:
  PfplReducer(const Parameters& para, const MemPoolGpu& pool, Detector& det);
  virtual ~PfplReducer() {}

  size_t payloadSize() const override { return m_compressor.maxSize(); }
  void   recordGraph(cudaStream_t       stream,
                     unsigned*    const state,
                     unsigned*    const index,
                     float const* const calibBuffers,
                     size_t       const calibBufsCnt,
                     uint8_t*     const dataBuffers,
                     size_t       const dataBufsCnt,
                     unsigned*    const retCode) override;
  int      configure(const nlohmann::json& configureMsg,
                     const nlohmann::json& connectMsg,
                     size_t                collectionId) override { return 0; }
  unsigned configure(XtcData::Xtc&, const void* bufEnd) override;
  void     event    (XtcData::Xtc&, const void* bufEnd, unsigned dataSize) override;
private:
  PFPL::Compressor m_compressor;
};

  } // Gpu
} // Drp
