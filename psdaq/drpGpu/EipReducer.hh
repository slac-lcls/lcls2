#pragma once

#include "ReducerAlgo.hh"

#include <eip/Compressor.hh>

namespace Drp {
  namespace Gpu {

class EipReducer : public ReducerAlgo
{
public:
  EipReducer(const Parameters& para, const MemPoolGpu& pool, Detector& det);
  virtual ~EipReducer();

  bool   hasGraph()    const override { return false; }
  size_t payloadSize() const override { return m_pool.calibBufsSize(); }
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
  int      configure(const nlohmann::json& configureMsg,
                     const nlohmann::json& connectMsg,
                     size_t                collectionId) override { return 0; }
  unsigned configure(XtcData::Xtc&, const void* bufEnd) override;
  void     event    (XtcData::Xtc&, const void* bufEnd, unsigned dataSize) override;
private:
  EIP::Compressor m_compressor;
};

  } // Gpu
} // Drp
