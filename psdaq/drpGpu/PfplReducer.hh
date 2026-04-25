#pragma once

#include "ReducerAlgo.hh"

#include <pfpl/f32_abs_comp_gpu.hh>

namespace Drp {
  namespace Gpu {

class PfplReducer : public ReducerAlgo
{
public:
  PfplReducer(const Parameters& para, const MemPoolGpu& pool, Detector& det);
  virtual ~PfplReducer() {}

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
  PFPL::PFPL_Compressor m_compressor;
};

  } // Gpu
} // Drp
