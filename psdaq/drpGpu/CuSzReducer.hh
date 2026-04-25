#pragma once

#include "ReducerAlgo.hh"

#include <cusz/api_v2.h>
#include <cusz/cusz.h>
#include <cusz/cusz/type.h>

namespace Drp {
  namespace Gpu {

class CuSzReducer : public ReducerAlgo
{
public:
  CuSzReducer(const Parameters& para, const MemPoolGpu& pool, Detector& det);
  virtual ~CuSzReducer();

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
  unsigned configure(XtcData::Xtc&, const void* bufEnd) override;
  void     event    (XtcData::Xtc&, const void* bufEnd, unsigned dataSize) override;
private:
  psz_predtype  m_predictor;
  psz_mode      m_mode;
  double        m_eb;
  psz_header    m_header;
  psz_resource* m_m;
};

  } // Gpu
} // Drp
