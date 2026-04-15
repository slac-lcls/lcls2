#pragma once

#include "ReducerAlgo_gpu.hh"

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
  void   recordGraph(cudaStream_t                        stream,
                     unsigned*                    const  index,
                     RingQueueHtoD<unsigned>*     const  inputQueue,
                     float const*                 const  calibBuffers,
                     size_t                       const  calibBufsCnt,
                     uint8_t*                     const  dataBuffers,
                     size_t                       const  dataBufsCnt,
                     RingQueueDtoH<ReducerTuple>* const  outputQueue,
                     uint64_t*                    const  state_d,
                     cuda::std::atomic<unsigned>  const& terminate_d) override;
  void     reduce   (cudaGraphExec_t,
                     cudaStream_t,
                     unsigned  index,
                     size_t*   dataSize,
                     unsigned* error) override;
  unsigned configure(XtcData::Xtc&, const void* bufEnd) override;
  void     event    (XtcData::Xtc&, const void* bufEnd, unsigned dataSize) override;
private:
  LC_framework::LC_Compressor m_compressor;
  unsigned*                   m_error_d;
};

  } // Gpu
} // Drp
