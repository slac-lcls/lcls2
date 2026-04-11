// This header is safe to include in CPU code.

#pragma once

#include "Detector.hh"                  // For NamesIndex enums
#include "drp/drp.hh"                   // For NamesIndex
#include "xtcdata/xtc/ShapesData.hh"    // For Alg
#include "xtcdata/xtc/NamesLookup.hh"

#include <cuda_runtime.h>

namespace XtcData {
  class Xtc;
} // XtcData

namespace Drp {
  struct Parameters;

  namespace Gpu {
    class MemPoolGpu;
    template <typename T> class RingQueueHtoD;
    template <typename T> class RingQueueDtoH;

struct ReducerTuple
{
  unsigned index;
  size_t   dataSize;
};

class ReducerAlgo
{
public:
  ReducerAlgo(const Parameters& para, const MemPoolGpu& pool, Detector& det) : m_para(para), m_pool(pool), m_det(det) {}
  virtual ~ReducerAlgo() {}

  virtual bool   hasGraph()    const = 0;
  virtual size_t payloadSize() const = 0;
  virtual void   recordGraph(cudaStream_t                       stream,
                             unsigned*                    const index,
                             RingQueueHtoD<unsigned>*     const inputQueue,
                             float const*                 const calibBuffers,
                             size_t                       const calibBufsCnt,
                             uint8_t*                     const dataBuffers,
                             size_t                       const dataBufsCnt,
                             RingQueueDtoH<ReducerTuple>* const outputQueue,
                             uint64_t*                    const state_d,
                             unsigned*                    const done) = 0;
  virtual void     reduce   (cudaGraphExec_t,
                             cudaStream_t,
                             unsigned  index,
                             size_t*   dataSize,
                             unsigned* error) = 0;
  virtual unsigned configure(XtcData::Xtc&, const void* bufEnd) = 0;                  // attach descriptions to xtc
  virtual void     event    (XtcData::Xtc&, const void* bufEnd, unsigned dataSize) {} // fill xtc data description
protected:
  const Parameters& m_para;
  const MemPoolGpu& m_pool;
  Detector&         m_det;
};

  } // Gpu
} // Drp


extern "C"
{
  typedef Drp::Gpu::ReducerAlgo* reducerAlgoFactoryFn_t(const Drp::Parameters&,
                                                        const Drp::Gpu::MemPoolGpu&,
                                                        Drp::Gpu::Detector&);

  Drp::Gpu::ReducerAlgo* createReducer(const Drp::Parameters&,
                                       const Drp::Gpu::MemPoolGpu&,
                                       Drp::Gpu::Detector&);
}
