// This header is safe to include in CPU code.

#pragma once

#include "Detector.hh"                  // For NamesIndex enums
#include "drp/drp.hh"                   // For NamesIndex
#include "xtcdata/xtc/ShapesData.hh"    // For Alg
#include "xtcdata/xtc/NamesLookup.hh"
#include <nlohmann/json.hpp>

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
  unsigned retCode;                     // Non-zero if the algorithm reported a problem
};

class ReducerAlgo
{
public:
  ReducerAlgo(const Parameters& para, const MemPoolGpu& pool, Detector& det) : m_para(para), m_pool(pool), m_det(det) {}
  virtual ~ReducerAlgo() {}

  virtual size_t payloadSize() const = 0;
  // retCode is zeroed before each event; an algorithm may set it non-zero to have the
  // recorder count and report a problem with that event.
  virtual void   recordGraph(cudaStream_t       stream,
                             unsigned*    const state,
                             unsigned*    const index,
                             float const* const calibBuffers,
                             size_t       const calibBufsCnt,
                             uint8_t*     const dataBuffers,
                             size_t       const dataBufsCnt,
                             unsigned*    const retCode) = 0;
  virtual int      configure(const nlohmann::json& configureMsg,
                             const nlohmann::json& connectMsg,
                             size_t                collectionId) = 0;                 // Retrieve configDb configuration
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
