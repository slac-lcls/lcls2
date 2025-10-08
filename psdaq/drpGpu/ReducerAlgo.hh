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

class ReducerAlgo
{
public:
  ReducerAlgo(const Parameters& para, const MemPoolGpu& pool, Detector& det) : m_para(para), m_pool(pool), m_det(det) {}
  virtual ~ReducerAlgo() {}

  virtual bool   hasGraph() const = 0;
  virtual size_t payloadSize() const = 0;
  virtual void   recordGraph(cudaStream_t       stream,
                             const unsigned&    index,
                             float const* const calibBuffers,
                             const size_t       calibBufsCnt,
                             uint8_t    * const dataBuffers,
                             const size_t       dataBufsCnt) = 0;
  virtual void     reduce   (cudaGraphExec_t, cudaStream_t, unsigned index, size_t* dataSize) = 0;
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
