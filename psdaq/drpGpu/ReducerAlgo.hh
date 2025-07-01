#pragma once

//#include "drp.hh"                       // For NamesIndex
#include "xtcdata/xtc/ShapesData.hh"    // For Alg

// @todo: Redefined? class cudaStream_t;

namespace XtcData {
  class Xtc;
  class ConfigIter;
} // XtcData

namespace Drp {
  struct Parameters;

  namespace Gpu {
    class MemPoolGpu;

#if 0 // @todo: Revisit
    enum { MaxSegsPerNode = 10 };
    enum {ConfigNamesIndex = Drp::NamesIndex::BASE,
          EventNamesIndex  = unsigned(ConfigNamesIndex) + unsigned(MaxSegsPerNode),
          UpdateNamesIndex = unsigned(EventNamesIndex)  + unsigned(MaxSegsPerNode) }; // index for xtc NamesId
#endif

class ReducerAlgo
{
public:
  ReducerAlgo(const Parameters& para, const MemPoolGpu& pool, const XtcData::Alg& alg) : m_para(para), m_pool(pool), m_alg(alg) {}
  virtual ~ReducerAlgo() {}

  virtual void recordGraph(cudaStream_t&   stream,
                           const unsigned& index,
                           float**   const calibBuffer,
                           uint8_t** const dataBuffer,
                           unsigned*       extent) = 0;
#if 0 // @todo: Revisit
  virtual unsigned configure(XtcData::Xtc&, const void* bufEnd, XtcData::ConfigIter&) = 0; // attach descriptions to xtc
  virtual void     event    (XtcData::Xtc&, const void* bufEnd) {}   // fill xtc data description
#endif
protected:
  const Parameters&  m_para;
  const MemPoolGpu&  m_pool;
  const XtcData::Alg m_alg;
};

  } // Gpu
} // Drp


extern "C"
{
  typedef Drp::Gpu::ReducerAlgo* reducerAlgoFactoryFn_t(const Drp::Parameters&, const Drp::Gpu::MemPoolGpu&);

  Drp::Gpu::ReducerAlgo* createReducer(const Drp::Parameters& para, const Drp::Gpu::MemPoolGpu& pool);
}
