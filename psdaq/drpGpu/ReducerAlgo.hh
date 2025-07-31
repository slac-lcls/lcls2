#pragma once

#include "drp/drp.hh"                   // For NamesIndex
#include "xtcdata/xtc/ShapesData.hh"    // For Alg
#include "xtcdata/xtc/NamesLookup.hh"

// @todo: Redefined? class cudaStream_t;

namespace XtcData {
  class Xtc;
} // XtcData

namespace Drp {
  struct Parameters;

  namespace Gpu {
    class MemPoolGpu;
    class Detector;

    // @todo: Revisit: Must match detector definition
    enum { MaxPnlsPerNode = 10 };       // From BEBDetector.hh
    enum { ConfigNamesIndex = Drp::NamesIndex::BASE,
           EventNamesIndex  = unsigned(ConfigNamesIndex) + unsigned(MaxPnlsPerNode),
           FexNamesIndex    = unsigned(EventNamesIndex)  + unsigned(MaxPnlsPerNode),
           ReducerNamesIndex };         // index for xtc NamesId

class ReducerAlgo
{
public:
  ReducerAlgo(const Parameters& para, const MemPoolGpu& pool, Detector& det) : m_para(para), m_pool(pool), m_det(det) {}
  virtual ~ReducerAlgo() {}

  virtual void recordGraph(cudaStream_t&   stream,
                           const unsigned& index,
                           float**   const calibBuffer,
                           uint8_t** const dataBuffer,
                           unsigned*       extent) = 0;
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
