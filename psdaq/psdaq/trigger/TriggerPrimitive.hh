#ifndef Pds_Trg_TriggerPrimitive_hh
#define Pds_Trg_TriggerPrimitive_hh

#include "psdaq/service/Dl.hh"
#include <nlohmann/json.hpp>

#include <cstdint>
#include <string>

struct CUstream_st;
typedef struct CUstream_st* cudaStream_t;

namespace XtcData {
  class Xtc;
}

namespace Drp {
  class MemPool;
}

namespace Pds {
  namespace Trg {

    class TriggerPrimitive
    {
    public:
      virtual ~TriggerPrimitive() {}
    public:
      virtual int    configure(const nlohmann::json& configureMsg,
                               const nlohmann::json& connectMsg,
                               size_t                collectionId) = 0;
      virtual void   event(const Drp::MemPool& pool,
                           uint32_t            index,
                           const XtcData::Xtc& contribution,
                           XtcData::Xtc&       xtc,
                           const void*         bufEnd) = 0;
      virtual void   event(cudaStream_t           stream,
                           float     const* const calibBuffers,
                           const size_t           calibBufsCnt,
                           uint32_t* const* const out,
                           const size_t           outBufsCnt,
                           const unsigned&        index,
                           const unsigned         nPanels) = 0;
      virtual size_t size() const = 0;
    };
  }
}

#endif
