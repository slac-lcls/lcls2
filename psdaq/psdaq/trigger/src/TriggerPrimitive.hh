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
      virtual void   configure(const XtcData::Xtc& xtc, const void* bufEnd) {}
      virtual void   event(const Drp::MemPool& pool,
                           uint32_t            index,
                           const XtcData::Xtc& contribution,
                           XtcData::Xtc&       xtc,
                           const void*         bufEnd) = 0;
      // GPU overload: enqueue a kernel that advances *state from 1 to 2, even when it
      // produces no TEB input, or TrgInpGen stalls.  Return non-zero on failure.
      virtual int    event(cudaStream_t           stream,
                           unsigned* const        state,
                           float     const* const calibBuffers,
                           size_t    const        calibBufsCnt,
                           uint32_t* const        outBuffers,
                           size_t    const        outBufsCnt,
                           unsigned  const* const index,
                           unsigned* const        retCode_d) { return -1; }
      virtual size_t size() const = 0;
    };
  }
}

#endif
