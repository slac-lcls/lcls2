#ifndef Pds_Trg_TriggerPrimitive_hh
#define Pds_Trg_TriggerPrimitive_hh

#include "psdaq/service/Dl.hh"
#include "psdaq/service/json.hpp"

#include "rapidjson/document.h"

#include <cstdint>
#include <string>

namespace XtcData {
  class Xtc;
};

namespace Drp {
  class MemPool;
};

namespace Pds {
  namespace Trg {

    class TriggerPrimitive
    {
    public:
      virtual ~TriggerPrimitive() {}
    public:
      virtual int    configure(const rapidjson::Document& top,
                               const nlohmann::json&      connectMsg,
                               size_t                     collectionId) = 0;
      virtual void   event(const Drp::MemPool& pool,
                           uint32_t            index,
                           const XtcData::Xtc& contribution,
                           XtcData::Xtc&       xtc) = 0;
      virtual size_t size() const = 0;
    };
  };
};

#endif
