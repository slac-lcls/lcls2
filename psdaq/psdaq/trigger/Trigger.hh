#ifndef Pds_Trg_Trigger_hh
#define Pds_Trg_Trigger_hh

#include "psdaq/service/Dl.hh"
#include "psdaq/service/json.hpp"
#include "psdaq/eb/eb.hh"               // For MAX_DRPS
#include "psdaq/eb/ResultDgram.hh"
#include "xtcdata/xtc/Dgram.hh"

#include "rapidjson/document.h"

#include <cstdint>
#include <string>

namespace Pds {
  namespace Trg {

    class Trigger
    {
    public:
      virtual ~Trigger() {}
    public:
      virtual int  configure(const nlohmann::json&      connectMsg,
                             const rapidjson::Document& top) = 0;
      virtual void event(const XtcData::Dgram* const* start,
                         const XtcData::Dgram**       end,
                         Pds::Eb::ResultDgram&        result) = 0;
    public:
      static size_t size() { return sizeof(Pds::Eb::ResultDgram); }
    };
  };
};

#endif
