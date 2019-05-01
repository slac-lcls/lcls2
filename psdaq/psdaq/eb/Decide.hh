#ifndef Pds_Eb_Decide_hh
#define Pds_Eb_Decide_hh

#include "psdaq/service/json.hpp"

#include <cstdint>

namespace XtcData {
  class Dgram;
  class Damage;
};

namespace Pds {
  namespace Eb {

    class Decide
    {
    public:
      Decide() {}
      virtual ~Decide() {}
    public:
      virtual int             configure(const nlohmann::json& msg)           = 0;
      virtual XtcData::Damage configure(const XtcData::Dgram* dgram)         = 0;
      virtual XtcData::Damage event    (const XtcData::Dgram* ctrb,
                                        uint32_t*             result,
                                        size_t                sizeofPayload) = 0;
    };

    // The types of the class factories
    typedef Decide* Create_t();
    typedef void Destroy_t(Decide*);
  };
};

#endif
