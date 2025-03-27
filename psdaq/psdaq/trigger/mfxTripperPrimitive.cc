#include "TriggerPrimitive.hh"
#include "TripperTebData.hh"
#include "utilities.hh"
#include "drp/drp.hh"
#include "xtcdata/xtc/Dgram.hh"
#include "xtcdata/xtc/Damage.hh"

#include <cstdint>
#include <stdio.h>

using namespace rapidjson;
using json = nlohmann::json;


namespace Pds {
  namespace Trg {

    class MfxTripperPrimitive : public TriggerPrimitive
    {
    public:
        int    configure(const Document& top,
                         const json&     connectMsg,
                         size_t          collectionId) override;
        void   event(const Drp::MemPool& pool,
                     uint32_t            idx,
                     const XtcData::Xtc& ctrb,
                     XtcData::Xtc&       xtc,
                     const void*         bufEnd) override;
        size_t size() const  { return sizeof(TripperTebData); }
    };
  };
};


using namespace Pds::Trg;

int Pds::Trg::MfxTripperPrimitive::configure(const Document& top,
                                             const json&     connectMsg,
                                             size_t          collectionId)
{
    return 0;
}

void Pds::Trg::MfxTripperPrimitive::event(const Drp::MemPool& pool,
                                          uint32_t            idx,
                                          const XtcData::Xtc& ctrb,
                                          XtcData::Xtc&       xtc,
                                          const void*         bufEnd)
{
    new (xtc.alloc(sizeof(TripperTebData), bufEnd)) TripperTebData(0, 0, "NOTRIP");
}

// The class factory

extern "C" Pds::Trg::TriggerPrimitive* create_producer()
{
    return new Pds::Trg::MfxTripperPrimitive;
}
