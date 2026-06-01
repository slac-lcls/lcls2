#include "TriggerPrimitive.hh"
#include "TripperTebData.hh"

#include "drp/TimingDef.hh"
#include "drp/drp.hh"

#include "xtcdata/xtc/Damage.hh"
#include "xtcdata/xtc/Dgram.hh"
#include "xtcdata/xtc/ShapesData.hh"
#include "xtcdata/xtc/ConfigIter.hh"

#include "nlohmann/json.hpp"

#include <cstdint>

using json = nlohmann::json;


namespace Pds {
  namespace Trg {

    class MfxTripperPrimitive_timing : public TriggerPrimitive
    {
    public:
        int    configure(const json& configureMsg,
                         const json& connectMsg,
                         size_t      collectionId) override;
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

int Pds::Trg::MfxTripperPrimitive_timing::configure(const json& configureMsg,
                                                    const json& connectMsg,
                                                    size_t      collectionId)
{
    return 0;
}

void Pds::Trg::MfxTripperPrimitive_timing::event(const Drp::MemPool& pool,
                                                 uint32_t            idx,
                                                 const XtcData::Xtc& ctrb,
                                                 XtcData::Xtc&       xtc,
                                                 const void*         bufEnd)
{
    XtcData::Xtc& shapesData = *reinterpret_cast<XtcData::Xtc*>(ctrb.payload());
    XtcData::Xtc& data = *reinterpret_cast<XtcData::Xtc*>(shapesData.payload());

    Drp::TimingData& timingData = *new (data.payload()) Drp::TimingData;

    uint16_t* seqInfo = reinterpret_cast<uint16_t*>(timingData.sequenceValues);

    new (xtc.alloc(sizeof(TripperTebData), bufEnd)) TripperTebData(seqInfo, "timing");
}

// The class factory

extern "C" Pds::Trg::TriggerPrimitive* create_producer_timing()
{
    return new Pds::Trg::MfxTripperPrimitive_timing;
}
