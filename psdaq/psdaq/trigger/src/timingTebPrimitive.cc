#include "TriggerPrimitive.hh"
#include "TimingTebData.hh"
#include "utilities.hh"
#include "drp/drp.hh"
#include "drp/TimingDef.hh"
#include "xtcdata/xtc/Dgram.hh"

#include <cstdint>
#include <stdio.h>

using json = nlohmann::json;

namespace Pds {
  namespace Trg {

    class TimingTebPrimitive : public TriggerPrimitive
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
      size_t size() const  { return sizeof(TimingTebData); }
    };
  };
};


using namespace Pds::Trg;

int Pds::Trg::TimingTebPrimitive::configure(const json& configureMsg,
                                            const json& connectMsg,
                                            size_t      collectionId)
{
    return 0;
}

void Pds::Trg::TimingTebPrimitive::event(const Drp::MemPool& pool,
                                         uint32_t            idx,
                                         const XtcData::Xtc& ctrb,
                                         XtcData::Xtc&       xtc,
                                         const void*         bufEnd)
{
    //  Step through the xtc: shapesdata->data
    const XtcData::Xtc& shdat = *reinterpret_cast<const XtcData::Xtc*>(ctrb.payload());
    const XtcData::Xtc& dat = *reinterpret_cast<const XtcData::Xtc*>(shdat.payload());
    const Drp::TimingData& tdat = *new(dat.payload()) Drp::TimingData;

    uint8_t ebeamDestn_ = tdat.ebeamDestn | (tdat.ebeamPresent ? 0x80 : 0);
    const uint32_t* eventcodes_ = reinterpret_cast<const uint32_t*>(tdat.sequenceValues);

    new(xtc.alloc(sizeof(TimingTebData), bufEnd)) TimingTebData(ebeamDestn_,
                                                                eventcodes_);
}

// The class factory

extern "C" Pds::Trg::TriggerPrimitive* create_producer_timing()
{
    return new Pds::Trg::TimingTebPrimitive;
}
