#include "TriggerPrimitive.hh"
#include "HrEncoderTebData.hh"
#include "utilities.hh"
#include "drp/drp.hh"
#include "xtcdata/xtc/Xtc.hh"
#include "psalg/utils/SysLog.hh"

#include <cstdint>
#include <stdio.h>

using json = nlohmann::json;
using logging = psalg::SysLog;

namespace Pds {
    namespace Trg {
        class HrEncoderTebPrimitive : public TriggerPrimitive {
        public:
            int    configure(const json& configureMsg,
                             const json& connectMsg,
                             size_t      collectionId) override;
            void   event(const Drp::MemPool& pool,
                         uint32_t            idx,
                         const XtcData::Xtc& ctrb,
                         XtcData::Xtc&       xtc,
                         const void*         bufEnd) override;
            size_t size() const  { return sizeof(HrEncoderTebData); }
        };

    };

};


using namespace Pds::Trg;
using namespace XtcData;

int Pds::Trg::HrEncoderTebPrimitive::configure(const json& configureMsg,
                                               const json& connectMsg,
                                               size_t      collectionId)
{
    logging::info("HrEncoderTebPrimitive::configure json");
    return 0;
}

void Pds::Trg::HrEncoderTebPrimitive::event(const Drp::MemPool& pool,
                                            uint32_t            idx,
                                            const Xtc&          ctrb,
                                            Xtc&                xtc,
                                            const void*         bufEnd)
{
    const XtcData::Xtc& shdat = *reinterpret_cast<const XtcData::Xtc*>(ctrb.payload());
    const XtcData::Xtc& sha   = *reinterpret_cast<const XtcData::Xtc*>(shdat.payload());
    const XtcData::Xtc& dat   = *reinterpret_cast<const XtcData::Xtc*>(sha.next());
    new(xtc.alloc(sizeof(HrEncoderTebData), bufEnd)) HrEncoderTebData((uint8_t*)dat.payload());
}

// The class factory

extern "C" Pds::Trg::TriggerPrimitive* create_producer_mono_hrencoder()
{
    return new Pds::Trg::HrEncoderTebPrimitive;
}
