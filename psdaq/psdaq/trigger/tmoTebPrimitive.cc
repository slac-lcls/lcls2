#include "TriggerPrimitive.hh"
#include "TmoTebData.hh"
#include "utilities.hh"
#include "drp/drp.hh"
#include "xtcdata/xtc/Dgram.hh"

#include <cstdint>
#include <stdio.h>

using namespace rapidjson;
using json = nlohmann::json;


namespace Pds {
  namespace Trg {

    class TmoTebPrimitive : public TriggerPrimitive
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
      size_t size() const  { return sizeof(TmoTebData); }
    };
  };
};


using namespace Pds::Trg;

int Pds::Trg::TmoTebPrimitive::configure(const Document& top,
                                         const json&     connectMsg,
                                         size_t          collectionId)
{
  return 0;
}

void Pds::Trg::TmoTebPrimitive::event(const Drp::MemPool& pool,
                                      uint32_t            idx,
                                      const XtcData::Xtc& ctrb,
                                      XtcData::Xtc&       xtc,
                                      const void*         bufEnd)
{
  uint32_t write_   = 0xdeadbeef;
  uint32_t monitor_ = 0x12345678;

  new(xtc.alloc(sizeof(TmoTebData), bufEnd)) TmoTebData(write_, monitor_);
}

// The class factory

extern "C" Pds::Trg::TriggerPrimitive* create_producer()
{
  return new Pds::Trg::TmoTebPrimitive;
}
