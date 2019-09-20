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
                   XtcData::Xtc&       xtc) override;
    public:
      static size_t size() { return sizeof(TmoTebData); }
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
                                      XtcData::Xtc&       xtc)
{
  void* buf  = xtc.alloc(sizeof(TmoTebData));
  auto  data = static_cast<TmoTebData*>(buf);

  data->write   = 0xdeadbeef;
  data->monitor = 0x12345678;
}

// The class factory

extern "C" Pds::Trg::TriggerPrimitive* create_producer()
{
  return new Pds::Trg::TmoTebPrimitive;
}
