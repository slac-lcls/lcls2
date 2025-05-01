#include "TriggerPrimitive.hh"
#include "utilities.hh"
#include "drp/drp.hh"

#include <cstdint>

using namespace rapidjson;
using json = nlohmann::json;


namespace Pds {
  namespace Trg {

    class CalibPrimitive : public TriggerPrimitive
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
      void   event(cudaStream_t&     stream,
                   float*            calibBuffers,
                   uint32_t** const* out,
                   unsigned&         index,
                   bool&             done) override;
      size_t size() const  { return 0; }
    };
  };
};


using namespace Pds::Trg;

int Pds::Trg::CalibPrimitive::configure(const Document& top,
                                        const json&     connectMsg,
                                        size_t          collectionId)
{
  return 0;
}

void Pds::Trg::CalibPrimitive::event(const Drp::MemPool& pool,
                                     uint32_t            idx,
                                     const XtcData::Xtc& ctrb,
                                     XtcData::Xtc&       xtc,
                                     const void*         bufEnd)
{
  // Nothing to do as TEB will accept all data
}

void Pds::Trg::CalibPrimitive::event(cudaStream_t&     stream,
                                     float*            calibBuffers,
                                     uint32_t** const* out,
                                     unsigned&         index,
                                     bool&             done)
{
  assert(false);  // Unused but must be defined
}

// The class factory

extern "C" Pds::Trg::TriggerPrimitive* create_producer()
{
  return new Pds::Trg::CalibPrimitive;
}
