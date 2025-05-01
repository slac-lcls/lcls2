#include "tmoTebPrimitive.hh"

#include "utilities.hh"
#include "drp/drp.hh"
#include "xtcdata/xtc/Dgram.hh"

using namespace rapidjson;
using json = nlohmann::json;
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
  // Analyze ctrb to determine TEB input data for the trigger
  // Example dummy input data:
  uint32_t write_   = 0xdeadbeef;
  uint32_t monitor_ = 0x12345678;

  new(xtc.alloc(sizeof(TmoTebData), bufEnd)) TmoTebData(write_, monitor_);
}

void Pds::Trg::TmoTebPrimitive::event(cudaStream_t&     stream,
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
  return new Pds::Trg::TmoTebPrimitive;
}
