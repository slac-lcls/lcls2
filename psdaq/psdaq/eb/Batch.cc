#include "Batch.hh"

#include "xtcdata/xtc/Dgram.hh"

using namespace XtcData;
using namespace Pds::Eb;


Batch::Batch() :
  _buffer (nullptr),
  _id     (0),
  _dg     (nullptr),
  _appPrms(nullptr),
  _result (nullptr)
{
}

Batch::Batch(void* buffer, AppPrm* appPrms) :
  _buffer (static_cast<Dgram*>(buffer)),
  _id     (0),
  _dg     (nullptr),
  _appPrms(appPrms),
  _result (nullptr)
{
}

void Batch::dump() const
{
  const Dgram* dg = static_cast<Dgram*>(_buffer);
  if (dg)
  {
    printf("Dump of Batch at index %d\n", index());
    while (true)
    {
      const char* svc = TransitionId::name(dg->seq.service());
      unsigned    ctl = dg->seq.pulseId().control();
      uint64_t    pid = dg->seq.pulseId().value();
      unsigned    idx = pid & (MAX_ENTRIES - 1);
      size_t      sz  = sizeof(*dg) + dg->xtc.sizeofPayload();
      unsigned    src = dg->xtc.src.value();
      unsigned    env = dg->env;
      uint32_t*   inp = (uint32_t*)dg->xtc.payload();
      //const Dgram* adg = (const Dgram*)retrieve(pid); // May not be a Dgram
      printf("  %2d, %15s  dg @ "
             "%16p, ctl %02x, pid %014lx, sz %4zd, src %2d, env %08x, inp [%08x, %08x], appPrm %p\n", //, %014lx\n",
             idx, svc, dg, ctl, pid, sz, src, env, inp[0], inp[1], retrieve(pid)); //, adg->seq.pulseId().value());

      // Last event in batch does not have Batch bit set
      if (!dg->seq.isBatch())  break;

      dg = reinterpret_cast<const Dgram*>(dg->xtc.next());
    }
  }
  else
  {
    printf("Batch %08x contains no datagrams\n", index());
  }
}
