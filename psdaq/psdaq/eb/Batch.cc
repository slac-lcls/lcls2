#include "Batch.hh"

#include "xtcdata/xtc/Dgram.hh"

using namespace XtcData;
using namespace Pds::Eb;


Batch::Batch() :
  _buffer (nullptr),
  _size   (0),
  _id     (0),
  _appPrms(nullptr),
  _result (nullptr)
{
}

Batch::Batch(void* buffer, size_t bufSize, AppPrm* appPrms) :
  _buffer (buffer),
  _size   (bufSize),
  _id     (0),
  _appPrms(appPrms),
  _result (nullptr)
{
}

void Batch::dump() const
{
  const char* buffer = static_cast<const char*>(_buffer);
  if (buffer)
  {
    printf("Dump of Batch %014lx at index %d (%p)\n", id(), index(), buffer);
    printf("  extent %zd, entry size %zd => # of entries %zd\n",
           extent(), size(), extent() / size());
    printf("  Readout groups remaining: %04x\n", rogs());
    printf("  Receiver ID list: %016lx\n", receivers());
    printf("  Result Dgram: %p\n", result());

    unsigned cnt = 0;
    while (true)
    {
      const Dgram* dg  = reinterpret_cast<const Dgram*>(buffer);
      const char*  svc = TransitionId::name(dg->seq.service());
      unsigned     ctl = dg->seq.pulseId().control();
      uint64_t     pid = dg->seq.pulseId().value();
      size_t       sz  = sizeof(*dg) + dg->xtc.sizeofPayload();
      unsigned     src = dg->xtc.src.value();
      unsigned     env = dg->env;
      uint32_t*    inp = (uint32_t*)dg->xtc.payload();
      printf("  %2d, %15s  dg @ "
             "%16p, ctl %02x, pid %014lx, sz %6zd, src %2d, env %08x, inp [%08x, %08x], appPrm %p\n",
             cnt, svc, dg, ctl, pid, sz, src, env, inp[0], inp[1], retrieve(pid));

      buffer += _size;
      dg      = reinterpret_cast<const Dgram*>(buffer);

      pid = dg->seq.pulseId().value();
      if ((++cnt == MAX_ENTRIES) || !pid)  break;
    }
  }
  else
  {
    printf("Batch %08x contains no datagrams\n", index());
  }
}
