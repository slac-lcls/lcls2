#include "Batch.hh"

#include "xtcdata/xtc/Dgram.hh"

using namespace XtcData;
using namespace Pds::Eb;


Batch::Batch(size_t bufSize) :
  _bufSize(bufSize),
  _id     (0),
  _extent (0)
{
}

void Batch::dump() const
{
  const char* buffer = static_cast<const char*>(_buffer);
  if (buffer)
  {
    printf("Dump of Batch %014lx at index %d (%p)\n", _id, index(), buffer);
    printf("  extent %u, entry size %zd => # of entries %zd\n",
           _extent, _bufSize, _extent / _bufSize);

    unsigned cnt = 0;
    while (true)
    {
      const EbDgram* dg  = reinterpret_cast<const EbDgram*>(buffer);
      const char*    svc = TransitionId::name(dg->service());
      unsigned       ctl = dg->control();
      uint64_t       pid = dg->pulseId();
      size_t         sz  = sizeof(*dg) + dg->xtc.sizeofPayload();
      unsigned       src = dg->xtc.src.value();
      unsigned       env = dg->env;
      uint32_t*      inp = (uint32_t*)dg->xtc.payload();
      printf("  %2d, %15s  dg @ "
             "%16p, ctl %02x, pid %014lx, env %08x, sz %6zd, src %2d, inp [%08x, %08x]\n",
             cnt, svc, dg, ctl, pid, env, sz, src, inp[0], inp[1]);

      buffer += _bufSize;
      dg      = reinterpret_cast<const EbDgram*>(buffer);

      pid = dg->pulseId();
      if ((++cnt == MAX_ENTRIES) || !pid)  break;
    }
  }
  else
  {
    printf("Batch %08x contains no datagrams\n", index());
  }
}
