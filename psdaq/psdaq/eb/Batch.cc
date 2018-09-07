#include "Batch.hh"

#include "xtcdata/xtc/Dgram.hh"

#include <new>

using namespace XtcData;
using namespace Pds::Eb;


Batch::Batch() :
  _index(0),
  _buffer(nullptr),
  _entries(0),
  _appPrms(nullptr)
{
}

void Batch::_fixup(unsigned index, void* buffer, std::atomic<uintptr_t>* appPrms)
{
  *const_cast<unsigned*>(&_index)  = index;
  *const_cast<void**>   (&_buffer) = buffer;
  _appPrms = appPrms;
}

void Batch::initialize(const Dgram* idg)
{
  Dgram* bdg = new(_buffer) Dgram(*idg);

  bdg->seq = Sequence(idg->seq.stamp(),
                      PulseId(idg->seq.pulseId(), Sequence::IsBatch));
  _entries = 0;
}
