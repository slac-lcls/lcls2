#include "Batch.hh"

#include "xtcdata/xtc/Dgram.hh"

using namespace XtcData;
using namespace Pds::Eb;


Batch::Batch() :
  _buffer (nullptr),
  _index  (0),
  _entries(0),
  _dg     (nullptr),
  _appPrms(nullptr)
{
}

Batch::Batch(unsigned index, void* buffer, AppPrm* appPrms) :
  _buffer (static_cast<Dgram*>(buffer)),
  _index  (index),
  _entries(0),
  _dg     (nullptr),
  _appPrms(appPrms)
{
}
