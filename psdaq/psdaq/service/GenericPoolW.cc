#include "GenericPoolW.hh"

using namespace Pds;

GenericPoolW::GenericPoolW(size_t sizeofObject, int numberofObjects) :
  GenericPool(sizeofObject, numberofObjects),
  _stopping(false)
{
}

GenericPoolW::~GenericPoolW()
{
}

void GenericPoolW::stop()
{
  std::lock_guard<std::mutex> lk(_mutex);
  _stopping = true;
  _condVar.notify_all();
}
