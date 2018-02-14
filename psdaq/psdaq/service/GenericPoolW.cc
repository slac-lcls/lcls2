#include "GenericPoolW.hh"

using namespace Pds;

GenericPoolW::GenericPoolW(size_t sizeofObject, int numberofObjects) :
  GenericPool(sizeofObject, numberofObjects)
{
}

GenericPoolW::~GenericPoolW()
{
}
