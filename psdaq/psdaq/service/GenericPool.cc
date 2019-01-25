#include "GenericPool.hh"

using namespace Pds;

void GenericPool::dump() const
{
  printf("  bounds %zu  buffer %p  current %zu\n",
	 _bounds, _buffer, _current);
  printf("  sizeofObject %zu  numofObjects %u  allocs %lu  frees %lu\n",
	 sizeofObject(), numberofObjects(),
	 numberofAllocs(), numberofFrees() );
}
