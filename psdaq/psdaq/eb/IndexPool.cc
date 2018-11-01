#include "IndexPool.hh"

using namespace Pds::Eb;

void IndexPoolBase::dump() const
{
  printf("  bounds %zu  buffer %p  mask 0x%0x\n",
	 size(), buffer(), _mask);
  printf("  sizeofObject %zd  numofObjects %zd  allocs %u  frees %u\n",
	 sizeofObject(), numberofObjects(),
	 numberofAllocs(), numberofFrees() );
}
