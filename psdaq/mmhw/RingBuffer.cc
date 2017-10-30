#include "psdaq/mmhw/RingBuffer.hh"

#include <stdio.h>

using namespace Pds::Mmhw;

void RingBuffer::enable(bool v)
{
  unsigned r = _csr;
  if (v)
    r |= (1<<31);
  else
    r &= ~(1<<31);
  _csr = r;
}

void RingBuffer::clear()
{
  unsigned r = _csr;
  _csr = r | (1<<30);
  _csr = r &~(1<<30);
}

void RingBuffer::dump()
{
  unsigned len = _csr & 0xfffff;
#if 0
  //  These memory accesses translate to more than just 32-bit reads
  uint32_t* buff = new uint32_t[len];
  for(unsigned i=0; i<len; i++)
    buff[i] = _dump[i];
  for(unsigned i=0; i<len; i++)
    printf("%08x%c", buff[i], (i&0x7)==0x7 ? '\n':' ');
  delete[] buff;
#else
  for(unsigned i=0; i<len; i++)
    printf("%08x%c", _dump[i], (i&0x7)==0x7 ? '\n':' ');
#endif
}
