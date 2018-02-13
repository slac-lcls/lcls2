#include "psdaq/cphw/RingBuffer.hh"

#include <unistd.h>
#include <stdio.h>

using namespace Pds::Cphw;

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
  usleep(1);
  _csr = r &~(1<<30);
}

void RingBuffer::dump(unsigned dataWidth)
{
  unsigned mask = dataWidth < 32 ? (1<<dataWidth)-1 : 0xffffffff;
  unsigned cmask = (dataWidth+3)/4;
  unsigned len = _csr & 0xfffff;
  if (len > 512) len=256;

  uint32_t* buff = new uint32_t[len];
  for(unsigned i=0; i<len; i++)
    buff[i] = _dump[i]&mask;

  printf("csr %08x  mask 0x%x  cmask %u  dataWidth %u\n", 
         unsigned(_csr), mask, cmask, dataWidth);
  for(unsigned i=0; i<len; i++)
    printf("%0*x%c", cmask, buff[i], (i&0x7)==0x7 ? '\n':' ');

  delete[] buff;
}
