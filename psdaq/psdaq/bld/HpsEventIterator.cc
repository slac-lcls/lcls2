#include "HpsEventIterator.hh"
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>

using namespace Bld;

void HpsEventIterator::_first(size_t sz)
{
  v.timeStamp = _buff[1]; v.timeStamp <<= 32; v.timeStamp += _buff[0];
  v.pulseId   = _buff[3]; v.pulseId   <<= 32; v.pulseId   += _buff[2];
  v.mask      = _buff[4];
  v.beam      = _buff[5];
  unsigned i=6;
  for(unsigned m=v.mask; m!=0; m&=(m-1))
    v.channels.push_back(_buff[i++]);
  v.valid     = _buff[i++];
  _next    = _buff+i;

  //
  //  Validate size of packet (sz = sizeof_first + n*sizeof_next)
  //
  unsigned sizeof_first = 4*i;
  unsigned sizeof_next  = 4*(i-4);
  if ( ((sz - sizeof_first)%sizeof_next) != 0 ) {
    printf("BldEventIterator:first() size of packet error : sz %zu, first %u, next %u\n",
           sz, sizeof_first, sizeof_next);
    abort();
  }
}

bool HpsEventIterator::next()
{
  if (_next >= _end)
    return false;

  v.timeStamp = _buff[1]; v.timeStamp <<= 32; v.timeStamp += _buff[0];
  v.pulseId   = _buff[3]; v.pulseId   <<= 32; v.pulseId   += _buff[2];
  v.timeStamp+= ((_next[0]>> 0)&0xfffff);
  v.pulseId  += ((_next[0]>>20)&0xfff);
  v.mask      = _buff[4];
  v.beam      = _next[1];
  unsigned i=2;
  for(unsigned m=v.mask; m!=0; m&=(m-1),i++)
    v.channels[i-2] = _next[i];
  v.valid     = _next[i++];

  _next   += i;

  return true;
}
