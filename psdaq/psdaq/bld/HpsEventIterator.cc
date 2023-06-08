#include "HpsEventIterator.hh"
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

using namespace Bld;

//#define DBUG

void HpsEventIterator::_first(size_t sz)
{
  _ts  = _buff[1]; _ts  <<= 32; _ts  += _buff[0];
  _pid = _buff[3]; _pid <<= 32; _pid += _buff[2];
  v.channels  = _buff[4];
  v.beam      = _buff[5];
  v.timeStamp = _ts;
  v.pulseId   = _pid;
  uint32_t* channels = const_cast<uint32_t*>(v.channelData());
  _nch=0;
  unsigned i=6;
  for(unsigned m=_buff[4]; m!=0; m&=(m-1))
    channels[_nch++] = _buff[i++];
  v.sevr   = _buff[i++]; v.sevr <<= 32;  v.sevr |= _buff[i++];
  _next    = _buff+i;

#ifdef DBUG
  printf("First ts 0x%llx  pid 0x%llx  beam %x  nch %u\n",
         v.timeStamp, v.pulseId, v.beam, _nch);
  for(unsigned j=0; j<32; j++)
    printf("%08x%c", _buff[j], (j&7)==7 ? '\n':' ');
#endif
  //
  //  Validate size of packet (sz = sizeof_first + n*sizeof_next)
  //
  unsigned sizeof_first = 4*i;
  unsigned sizeof_next  = sizeof_first - 16;
  if ( ((sz - sizeof_first)%sizeof_next) != 0 ) {
    printf("HpsEventIterator:first() size of packet error : sz %zu, first %u, next %u\n",
           sz, sizeof_first, sizeof_next);
    _valid = false;
  }
  else
    _valid = true;
}

bool HpsEventIterator::next()
{
  if (_next >= _end)
    return false;

  // v.timeStamp = _buff[1]; v.timeStamp <<= 32; v.timeStamp += _buff[0];
  // v.pulseId   = _buff[3]; v.pulseId   <<= 32; v.pulseId   += _buff[2];
  v.timeStamp = _ts;
  v.timeStamp+= ((_next[0]>> 0)&0xfffff);
  v.pulseId   = _pid;
  v.pulseId  += ((_next[0]>>20)&0xfff);
  v.beam      = _next[1];
  memcpy(&v+1, &_next[2], _nch*sizeof(uint32_t));
  v.sevr      = uint64_t(_next[3+_nch])<<32;
  v.sevr     |= _next[2+_nch];
  _next      += 4+_nch;

  return true;
}
