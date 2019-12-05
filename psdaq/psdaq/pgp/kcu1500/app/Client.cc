#include "Client.hh"
#include "DataDriver.h"
#include "psdaq/service/EbDgram.hh"
using XtcData::Transition;

using namespace Pds::Kcu;

Client::Client(const char* dev) : 
  _dmaBuffers(0), _current(0), _ret(0), 
  _retry_intv(1000), _retry_num(10), // 10 ms
  _next    (0),
  _rxFlags (new uint32_t[max_ret_cnt]),
  _dest    (new uint32_t[max_ret_cnt]),
  _dmaIndex(new uint32_t[max_ret_cnt]),
  _dmaRet  (new int32_t [max_ret_cnt]),
  _skips       (0),
  _retries     (0)
{
  uint8_t*      mask    = new uint8_t [DMA_MASK_SIZE];
  dmaInitMaskBytes(mask);
  dmaAddMaskBytes((uint8_t*)mask,dmaDest(0,0));  // add lane 0

  _fd = ::open(dev, O_RDWR);
  if (_fd<0) {
    printf("Error opening %s\n",dev);
    exit(1);
  }

  uint32_t      dmaCount;
  uint32_t      dmaSize;
  if ( (_dmaBuffers = dmaMapDma(_fd,&dmaCount,&dmaSize)) == NULL ) {
    perror("Failed to map dma buffers!");
    exit(1);
  }

  printf("Found %u buffers of size %u\n", dmaCount, dmaSize);
  for(unsigned i=0; i<4; i++)
    printf("%u : %p : %08x\n", i, _dmaBuffers[i],
           *reinterpret_cast<uint32_t*>(_dmaBuffers[i]));

  dmaSetMaskBytes(_fd,mask);

  delete[] mask;
}

Client::~Client() { stop(); close(_fd); }

void Client::set_retry(unsigned intv_us,
                       unsigned retries) 
{
  _retry_intv = intv_us;
  _retry_num  = retries;
}

  //  Enable the trigger
void Client::start(unsigned group)
{
  //  Configure the simulated camera
  unsigned links = 1;
  unsigned length = 1;
  bool     frameRst = true;

  if (frameRst) {
    unsigned v; dmaReadRegister(_fd,0x00a00000,&v);
    unsigned w = v;
    w &= ~(0xf<<28);    // disable and drain
    dmaWriteRegister(_fd,0x00a00000,w);
    usleep(1000);
    w |=  (1<<3);       // reset
    dmaWriteRegister(_fd, 0x00a00000,w);
    usleep(1);         
    dmaWriteRegister(_fd, 0x00a00000,v);
  }

  unsigned v = ((group&0xf)<<0) |
    ((length&0xffffff)<<4) |
    (links<<28);
  dmaWriteRegister(_fd,0x00a00000, v);
  unsigned w;
  dmaReadRegister(_fd,0x00a00000,&w);
  printf("Configured group [%u], length [%u], links [%x]: [%x](%x)\n",
         group, length, links, v, w);
  for(unsigned i=0; i<4; i++)
    if (links&(1<<i))
      dmaWriteRegister(_fd,0x00800084+32*i, 0x1f00);

  _skips = 0;
}

//  Disable the trigger
void Client::stop()
{
  unsigned v;
  dmaReadRegister(_fd,0x00a00000,&v);
  v &= ~(0xf<<28);
  dmaWriteRegister(_fd,0x00a00000, v);
}

  //
  //  Seek timing frame by pulseId
  //  Return address on success or 0
  //
const Transition* Client::advance(uint64_t pulseId)
{
  const Transition* result = 0;

  //
  //  Fast forward to _next pulseId
  //
  if (pulseId < _next) {
    _skips++;
    return 0;
  }

  uint32_t getCnt = max_ret_cnt;
  while(1) {
    //
    //  Check if retrieved buffers are exhausted
    //
    if (_current == _ret) {
      _current = 0; 
      unsigned retries = 0;
      //
      //  Attempt to retrieve more buffers
      //
      while( (_ret = dmaReadBulkIndex(_fd, getCnt, _dmaRet, _dmaIndex, _rxFlags, NULL, _dest)) == 0 ) {
        usleep(_retry_intv);
        if (retries++ < _retry_num)
          continue;
        else {
          //
          //  Timeout : calculate _next pulseId to fast forward
          //
          // _next = pulseId + (_retry_num*_retry_intv)>>1;
          _next = pulseId + (_retry_num*_retry_intv);
          _retries += retries;
          return 0;
        }
      }
      _retries += retries;
    }
  
    int x = _current;
    int last = _dmaRet[x];
    if (last > 0) {
      const Pds::EbDgram* b = reinterpret_cast<const Pds::EbDgram*>
        (_dmaBuffers[_dmaIndex[x]]);
      if (b->pulseId() == pulseId) {
        _current++;
        result = b;
        break;
      }
      if (b->pulseId() > pulseId) {
        _next = b->pulseId();
        break;
      }
    }

    if (++_current == _ret)  //  Return exhausted buffers
      dmaRetIndexes(_fd,_ret,_dmaIndex);
  }

  if (result) { //  Cache result; buffer will be returned to DMA engine
    _tr = *result;
    result = &_tr;
  }

  if (_current == _ret)  //  Return exhausted buffers
    dmaRetIndexes(_fd,_ret,_dmaIndex);

  return result;
}

void Client::dump()
{
  printf("skips: %u  retries %u\n",_skips,_retries);
  _skips = _retries = 0;
}
