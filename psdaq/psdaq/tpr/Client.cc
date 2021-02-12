#include "Client.hh"
#include "Frame.hh"
#include "Module.hh"
#include "Queues.hh"

#include <sys/mman.h>
#include <fcntl.h>

using namespace Pds::Tpr;

Client::Client(const char* devname,
               unsigned    channel,
               bool        lcls2) :
  _channel(channel)
{
  _fd = ::open(devname, O_RDWR);
  if (_fd<0) {
    perror("Could not open tpr");
    throw std::string("Could not open tpr");
  }

  void* ptr = mmap(0, sizeof(Pds::Tpr::TprReg), PROT_READ|PROT_WRITE, MAP_SHARED, _fd, 0);
  if (ptr == MAP_FAILED) {
    perror("Failed to map master");
    throw std::string("Failed to map master");
  }

  _dev = reinterpret_cast<Pds::Tpr::TprReg*>(ptr);

  if (_dev->tpr.clkSel() != lcls2) {
    _dev->tpr.clkSel(lcls2);
    printf("Changed clk xbar to %s mode\n", lcls2 ? "LCLS2":"LCLS1");
  }

  _dev->xbar.setTpr( XBar::StraightIn );
  _dev->xbar.setTpr( XBar::StraightOut);
  
  { Pds::Tpr::RingB& ring = _dev->ring0;
    ring.clear();
    ring.enable(true);
    usleep(1000);
    ring.enable(false);
    ring.dumpFrames(); }

  char dev[16];
  sprintf(dev,"%s%x",devname,_channel);
  
  _fdsh = open(dev, O_RDWR);
  if (_fdsh < 0) {
    perror("Could not open sh");
    throw std::string("Could not open sh");
  }

  ptr = mmap(0, sizeof(Pds::Tpr::Queues), PROT_READ, MAP_SHARED, _fdsh, 0);
  if (ptr == MAP_FAILED) {
    perror("Failed to map slave");
    throw std::string("Failed to map slave");
  }

  _queues = reinterpret_cast<Pds::Tpr::Queues*>(ptr);

}

Client::~Client() { release(); }

void Client::release() { 
  if (_fd>=0) {
    close(_fd); 
    munmap(_dev,sizeof(Pds::Tpr::TprReg));
  }
  _fd=-1; 
}

void Client::_dump() const 
{
  printf("TprBase::channel[%u] evtsel %08x control %x\n",
         _channel,
         _dev->base.channel[_channel].evtSel,
         _dev->base.channel[_channel].control);
}

//  Setup the trigger channel
void Client::setup(unsigned output, unsigned delay, unsigned width, unsigned polarity)
{
  _dev->base.setupTrigger(output, _channel, polarity, delay, width, 0);
}

  //  Enable the trigger
void Client::start(TprBase::Partition partn)
{
  _dump();
  _dev->base.setupChannel(_channel, (TprBase::Partition)partn, 0, 0, 1);
  _dump();

  char buff[32];
  read(_fdsh, &buff, 32 );
  _rp = _queues->allwp[_channel];
}

//  Enable the trigger
void Client::start(TprBase::FixedRate rate)
{
  _dump();
  _dev->base.setupChannel(_channel, TprBase::Any, rate, 0, 0, 1);
  _dump();

  char buff[32];
  read(_fdsh, &buff, 32 );
  _rp = _queues->allwp[_channel];
}

//  Enable the trigger
void Client::start(TprBase::ACRate rate, unsigned timeSlotMask)
{
  _dump();
  _dev->base.setupChannel(_channel, TprBase::Any, rate, timeSlotMask, 0, 0, 1);
  _dump();

  char buff[32];
  read(_fdsh, &buff, 32 );
  _rp = _queues->allwp[_channel];
}

//  Enable the trigger
void Client::start(TprBase::EventCode evcode)
{
  _dump();
  _dev->base.setupChannel(_channel, evcode, 0, 0, 1);
  _dump();

  char buff[32];
  read(_fdsh, &buff, 32 );
  _rp = _queues->allwp[_channel];
}

//  Disable the trigger
void Client::stop()
{
  _dev->base.channel[_channel].control = 0;
}

  //
  //  Seek timing frame by pulseId
  //  Return address on success or 0
  //
const Pds::Tpr::Frame* Client::advance(uint64_t pulseId)
{
  char buff[32];
  const Pds::Tpr::Queues& q = *_queues;
  const Pds::Tpr::Frame* f=0;
  while(1) {
    while (_rp < q.allwp[_channel]) {
      f = reinterpret_cast<const Pds::Tpr::Frame*>(&q.allq [ q.allrp[_channel].idx[_rp &(MAX_TPR_ALLQ-1)] & (MAX_TPR_ALLQ-1)]);
      if (f->pulseId >  pulseId) return 0;
      _rp++;
      if (f->pulseId == pulseId) return f;
    }
    read(_fdsh, buff, 32);
  }
  return 0;
}

const Pds::Tpr::Frame* Client::advance()
{
  char buff[32];
  const Pds::Tpr::Queues& q = *_queues;
  const Pds::Tpr::Frame* f=0;
  while(1) {
    while (_rp < q.allwp[_channel]) {
      f = reinterpret_cast<const Pds::Tpr::Frame*>(&q.allq [ q.allrp[_channel].idx[_rp &(MAX_TPR_ALLQ-1)] & (MAX_TPR_ALLQ-1)]);
      _rp++;
      return f;
    }
    read(_fdsh, buff, 32);
  }
  return 0;
}
