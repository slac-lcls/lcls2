#include "Client.hh"
#include "Frame.hh"
#include "Module.hh"
#include "Queues.hh"

#include <sys/mman.h>
#include <fcntl.h>

using namespace Pds::Tpr;

Client::Client(const char* devname)
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

  _dev->xbar.outMap[2]=0;
  _dev->xbar.outMap[3]=1;

  { Pds::Tpr::RingB& ring = _dev->ring0;
    ring.clear();
    ring.enable(true);
    usleep(1000);
    ring.enable(false);
    ring.dumpFrames(); }

  char dev[16];
  sprintf(dev,"%s%c",devname,'0');
  
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

Client::~Client() { close(_fd); }

  //  Enable the trigger
void Client::start(unsigned partn)
{
  printf("TprBase::channel[0] evtsel %08x control %x\n",
         _dev->base.channel[0].evtSel,
         _dev->base.channel[0].control);
  //  _dev->base.setupDaq(0,partn);
  _dev->base.setupChannel(0, TprBase::Any, (TprBase::FixedRate)partn, 0, 0, 0);
  printf("TprBase::channel[0] evtsel %08x control %x\n",
         _dev->base.channel[0].evtSel,
         _dev->base.channel[0].control);
  char buff[32];
  read(_fdsh, &buff, 32 );
  _rp = _queues->allwp[0];
}

//  Enable the trigger
void Client::start(TprBase::FixedRate rate)
{
  printf("TprBase::channel[0] evtsel %08x control %x\n",
         _dev->base.channel[0].evtSel,
         _dev->base.channel[0].control);
  _dev->base.setupChannel(0, TprBase::Any, rate, 0, 0, 1);
  printf("TprBase::channel[0] evtsel %08x control %x\n",
         _dev->base.channel[0].evtSel,
         _dev->base.channel[0].control);
  char buff[32];
  read(_fdsh, &buff, 32 );
  _rp = _queues->allwp[0];
}

//  Disable the trigger
void Client::stop()
{
  _dev->base.channel[0].control = 0;
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
    while (_rp < q.allwp[0]) {
      f = reinterpret_cast<const Pds::Tpr::Frame*>(&q.allq [ q.allrp[0].idx[_rp &(MAX_TPR_ALLQ-1)] & (MAX_TPR_ALLQ-1)]);
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
    while (_rp < q.allwp[0]) {
      f = reinterpret_cast<const Pds::Tpr::Frame*>(&q.allq [ q.allrp[0].idx[_rp &(MAX_TPR_ALLQ-1)] & (MAX_TPR_ALLQ-1)]);
      _rp++;
      return f;
    }
    read(_fdsh, buff, 32);
  }
  return 0;
}
