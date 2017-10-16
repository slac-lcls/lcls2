#include "psdaq/cphw/Reg.hh"

#include <cpsw_api_builder.h>
#include <cpsw_mmio_dev.h>

//#define DBUG

static ScalVal _reg;

using namespace Pds::Cphw;

void Reg::setBit  (unsigned b)
{
  unsigned r = *this;
  *this = r | (1<<b);
}

void Reg::clearBit(unsigned b)
{
  unsigned r = *this;
  *this = r &~(1<<b);
}

void Reg::set(const char* ip,
              unsigned short port,
              unsigned mem,
              unsigned long long memsz)
{
  //
  //  Build
  //
  NetIODev  root = INetIODev::create("fpga", ip);

  {  //  Register access
    ProtoStackBuilder bldr = IProtoStackBuilder::create();
    bldr->setSRPVersion              ( IProtoStackBuilder::SRP_UDP_V3 );
    bldr->setUdpPort                 (                  8193 );
    bldr->useRssi                    (                  true );
    bldr->setSRPTimeoutUS            (                 90000 );
    bldr->setSRPRetryCount           (                     5 );
    bldr->setSRPMuxVirtualChannel    (                     0 );
    bldr->useDepack                  (                  true );
    bldr->setTDestMuxTDEST           (                     0 );

    MMIODev   mmio = IMMIODev::create ("mmio", memsz);
    Field f = IIntField::create("reg", 32, false, 0);
    mmio->addAtAddress( f, mem, memsz/4, 4 );
    root->addAtAddress( mmio, bldr);
  }

  Path pre = IPath::create(root);
  _reg = IScalVal::create( pre->findByName("mmio/reg") );
}

Reg& Reg::operator=(const unsigned r)
{
  uint32_t addr = ((uint64_t(this))>>2)&0x3fffffff; 
  IndexRange rng(addr);
  uint32_t v(r);
  _reg->setVal(&v,1,&rng);
  return *this;
}

Reg::operator unsigned() const
{
  uint32_t addr = ((uint64_t(this))>>2)&0x3fffffff; 
  IndexRange rng(addr);
  unsigned v;
  _reg->getVal(&v,1,&rng);
  return v;
}
