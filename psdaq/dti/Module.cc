#include "Module.hh"

#include <unistd.h>
#include <stdio.h>
#include <new>

using namespace Pds::Dti;
using Pds::Cphw::Reg;

static inline unsigned getf(const Pds::Cphw::Reg& i, unsigned n, unsigned sh)
{
  unsigned v = i;
  return (v>>sh)&((1<<n)-1);
}

static inline unsigned setf(Pds::Cphw::Reg& o, unsigned v, unsigned n, unsigned sh)
{
  unsigned r = unsigned(o);
  unsigned q = r;
  q &= ~(((1<<n)-1)<<sh);
  q |= (v&((1<<n)-1))<<sh;
  o = q;
  return q;
}


void Stats::dump() const
{
#define PU32(title,stat) printf("%9.9s: %x\n",#title,stat)

  printf("Link Up:  US: %x  BP: %x  DS: %x\n", usLinkUp, bpLinkUp, dsLinkUp);

  printf("UsLink  RxErrs    RxFull    IbRecv    IbEvt     ObRecv    ObSent\n");
  //      0      00000000  00000000  00000000  00000000  00000000  00000000
  for (unsigned i = 0; i < Module::NUsLinks; ++i)
  {
    printf("%d      %08x  %08x  %08x  %08x  %08x  %08x\n", i,
           us[i].rxErrs,
           us[i].rxFull,
           us[i].ibRecv,
           us[i].ibEvt,
           us[i].obRecv,
           us[i].obSent);
  }

  PU32(bpObSent, bpObSent);

  printf("DsLink  RxErrs    RxFull      ObSent\n");
  //      0      00000000  00000000  000000000000
  for (unsigned i = 0; i < Module::NDsLinks; ++i)
  {
    printf("%d      %08x  %08x  %012lx\n", i,
           ds[i].rxErrs,
           ds[i].rxFull,
           ds[i].obSent);
  }

  PU32(qpllLock, qpllLock);

  printf("MonClk   Rate   Slow  Fast  Lock\n");
  //      0      00000000  0     0     0
  for (unsigned i = 0; i < 4; ++i)
  {
    printf("%d  %08x  %d     %d     %d\n", i,
           monClk[i].rate,
           monClk[i].slow,
           monClk[i].fast,
           monClk[i].lock);
  }

  // Revisit: Not provided by the f/w yet
  //printf("usLinkOb  L0 %08x  L1A %08x  L1R %08x\n", usLinkObL0, usLinkObL1A, usLinkObL1R);

  printf("PGP  RxFrmErrs  RxFrmCnt   RxOpcodes  TxFrmErrs  TxFrmCnt   TxOpcodes\n");
  //      0    00000000   00000000   00000000   00000000   00000000   00000000
  for (unsigned i = 0; i < 2; ++i)      // Revisit: 2 -> NUsLinks + NDsLinks
  {
    printf("%d    %08x   %08x   %08x   %08x   %08x   %08x\n", i,
           pgp[i].rxFrameErrs,
           pgp[i].rxFrames,
           pgp[i].rxOpcodes,
           pgp[i].txFrameErrs,
           pgp[i].txFrames,
           pgp[i].txOpcodes);
  }
}


Module* Module::locate()
{
  return new((void*)0) Module;
}

Module::Module()
{
  clearCounters();
  updateCounters();                     // Revisit: Necessary?

  /*init();*/
}

void Module::init()
{
  printf("Index:      UsLink %u  DsLink %u\n", usLink(), dsLink());
  printf("Link Up:    Us %02x  Bp %d  Ds %02x\n", usLinkUp(), bpLinkUp(), dsLinkUp());

  printf("UsLnkCfg:   %4.4s %8.8s %8.8s %8.8s %8.8s %8.8s %8.8s %8.8s\n",
         "Link", "Enable", "TagEnbl", "L1Enbl", "Partn", "TrgDelay", "FwdMask", "FwdMode");
  for(unsigned i = 0; i<NUsLinks; ++i) {
    printf("            %4u %8u %8u %8u %8u %8u %8u %8u\n",
           i,
           _usLinkConfig[i].enabled(),
           _usLinkConfig[i].tagEnabled(),
           _usLinkConfig[i].l1Enabled(),
           _usLinkConfig[i].partition(),
           _usLinkConfig[i].trigDelay(),
           _usLinkConfig[i].fwdMask(),
           _usLinkConfig[i].fwdMode());
  }

  printf("UsLnkStat:  %4.4s %8.8s %8.8s %10.10s %10.10s %10.10s %10.10s %10.10s\n",
         "Link", "RxErr", "RemLnkId", "RxFull", "IbDump", "IbEvt", "ObRecv", "ObSent");
  for(unsigned i = 0; i<NUsLinks; ++i) {
    usLink(i);
    printf("            %4u %8u %8x %10u %10u %10u %10u %10u\n",
           i,
           _usStatus.rxErrs(),
           _usStatus.remLinkId(),
           unsigned(_usStatus._rxFull),
           unsigned(_usStatus._ibDump),
           unsigned(_usStatus._ibEvt),
           unsigned(_usObRecv),
           unsigned(_usObSent));
  }

  printf("DsLnkStat:  %4.4s %8.8s %8.8s %10.10s %10.10s\n",
         "Link", "RxErr", "RemLnkId", "RxFull", "ObSent");
  for(unsigned i = 0; i<NDsLinks; ++i) {
    dsLink(i);
    printf("            %4u %8u %8x %10u %10u\n",
           i,
           _dsStatus.rxErrs(),
           _dsStatus.remLinkId(),
           unsigned(_dsStatus._rxFull),
           unsigned(_dsStatus._obSent));
  }

  printf("BpObSent:   %u\n", unsigned(_bpObSent));
  printf("QPLL:       Lock %u  BpTxPd %u\n", qpllLock(), bpTxInterval());
  printf("MonClk:     %4.4s %4.4s %4.4s %4.4s\n", "Rate", "Slow", "Fast", "Lock");
  for(unsigned i = 0; i < sizeof(_monClk)/sizeof(*_monClk); ++i) {
    printf("       %9u %4u %4u %4u\n",
           monClkRate(i), monClkSlow(i), monClkFast(i), monClkLock(i));
  }

  printf("PGP:        %4.4s %9.9s %9.9s %9.9s %9.9s %9.9s %9.9s\n",
         "Link", "RxFrmErrs", "RxFrmCnt", "RxOpcodes", "TxFrmErrs", "TxFrmCnt", "TxOpcodes");
  for (unsigned i = 0; i < 2; ++i) {    // Revisit: 2 -> NUsLinks + NDsLinks
    printf("            %4u %9u %9u %9u %9u %9u %9u\n",
           i,
           unsigned(_pgp[i]._rxFrameErrs),
           unsigned(_pgp[i]._rxFrames),
           unsigned(_pgp[i]._rxOpcodes),
           unsigned(_pgp[i]._txFrameErrs),
           unsigned(_pgp[i]._txFrames),
           unsigned(_pgp[i]._txOpcodes));
  }
}


bool     Module::UsLink::enabled()       const { return getf(_config,     1,  0); }
void     Module::UsLink::enable(bool v)        {        setf(_config, v,  1,  0); }

bool     Module::UsLink::tagEnabled()    const { return getf(_config,     1,  1); }
void     Module::UsLink::tagEnable(bool v)     {        setf(_config, v,  1,  1); }

bool     Module::UsLink::l1Enabled()     const { return getf(_config,     1,  2); }
void     Module::UsLink::l1Enable(bool v)      {        setf(_config, v,  1,  2); }

unsigned Module::UsLink::partition()     const { return getf(_config,     4,  4); }
void     Module::UsLink::partition(unsigned v) {        setf(_config, v,  4,  4); }

unsigned Module::UsLink::trigDelay()     const { return getf(_config,     8,  8); }
void     Module::UsLink::trigDelay(unsigned v) {        setf(_config, v,  8,  8); }

unsigned Module::UsLink::fwdMask()       const { return getf(_config,    13, 16); }
void     Module::UsLink::fwdMask(unsigned v)   {        setf(_config, v, 13, 16); }

bool     Module::UsLink::fwdMode()       const { return getf(_config,     1, 31); }
void     Module::UsLink::fwdMode(bool v)       {        setf(_config, v,  1, 31); }


unsigned Module::usLinkUp() const      { return getf(_linkUp,     7,  0); }
bool     Module::bpLinkUp() const      { return getf(_linkUp,     1, 15); }
unsigned Module::dsLinkUp() const      { return getf(_linkUp,     7, 16); }

unsigned Module::usLinkEnabled(unsigned idx) const
{
  return (unsigned) _usLinkConfig[idx].enabled();
}
void     Module::usLinkEnabled(unsigned idx, unsigned v)
{
  _usLinkConfig[idx].enable((bool) v);
}

unsigned Module::usLinkTrigDelay(unsigned idx) const
{
  return (unsigned) _usLinkConfig[idx].trigDelay();
}
void     Module::usLinkTrigDelay(unsigned idx, unsigned v)
{
  _usLinkConfig[idx].trigDelay(v);
}

unsigned Module::usLinkFwdMask(unsigned idx) const
{
  return (unsigned) _usLinkConfig[idx].fwdMask();
}
void     Module::usLinkFwdMask(unsigned idx, unsigned v)
{
  _usLinkConfig[idx].fwdMask(v);
}

unsigned Module::usLinkPartition(unsigned idx) const
{
  return (unsigned) _usLinkConfig[idx].partition();
}
void     Module::usLinkPartition(unsigned idx, unsigned v)
{
  _usLinkConfig[idx].partition(v);
}

unsigned Module::usLink() const        { return getf(_index,      4,  0); }
void     Module::usLink(unsigned v) const
{
  setf(const_cast<Pds::Cphw::Reg&>(_index), v, 4,  0);
}
unsigned Module::dsLink() const        { return getf(_index,      4, 16); }
void     Module::dsLink(unsigned v) const
{
  setf(const_cast<Pds::Cphw::Reg&>(_index), v, 4, 16);
}
void     Module::clearCounters() const
{
  setf(const_cast<Pds::Cphw::Reg&>(_index), 1, 1, 30);
  usleep(10);
  setf(const_cast<Pds::Cphw::Reg&>(_index), 0, 1, 30);
}
void     Module::updateCounters() const
{
  setf(const_cast<Pds::Cphw::Reg&>(_index), 1, 1, 31);
  usleep(10);
  setf(const_cast<Pds::Cphw::Reg&>(_index), 0, 1, 31);
}

unsigned Module::UsStatus::rxErrs()    const  { return getf(_rxErrs,    24,  0); }
unsigned Module::UsStatus::remLinkId() const  { return getf(_rxErrs,     8, 24); }

unsigned Module::DsStatus::rxErrs()    const  { return getf(_rxErrs,    24,  0); }
unsigned Module::DsStatus::remLinkId() const  { return getf(_rxErrs,     8, 24); }

unsigned Module::qpllLock()     const         { return getf(_qpllLock,       2,  0); }
unsigned Module::bpTxInterval() const         { return getf(_qpllLock,       8, 16); }
void     Module::bpTxInterval(unsigned v)     {        setf(_qpllLock,   v,  8, 16); }

unsigned Module::monClkRate(unsigned i) const { return getf(_monClk[i], 29,  0); }
bool     Module::monClkSlow(unsigned i) const { return getf(_monClk[i],  1, 29); }
bool     Module::monClkFast(unsigned i) const { return getf(_monClk[i],  1, 30); }
bool     Module::monClkLock(unsigned i) const { return getf(_monClk[i],  1, 31); }

void Module::Pgp2bAxi::clearCounters() const
{
  const_cast<Pds::Cphw::Reg&>(_countReset) = 1;
  usleep(10);
  const_cast<Pds::Cphw::Reg&>(_countReset) = 0;
}

void Module::TheRingBuffer::acqNdump()
{
  printf("RingBuffer @ %p\n", this);
  enable(false);
  clear();
  enable(true);
  usleep(1000);
  enable(false);
  dump();
  printf("\n");
}

Stats Module::stats() const
{
  Stats s;

  // Revisit: Useful?  s.enabled    = enabled();
  // Revisit: Useful?  s.tagEnabled = tagEnabled();
  // Revisit: Useful?  s.l1Enabled  = l1Enabled();

  s.usLinkUp = usLinkUp();
  s.bpLinkUp = bpLinkUp();
  s.dsLinkUp = dsLinkUp();

  for (unsigned i = 0; i < NUsLinks; ++i)
  {
    usLink(i);

    s.us[i].rxErrs = _usStatus.rxErrs();
    s.us[i].rxFull = _usStatus._rxFull;
    //s.us[i].ibRecv = _usStatus._ibRecv;
    s.us[i].ibRecv = _usStatus._ibDump;
    s.us[i].ibEvt  = _usStatus._ibEvt;
    s.us[i].obRecv = _usObRecv;
    s.us[i].obSent = _usObSent;
  }

  s.bpObSent = _bpObSent;

  for (unsigned i = 0; i < NDsLinks; ++i)
  {
    dsLink(i);

    s.ds[i].rxErrs = _dsStatus.rxErrs();
    s.ds[i].rxFull = _dsStatus._rxFull;
    s.ds[i].obSent = _dsStatus._obSent;
  }

  s.qpllLock = qpllLock();

  updateCounters();                     // Revisit: Is this needed?

  for (unsigned i = 0; i < 4; ++i)
  {
    s.monClk[i].rate = monClkRate(i);
    s.monClk[i].slow = monClkSlow(i);
    s.monClk[i].fast = monClkFast(i);
    s.monClk[i].lock = monClkLock(i);
  }

  // @todo: These are not available yet
  //s.usLinkObL0  = _usLinkObL0;
  //s.usLinkObL1A = _usLinkObL1A;
  //s.usLinkObL1R = _usLinkObL1R;

  for (unsigned i = 0; i < 2; ++i)      // Revisit: 2 -> NUsLinks + NDsLinks
  {
    s.pgp[i].rxFrameErrs = _pgp[i]._rxFrameErrs;
    s.pgp[i].rxFrames    = _pgp[i]._rxFrames;
    s.pgp[i].txFrameErrs = _pgp[i]._txFrameErrs;
    s.pgp[i].txFrames    = _pgp[i]._txFrames;
    s.pgp[i].txOpcodes   = _pgp[i]._txOpcodes;
    s.pgp[i].rxOpcodes   = _pgp[i]._rxOpcodes;
  }

  return s;
}
