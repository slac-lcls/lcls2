#include "psdaq/mmhw/RegProxy.hh"
#include "psdaq/service/Semaphore.hh"

#include <unistd.h>
#include <stdio.h>

static uint64_t _base = 0;
static uint32_t* _csr = 0;
static Pds::Semaphore _sem(Pds::Semaphore::FULL);

using namespace Pds::Mmhw;

void RegProxy::initialize(void* base, void* csr)
{
  _base = reinterpret_cast<uint64_t>(base);
  _csr  = reinterpret_cast<uint32_t*>(csr);
}

RegProxy& RegProxy::operator=(const unsigned r)
{
  _sem.take();

  //  launch transaction
  _csr[3] = r;
  _csr[2] = reinterpret_cast<uint64_t>(this)-_base;
  _csr[0] = 0;

  //  wait until transaction is complete
  unsigned tmo=0;
  unsigned tmo_mask = 0x3;
  do { 
    usleep(1000);
    if ((++tmo&tmo_mask) ==  tmo_mask) {
      tmo_mask = (tmo_mask<<1) | 1;
      printf("RegProxy tmo (%x) writing 0x%x to %llx\n", 
             tmo, r, reinterpret_cast<uint64_t>(this)-_base);
    }
  } while ( (_csr[1]&1)==0 );

  _sem.give();

  return *this;
}

RegProxy::operator unsigned() const 
{
  _sem.take();

  //  launch transaction
  _csr[2] = reinterpret_cast<uint64_t>(this)-_base;
  _csr[0] = 1;

  //  wait until transaction is complete
  unsigned tmo=0;
  unsigned tmo_mask = 0x3;
  do { 
    usleep(1000);
    if ((++tmo&tmo_mask) == tmo_mask) {
      tmo_mask = (tmo_mask<<1) | 1;
      printf("RegProxy tmo (%x) read from %llx\n", 
             tmo, reinterpret_cast<uint64_t>(this)-_base);
    }
  } while ( (_csr[1]&1)==0 );
  
  unsigned r = _csr[3];

  _sem.give();

  return r;
}
