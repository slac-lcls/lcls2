#ifndef PDS_LOCK_HH
#define PDS_LOCK_HH

#include "Semaphore.hh"

namespace Pds {
class Lock {
public:
  Lock(unsigned retries);
  void             release();
  void             get();
  virtual void     cantLock() = 0;
  virtual          ~Lock() {}

private:
  unsigned tryOnce();
#ifdef VXWORKS
  unsigned _lock;
#else
  Semaphore _lock;
#endif
  unsigned _retries;
  enum {Free=0}; // assembly code needs this to be zero.

};
}
#endif
