#include "Lock.hh"
#ifdef VXWORKS
#  error VXWORKS NOT SUPPORTED
#endif

using namespace Pds;

#ifdef VXWORKS
#  error VXWORKS NOT SUPPORTED
#else
Lock::Lock(unsigned retries) : _lock(Semaphore::FULL),
  _retries(retries) {}
#endif

#define _asm asm volatile

unsigned Lock::tryOnce() {
#ifdef VXWORKS
#  error VXWORKS NOT SUPPORTED
#else
  _lock.take();
  return 0;
#endif
}

void Lock::get() {
  unsigned value;
  if ((value = tryOnce())) {
    unsigned count = _retries;
    do {
#ifdef VXWORKS
#  error VXWORKS NOT SUPPORTED
#endif
      if ((value = tryOnce()) == 0) return;
    } while (--count);
    cantLock();
  }
}

void Lock::release() {
#ifdef VXWORKS
#  error VXWORKS NOT SUPPORTED
#else
  _lock.give();
#endif
}
