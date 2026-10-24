/**
 * @file
 * @brief Lock, an abstract lock built on a binary Semaphore with a cantLock() hook.
 */
#ifndef PDS_LOCK_HH
#define PDS_LOCK_HH

#include "Semaphore.hh"

namespace Pds {
/** Abstract lock on a binary Semaphore (the VxWorks variant is not supported). Subclasses implement cantLock(). */
class Lock {
public:
  /** Create the lock free (semaphore full) and keep the retry count retries. */
  Lock(unsigned retries);
  /** Release the lock (give the semaphore). */
  void             release();
  /** Take the lock, blocking on the semaphore. On non-VxWorks builds the retry loop and cantLock() are never reached, because each attempt blocks until it succeeds. */
  void             get();
  /** Pure virtual: meant to be called when the lock cannot be obtained after the retries. */
  virtual void     cantLock() = 0;
  /** Does nothing; empty virtual destructor. */
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
