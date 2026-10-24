/**
 * @file
 * @brief SemLock, a lock built on a Semaphore that starts empty.
 */
#ifndef PDS_SEMLOCK_HH
#define PDS_SEMLOCK_HH

#include "Semaphore.hh"

namespace Pds {
/** Lock built on a Semaphore. The semaphore starts EMPTY, so lock() blocks until release() has been called once. */
class SemLock {
 public:
  /** Create the semaphore empty (count 0). */
  SemLock();
  /** Take the semaphore, blocking while it is 0. */
  virtual void lock();
  /** Give the semaphore. */
  virtual void release();
private:
  Semaphore _sem;
};
}
#endif



















