#ifndef PDS_SEMLOCK_HH
#define PDS_SEMLOCK_HH

#include "psdaq/service/Semaphore.hh"

namespace Pds {
class SemLock {
 public:
  SemLock();
  virtual void lock();
  virtual void release();
private:
  Semaphore _sem;
};
}
#endif



















