#ifndef PDS_SEMAPHORE_HH
#define PDS_SEMAPHORE_HH


#ifdef VXWORKS
#  include "semLib.h"
#else
#  include <pthread.h>
#  include <semaphore.h>
#endif

namespace Pds {
class Semaphore {
 public:
  enum semState { EMPTY, FULL};
  Semaphore(semState initial);
  ~Semaphore();
  void take();
  void give();

 private:

#ifdef VXWORKS
  SEM_ID _sem;
#else
  sem_t _sem;
#endif

};
}
#endif



















