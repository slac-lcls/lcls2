#ifndef PDS_SEMAPHORE_HH
#define PDS_SEMAPHORE_HH

#include <semaphore.h>

namespace Pds {
class Semaphore {
 public:
  enum semState { EMPTY, FULL};
  Semaphore(semState initial);
  ~Semaphore();
  void take();
  void give();

 private:

  sem_t _sem;
};
}

inline void Pds::Semaphore::take() { sem_wait(&_sem); }

inline void Pds::Semaphore::give() { sem_post(&_sem); }

#endif



















