/**
 * @file
 * @brief Semaphore, a thin wrapper around an unnamed POSIX semaphore.
 */
#ifndef PDS_SEMAPHORE_HH
#define PDS_SEMAPHORE_HH

#include <semaphore.h>

namespace Pds {
/** Unnamed POSIX semaphore (sem_t) shared between threads of one process. */
class Semaphore {
 public:
  /** Initial state of the semaphore. */
  enum semState { EMPTY, /**< Initial count 0: take() blocks until give() is called. */ FULL /**< Initial count 1. */ };
  /** Initialize the semaphore with count 1 for FULL, otherwise 0. */
  Semaphore(semState initial);
  /** Destroy the semaphore. */
  ~Semaphore();
  /** Decrement the semaphore, blocking while it is 0 (sem_wait()). */
  void take();
  /** Increment the semaphore (sem_post()). */
  void give();

 private:

  sem_t _sem;
};
}

inline void Pds::Semaphore::take() { sem_wait(&_sem); }

inline void Pds::Semaphore::give() { sem_post(&_sem); }

#endif



















