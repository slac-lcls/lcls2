/*
 * semaphore routines implemented with solaris semapores
 * posix semaphores would be almost the same
 *
 */


#include "Semaphore.hh"

using namespace Pds;

Semaphore::Semaphore(semState initial)
{
  unsigned int count=0;

  if(initial == Semaphore::FULL) count = 1;

  sem_init(&_sem, 0, count);

}


Semaphore::~Semaphore()
{
  sem_destroy(&_sem);
}



void Semaphore::take()
{
  sem_wait(&_sem);
}

void Semaphore::give() 
{
  sem_post(&_sem);
}

