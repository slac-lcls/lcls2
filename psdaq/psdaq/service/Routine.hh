/**
 * @file
 * @brief Routine, the job interface queued to and run by a Task.
 */
#ifndef ODF_ROUTINE_H
#define ODF_ROUTINE_H

#include "Queue.hh"

/*
 * Functor specification for jobs that can be sent to the task
 * on its processing queue. The call() method of Task allows 
 * inserting anything derived from Routine to be put on the 
 * queue.
 */
namespace Pds {
/** Job that can be queued to a Task with Task::call() (a queue Entry); the task thread calls routine(). */
class Routine : public Entry {
 public:
  /** Does nothing; empty virtual destructor. */
  virtual ~Routine() {}
  /** Pure virtual: the work to run on the task thread. */
  virtual void routine(void) = 0;
};
}
#endif
