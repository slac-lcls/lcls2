/**
 * @file
 * @brief TaskWait, a Routine used to wait until a Task has processed everything queued before it.
 */
// This class can be used to be sure that a given task has finished
// with the elements in its queue. This is implemented by inserting
// TaskWait in the task queue and then by calling wait() to block
// until TaskWait has done.

#ifndef PDS_TASKWAIT_HH
#define PDS_TASKWAIT_HH

#include "Task.hh"

namespace Pds {

/** Routine that, once queued on a Task, lets the caller block until the task has run all jobs queued before it (per the file comment). */
class TaskWait : private Routine {
public:
  /** Create the semaphore empty and queue this object on task. */
  TaskWait(Task* task) 
    : _sem(Semaphore::EMPTY) 
  {
    task->call(this);
  }
  /** Does nothing; empty virtual destructor. */
  virtual ~TaskWait() {}

  /** Block until the task has run this routine (takes the semaphore that routine() gives). */
  void wait() {_sem.take();}
  
private:
  virtual void routine() {_sem.give();}
  Semaphore _sem;
};

}
#endif
