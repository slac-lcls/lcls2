// This class can be used to be sure that a given task has finished
// with the elements in its queue. This is implemented by inserting
// TaskWait in the task queue and then by calling wait() to block
// until TaskWait has done.

#ifndef PDS_TASKWAIT_HH
#define PDS_TASKWAIT_HH

#include "Task.hh"

namespace Pds {

class TaskWait : private Routine {
public:
  TaskWait(Task* task) 
    : _sem(Semaphore::EMPTY) 
  {
    task->call(this);
  }
  virtual ~TaskWait() {}

  void wait() {_sem.take();}
  
private:
  virtual void routine() {_sem.give();}
  Semaphore _sem;
};

}
#endif
