/*
 * UNIX specific Task class implementation.
 *
 *
 */

#include "Task.hh"
#include <signal.h>
#include <sched.h>

using namespace Pds;

/*
 * actually make the call to start the thread
 *
 * see TaskObjectUx.cc for a note on the priority value and
 * the stupid sentinal trick.
 */
int Task::createTask(TaskObject& tobj, TaskFunction aRoutine)
{
  struct sched_param param;
  param.sched_priority=tobj.priority();
  pthread_attr_setstacksize(&tobj._flags,tobj.stackSize()); 
  pthread_attr_setschedparam(&tobj._flags,&param);
 
  int status = pthread_create(&tobj._threadID,&tobj._flags,aRoutine,this);

  //  printf("Task::createTask id %d name %s\n", tobj._threadID, tobj._name);

  return status;
}


/*
 * actually make the call to exit the thread. This routine can only 
 * be called from within the context of the thread itself. Therefore,
 * only the destroy() method of Task will really use it.
 */
void Task::deleteTask()
{
  //  printf("Task::deleteTask id %d\n", _taskObj->_threadID);

  if(*_refCount != 0) {
    // error as this should be the last message 
    // from the last task object for this task
  }
  delete _pending;
  delete _jobs;
  delete _refCount;
  delete _taskObj;
  delete _destroyRoutine;

  delete this;

  int status;
  pthread_exit((void*)&status);
}


void Task::signal(int signal){
  pthread_kill(_taskObj->_threadID, signal);
}

bool Task::is_self() const
{
  return pthread_self() == (pthread_t)_taskObj->taskID();
}
