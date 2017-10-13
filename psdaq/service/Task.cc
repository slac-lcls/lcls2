
#include "Semaphore.hh"

#include "Task.hh"

using namespace Pds;


/*
 *
 *
 *
 */
Task::Task(const TaskObject& tobj)
{
  _taskObj = new TaskObject(tobj);
  _refCount = new int(1);
  _jobs = new Queue<Routine>;
  _pending = new Semaphore(Semaphore::EMPTY);

  int err = createTask(*_taskObj, (TaskFunction) TaskMainLoop );
  if ( err != 0 ) {
    //error occured, throw exception
  }
}


/*
 *
 */
Task::Task(const Task& aTask)
{
  _refCount = aTask._refCount; ++*_refCount;
  _taskObj = aTask._taskObj;
  _jobs = aTask._jobs;
  _pending = aTask._pending;
}

/*
 *
 */
void Task::operator =(const Task& aTask)
{
  _refCount = aTask._refCount;
  _taskObj = aTask._taskObj;
  _jobs = aTask._jobs;
  _pending = aTask._pending;
}

/*
 *
 */
Task::Task(MakeThisATaskFlag dummy)
{
  _taskObj = new TaskObject();
  _refCount = new int(1);
  _jobs = new Queue<Routine>;
  _pending = new Semaphore(Semaphore::EMPTY);
}

/*
 *
 */
Task::~Task()
{
  // deletes only the memory for the object itself
  // memory for the contained by pointer objects is deleted 
  // by the TaskDelete object's routine()  
}


/*
 *
 *
 *
 */
void Task::destroy()
{
  --*_refCount;
  if (*_refCount > 0) {
    delete this;
  }
  if (*_refCount == 0) {
    _destroyRoutine = new TaskDelete(this);
    call(_destroyRoutine);
  }
  else {
    // severe error, probably bug check (assert)
  }
}




/*
 *
 *
 *
 */
const TaskObject& Task::parameters() const
{
  return *_taskObj;
};


/*
 * Inserts an entry on the tasks processing queue.
 * give the jobs pending semaphore only when the queue goes empty to non-empty
 *
 */
void Task::call(Routine* routine)
{
  if( _jobs->insert(routine) == _jobs->empty()) {
    _pending->give();
  }
}




/*
 * Main Loop of a task.
 *
 * It is a global function with c linkage so it can be correctly
 * called by the underlying system service layer. It is friend 
 * to the Task class so that it can act like a private member 
 * function. This function should never be called directly.
 *
 * Process jobs while there are entries on the queue then take 
 * the jobs pending semaphore and wait for new entries.
 *
 */
void* Pds::TaskMainLoop(void* task)
{
  Task* t = (Task*) task;
  Routine *aJob;

  for(;;) {
    while( (aJob=t->_jobs->remove()) != t->_jobs->empty() ) {
      aJob->routine();
    }
    t->_pending->take();
  }
  return NULL;
}


/*
 * Public Callable Version of taskMainLoop()
 *
 */
void Task::mainLoop()
{
  TaskMainLoop((void*)this);
}


void TaskDelete::routine(void) 
{ 
  _taskToKill->deleteTask(); 
}

