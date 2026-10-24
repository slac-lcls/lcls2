/**
 * @file
 * @brief Task, a thread with a queue of Routine jobs, and its helpers.
 */
#ifndef PDS_TASK_H
#define PDS_TASK_H

#include "QueueSlowSafe.hh"
#include "Semaphore.hh"
#include "TaskObject.hh"
#include "Routine.hh"


namespace Pds {

extern "C" {
  /** C-linkage thread entry function type: takes and returns a void pointer. */
  typedef void* (*TaskFunction)(void*);
  /** Thread body of a Task (task is the Task): run every queued Routine, then wait on the pending semaphore for more, forever. Never returns. */
  void* TaskMainLoop(void*);
}


/*
 * an Task specific class that is used privately by Task so that
 * it can properly delete itself.
 *
 */

class Task;
/** Routine queued by Task::destroy() so that the task deletes itself from its own thread (per the code comment). */
class TaskDelete : public Routine {
 public:
  /** Remember the task t to delete. */
  TaskDelete(Task* t) { _taskToKill = t; }
  /** Call deleteTask() on the task, which frees its resources and exits the calling thread. */
  void routine(void);
 private:
  Task* _taskToKill;
};

/*
 * The Task callable interface. Customization of the code 
 *
 *
 */

/** Thread that runs Routine jobs from a queue in order (TaskMainLoop). Copies share the same thread, queue and reference count. */
class Task {
 public:

  /** Copy the TaskObject parameters, create the job queue and pending semaphore and start the thread with pthread_create() (stack size and priority from the parameters); a creation error is ignored. */
  Task(const TaskObject&);
  /** Share the thread, queue, semaphore and parameters of aTask and increment the shared reference count. */
  Task(const Task&);

  // this ctor makes current task the Task.  make c++ signature
  // distinct so it isn't used by accident.
  /** Tag type that selects the constructor which wraps the calling thread instead of creating one. */
  enum MakeThisATaskFlag {MakeThisATask /**< The only value of the tag. */ };
  /** Wrap the calling thread: create default TaskObject parameters (thread ID of the caller), a job queue and a pending semaphore, without starting a thread. */
  Task(MakeThisATaskFlag);

  /** Share the thread, queue, semaphore and parameters of aTask; unlike the copy constructor, the reference count is not incremented. */
  void operator= (const Task&);

  /** Return the TaskObject parameters. */
  const TaskObject& parameters() const;
  /** Insert routine at the tail of the job queue and give the pending semaphore if the queue was empty. */
  void call(Routine*);
  /** Decrement the shared reference count; if others remain, delete this handle, and if it reaches 0, queue a TaskDelete so that the thread deletes everything and exits. */
  void destroy();
  /** Run TaskMainLoop() on the calling thread (used with the MakeThisATask constructor); never returns. */
  void mainLoop();
  /** Send signal to the task thread with pthread_kill(). */
  void signal(int signal);

  /** Return true if the calling thread is the task thread. */
  bool is_self() const;

  /** Lets TaskDelete call the private deleteTask(). */
  friend class TaskDelete;
  /** Lets TaskMainLoop() reach the private job queue and semaphore. */
  friend void* TaskMainLoop(void*);

 private:
  Task() {}
  ~Task();
  int createTask( TaskObject&, TaskFunction );
  void deleteTask();

  TaskObject*        _taskObj;
  int*                  _refCount;
  Queue<Routine>* _jobs;
  Semaphore*         _pending;
  Routine*           _destroyRoutine;
};
}

#endif
