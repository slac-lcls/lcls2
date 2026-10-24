/**
 * @file
 * @brief TaskObject, the thread parameters (name, priority, stack, attributes) of a Task.
 */
#ifndef PDS_TASK_OBJECT_H
#define PDS_TASK_OBJECT_H

#include <string.h>
#include <pthread.h>

namespace Pds {

class Task;

/*
 *
 * Note on priority: 0xffffffff is a sentinal value meaning use the
 *     default priority for the thread. The value of the priority is
 *     entered as though it were for a vxWorks task ( 0 highest to 127
 *     lowest ). This also limits the vxWorks range to 0 to 127 instead
 *     of 0 to 255. The sense of this priority is opposite that for
 *     threads so the value is renormalized to the unix version inside
 *     the constructor.
 */

/** Thread parameters of a Task: name, priority, stack size and base, queue size and pthread attributes, plus the thread ID once created. Priorities are given in the range 0 (highest) to 127 and stored inverted (127 minus the value), per the header note. */
class TaskObject {
 public:
  /** Thread creation flags. */
  enum { //THR_SUSPENDED = 1 << 0,
         THR_DETACHED  = 1 << 1, /**< Create the thread detached (1 shifted left by 1); otherwise joinable. */        // Else joinable
         //THR_BOUND     = 1 << 2,
         //THR_NEW_LWP   = 1 << 3,
         //THR_DAEMON    = 1 << 4,
       };
 public:
  /** Copy name, clamp priority to 0-127 and store 127 minus it, keep the stack size (raised to the pthread default minimum if smaller and non-zero), stack base and queue size, and set the detached attribute if flags has THR_DETACHED. */
  TaskObject( const char* name,
              int priority=127,
              int stackSize=20*1024, char* stackBase=NULL,
              int queueSize=0,
              unsigned flags=THR_DETACHED);
  /** Copy all parameters of tparam, including the pthread attributes and thread ID (the name is duplicated). */
  TaskObject(const TaskObject& tparam );
  /** Parameters of the calling thread: no name, zero stack size, default attributes, the caller thread ID and the default scheduling priority (clamped and inverted). */
  TaskObject();
  /** Replace all parameters with copies of those of the argument (the name is duplicated). */
  void operator= (const TaskObject&);
  /** Destroy the pthread attributes and free the name. */
  ~TaskObject();

  /** Return the task name (null for the default-constructed object). */
  char* name() const { return _name;}
  /** Return the stored priority (127 minus the requested value). */
  int priority() const { return _priority;}
  //  int flags() const {return _flags;}
  /** Return the stack base given to the constructor. */
  char* stackBase() const {return (char*)_stackbase;}
  /** Return the stack size in bytes. */
  int stackSize() const {return _stacksize;}
  /** Return the pthread ID (set when the thread is created, or the caller ID for the default object). */
  pthread_t taskID() const {return _threadID;}
  /** Return the queue size given to the constructor. */
  int queueSize() const { return _queueSize; }

  /** Lets Task use the private attributes and thread ID. */
  friend class Task;
  /** Lets SpinTask use the private attributes and thread ID. */
  friend class SpinTask;
 private:
  char* _name;
  size_t _stacksize;
  void* _stackbase;
  int _priority;
  pthread_attr_t _flags;
  pthread_t _threadID;
  int _queueSize;
};

}
#endif
