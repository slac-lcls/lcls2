/**
 * @file
 * @brief Timer, which calls expired() in a user Task after duration() milliseconds, once or repeatedly, and its UNIX service routine.
 */
// ---------------------------------------------------------------------------
// Description:
//
//  The timer is used to execute a defined function at regular
//  intervals. The interval length, in ms, is defined by
//  duration(). When the timer goes off the function expired() is
//  called back. The task in which expired() is executed is defined by
//  the pointer returned by task(). In general the user builds a timer
//  by deriving by this class and implementing the pure virtual
//  functions expired(), duration() and task().
//
//  The timer is started by start() and stopped by cancel(). It's
//  possible to start-stop the timer more than once.
//
//  Note: It's required that cancel() is called before the
//  destructor. In the UNIX implementation we are guaranteed that,
//  when cancel() returns, no more call backs to expired() are
//  done. This cannot be done in the timer destructor because it could
//  be too late: the derived virtual class which implements the timer
//  has already been destroyed when the timer destructor is executed
//  and we get a crash if expired() is called back.
//
// ---------------------------------------------------------------------------


#ifndef PDS_TIMER_HH
#define PDS_TIMER_HH

#include "Routine.hh"

#ifdef VXWORKS
#include <wdLib.h>
#include <vxWorks.h>
#include <sysLib.h>
#include "Lock.hh"
#else
#include <pthread.h>
#include <time.h>
#endif

namespace Pds {

class Timer;
class Task;

#ifdef VXWORKS
class TimerLock : public Lock {
public:
  TimerLock() : Lock(_lockRetries) {};
  void cantLock() {printf("*** Timer: unable to obtain timer lock\n");}

private:
  enum{_lockRetries=3};
};
#else
/** UNIX implementation of a Timer: a separate service Task waits on a condition variable with a deadline and, on timeout, queues the Timer on its user task. */
class TimerServiceRoutine : private Routine {
public:
  /** Keep timer and initialize the status mutex and condition variable; the service task is created on the first armTimer(). */
  TimerServiceRoutine(Timer* timer);
  /** Destroy the condition variable and mutex, and destroy the service task if one was created. */
  virtual ~TimerServiceRoutine();

  /** Create the service task on first use (named after the user task, same priority), then if the timer is off, turn it on, set the delay to duration() (minus 10 ms when above 10 ms) and queue the wait. Returns 0 if armed, 1 if it was already on. */
  unsigned armTimer();
  /** If the timer is on, turn it off, wake the service task and wait until it has finished; returns 0, or 1 if it was already off. */
  unsigned disarmTimer();

  /** Queue the wait routine on the service task if the timer is on. */
  void submit();

private:
  Timer* _timer;

  // Service task pointer
  Task*        _service_task;  // Blocked by pthread_cond_timedwait

  // Needed by the pthread mechanism we use for the Unix timer
  enum Status {Off, On};
  Status          _status;
  pthread_mutex_t _status_mutex;
  pthread_cond_t  _status_cond;

  // Calculate end time of pthread_cond_timedwait
  struct timespec _delay;
  inline timespec wakeup(); 

  // Implements Routine for the service task
  virtual void routine();
};
#endif

/** Base of timers: after start(), expired() is run in the Task returned by task() every duration() milliseconds (or once if repetitive() is 0) until cancel(). The header comment requires cancel() before destruction. */
class Timer: public Routine {
public:
  /** Create the service routine for this timer. */
  Timer();
  /** Does nothing; empty virtual destructor. */
  virtual ~Timer();

  // Start timer
  /** Arm the timer (TimerServiceRoutine::armTimer()); returns 0, or 1 if it was already running. */
  unsigned start();

  // Stop timer
  /** Disarm the timer and, if it was running, make sure no expired() call is pending: remove it from the queue when called from the user task, else wait until the user task has drained its queue. Returns 0, or 1 if the timer was not running. */
  unsigned cancel();

  // User's code executed in the task's context
  /** Pure virtual: the user code run in task() when the timer expires. */
  virtual void     expired()          = 0;

  // Task the timer is connected to
  /** Pure virtual: the Task in which expired() runs. */
  virtual Task* task()             = 0;

  // Value in milliseconds of the duration of the timer
  /** Pure virtual: the timer period in milliseconds (per the comment). */
  virtual unsigned duration()   const = 0;

  // Return 0 if one-shot, != 0 if repetitive
  /** Pure virtual: 0 for a one-shot timer, non-zero for a repeating one (per the comment). */
  virtual unsigned repetitive() const = 0;

private:
  // Implements Routine for the timer task
  virtual void routine();

#ifdef VXWORKS
  TimerLock _timerLock;
  WDOG_ID _timer;

  unsigned armTimer();
  unsigned disarmTimer();

  int _active;
#else
  TimerServiceRoutine _service;
#endif
};

}
#endif


