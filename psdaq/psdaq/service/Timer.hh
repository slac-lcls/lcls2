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
class TimerServiceRoutine : private Routine {
public:
  TimerServiceRoutine(Timer* timer);
  virtual ~TimerServiceRoutine();

  unsigned armTimer();
  unsigned disarmTimer();

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

class Timer: public Routine {
public:
  Timer();
  virtual ~Timer();

  // Start timer
  unsigned start();

  // Stop timer
  unsigned cancel();

  // User's code executed in the task's context
  virtual void     expired()          = 0;

  // Task the timer is connected to
  virtual Task* task()             = 0;

  // Value in milliseconds of the duration of the timer
  virtual unsigned duration()   const = 0;

  // Return 0 if one-shot, != 0 if repetitive
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


