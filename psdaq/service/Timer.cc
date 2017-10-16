//  This timer has a minimum duration of 20 ms with a resolution of 10
//  ms (any fraction of 10 ms is rounded up).  It was noticed during
//  testing that multiple timers within the same process interacted in
//  some way.  A timer with duration which is an odd multiple of 10 ms
//  was seen to act as a timer with duration 10 ms greater than
//  specified, when running with a second timer whose duration was an
//  even multiple of 10.  Multiple timers whose duration was an even
//  multiple of 10 did not interact.  When there were multiple timers
//  with durations that were odd multiples of 10 the behaviour was
//  unpredictable, although they seemed to have a duration which
//  fluctuated between the specified and 10 + the specified.  This was
//  traced to a feature of cond_timed_wait.


#include <errno.h>
#include <sys/time.h>

#include "Timer.hh"
#include "TaskWait.hh"

using namespace Pds;

Timer::Timer()
  : _service(this)
{}

Timer::~Timer() {}

unsigned Timer::start() {
  return _service.armTimer();
}

unsigned Timer::cancel() {
  unsigned disarm = _service.disarmTimer();
  if (disarm == 0) {
    if (task()->is_self()) {
      // Calling remove() is safe if we are running in the timer task;
      // we first check that there is instance of `this' in the timer
      // task queue, otherwise there is no need to do anything
      if (next() != previous()) remove();
    } else {
      // We want to be sure that `this' is not in the timer task queue
      // when we return from cancel
      TaskWait block(task());
      block.wait();
    }
  }
  return disarm;
}

// Executed in the timer task
void Timer::routine() {
  if (repetitive()) {
    expired();
    _service.submit();
  } else {
     // Disarm before expired so that user can call start from expired
    _service.disarmTimer();
    expired();
  }
}

TimerServiceRoutine::TimerServiceRoutine(Timer* timer) 
  : _timer(timer)  
  , _service_task(0)
  , _status(Off)
{
  pthread_mutex_init(&_status_mutex, 0);
  pthread_cond_init(&_status_cond, 0);
}

// Executed in the work task (main task)
TimerServiceRoutine::~TimerServiceRoutine()
{
  pthread_cond_destroy(&_status_cond);
  pthread_mutex_destroy(&_status_mutex);
  if (_service_task) _service_task->destroy();
}

// Executed in the work task (main task)
unsigned TimerServiceRoutine::armTimer() {
  if (!_service_task) {
    char taskname[128];
    if (_timer->task()->parameters().name()) {
      sprintf(taskname, "%sTimer", _timer->task()->parameters().name());
    } else {
      sprintf(taskname, "%d_Timer", _timer->task()->parameters().taskID());
    }
    int taskpriority = _timer->task()->parameters().priority();
    _service_task = new Task(TaskObject(taskname, taskpriority));
  }
  pthread_mutex_lock(&_status_mutex);
  if (_status == Off) {
    _status = On;
    pthread_mutex_unlock(&_status_mutex);
    // cond_timedwait rounds up to the nearest 10 ms and adds 10 ms, so
    // we remove 10 ms from the duration to get behaviour closer to what
    // the user expects
    enum {MilliSeconds = 1000, Nsperms = 1000000};
    unsigned delay = _timer->duration();
    if (delay > 10) delay -= 10;
    _delay.tv_sec  =  delay / MilliSeconds;
    _delay.tv_nsec = (delay - (_delay.tv_sec * MilliSeconds)) * Nsperms;
    submit();
    return 0;
  }
  pthread_mutex_unlock(&_status_mutex);
  return 1;
}

// Executed in the work task (main task) or in the timer task
unsigned TimerServiceRoutine::disarmTimer() {
  pthread_mutex_lock(&_status_mutex);
  if (_status == On) {
    _status = Off;
    pthread_mutex_unlock(&_status_mutex);
    TaskWait block(_service_task);
    // Wake up service thread waiting on _status_cond 
    pthread_cond_signal(&_status_cond);  
    block.wait();
    return 0;
  }
  pthread_mutex_unlock(&_status_mutex);
  return 1;
}

// Executed in the timer task
void  TimerServiceRoutine::submit() {
  pthread_mutex_lock(&_status_mutex);
  if (_status == On) {
    _service_task->call(this);
  }
  pthread_mutex_unlock(&_status_mutex);
}

// Executed in the service task
void TimerServiceRoutine::routine() {
  timespec wakeuptime = wakeup();
  int error;
  pthread_mutex_lock(&_status_mutex);
  while (_status == On) {
    error = pthread_cond_timedwait(&_status_cond, &_status_mutex, &wakeuptime);
    if (error == ETIMEDOUT) {
      pthread_mutex_unlock(&_status_mutex);
      _timer->task()->call(_timer);
      return;
    }
  }
  pthread_mutex_unlock(&_status_mutex);  
}

// Calculation of absolute time for timer expiration
timespec TimerServiceRoutine::wakeup() {
  timespec wakeuptime, now;

  clock_gettime(CLOCK_REALTIME, &now);

  wakeuptime.tv_sec  = now.tv_sec  + _delay.tv_sec;
  wakeuptime.tv_nsec = now.tv_nsec + _delay.tv_nsec; 

  static const long int NanoSeconds  = 1000000000;

  if (wakeuptime.tv_nsec < 0) {
    wakeuptime.tv_sec--;
    wakeuptime.tv_nsec += NanoSeconds;
  } else {
    if (wakeuptime.tv_nsec >= NanoSeconds) {
      wakeuptime.tv_sec++;
      wakeuptime.tv_nsec -= NanoSeconds;
    }
  }
  return wakeuptime;
}
