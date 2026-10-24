/**
 * @file
 * @brief QueueSS, a List-based queue whose operations hold a pthread mutex (the UNIX version of QueueSlowSafe).
 */
#ifndef PDS_QUEUE_SS
#define PDS_QUEUE_SS

/*
 * A UNIX specific version of LinkedList that that is re-entrant
 * but slow due to using a locking mechanism arround all operations.
 *
 *
 */


#include "Queue.hh"
#include <pthread.h>


/*
 * the class is just like Queue except that it is locked into
 * running during critical operations.
 *
 * QueueSS is short for QueueSlowSafe
 */

namespace Pds {
template<class T>
/** Queue of T over List (private base) whose insert, remove, atHead and atTail hold a pthread mutex (per the header comment, re-entrant but slow). Note that atHead() and atTail() are const but call the non-const lock helpers. */
class QueueSS : private List
{
  public:
    /** Initialize the mutex. */
    QueueSS()
    {
      int status = pthread_mutex_init( &_lockkey, NULL);
      // assert(!status);
    }
    /** Destroy the mutex. */
    ~QueueSS()
    {
      int status = pthread_mutex_destroy( &_lockkey);
      // assert(!status);
    }
    /** Return the list label as T*, without locking. */
    T* empty() const
    {
      return (T*) List::empty();
    }
    /** Append entry at the tail while holding the mutex; returns the entry it was inserted after. */
    T* insert(Entry *entry)
    {
      T* t;
      lock();
      t = (T*) List::insert(entry);
      unlock();
      return t;
    }
    /** Remove and return the head entry while holding the mutex (the list label if empty). */
    T* remove()
    {
      T* t;
      lock();
      t = (T*) List::remove();
      unlock();
      return t; 
    }
    /** Return the head entry while holding the mutex. */
    T* atHead() const
    {
      T* t;
      lock();
      t = (T*) List::atHead();
      unlock();
      return t; 
    }
    /** Return the tail entry while holding the mutex. */
    T* atTail() const
    {
      T* t;
      lock();
      t = (T*) List::atTail();
      unlock();
      return t; 
    }
private:
  pthread_mutex_t _lockkey;
  void lock() { int status = pthread_mutex_lock(&_lockkey); }
  void unlock() { int status = pthread_mutex_unlock(&_lockkey);  }
};
}
#endif
