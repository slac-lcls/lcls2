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
class QueueSS : private List
{
  public:
    QueueSS()
    {
      int status = pthread_mutex_init( &_lockkey, NULL);
      // assert(!status);
    }
    ~QueueSS()
    {
      int status = pthread_mutex_destroy( &_lockkey);
      // assert(!status);
    }
    T* empty() const
    {
      return (T*) List::empty();
    }
    T* insert(Entry *entry)
    {
      T* t;
      lock();
      t = (T*) List::insert(entry);
      unlock();
      return t;
    }
    T* remove()
    {
      T* t;
      lock();
      t = (T*) List::remove();
      unlock();
      return t; 
    }
    T* atHead() const
    {
      T* t;
      lock();
      t = (T*) List::atHead();
      unlock();
      return t; 
    }
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
