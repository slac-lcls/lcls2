#ifndef PDS_QUEUE
#define PDS_QUEUE

/*
** ++
**  Package:
**	Service
**
**  Abstract:
**	Various classes to maniplate doubly-lined lists
**
**  Author:
**      Michael Huffer, SLAC, (415) 926-4269
**
**  Creation Date:
**	000 - June 20 1,1997
**
**  Revision History:
**	November 18, 1998 - (RiC) Fixed problem with calling some of these
**                          routines from IPL where Lock is ignored.  As a
**                          work-around, the Locks have been replaced with
**                          intLock() and intUnlock() statements, making these
**                          functions unfit for inter-task communication on
**                          VxWorks.  On UNIX, the behaviour is unchanged.
**
** --
*/

#include "psdaq/service/SpinLock.hh"

#include <mutex>
#include <condition_variable>

#define CACHE_LINE_SIZE 64

namespace Pds {

class Entry
  {
  public:
    Entry();
    Entry* insert(Entry* after);
    Entry* insertList(Entry* after);
    Entry* remove();
    Entry* next()     const;
    Entry* previous() const;
  private:
    //char   _pad0[CACHE_LINE_SIZE];
    Entry* _flink;
    //char   _pad1[CACHE_LINE_SIZE];
    Entry* _blink;
    //char   _pad2[CACHE_LINE_SIZE];
  };

class List
  {
  public:
    List();
    Entry* empty() const;
    Entry* insert(Entry*);
    Entry* insertN(Entry*);
    Entry* jam(Entry*);
    Entry* remove();
    Entry* removeW();
    Entry* remove(Entry*);
    Entry* atHead() const;
    Entry* atTail() const;
  private:
    mutable Pds::SpinLock   _lock;
    mutable std::mutex      _mutex;
    std::condition_variable _condVar;
  private:
    //char   _pad0[CACHE_LINE_SIZE];
    Entry* _flink;
    //char   _pad1[CACHE_LINE_SIZE];
    Entry* _blink;
    //char   _pad2[CACHE_LINE_SIZE];
  };
}

/*
** ++
**
** Return the queue's label (usefull for checking for an empty queue)
**
** --
*/

inline Pds::Entry* Pds::List::empty() const
  {
  return (Entry*)&_flink;
  }

/*
** ++
**
** constructor sets up the queue's listhead to point to itself...
**
** --
*/

inline Pds::List::List() : _flink(empty()), _blink(empty()) {}


/*
** ++
**
** Return the entry at either the head or tail of the queue...
**
** --
*/

inline Pds::Entry* Pds::List::atHead() const {return _flink;}
inline Pds::Entry* Pds::List::atTail() const {return _blink;}

/*
** ++
**
** constructor sets up the queue's listhead to point to itself...
**
** --
*/

inline Pds::Entry::Entry() :
_flink((Entry*) &_flink), _blink((Entry*)&_flink) {}

/*
** ++
**
** insert an item on a doubly-linked list. The method has one argument:
**   after - A pointer to the item AFTER which the entry is to be inserted.
**
** --
*/

// WARNING: if `this' and `after' point to the same entry, the
// following function will destroy the list consistency. Last entry
// will point to itself instead of pointing to the list label and
// looping on the list will result in a infinite loop

inline Pds::Entry* Pds::Entry::insert(Entry* after)
  {
  Pds::Entry* next = after->_flink;
  _flink         = next;
  _blink         = after;
  next->_blink   = this;
  after->_flink  = this;
  return after;
  }

/*
** ++
**
** insert a list into a doubly-linked list. The method has one argument:
**   after - A pointer to the item AFTER which the list is to be inserted.
** 'this' is treated as being the head of a linked list.
**
** --
*/

inline Pds::Entry* Pds::Entry::insertList(Entry* after)
  {
  Pds::Entry* next = after->_flink;
  _blink->_flink = next;
  next->_blink   = _blink;
  _blink         = after;
  after->_flink  = this;
  return after;
  }

/*
** ++
**
** remove the entry from a doubly linked list...
**
** --
*/

inline Pds::Entry* Pds::Entry::remove()
  {
  register Pds::Entry* next = _flink;
  register Pds::Entry* prev = _blink;
  prev->_flink            = next;
  next->_blink            = prev;
  _flink                  = (Entry*) &_flink;
  _blink                  = (Entry*) &_flink;
  return this;
  }

/*
** ++
**
** insert an item on a doubly-linked list. The method has one argument:
**   after - A pointer to the item AFTER which the entry is to be inserted.
**
** --
*/

// WARNING: see note before Pds::Entry::insert(Entry* after)

inline Pds::Entry* Pds::List::insert(Entry* entry)
  {
  //std::lock_guard<std::mutex> lk(_mutex);
  std::lock_guard<Pds::SpinLock> lk(_lock);
  Pds::Entry* afterentry = entry->insert(atTail());
  return afterentry;
  }

inline Pds::Entry* Pds::List::insertN(Entry* entry)
  {
  std::lock_guard<std::mutex> lk(_mutex);
  Pds::Entry* afterentry = entry->insert(atTail());
  _condVar.notify_one();
  return afterentry;
  }

/*
** ++
**
** insert an item on a doubly-linked list. The method has one argument:
**   after - A pointer to the item AFTER which the entry is to be inserted.
**
** --
*/

inline Pds::Entry* Pds::List::jam(Entry* entry)
  {
  //std::lock_guard<std::mutex> lk(_mutex);
  std::lock_guard<Pds::SpinLock> lk(_lock);
  Pds::Entry* afterentry = entry->insert(atHead());
  return afterentry;
  }

/*
** ++
**
**
** --
*/

inline Pds::Entry* Pds::Entry::next()     const {return _flink;}
inline Pds::Entry* Pds::Entry::previous() const {return _blink;}

/*
** ++
**
** remove the entry from a doubly linked list...
**
** --
*/

inline Pds::Entry* Pds::List::remove()
  {
  //std::unique_lock<std::mutex> lk(_mutex);
  std::lock_guard<Pds::SpinLock> lk(_lock);
  Pds::Entry* entry = atHead()->remove();
  return entry;
  }

inline Pds::Entry* Pds::List::removeW()
  {
  std::unique_lock<std::mutex> lk(_mutex);
  _condVar.wait(lk, [this]{ return atHead() != empty(); });
  Pds::Entry* entry = atHead()->remove();
  return entry;
  }

/*
** ++
**
** remove a specific entry from a doubly linked list...
**
** --
*/

inline Pds::Entry* Pds::List::remove(Entry* entry)
  {
  //std::unique_lock<std::mutex> lk(_mutex);
  std::lock_guard<Pds::SpinLock> lk(_lock);
  Pds::Entry* theEntry = entry->remove();
  return theEntry;
  }


namespace Pds {
template<class T>
class Queue : private List
  {
  public:
    ~Queue()                {}
    Queue()                 {}
    T* empty() const        {return (T*) List::empty();}
    T* insert(T* entry)     {return (T*) List::insert((Entry*)entry);}
    T* insertN(T* entry)    {return (T*) List::insertN((Entry*)entry);}
    T* jam(T* entry)        {return (T*) List::jam((Entry*)entry);}
    T* remove()             {return (T*) List::remove();}
    T* removeW()            {return (T*) List::removeW();}
    T* remove(T* entry)     {return (T*) List::remove((Entry*)entry);}
    T* atHead() const       {return (T*) List::atHead();}
    T* atTail() const       {return (T*) List::atTail();}
  };
}
#endif

