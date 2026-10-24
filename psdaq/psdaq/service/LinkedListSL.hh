/**
 * @file
 * @brief LinkedListSL, a doubly linked list element whose operations are guarded by a spin lock.
 */
#ifndef PDS_LINKEDLISTSL
#define PDS_LINKEDLISTSL

#include "LinkedList.hh"
#include "SpinLock.hh"

#include <mutex>

/** Declares a std::lock_guard on the member spin lock _lock for the rest of the enclosing scope; undefined at the end of this header. */
#define SL() std::lock_guard<Pds::SpinLock> lk(_lock)

namespace Pds {

  template<class T>
  /** Typed wrapper over Pds::ListBase (private base) in which every operation holds a per-object Pds::SpinLock. */
  class LinkedListSL : private ListBase
  {
  public:
    /** Does nothing; empty body. */
    ~LinkedListSL()                                {}
    /** Construct an element linked to itself (ListBase default constructor). */
    LinkedListSL()            : ListBase()         {}
    /** Construct with ListBase(listhead). */
    LinkedListSL(T* listhead) : ListBase(listhead) {}
    /** Under the lock, insert this element after after (ListBase::connect()) and return after as T*. */
    T* connect(T* after)          {SL(); return (T*)ListBase::connect(after);}
    /** Under the lock, unlink this element (ListBase::disconnect()) and return it as T*. */
    T* disconnect()               {SL(); return (T*)ListBase::disconnect();}
    /** Under the lock, call ListBase::insert(entry) and return the result as T*. */
    T* insert(ListBase* entry)    {SL(); return (T*)ListBase::insert(entry);}
    /** Under the lock, call ListBase::insertList(list) and return the result as T*. */
    T* insertList(ListBase* list) {SL(); return (T*)ListBase::insertList(list);}
    /** Under the lock, call ListBase::remove() and return the result as T*. */
    T* remove()                   {SL(); return (T*)ListBase::remove();}
    /** Under the lock, return ListBase::empty() as T*. */
    T* empty()  const             {SL(); return (T*)ListBase::empty();}
    /** Under the lock, return the next element as T*. */
    T* forward() const            {SL(); return (T*)ListBase::forward();}
    /** Under the lock, return the previous element as T*. */
    T* reverse() const            {SL(); return (T*)ListBase::reverse();}
  private:
    mutable Pds::SpinLock _lock;
  };
};

#undef SL

#endif
