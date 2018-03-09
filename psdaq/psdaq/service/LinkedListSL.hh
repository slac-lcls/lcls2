#ifndef PDS_LINKEDLISTSL
#define PDS_LINKEDLISTSL

#include "LinkedList.hh"
#include "SpinLock.hh"

#include <mutex>

#define SL() std::lock_guard<Pds::SpinLock> lk(_lock)

namespace Pds {

  template<class T>
  class LinkedListSL : private ListBase
  {
  public:
    ~LinkedListSL()                                {}
    LinkedListSL()            : ListBase()         {}
    LinkedListSL(T* listhead) : ListBase(listhead) {}
    T* connect(T* after)          {SL(); return (T*)ListBase::connect(after);}
    T* disconnect()               {SL(); return (T*)ListBase::disconnect();}
    T* insert(ListBase* entry)    {SL(); return (T*)ListBase::insert(entry);}
    T* insertList(ListBase* list) {SL(); return (T*)ListBase::insertList(list);}
    T* remove()                   {SL(); return (T*)ListBase::remove();}
    T* empty()  const             {SL(); return (T*)ListBase::empty();}
    T* forward() const            {SL(); return (T*)ListBase::forward();}
    T* reverse() const            {SL(); return (T*)ListBase::reverse();}
  private:
    mutable Pds::SpinLock _lock;
  };
};

#undef SL

#endif
