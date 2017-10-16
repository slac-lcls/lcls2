#ifndef Pds_GenericPoolW_hh
#define Pds_GenericPoolW_hh

#include "GenericPool.hh"

#include "Semaphore.hh"

namespace Pds {

  class GenericPoolW : public GenericPool {
  public:
    GenericPoolW(size_t sizeofObject, int numberofObjects);
    virtual ~GenericPoolW();
    int           depth()           const;
  protected:
    virtual void* deque(); 
    virtual void  enque(PoolEntry*);
  private:
    Semaphore    _sem;
  };

}


inline int Pds::GenericPoolW::depth() const
{
  int val;
  sem_getvalue(const_cast<sem_t*>(reinterpret_cast<const sem_t*>(&_sem)),&val);
  return val; 
}

#endif
