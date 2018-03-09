#ifndef Pds_GenericPoolW_hh
#define Pds_GenericPoolW_hh

#include "GenericPool.hh"

#include <mutex>
#include <condition_variable>

namespace Pds {

  class GenericPoolW : public GenericPool {
  public:
    GenericPoolW(size_t sizeofObject, int numberofObjects);
    virtual ~GenericPoolW();
  protected:
    virtual void* deque();
    virtual void  enque(PoolEntry*);
  private:
    mutable std::mutex      _mutex;
    std::condition_variable _condVar;
  };

}


inline void* Pds::GenericPoolW::deque()
{
  std::unique_lock<std::mutex> lk(_mutex);
  _condVar.wait(lk, [this]{ return atHead() != empty(); });
  Pds::PoolEntry* entry = removeNL();
  return (void*)&entry[1];
}

inline void Pds::GenericPoolW::enque(PoolEntry* entry)
{
  std::lock_guard<std::mutex> lk(_mutex);
  insertNL(entry);
  _condVar.notify_one();
}

#endif
