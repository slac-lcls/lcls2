/**
 * @file
 * @brief GenericPoolW, a GenericPool whose allocation waits for a free buffer instead of failing.
 */
#ifndef Pds_GenericPoolW_hh
#define Pds_GenericPoolW_hh

#include "GenericPool.hh"

#include <atomic>
#include <mutex>
#include <condition_variable>

namespace Pds {

  /** GenericPool whose allocation blocks on a condition variable until a buffer is freed (or stop() is called); frees wake one waiter. */
  class GenericPoolW : public GenericPool {
  public:
    /** Construct the base pool of numberofObjects buffers of sizeofObject bytes. */
    GenericPoolW(size_t sizeofObject, int numberofObjects);
    /** Construct the base pool with buffers aligned to alignBoundary. */
    GenericPoolW(size_t sizeofObject, int numberofObjects, unsigned alignBoundary);
    /** Does nothing; empty virtual destructor. */
    virtual ~GenericPoolW();
  public:
    /** Set the stopping flag and wake all waiting allocations, which then return nullptr. */
    void stop();
  protected:
    virtual void* deque();
    virtual void  enque(PoolEntry*);
  private:
    std::atomic<bool>       _stopping;
    mutable std::mutex      _mutex;
    std::condition_variable _condVar;
  };

}


inline void* Pds::GenericPoolW::deque()
{
  std::unique_lock<std::mutex> lk(_mutex);
  if (atHead() == empty())              // Don't involve kernel if not needed
  {
    _condVar.wait(lk, [&](){ return (atHead() != empty()) || _stopping; });
  }
  if (!_stopping)
  {
    Pds::PoolEntry* entry = removeNL();
    return (void*)&entry[1];
  }
  return nullptr;
}

inline void Pds::GenericPoolW::enque(PoolEntry* entry)
{
  std::lock_guard<std::mutex> lk(_mutex);
  insertNL(entry);
  _condVar.notify_one();
}

#endif
