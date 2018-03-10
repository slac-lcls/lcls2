#ifndef PDS_SPINLOCK_HH
#define PDS_SPINLOCK_HH

#include <atomic>

namespace Pds {

class SpinLock {
public:
  SpinLock();
  ~SpinLock() = default;

  SpinLock(const SpinLock&) = delete;
  SpinLock& operator=(const SpinLock&) = delete;

public:
  void lock();
  void unlock();

private:
  void _pause();
private:
  std::atomic_flag _lock;
};


inline Pds::SpinLock::SpinLock() : _lock(ATOMIC_FLAG_INIT)
{
}

/* Pause instruction to prevent excess processor bus usage */
inline void Pds::SpinLock::_pause()
{
  asm volatile("pause\n": : :"memory");
}

inline void Pds::SpinLock::lock()
{
  while ( _lock.test_and_set(std::memory_order_acquire) ) { _pause(); }
}

inline void Pds::SpinLock::unlock()
{
  _lock.clear(std::memory_order_release);
}

}

#endif
