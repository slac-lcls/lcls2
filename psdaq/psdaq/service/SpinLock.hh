/**
 * @file
 * @brief SpinLock, a busy-waiting lock on std::atomic_flag, usable with std::lock_guard.
 */
#ifndef PDS_SPINLOCK_HH
#define PDS_SPINLOCK_HH

#include <atomic>
#include <thread>

namespace Pds
{
  /** Busy-waiting lock on an atomic flag; satisfies the lock() and unlock() interface of std::lock_guard. Not copyable. */
  class SpinLock
  {
  public:
    /** Construct unlocked. */
    SpinLock();
    /** Defaulted destructor. */
    ~SpinLock() = default;

    /** Deleted: a lock cannot be copied. */
    SpinLock(const SpinLock&) = delete;
    /** Deleted: a lock cannot be copy-assigned. */
    SpinLock& operator=(const SpinLock&) = delete;

  public:
    /** Spin until the flag is acquired, executing a pause instruction and yielding the thread between attempts. */
    void lock();
    /** Clear the flag. */
    void unlock();

  private:
    void _pause();
  private:
    std::atomic_flag _lock;
  };
};


inline Pds::SpinLock::SpinLock() : _lock(ATOMIC_FLAG_INIT)
{
}

/* Pause instruction to prevent excess processor bus usage */
inline void Pds::SpinLock::_pause()
{
  asm volatile("pause\n": : :"memory");
  std::this_thread::yield();
}

inline void Pds::SpinLock::lock()
{
  while ( _lock.test_and_set(std::memory_order_acquire) ) { _pause(); }
}

inline void Pds::SpinLock::unlock()
{
  _lock.clear(std::memory_order_release);
}

#endif
