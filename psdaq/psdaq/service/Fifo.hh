/**
 * @file
 * @brief Fifo, a fixed-size ring buffer, with a lock-guarded variant (FifoMT) and a variant whose readers can wait (FifoW).
 */
#ifndef Pds_Fifo_hh
#define Pds_Fifo_hh

#include <atomic>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <chrono>

#include "psdaq/service/SpinLock.hh"


namespace Pds
{
  /** Fixed-size ring buffer of T, not thread-safe. push() and pop() return true when they fail (full or empty). */
  template <class T>
  class Fifo
  {
  public:
    /** Allocate size slots and start empty. */
    Fifo(size_t size);
  public:
    /** Make the FIFO empty (the slots are not cleared). */
    void          clear();
    /** Copy item to the back; returns true, without storing, if the FIFO is full, else false. */
    bool          push(const T& item);
    /** Move the front element into item; returns true, without changing item, if the FIFO is empty, else false. */
    bool          pop(T& item);
    /** Return a reference to the front slot (not checked for emptiness). */
    T&            front();
    /** Return a const reference to the front slot (not checked for emptiness). */
    const T&      front() const;
    /** Return the slot offset positions after the front (wrapping; not checked against count()). */
    const T&      peek(size_t offset) const;
    /** Return a reference to the most recently pushed slot. */
    T&            back();
    /** Return a const reference to the most recently pushed slot. */
    const T&      back()  const;
    /** Return true if the FIFO holds no elements. */
    bool          empty() const;
    /** Return the capacity. */
    size_t        size()  const;
    /** Return a reference to the number of elements held. */
    const size_t& count() const;
  private:
    size_t         _head;
    size_t         _tail;
    size_t         _count;
    std::vector<T> _array;
  };
};


template <class T>
inline
Pds::Fifo<T>::Fifo(size_t size) :
  _head (0),
  _tail (size - 1),
  _count(0),
  _array(size)
{
}

template <class T>
inline
void Pds::Fifo<T>::clear()
{
  _head  = 0;
  _tail  = size() - 1;
  _count = 0;
}

template <class T>
inline
bool Pds::Fifo<T>::push(const T& item)
{
  size_t sz = size();
  if (_count >= sz)  return true;  // Can't push when full

  _tail = (_tail + 1) % sz;        // Optimized to & when size is a power-of-2

  _array[_tail] = item;

  ++_count;

  return false;
}

template <class T>
inline
bool Pds::Fifo<T>::pop(T& item)
{
  if (_count == 0)  return true;   // Can't pop when empty

  item = _array[_head];

  _head = (_head + 1) % size();    // Optimized to & when size is a power-of-2

  --_count;

  return false;
}

template <class T>
inline
T& Pds::Fifo<T>::front()
{
  return _array[_head];
}

template <class T>
inline
const T& Pds::Fifo<T>::front() const
{
  return _array[_head];
}

template <class T>
inline
const T& Pds::Fifo<T>::peek(size_t offset) const
{
  return _array[(_head + offset) % size()];
}

template <class T>
inline
T& Pds::Fifo<T>::back()
{
  return _array[_tail];
}

template <class T>
inline
const T& Pds::Fifo<T>::back() const
{
  return _array[_tail];
}

template <class T>
inline
bool Pds::Fifo<T>::empty() const
{
  return _count == 0;
}

template <class T>
inline
size_t Pds::Fifo<T>::size() const
{
  return _array.size();
}

template <class T>
inline
const size_t& Pds::Fifo<T>::count() const
{
  return _count;
}


namespace Pds
{
  /** Fifo of T whose every operation holds a lock of type L (default Pds::SpinLock) for multi-threaded use; references returned by front(), back() and peek() are used after the lock is released. */
  template <class T, class L = Pds::SpinLock>
  class FifoMT : private Fifo<T>        // MT = Multi-Threading
  {
  public:
    /** Allocate size slots and start empty. */
    FifoMT(size_t size) : Fifo<T>(size) { }
  public:
/** Declares a std::lock_guard on the member lock for the rest of the enclosing body; undefined after the member functions. */
#define           LCK                       std::lock_guard<L> lk(_lock)
    /** Under the lock, make the FIFO empty. */
    void          clear()                 { LCK;        Fifo<T>::clear();    }
    /** Under the lock, copy item to the back; returns true if the FIFO was full. */
    bool          push(const T& item)     { LCK; return Fifo<T>::push(item); }
    /** Under the lock, move the front element into item; returns true if the FIFO was empty. */
    bool          pop(T& item)            { LCK; return Fifo<T>::pop(item);  }
    /** Under the lock, return a reference to the front slot. */
    T&            front()                 { LCK; return Fifo<T>::front();    }
    /** Under the lock, return a const reference to the front slot. */
    const T&      front() const           { LCK; return Fifo<T>::front();    }
    /** Under the lock, return the slot ofs positions after the front. */
    const T&      peek(size_t ofs) const  { LCK; return Fifo<T>::peek(ofs);  }
    /** Under the lock, return a reference to the most recently pushed slot. */
    T&            back()                  { LCK; return Fifo<T>::back();     }
    /** Under the lock, return a const reference to the most recently pushed slot. */
    const T&      back()  const           { LCK; return Fifo<T>::back();     }
    /** Under the lock, return true if the FIFO is empty. */
    bool          empty() const           { LCK; return Fifo<T>::empty();    }
    /** Under the lock, return the capacity. */
    size_t        size()  const           { LCK; return Fifo<T>::size();     }
    /** Under the lock, return a reference to the number of elements. */
    const size_t& count() const           { LCK; return Fifo<T>::count();    }
#undef            LCK
  private:
    mutable L _lock;
  };
};


namespace Pds
{
  /** Fifo of T whose push() and pop() hold a lock of type L (default std::mutex) and notify a condition variable, so other threads can wait in pend() for data or in pendn() for the FIFO to drain. The inherited accessors do not lock. */
  template <class T, class L = std::mutex>
  class FifoW : public Fifo<T>
  {
  public:
    /** Allocate size slots and start empty. */
    FifoW(size_t size);
  public:
    /** Under the lock, copy item to the back and wake one waiter; returns true if the FIFO was full. */
    bool push(const T& item);
    /** Under the lock, move the front element into item and wake one waiter; returns true if the FIFO was empty. */
    bool pop(T& item);
    /** Block until the FIFO is not empty. */
    void pend() const;
    /** Block until the FIFO is not empty or tmo has elapsed. */
    void pend(const std::chrono::milliseconds& tmo) const;
    /** Block until the FIFO is empty. */
    void pendn() const;
    /** Block until the FIFO is empty or tmo has elapsed. */
    void pendn(const std::chrono::milliseconds& tmo) const;
  private:
    mutable L               _lock;
    std::condition_variable _cv;
  };
};


template <class T, class L>
inline
Pds::FifoW<T, L>::FifoW(size_t size) :
  Fifo<T>(size),
  _lock(),
  _cv()
{
}

template <class T, class L>
inline
bool Pds::FifoW<T, L>::push(const T& item)
{
  std::lock_guard<L> lock(_lock);

  bool full = Fifo<T>::push(item);
  _cv.notify_one();

  return full;
}

template <class T, class L>
inline
bool Pds::FifoW<T, L>::pop(T& item)
{
  std::lock_guard<L> lock(_lock);

  bool isempty = Fifo<T>::pop(item);
  _cv.notify_one();

  return isempty;
}

template <class T, class L>
inline
void Pds::FifoW<T, L>::pend() const
{
  std::unique_lock<L> lock(_lock);
  const_cast<std::condition_variable&>(_cv).wait(lock, [this] { return !Fifo<T>::empty(); }); // Block when empty
}

template <class T, class L>
inline
void Pds::FifoW<T, L>::pend(const std::chrono::milliseconds& tmo) const
{
  std::unique_lock<L> lock(_lock);
  const_cast<std::condition_variable&>(_cv).wait_for(lock, tmo, [this] { return !Fifo<T>::empty(); }); // Block when empty
}

template <class T, class L>
inline
void Pds::FifoW<T, L>::pendn() const
{
  std::unique_lock<L> lock(_lock);
  const_cast<std::condition_variable&>(_cv).wait(lock, [this] { return Fifo<T>::empty(); }); // Block when not empty
}

template <class T, class L>
inline
void Pds::FifoW<T, L>::pendn(const std::chrono::milliseconds& tmo) const
{
  std::unique_lock<L> lock(_lock);
  const_cast<std::condition_variable&>(_cv).wait_for(lock, tmo, [this] { return Fifo<T>::empty(); }); // Block when not empty
}



#endif
