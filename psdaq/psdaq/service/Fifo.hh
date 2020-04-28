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
  template <class T>
  class Fifo
  {
  public:
    Fifo(size_t size);
  public:
    void          clear();
    bool          push(const T& item);
    bool          pop(T& item);
    T&            front();
    const T&      front() const;
    const T&      peek(size_t offset) const;
    T&            back();
    const T&      back()  const;
    bool          empty() const;
    size_t        size()  const;
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
  template <class T, class L = Pds::SpinLock>
  class FifoMT : private Fifo<T>        // MT = Multi-Threading
  {
  public:
    FifoMT(size_t size) : Fifo<T>(size) { }
  public:
#define           LCK                       std::lock_guard<L> lk(_lock)
    void          clear()                 { LCK;        Fifo<T>::clear();    }
    bool          push(const T& item)     { LCK; return Fifo<T>::push(item); }
    bool          pop(T& item)            { LCK; return Fifo<T>::pop(item);  }
    T&            front()                 { LCK; return Fifo<T>::front();    }
    const T&      front() const           { LCK; return Fifo<T>::front();    }
    const T&      peek(size_t ofs) const  { LCK; return Fifo<T>::peek(ofs);  }
    T&            back()                  { LCK; return Fifo<T>::back();     }
    const T&      back()  const           { LCK; return Fifo<T>::back();     }
    bool          empty() const           { LCK; return Fifo<T>::empty();    }
    size_t        size()  const           { LCK; return Fifo<T>::size();     }
    const size_t& count() const           { LCK; return Fifo<T>::count();    }
#undef            LCK
  private:
    mutable L _lock;
  };
};


namespace Pds
{
  template <class T, class L = std::mutex>
  class FifoW : public Fifo<T>
  {
  public:
    FifoW(size_t size);
  public:
    bool push(const T& item);
    void pend() const;
    void pend(const std::chrono::milliseconds& tmo) const;
    void pendn() const;
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
