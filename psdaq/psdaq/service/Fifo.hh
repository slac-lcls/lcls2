#ifndef Pds_Fifo_hh
#define Pds_Fifo_hh

#include <atomic>
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
    ~Fifo();
  public:
    bool   push(T item);
    T      pop();
    bool   empty() const;
    size_t size()  const;
    size_t count() const;
  private:
    std::atomic<unsigned>   _head;
    std::atomic<unsigned>   _tail;
  private:
    mutable Pds::SpinLock   _lock;
    size_t                  _size;
    T*                      _vector;
  };
};


template <class T>
inline
Pds::Fifo<T>::Fifo(size_t size) :
  _head(0),
  _tail(0),
  _lock(),
  _size(size + 1),                      // +1 for Full detection
  _vector(new T[_size])
{
}

template <class T>
inline
Pds::Fifo<T>::~Fifo()
{
  delete [] _vector;
}

template <class T>
inline
bool Pds::Fifo<T>::push(T item)
{
  std::lock_guard<Pds::SpinLock> lk(_lock);

  unsigned tail(_tail);
  tail = (tail + 1) % _size;
  bool rc = tail != _head;
  if (rc)                               // Can't push when full
  {
    _vector[_tail] = item;

    _tail = tail;
  }

  return rc;
}

template <class T>
inline
T Pds::Fifo<T>::pop()
{
  std::lock_guard<Pds::SpinLock> lk(_lock);

  unsigned head(_head);
  _head = (head + 1) % _size;

  return _vector[head];
}

template <class T>
inline
bool Pds::Fifo<T>::empty() const
{
  std::lock_guard<Pds::SpinLock> lk(_lock);

  return _head == _tail;
}

template <class T>
inline
size_t Pds::Fifo<T>::size() const
{
  return _size - 1;
}

template <class T>
inline
size_t Pds::Fifo<T>::count() const
{
  std::lock_guard<Pds::SpinLock> lk(_lock);

  return (_size + _tail - _head) % _size;
}


namespace Pds
{
  template <class T, class L = std::mutex>
  class FifoW : public Fifo<T>
  {
  public:
    FifoW(size_t size);
    FifoW(size_t size, T empty);
    ~FifoW();
  public:
    bool   push(T item);
    T      pop();
    T      pop(const std::chrono::milliseconds& tmo);
  private:
    mutable L               _lock;
    std::condition_variable _cv;
    T                       _empty;
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
Pds::FifoW<T, L>::FifoW(size_t size, T empty) :
  Fifo<T>(size),
  _lock(),
  _cv(),
  _empty(empty)
{
}

template <class T, class L>
inline
Pds::FifoW<T, L>::~FifoW()
{
}

template <class T, class L>
inline
bool Pds::FifoW<T, L>::push(T item)
{
  std::lock_guard<L> lock(_lock);

  bool rc = Fifo<T>::push(item);
  if (rc)
    _cv.notify_one();

  return rc;
}

template <class T, class L>
inline
T Pds::FifoW<T, L>::pop()
{
  std::unique_lock<L> lock(_lock);
  _cv.wait(lock, [this] { return !Fifo<T>::empty(); }); // Block when empty

  return Fifo<T>::pop();
}

template <class T, class L>
inline
T Pds::FifoW<T, L>::pop(const std::chrono::milliseconds& tmo)
{
  std::unique_lock<L> lock(_lock);
  if (!_cv.wait_for(lock, tmo, [this] { return !Fifo<T>::empty(); })) // Block when empty
    return _empty;

  return Fifo<T>::pop();
}

#endif
