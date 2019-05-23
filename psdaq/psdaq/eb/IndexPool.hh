#ifndef Pds_Eb_IndexPool_hh
#define Pds_Eb_IndexPool_hh

#include "psdaq/service/fast_monotonic_clock.hh"
#include "psdaq/service/AlignmentAllocator.hh"

#include <cassert>
#include <cstddef>                      // For size_t
#include <vector>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <atomic>


namespace Pds
{
  namespace Eb
  {
    class IndexPoolBase
    {
    public:
      IndexPoolBase(size_t sizeofObject, unsigned numberofObjects);
      ~IndexPoolBase();
    public:
      bool            isAllocated(unsigned key) const;
      void*           allocate(unsigned key);
      void            free(unsigned key);
      void            free(const void* buf);
    public:
      const void*     buffer() const;
      size_t          size  () const;
      unsigned        mask  () const;
    public:
      char&           operator[](unsigned key);
      const char&     operator[](unsigned key) const;
      unsigned        index(const void* buf) const;
    public:
      size_t          sizeofObject()              const;
      size_t          numberofObjects()           const;
      const uint64_t& numberofAllocs()            const;
      const uint64_t& numberofFrees()             const;
      int64_t         numberofAllocatedObjects()  const;
      int64_t         numberofFreeObjects()       const;
    public:
      void            clear();
      void            dump() const;
    private:
      const unsigned                                   _mask;
      const size_t                                     _sizeofObject;
      std::vector<bool>                                _allocated;
      std::vector<char, AlignmentAllocator<char, 16> > _buffer;
      uint64_t                                         _numberofAllocs;
      uint64_t                                         _numberofFrees;
    };
  };
};


inline
Pds::Eb::IndexPoolBase::IndexPoolBase(size_t   sizeofObject,
                                      unsigned numberofObjects) :
  _mask(numberofObjects - 1),
  _sizeofObject(sizeofObject),
  _allocated(numberofObjects, false),
  _buffer(numberofObjects * sizeofObject),
  _numberofAllocs(0),
  _numberofFrees(0)
{
  if (numberofObjects & _mask)
  {
    fprintf(stderr, "%s: numberofObjects (0x%0x = %d) must be a power of 2\n",
            __func__, numberofObjects, numberofObjects);
    abort();
  }
}

inline
Pds::Eb::IndexPoolBase::~IndexPoolBase()
{
}

inline
const void*  Pds::Eb::IndexPoolBase::buffer() const
{
  return _buffer.data();
}

inline
size_t Pds::Eb::IndexPoolBase::size() const
{
  return _buffer.size();
}

inline
unsigned Pds::Eb::IndexPoolBase::mask() const
{
  return _mask;
}

inline
char& Pds::Eb::IndexPoolBase::operator[](unsigned key)
{
  return _buffer[(key & _mask) * _sizeofObject];
}

inline
const char& Pds::Eb::IndexPoolBase::operator[](unsigned key) const
{
  return _buffer[(key & _mask) * _sizeofObject];
}

inline
unsigned Pds::Eb::IndexPoolBase::index(const void* buf) const
{
  unsigned key = (static_cast<const char*>(buf) - _buffer.data()) / _sizeofObject;

  assert (key & ~_mask);

  return key;
}

inline
size_t Pds::Eb::IndexPoolBase::sizeofObject() const
{
  return _sizeofObject;
}

inline
size_t Pds::Eb::IndexPoolBase::numberofObjects() const
{
  return _mask + 1;
}

inline
const uint64_t& Pds::Eb::IndexPoolBase::numberofAllocs() const
{
  return _numberofAllocs;
}

inline
const uint64_t& Pds::Eb::IndexPoolBase::numberofFrees() const
{
  return _numberofFrees;
}

inline
int64_t Pds::Eb::IndexPoolBase::numberofAllocatedObjects() const
{
   return _numberofAllocs - _numberofFrees;
}

inline
int64_t Pds::Eb::IndexPoolBase::numberofFreeObjects() const
{
  return numberofObjects() - numberofAllocatedObjects();
}

inline
bool Pds::Eb::IndexPoolBase::isAllocated(unsigned key) const
{
  return _allocated[key & _mask];
}

inline
void* Pds::Eb::IndexPoolBase::allocate(unsigned key)
{
  key &= _mask;

  if (_allocated[key])  return nullptr;

  _allocated[key] = true;

  ++_numberofAllocs;

  return &_buffer[key * _sizeofObject];
}

inline
void Pds::Eb::IndexPoolBase::free(unsigned key)
{
  _allocated[key & _mask] = false;

  ++_numberofFrees;
}

inline
void Pds::Eb::IndexPoolBase::free(const void* buf)
{
  free(index(buf));
}


namespace Pds
{
  namespace Eb
  {
    template <class L = std::mutex>
    class IndexPoolBaseW : public IndexPoolBase
    {
    public:
      IndexPoolBaseW(size_t sizeofObject, unsigned numberofObjects);
      ~IndexPoolBaseW();
    public:
      void  stop();
    public:
      void* allocate(unsigned key);
      void  free(unsigned key);
      void  free(const void* buf);
    public:
      const uint64_t& waiting() const;
    private:
      mutable L                   _lock;
      std::condition_variable_any _cv;
      uint64_t                    _waiting;
      std::atomic<bool>           _stopping;
    };
  };
};


template <class L>
inline
Pds::Eb::IndexPoolBaseW<L>::IndexPoolBaseW(size_t   sizeofObject,
                                           unsigned numberofObjects) :
  IndexPoolBase(sizeofObject, numberofObjects),
  _lock(),
  _cv(),
  _waiting(0),
  _stopping(false)
{
}

template <class L>
inline
Pds::Eb::IndexPoolBaseW<L>::~IndexPoolBaseW()
{
}

template <class L>
inline
void Pds::Eb::IndexPoolBaseW<L>::stop()
{
  std::lock_guard<L> lk(_lock);
  _stopping = true;
  _cv.notify_all();
}

template <class L>
inline
void* Pds::Eb::IndexPoolBaseW<L>::allocate(unsigned key)
{
  if (isAllocated(key))
  {
    ++_waiting;

    auto t0 = Pds::fast_monotonic_clock::now();
    while (isAllocated(key))
    {
      using ms_t = std::chrono::milliseconds;
      auto  t1   = Pds::fast_monotonic_clock::now();

      if (std::chrono::duration_cast<ms_t>(t1 - t0).count() > 100)
      {
        std::unique_lock<L> lock(_lock);
        _waiting += 1;
        _cv.wait(lock, [this, key] {
                       return !isAllocated(key) || _stopping; });
        _waiting -= 2;

        return IndexPoolBase::allocate(key);
      }
    }

    --_waiting;
  }

  std::unique_lock<L> lock(_lock);
  return IndexPoolBase::allocate(key);
}

template <class L>
inline
void Pds::Eb::IndexPoolBaseW<L>::free(unsigned key)
{
  {
    std::lock_guard<L> lock(_lock);

    IndexPoolBase::free(key);
  }
  _cv.notify_one();
}

template <class L>
inline
void Pds::Eb::IndexPoolBaseW<L>::free(const void* buf)
{
  {
    std::lock_guard<L> lock(_lock);

    IndexPoolBase::free(buf);
  }
  _cv.notify_one();
}

template <class L>
inline
const uint64_t& Pds::Eb::IndexPoolBaseW<L>::waiting() const
{
  return _waiting;
}


namespace Pds
{
  namespace Eb
  {
    template <class T, class IPB = IndexPoolBase>
    class IndexPool : public IPB
    {
    public:
      IndexPool(unsigned numberofObjects,
                size_t   size = 0) : IPB(sizeof(T) + size, numberofObjects) { }
      ~IndexPool() { }
    public:
      // Revisit: These are 'static', so 'this' is undefined
      //void* operator new   (size_t size,
      //                      unsigned key)     { return IndexPoolBase::allocate(key); }
      //void  operator delete(void* buf)        { IndexPoolBase::free(buf); }
      T*       allocate(unsigned key)         { return (T*)IPB::allocate(key); }
      const T* buffer() const                 { return (T*)IPB::buffer(); }
      T&       operator[](unsigned key)       { return (T&)IPB::operator[](key); }
      const T& operator[](unsigned key) const { return (T&)IPB::operator[](key); }
    };
  };
};


namespace Pds
{
  namespace Eb
  {
    template <class T, class L = std::mutex>
    class IndexPoolW : public IndexPool<T, IndexPoolBaseW<L> >
    {
    public:
      IndexPoolW(unsigned numberofObjects,
                 size_t   size = 0) :
        IndexPool<T, IndexPoolBaseW<L> >(numberofObjects, size) { }
      ~IndexPoolW() {}
    public:
      T* allocate(unsigned key)
      {
        return (T*)IndexPoolBaseW<L>::allocate(key);
      }
    };
  };
};

#endif
