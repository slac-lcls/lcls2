/**
 * @file
 * @brief Index pools: fixed arrays of equal-size objects addressed by a masked key, with non-waiting (IndexPoolBase, IndexPool) and waiting (IndexPoolBaseW, IndexPoolW) variants.
 */
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
#include <cstdio>


namespace Pds
{
  namespace Eb
  {
    /** Fixed array of a power-of-2 number of equal-size objects, addressed by a key masked to the array size, with a per-slot allocated flag and allocation counters. Not thread-safe; see IndexPoolBaseW. */
    class IndexPoolBase
    {
    public:
      /** Allocate numberofObjects slots of sizeofObject bytes (16-byte aligned storage), all free. Throws a C string if numberofObjects is not a power of 2. */
      IndexPoolBase(size_t sizeofObject, unsigned numberofObjects);
      /** Does nothing; empty body. */
      ~IndexPoolBase();
    public:
      /** Return true if the slot for key (masked) is allocated. */
      bool            isAllocated(unsigned key) const;
      /** Mark the slot for key (masked) allocated and return its address, or return nullptr if it is already allocated. */
      void*           allocate(unsigned key);
      /** Mark the slot for key (masked) free and count a free (also counted if the slot was not allocated). */
      void            free(unsigned key);
      /** Free the slot that contains buf (found with index()). */
      void            free(const void* buf);
    public:
      /** Return the start of the object storage. */
      const void*     buffer() const;
      /** Return the storage size in bytes. */
      size_t          size  () const;
      /** Return the key mask (number of objects minus 1). */
      unsigned        mask  () const;
    public:
      /** Return a reference to the first byte of the slot for key (masked); the allocated flag is not checked. */
      char&           operator[](unsigned key);
      /** Return a const reference to the first byte of the slot for key (masked); the allocated flag is not checked. */
      const char&     operator[](unsigned key) const;
      /** Return the slot index of buf (its byte offset from the storage start divided by the object size). Note that the body's assert(key & ~_mask) fails for in-range keys when assertions are enabled. */
      unsigned        index(const void* buf) const;
    public:
      /** Return the object size in bytes. */
      size_t          sizeofObject()              const;
      /** Return the number of slots (mask + 1). */
      size_t          numberofObjects()           const;
      /** Return a reference to the allocation counter. */
      const uint64_t& numberofAllocs()            const;
      /** Return a reference to the free counter. */
      const uint64_t& numberofFrees()             const;
      /** Return the number of allocations minus the number of frees. */
      int64_t         numberofAllocatedObjects()  const;
      /** Return numberofObjects() minus numberofAllocatedObjects(). */
      int64_t         numberofFreeObjects()       const;
    public:
      /** Mark all slots free and zero both counters. */
      void            clear();
      /** Print the storage size and address, the mask, the object size, the slot count and the counters to stdout. */
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
    throw "numberofObjects must be a power of 2";
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
    /** IndexPoolBase whose allocate() waits for a busy slot to be freed, using a lock of type L and a condition variable. */
    template <class L = std::mutex>
    class IndexPoolBaseW : public IndexPoolBase
    {
    public:
      /** Construct the base pool with all slots free and the waiting count at 0. */
      IndexPoolBaseW(size_t sizeofObject, unsigned numberofObjects);
      /** Does nothing; empty body. */
      ~IndexPoolBaseW();
    public:
      /** Set the stopping flag and wake all callers waiting in allocate(). */
      void  stop();
    public:
      /** Allocate the slot for key. If it is busy, spin for up to 100 ms, then block until it is freed or stop() is called. Returns the slot address, or nullptr if the slot is still allocated (for example after stop()). */
      void* allocate(unsigned key);
      /** Free the slot for key while holding the lock, then wake one waiter. */
      void  free(unsigned key);
      /** Free the slot containing buf while holding the lock, then wake one waiter. */
      void  free(const void* buf);
    public:
      /** Return a reference to the count of callers currently waiting in allocate(). */
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
    /** Typed index pool over base IPB (default IndexPoolBase) whose slots each hold a T plus size extra bytes. */
    template <class T, class IPB = IndexPoolBase>
    class IndexPool : public IPB
    {
    public:
      /** Construct the base pool with numberofObjects slots of sizeof(T) + size bytes. */
      IndexPool(unsigned numberofObjects,
                size_t   size = 0) : IPB(sizeof(T) + size, numberofObjects) { }
      /** Does nothing; empty body. */
      ~IndexPool() { }
    public:
      // Revisit: These are 'static', so 'this' is undefined
      //void* operator new   (size_t size,
      //                      unsigned key)     { return IndexPoolBase::allocate(key); }
      //void  operator delete(void* buf)        { IndexPoolBase::free(buf); }
      /** Return IPB::allocate(key) cast to T*. */
      T*       allocate(unsigned key)         { return (T*)IPB::allocate(key); }
      /** Return the storage start cast to const T*. */
      const T* buffer() const                 { return (T*)IPB::buffer(); }
      /** Return the slot for key as a T reference. */
      T&       operator[](unsigned key)       { return (T&)IPB::operator[](key); }
      /** Return the slot for key as a const T reference. */
      const T& operator[](unsigned key) const { return (T&)IPB::operator[](key); }
    };
  };
};


namespace Pds
{
  namespace Eb
  {
    /** Typed index pool over `IndexPoolBaseW<L>`, so allocate() waits for busy slots. */
    template <class T, class L = std::mutex>
    class IndexPoolW : public IndexPool<T, IndexPoolBaseW<L> >
    {
    public:
      /** Construct the base pool with numberofObjects slots of sizeof(T) + size bytes. */
      IndexPoolW(unsigned numberofObjects,
                 size_t   size = 0) :
        IndexPool<T, IndexPoolBaseW<L> >(numberofObjects, size) { }
      /** Does nothing; empty body. */
      ~IndexPoolW() {}
    public:
      /** Return `IndexPoolBaseW<L>::allocate(key)`, the waiting allocation, cast to T*. */
      T* allocate(unsigned key)
      {
        return (T*)IndexPoolBaseW<L>::allocate(key);
      }
    };
  };
};

#endif
