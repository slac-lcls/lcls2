#ifndef Pds_Eb_IndexPool_hh
#define Pds_Eb_IndexPool_hh

#include <cassert>
#include <cstddef>                      // For size_t
#include <vector>
#include <mutex>
#include <condition_variable>
#include <chrono>

#include "psdaq/service/AlignmentAllocator.hh"


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
      void            free(void* buf);
      void            free(const void* buf);
    public:
      const void*     buffer() const;
      size_t          size  () const;
      unsigned        mask  () const;
    public:
      char&           operator[](unsigned key);
      const char&     operator[](unsigned key) const;
      unsigned        index(const char* buf) const;
      unsigned        index(const char& buf) const;
    public:
      size_t          sizeofObject()              const;
      size_t          numberofObjects()           const;
      const uint64_t& numberofAllocs()            const;
      const uint64_t& numberofFrees()             const;
      int64_t         numberofAllocatedObjects()  const;
      int64_t         numberofFreeObjects()       const;
    public:
      void            dump() const;
    protected:
      unsigned                                         _mask;
      size_t                                           _sizeofObject;
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
char&  Pds::Eb::IndexPoolBase::operator[](unsigned key)
{
  return _buffer[(key & _mask) * _sizeofObject];
}

inline
const char&  Pds::Eb::IndexPoolBase::operator[](unsigned key) const
{
  return _buffer[(key & _mask) * _sizeofObject];
}

inline
unsigned Pds::Eb::IndexPoolBase::index(const char* buf) const
{
  unsigned key = (buf - _buffer.data()) / _sizeofObject;

  assert (key & ~_mask);

  return key;
}

inline
unsigned Pds::Eb::IndexPoolBase::index(const char& buf) const
{
  return index(&buf);
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

  if (!_allocated[key])
  {
    ++_numberofAllocs;

    _allocated[key] = true;

    return &_buffer[key * _sizeofObject];
  }
  else
    return nullptr;
}

inline
void Pds::Eb::IndexPoolBase::free(unsigned key)
{
  _allocated[key & _mask] = false;

  ++_numberofFrees;
}

inline
void Pds::Eb::IndexPoolBase::free(void* buf)
{
  free(index(static_cast<char*>(buf)));
}

inline
void Pds::Eb::IndexPoolBase::free(const void* buf)
{
  free(const_cast<void*>(buf));
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
      void* allocate(unsigned key);
      void* allocate(unsigned key, const std::chrono::milliseconds& tmo);
      void  free(unsigned key);
      void  free(void* buf);
      void  free(const void* buf);
    public:
      const uint64_t& waiting() const;
    private:
      mutable L               _lock;
      std::condition_variable _cv;
      uint64_t                _waiting;
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
  _waiting(0)
{
}

template <class L>
inline
Pds::Eb::IndexPoolBaseW<L>::~IndexPoolBaseW()
{
}

template <class L>
inline
void* Pds::Eb::IndexPoolBaseW<L>::allocate(unsigned key)
{
  ++_waiting;

  std::unique_lock<L> lock(_lock);
  _cv.wait(lock, [this, key] { return !isAllocated(key); });

  --_waiting;

  return IndexPoolBase::allocate(key);
}

template <class L>
inline
void* Pds::Eb::IndexPoolBaseW<L>::allocate(unsigned key,
                                           const std::chrono::milliseconds& tmo)
{
  ++_waiting;

  std::unique_lock<L> lock(_lock);
  bool ret(!_cv.wait_for(lock, tmo, [this, key] { return !isAllocated(key); }));

  --_waiting;

  if (ret)  return nullptr;

  return IndexPoolBase::allocate(key);
}

template <class L>
inline
void Pds::Eb::IndexPoolBaseW<L>::free(unsigned key)
{
  std::lock_guard<L> lock(_lock);

  IndexPoolBase::free(key);

  _cv.notify_one();
}

template <class L>
inline
void Pds::Eb::IndexPoolBaseW<L>::free(void* buf)
{
  std::lock_guard<L> lock(_lock);

  IndexPoolBase::free(buf);

  _cv.notify_one();
}

template <class L>
inline
void Pds::Eb::IndexPoolBaseW<L>::free(const void* buf)
{
  free(const_cast<void*>(buf));
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
    public:
      const T* buffer() const                 { return (T*)IPB::buffer(); }
      size_t   size  () const                 { return IPB::size(); }
      unsigned mask  () const                 { return IPB::mask(); }
    public:
      T&       operator[](unsigned key)       { return (T&)IPB::operator[](key); }
      const T& operator[](unsigned key) const { return (T&)IPB::operator[](key); }
      unsigned index(const T* buf) const      { return IPB::index((const char*)buf);  }
      unsigned index(const T& buf) const      { return IPB::index((const char*)&buf); }
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
      T* allocate(unsigned key, const std::chrono::milliseconds& tmo)
      {
        return (T*)IndexPoolBaseW<L>::allocate(key, tmo);
      }
    };
  };
};

#endif
