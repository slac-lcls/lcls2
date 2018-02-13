#ifndef Pds_IovElement_hh
#define Pds_IovElement_hh

#include "Endpoint.hh"

#include <sys/socket.h>                 // For struct iovec...

namespace Pds {

  class IovElement : public iovec
  {
  public:
    IovElement() {};
    IovElement(const void* payload, size_t size)
    {
      iov_base = const_cast<void*>(payload);
      iov_len  = size;
    }
    ~IovElement() { };
  public:
    void* operator new   (size_t, Pds::Fabrics::LocalIOVec& pool);
    void  operator delete(void*);
  public:
    void*  payload() const { return iov_base; }
    size_t size()    const { return iov_len;  }
  public:
    void*  next()    const { return (char*)iov_base + iov_len; }
    void   extend(size_t size) { iov_len += size; }
  };
};


inline void* Pds::IovElement::operator new(size_t, Pds::Fabrics::LocalIOVec& pool)
{
  return pool.allocate();
}

inline void Pds::IovElement::operator delete(void*)
{
  // No deallocation
}

#endif
