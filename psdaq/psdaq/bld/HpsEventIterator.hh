#ifndef HpsEventIterator_hh
#define HpsEventIterator_hh

#include "HpsEvent.hh"

#include <unistd.h>

namespace Bld {
  class HpsEventIterator {
  public:
    HpsEventIterator(const char* b, size_t sz) : 
      _buff(reinterpret_cast<const uint32_t*>(b)),
      _end (reinterpret_cast<const uint32_t*>(b+sz)),
      _next(_buff)
    { _first(sz); }
  public:
    bool next ();
    HpsEvent        operator*() { return v; }
  private:
    void _first(size_t);
  private:
    const uint32_t* _buff;
    const uint32_t* _end;
    const uint32_t* _next;
    HpsEvent v;
  };
};

#endif
