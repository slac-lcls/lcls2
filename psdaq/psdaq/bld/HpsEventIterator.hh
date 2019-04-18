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
      _next(_buff),
      _nch (0),
      _id  (0)
    { _first(sz); }
  public:
    bool valid() const { return _valid; }
    bool next ();
    const HpsEvent& operator*() { return v; }
    unsigned id       () const { return _id; }
    unsigned nchannels() const { return _nch; }
    size_t   eventSize() const { return sizeof(HpsEvent)+_nch*sizeof(uint32_t); }
  private:
    void _first(size_t);
  private:
    const uint32_t* _buff;
    const uint32_t* _end;
    const uint32_t* _next;
    uint64_t        _ts;
    uint64_t        _pid;
    uint32_t        _nch;
    uint32_t        _id;
    bool            _valid;
    HpsEvent v;
    uint32_t _reserved[31];
  };
};

#endif
