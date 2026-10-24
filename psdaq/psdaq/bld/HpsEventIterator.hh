/**
 * @file
 * @brief Bld::HpsEventIterator, which decodes the events of an HPS BLD packet.
 */
#ifndef HpsEventIterator_hh
#define HpsEventIterator_hh

#include "HpsEvent.hh"

#include <unistd.h>

namespace Bld {
  /** Decodes the events of an HPS BLD packet into one HpsEvent, which can hold up to 31 channel values after it. The first event is decoded by the constructor; next() decodes each later one. */
  class HpsEventIterator {
  public:
    /** Decode the first event of the sz-byte packet at b and check that sz equals the first event's size plus a whole number of later events (see valid()). */
    HpsEventIterator(const char* b, size_t sz) : 
      _buff(reinterpret_cast<const uint32_t*>(b)),
      _end (reinterpret_cast<const uint32_t*>(b+sz)),
      _next(_buff),
      _nch (0),
      _id  (0)
    { _first(sz); }
  public:
    /** Return the result of the packet size check made by the constructor. */
    bool valid() const { return _valid; }
    /** Decode the next event into the current HpsEvent, adding its pulse ID and time stamp offsets to the first event's values. Returns false at the end of the packet. */
    bool next ();
    /** Return the current decoded event. */
    const HpsEvent& operator*() { return v; }
    /** Return the ID, which is initialized to 0 and never set. */
    unsigned id       () const { return _id; }
    /** Return the number of channels, the number of bits set in the first event's channel mask. */
    unsigned nchannels() const { return _nch; }
    /** Return the size of an HpsEvent plus its channel values, in bytes. */
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
