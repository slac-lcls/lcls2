#ifndef Pds_Eb_Batch_hh
#define Pds_Eb_Batch_hh

#include "xtcdata/xtc/Dgram.hh"

#include <assert.h>
#include <cstdint>                      // For uint64_t
#include <cstddef>                      // for size_t
#include <atomic>


namespace Pds {
  namespace Eb {

    using AppPrm = std::atomic<uintptr_t>;

    class BatchManager;

    class Batch
    {
    public:
      explicit Batch();
      ~Batch();
    public:
      void            initialize(const XtcData::Dgram* contrib);
    public:
      void*           allocate(size_t);
      void*           allocate(size_t, const void* appPrm);
      void            store(const XtcData::Dgram*);
      unsigned        entries() const;
      uint64_t        id() const;
      bool            expired(uint64_t pid, uint64_t mask) const;
      unsigned        index() const;
      size_t          extent() const;
      XtcData::Dgram* datagram() const;
      const void*     appParm(unsigned idx) const;
    private:
      friend BatchManager;
      void           _fixup(unsigned index, void* buffer, AppPrm* appPrms);
    private:
      unsigned const _index;
      unsigned       _entries;
      void*    const _buffer;
      AppPrm*        _appPrms;
    };
  };
};


inline Pds::Eb::Batch::~Batch()
{
}

inline unsigned Pds::Eb::Batch::index() const
{
  return _index;
}

inline XtcData::Dgram* Pds::Eb::Batch::datagram() const
{
  return (XtcData::Dgram*)_buffer;
}

inline const void* Pds::Eb::Batch::appParm(unsigned idx) const
{
  assert(idx < _entries);

  return (const void*)(uintptr_t)_appPrms[idx];
}

inline void* Pds::Eb::Batch::allocate(size_t size)
{
  ++_entries;

  return datagram()->xtc.alloc(size);
}

inline void* Pds::Eb::Batch::allocate(size_t size, const void* appPrm)
{
  _appPrms[_entries++] = (uintptr_t)appPrm;

  return datagram()->xtc.alloc(size);
}

inline size_t Pds::Eb::Batch::extent() const
{
  const XtcData::Dgram* dg = datagram();
  return sizeof(*dg) + dg->xtc.sizeofPayload();
}

inline unsigned Pds::Eb::Batch::entries() const
{
  return _entries;
}

inline uint64_t Pds::Eb::Batch::id() const
{
  return datagram()->seq.pulseId().value();
}

inline bool Pds::Eb::Batch::expired(uint64_t pid, uint64_t mask) const
{
  return ((id() ^ pid) & mask) != 0;
}

#endif
