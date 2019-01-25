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
      Batch();
      Batch(unsigned index, void* buffer, AppPrm* appPrms);
      ~Batch();
    public:
      void*           operator new(size_t, Pds::Eb::Batch&);
    public:
      XtcData::Dgram* buffer();
      XtcData::Dgram* buffer(const void* appPrm);
      void            release();
      unsigned        entries() const;
      uint64_t        id() const;
      bool            expired(uint64_t pid, uint64_t mask) const;
      unsigned        index() const;
      size_t          extent() const;
      const void*     batch() const;
      const void*     appParm(unsigned idx) const;
    private:
      XtcData::Dgram* const _buffer;
      unsigned const        _index;
      unsigned              _entries;
      XtcData::Dgram*       _dg;
      AppPrm*               _appPrms;
    };
  };
};


inline Pds::Eb::Batch::~Batch()
{
}

inline void* Pds::Eb::Batch::operator new(size_t, Pds::Eb::Batch& batch)
{
  return &batch;
}

inline unsigned Pds::Eb::Batch::index() const
{
  return _index;
}

inline const void* Pds::Eb::Batch::batch() const
{
  return _buffer;
}

inline const void* Pds::Eb::Batch::appParm(unsigned idx) const
{
  assert(idx < _entries);

  return reinterpret_cast<void*>((uintptr_t)_appPrms[idx]);
}

inline XtcData::Dgram* Pds::Eb::Batch::buffer()
{
  XtcData::Dgram* dg = _dg;

  if (dg)
  {
    dg->seq.markBatch();

    dg = reinterpret_cast<XtcData::Dgram*>(dg->xtc.next());
  }
  else
  {
    dg = _buffer;
  }

  _dg = dg;

  ++_entries;

  return dg;
}

inline XtcData::Dgram* Pds::Eb::Batch::buffer(const void* appPrm)
{
  _appPrms[_entries] = (uintptr_t)appPrm;

  return buffer();
}

inline void Pds::Eb::Batch::release()
{
  _entries = 0;
  _dg      = nullptr;
}

inline size_t Pds::Eb::Batch::extent() const
{
  return (reinterpret_cast<char*>(_dg->xtc.next()) -
          reinterpret_cast<char*>(_buffer));
}

inline unsigned Pds::Eb::Batch::entries() const
{
  return _entries;
}

inline uint64_t Pds::Eb::Batch::id() const
{
  return _buffer->seq.pulseId().value();
}

inline bool Pds::Eb::Batch::expired(uint64_t pid, uint64_t mask) const
{
  return ((id() ^ pid) & mask) != 0;
}

#endif
