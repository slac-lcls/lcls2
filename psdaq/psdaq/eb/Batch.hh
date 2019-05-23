#ifndef Pds_Eb_Batch_hh
#define Pds_Eb_Batch_hh

#include "eb.hh"

#include "xtcdata/xtc/Dgram.hh"

#ifdef NDEBUG
//#undef NDEBUG
#endif

#include <cassert>
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
      Batch(void* buffer, AppPrm* appPrms);
    public:
      void*                 operator new(size_t, Pds::Eb::Batch&);
    public:
      static uint64_t       batchId(uint64_t id);
    public:
      XtcData::Dgram*       allocate();
      Batch*                initialize(uint64_t id);
      void                  release();
      void                  store(uint64_t pid, const void* appPrm);
      const void*           retrieve(uint64_t pid) const;
      void                  result(const XtcData::Dgram*);
      const XtcData::Dgram* result() const;
      uint64_t              id() const;
      unsigned              index() const;
      size_t                extent() const;
      const void*           buffer() const;
      const bool            empty() const;
      bool                  expired(uint64_t pid) const;
      void                  dump() const;
    private:
      XtcData::Dgram* const _buffer;  // Pointer to RDMA space for this Batch
      uint64_t              _id;      // Id of Batch, in case it remains empty
      XtcData::Dgram*       _dg;      // Pointer to the current Dgram entry
      AppPrm* const         _appPrms; // Pointer to AppPrms array for this Batch
      const XtcData::Dgram* _result;  // For when Batch is handled out of order
    };
  };
};


inline
void* Pds::Eb::Batch::operator new(size_t, Pds::Eb::Batch& batch)
{
  return &batch;
}

inline
uint64_t Pds::Eb::Batch::batchId(uint64_t id)
{
  return (id >> __builtin_ctzl(BATCH_DURATION)) & (MAX_BATCHES - 1); // Batch number
}

inline
unsigned Pds::Eb::Batch::index() const
{
  return batchId(_id);
}

inline
const void* Pds::Eb::Batch::buffer() const
{
  return _buffer;
}

inline
const bool Pds::Eb::Batch::empty() const
{
  return _dg == nullptr;
}

inline
Pds::Eb::Batch* Pds::Eb::Batch::initialize(uint64_t id)
{
  // Multiple batches can exist with the same BatchId, but different PIDs
  _id = id;                             // Full PID, not BatchId
  _dg = nullptr;

  return this;
}

inline
void Pds::Eb::Batch::release()
{
  _result = nullptr;
}

inline
XtcData::Dgram* Pds::Eb::Batch::allocate()
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

  return dg;
}

inline
void Pds::Eb::Batch::store(uint64_t pid, const void* appPrm)
{
  unsigned idx = pid & (MAX_ENTRIES - 1);

  _appPrms[idx] = reinterpret_cast<uintptr_t>(appPrm);
}

inline
const void* Pds::Eb::Batch::retrieve(uint64_t pid) const
{
  unsigned idx = pid & (MAX_ENTRIES - 1);

  return reinterpret_cast<void*>((uintptr_t)_appPrms[idx]);
}

inline
void Pds::Eb::Batch::result(const XtcData::Dgram* batch)
{
  _result = batch;
}

inline
const XtcData::Dgram* Pds::Eb::Batch::result() const
{
  return _result;
}

inline
size_t Pds::Eb::Batch::extent() const
{
  return (reinterpret_cast<char*>(_dg->xtc.next()) -
          reinterpret_cast<char*>(_buffer));
}

inline
uint64_t Pds::Eb::Batch::id() const
{
  // Multiple batches can exist with the same BatchId, but different PIDs
  return _id;                           // Full PID, not BatchId
}

inline
bool Pds::Eb::Batch::expired(uint64_t pid) const
{
  return ((id() ^ pid) & ((MAX_BATCHES - 1) << __builtin_ctzl(BATCH_DURATION))) != 0;
}

#endif
