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
      Batch(void* buffer, size_t bufSize, AppPrm* appPrms);
    public:
      void*                 operator new(size_t, Pds::Eb::Batch&);
      bool                  operator()(Batch* const& lhs, Batch* const& rhs) const;
    public:
      static uint64_t       batchNum(uint64_t pid);
    public:
      XtcData::Dgram*       allocate();
      Batch*                initialize(const XtcData::Transition&);
      void                  accumRogs(const XtcData::Transition&);
      uint16_t              rogsRem(const XtcData::Transition&);
      void                  accumRcvrs(uint64_t receivers);
      uint64_t              receivers() const;
      size_t                terminate() const;
      void                  release();
      void                  store(uint64_t pid, const void* appPrm);
      const void*           retrieve(uint64_t pid) const;
      void                  result(const XtcData::Dgram*);
      const XtcData::Dgram* result() const;
      size_t                size() const;
      uint64_t              id() const;
      unsigned              index() const;
      const void*           buffer() const;
      const bool            empty() const;
      bool                  expired(uint64_t pid) const;
      void                  dump() const;
    private:
      void* const           _buffer;   // Pointer to RDMA space for this Batch
      size_t                _size;     // Size of entries
      uint64_t              _id;       // Id of Batch, in case it remains empty
      unsigned              _entries;  // Number of entries in this batch
      unsigned              _rogs;     // Readout groups that contributed
      uint64_t              _receivers;// Destinations for this batch
      AppPrm* const         _appPrms;  // Pointer to AppPrms array for this Batch
      const XtcData::Dgram* _result;   // For when Batch is handled out of order
    };
  };
};


inline
void* Pds::Eb::Batch::operator new(size_t, Pds::Eb::Batch& batch)
{
  return &batch;
}

inline
bool Pds::Eb::Batch::operator()(Batch* const& lhs, Batch* const& rhs) const
{
  return lhs->_id < rhs->_id;
}

inline
uint64_t Pds::Eb::Batch::batchNum(uint64_t pid)
{
  return (pid >> __builtin_ctzl(BATCH_DURATION)) & (MAX_BATCHES - 1); // Batch number
}

inline
unsigned Pds::Eb::Batch::index() const
{
  return batchNum(_id);
}

inline
const void* Pds::Eb::Batch::buffer() const
{
  return _buffer;
}

inline
size_t Pds::Eb::Batch::size() const
{
  return _size;
}

inline
const bool Pds::Eb::Batch::empty() const
{
  return _entries == 0;
}

inline
Pds::Eb::Batch* Pds::Eb::Batch::initialize(const XtcData::Transition& hdr)
{
  // Multiple batches can exist with the same BatchNum, but different PIDs
  _id        = hdr.seq.pulseId().value(); // Full PID, not BatchNum
  _rogs      = hdr.readoutGroups();
  _receivers = 0;
  _entries   = 0;

  return this;
}

inline
void Pds::Eb::Batch::accumRogs(const XtcData::Transition& hdr)
{
  _rogs |= hdr.readoutGroups();
}

inline
uint16_t Pds::Eb::Batch::rogsRem(const XtcData::Transition& hdr)
{
  _rogs &= ~hdr.readoutGroups();

  return _rogs;
}

inline
void Pds::Eb::Batch::accumRcvrs(uint64_t receivers)
{
  _receivers |= receivers;
}

inline
uint64_t Pds::Eb::Batch::receivers() const
{
  return _receivers;
}

inline
void Pds::Eb::Batch::release()
{
  _result = nullptr;
}

inline
XtcData::Dgram* Pds::Eb::Batch::allocate()
{
  char* buf = static_cast<char*>(_buffer) + _entries++ * _size;
  return reinterpret_cast<XtcData::Dgram*>(buf);
}

inline
size_t Pds::Eb::Batch::terminate() const
{
  size_t size = _entries * _size;

  if (_entries < MAX_ENTRIES)
  {
    char*           buf = static_cast<char*>(_buffer) + size;
    XtcData::Dgram* dg  = reinterpret_cast<XtcData::Dgram*>(buf);
    dg->seq = XtcData::Sequence(XtcData::TimeStamp(), XtcData::PulseId());
    size += sizeof(XtcData::PulseId);
  }
  return size;
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
uint64_t Pds::Eb::Batch::id() const
{
  // Multiple batches can exist with the same BatchNum, but different PIDs
  return _id;                           // Full PID, not BatchNum
}

inline
bool Pds::Eb::Batch::expired(uint64_t pid) const
{
  //return ((id() ^ pid) & ((MAX_BATCHES - 1) << __builtin_ctzl(BATCH_DURATION))) != 0;

  return (pid & ~(BATCH_DURATION - 1)) > (id() & ~(BATCH_DURATION - 1));
}

#endif
