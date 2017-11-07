#ifndef Pds_Eb_Batch_hh
#define Pds_Eb_Batch_hh

#include "Endpoint.hh"

#include "xtcdata/xtc/Dgram.hh"
#include "psdaq/service/Queue.hh"
#include "psdaq/service/Pool.hh"

#include <stdint.h>
#include <cstddef>


namespace Pds {

  class GenericPoolW;

  namespace Eb {

#define BatchList Queue<Batch>     // Notational convenience...

    class Batch1;

    class Batch : public Pds::Entry
    {
    public:
      Batch(const XtcData::Src&, const XtcData::Dgram&, uint64_t pid);
      ~Batch();
    public:
      static size_t size();
      static void   init(GenericPoolW&,
                         unsigned               batchDepth,
                         unsigned               iovPoolDepth,
                         Fabrics::MemoryRegion* mr[2]);
    public:
      PoolDeclare;
    public:
      void                  append(const XtcData::Dgram&);
      uint64_t              id() const;
      void                  id(uint64_t pid);
      bool                  expired(uint64_t pid);
      unsigned              index() const;
      size_t                extent() const;
      Fabrics::LocalIOVec&  pool() const;
      const XtcData::Dgram* datagram() const;
    private:
      Batch1&              _batch1() const;
    private:
      XtcData::Dgram       _datagram;   // Batch descriptor
    };
  };
};

inline uint64_t Pds::Eb::Batch::id() const
{
  return _datagram.seq.stamp().pulseId();
}

inline void Pds::Eb::Batch::id(uint64_t pid)
{
  XtcData::TimeStamp ts(pid, _datagram.seq.stamp().control());
  _datagram.seq = XtcData::Sequence(_datagram.seq.clock(), ts);
}

inline bool Pds::Eb::Batch::expired(uint64_t pid)
{
  if (id() == pid)  return false;

  if (id() != 0UL)  return true;

  id(pid);                     // Revisit: Happens only on the very first batch
  return false;
}

inline const XtcData::Dgram* Pds::Eb::Batch::datagram() const
{
  return &_datagram;
}

#endif
