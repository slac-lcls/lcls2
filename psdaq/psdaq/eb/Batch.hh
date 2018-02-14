#ifndef Pds_Eb_Batch_hh
#define Pds_Eb_Batch_hh

#include "xtcdata/xtc/Dgram.hh"
#include "psdaq/service/Pool.hh"

#include <stdint.h>
#include <cstddef>


namespace Pds {

  class GenericPoolW;

  namespace Eb {

    class Batch1;

    class Batch
    {
    public:
      enum IsLastFlag { IsLast };
    public:
      Batch(IsLastFlag);
      Batch(const XtcData::Dgram&, uint64_t pid);
      ~Batch();
    public:
      static size_t size();
      static void   init(GenericPoolW&          pool,
                         char*                  buffer,
                         unsigned               depth,
                         size_t                 maxSize,
                         const XtcData::Dgram** datagrams,
                         unsigned               maxEntries,
                         Batch**                batches);
    public:
      PoolDeclare;
    public:
      void*                 allocate(size_t);
      void                  store(const XtcData::Dgram*);
      uint64_t              id() const;
      void                  id(uint64_t pid);
      bool                  expired(uint64_t pid) const;
      unsigned              index() const;
      size_t                extent() const;
      void*                 buffer() const;
      XtcData::Dgram*       datagram() const;
      void                  parameter(void*);
      void*                 parameter() const;
      bool                  isLast() const;
      const XtcData::Dgram* datagram(unsigned i) const;
    private:
      Batch1* _batch1() const;
    private:
      void*   _parameter;
    };
  };
};


inline XtcData::Dgram* Pds::Eb::Batch::datagram() const
{
  return (XtcData::Dgram*)buffer();
}

inline void Pds::Eb::Batch::parameter(void* parameter)
{
  _parameter = parameter;
}

inline void* Pds::Eb::Batch::parameter() const
{
  return _parameter;
}

inline bool Pds::Eb::Batch::isLast() const
{
  return _parameter == this;
}

inline uint64_t Pds::Eb::Batch::id() const
{
  return datagram()->seq.pulseId().value();
}

inline void Pds::Eb::Batch::id(uint64_t pid)
{
  XtcData::Dgram*  dg = datagram();
  XtcData::PulseId pulseId(pid, dg->seq.pulseId().control());
  dg->seq = XtcData::Sequence(dg->seq.stamp(), pulseId);
}

inline bool Pds::Eb::Batch::expired(uint64_t pid) const
{
  return id() != pid;
}

#endif
