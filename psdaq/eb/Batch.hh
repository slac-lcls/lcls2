#ifndef Pds_Eb_Batch_hh
#define Pds_Eb_Batch_hh

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
      static void   init(GenericPoolW& pool,
                         char*         buffer,
                         unsigned      depth,
                         size_t        maxSize);
    public:
      PoolDeclare;
    public:
      void*            allocate(size_t);
      uint64_t         id() const;
      void             id(uint64_t pid);
      bool             expired(uint64_t pid);
      unsigned         index() const;
      size_t           extent() const;
      void*            pool() const;
      XtcData::Dgram*  datagram() const;
      void             parameter(void*);
      void*            parameter() const;
    private:
      Batch1&         _batch1() const;
    private:
      void*           _parameter;
    };
  };
};


inline XtcData::Dgram* Pds::Eb::Batch::datagram() const
{
  return (XtcData::Dgram*)pool();
}

inline void Pds::Eb::Batch::parameter(void* parameter)
{
  _parameter = parameter;
}

inline void* Pds::Eb::Batch::parameter() const
{
  return _parameter;
}

inline uint64_t Pds::Eb::Batch::id() const
{
  return datagram()->seq.stamp().pulseId();
}

inline void Pds::Eb::Batch::id(uint64_t pid)
{
  XtcData::Dgram*    dg = datagram();
  XtcData::TimeStamp ts(pid, dg->seq.stamp().control());
  dg->seq = XtcData::Sequence(dg->seq.clock(), ts);
}

inline bool Pds::Eb::Batch::expired(uint64_t pid)
{
  return id() != pid;
}

#endif
