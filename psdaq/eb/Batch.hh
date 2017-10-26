#ifndef Pds_Eb_Batch_hh
#define Pds_Eb_Batch_hh

#include "IovPool.hh"

//#include "psdaq/xtc/Datagram.hh"
#include "xtcdata/xtc/Dgram.hh"
#include "psdaq/service/Queue.hh"
#include "psdaq/service/Pool.hh"

#include <rdma/fi_rma.h>

#include <stdint.h>
#include <cstddef>


namespace Pds {

  class GenericPoolW;

  namespace Eb {

#define BatchList Queue<Batch>     // Notational convenience...

    class Batch : public Pds::Entry
    {
    public:
      Batch(const XtcData::Dgram&, uint64_t pid);
      ~Batch();
    public:
      static size_t size();
      static void   init(GenericPoolW&, unsigned batchDepth, unsigned iovPoolDepth);
    public:
      PoolDeclare;
    public:
      void               append(const XtcData::Dgram&);
      uint64_t           id() const;
      void               id(uint64_t pid);
      bool               expired(uint64_t pid);
      struct fi_msg_rma* finalize();
      unsigned           index() const;
    private:
      XtcData::Dgram    _datagram;      // Batch descriptor
      struct fi_rma_iov _rmaIov;        // Destination descriptor
      struct fi_msg_rma _rmaMsg;
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

  if (pid == 0UL)   return true;

  id(pid);                     // Revisit: Happens only on the very first batch
  return false;
}

#endif
