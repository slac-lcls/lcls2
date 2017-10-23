#ifndef Pds_Eb_Batch_hh
#define Pds_Eb_Batch_hh

#include "IovPool.hh"

#include "psdaq/xtc/Datagram.hh"
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
      Batch(const Datagram&, XtcData::ClockTime&);
      ~Batch();
    public:
      static size_t size();
      static void   init(GenericPoolW&, unsigned batchDepth, unsigned iovPoolDepth);
    public:
      PoolDeclare;
    public:
      void               append(const Datagram&);
      const XtcData::ClockTime&   clock() const;
      void               clock(const XtcData::ClockTime& start);
      bool               expired(const XtcData::ClockTime&);
      struct fi_msg_rma* finalize();
      unsigned           index() const;
    private:
      Datagram          _datagram;      // Batch descriptor
      struct fi_rma_iov _rmaIov;        // Destination descriptor
      struct fi_msg_rma _rmaMsg;
    };
  };
};

inline const XtcData::ClockTime& Pds::Eb::Batch::clock() const
{
  return _datagram.seq.clock();
}

inline void Pds::Eb::Batch::clock(const XtcData::ClockTime& start)
{
  _datagram.seq = XtcData::Sequence(start, _datagram.seq.stamp());
}

inline bool Pds::Eb::Batch::expired(const XtcData::ClockTime& time)
{
  if (clock() == time)    return false;

  if (!clock().isZero())  return true;

  clock(time);                 // Revisit: Happens only on the very first batch
  return false;
}

#endif
