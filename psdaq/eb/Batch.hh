#ifndef Pds_Eb_Batch_hh
#define Pds_Eb_Batch_hh

#include <stdint.h>
#include <cstddef>

#include <rdma/fi_rma.h>

#include "psdaq/xtc/Datagram.hh"
#include "psdaq/service/LinkedList.hh"


namespace Pds {

  class IovecPool;

  namespace Eb {

#define BatchList LinkedList<Batch>    // Notational convenience...

    class Batch : public BatchList
    {
    public:
      Batch();
      Batch(const Batch&, IovecPool*);
      Batch(Sequence&);
      ~Batch();
    public:
      PoolDeclare;
    public:
      Batch&             append(const Datagram&);
      ClockTime&         clock() const;
      bool               expired(const ClockTime) const;
      struct fi_msg_rma* finalize(uint64_t dstAddr,
                                  uint64_t key,
                                  void**   desc,
                                  uint64_t immData);
      unsigned           index() const;
    private:
      const unsigned    _index;         // Identifies the location of this batch
      ClockTime         _hwm;           // Instances older than this may be freed
      Datagram          _datagram;      // Batch descriptor
      IovecPool*        _pool;          // Pool of iovecs
      struct fi_rma_iov _rmaIov;        // Destination descriptor
      struct fi_msg_rma _msg;
    };
  };
};


inline ClockTime& Pds::Eb::Batch::clock() const
{
  return _datagram.seq.clock();
}

inline bool Pds::Eb::Batch::expired(const ClockTime time) const
{
  return clock() != time;
}

inline unsigned Pds::Eb::Batch::index() const
{
  return _index;
}

#endif
