#include "Batch.hh"

#include "psdaq/eb/IovPool.hh"

namespace Pds {
  namespace Eb {

    class BatchEntry : public IovEntry
    {
    public:
      BatchEntry(const Datagram& contrib) :
        IovEntry(&contrib, sizeof(Datagram) + contrib.sizeofPayload())
      {
      }
      ~BatchEntry();
    };
  };
};

using namespace Pds::Eb;

Batch::Batch()
{
  // Revisit: That the following ctors don't provide this suggests this is the wrong idea
  ClockTime clk(0, 0);
  TimeStamp ts();
  _datagram.seq        = Sequence(clk, ts);
  _datagram.xtc.extent = sizeof(Xtc);

  BatchEntry* entry = new(_pool) BatchEntry(this);
  assert(entry != NULL);

  //_datagram.xtc.extent += sizeof(_datagram);
}

Batch::Batch(const Datagram& contrib, ClockTime& start)
{
  _datagram            = contrib;
  _datagram.seq        = Sequence(start, contrib.seq.stamp());
  _datagram.xtc.extent = sizeof(Xtc);

  BatchEntry* entry = new(_pool) BatchEntry(this);
  assert(entry != NULL);

  //_datagram.xtc.extent += sizeof(_datagram);
}

Batch::Batch(const Batch& base, IovecPool* pool) :
  _index(this - &base),
  _pool(pool)
{
  memset(&rmaMsg, 0, sizeof(rmaMsg));

  _rmaMsg.rma_iov       = &_rmaIov;
  _rmaMsg.rma_iov_count = 1;
  _rmaMsg.context       = &_ctx;        // Revisit: What is this for?
}

Batch::~Batch()
{
  _datagram.xtc.extent = sizeof(Xtc);

  _pool->clear();
}

void Batch::append(const Datagram& contrib)
{
  BatchEntry* entry = new(_pool) BatchEntry(contrib);
  assert(entry != NULL);

  _datagram.xtc.alloc(entry->size());
}

struct fi_msg_rma* Batch::finalize(uint64_t dstAddr,
                                   uint64_t key,
                                   void**   desc,
                                   uint64_t immData)
{
  _rmaIov.addr = dstAddr;
  _rmaIov.len  = (char*)_datagram.xtc.next() - (char*)&_datagram;
  _rmaIov.key  = key;

  _rmaMsg.msg_iov   = _pool->iovs();
  _rmaMsg.iov_count = _pool->iovSize();
  _rmaMsg.desc      = desc;
  _rmaMsg.data      = immData;

  return &_rmaMsg;
}
