#include "Batch.hh"
#include "IovPool.hh"

#include "psdaq/service/GenericPoolW.hh"

#include <assert.h>
#include <new>


namespace Pds {
  namespace Eb {

    class Batch1
    {
    public:
      Batch1(unsigned index, unsigned iovPoolDepth) :
        _index(index),
        _pool(iovPoolDepth),
        _desc(new void*[iovPoolDepth])
      {
      }
      ~Batch1() {};
    private:
      friend class Batch;
    private:
      const unsigned _index;
      IovPool        _pool;
      void**         _desc;
    };

    class BatchEntry : public IovEntry
    {
    public:
      BatchEntry(const XtcData::Dgram& contrib) :
        IovEntry(&contrib, sizeof(contrib) + contrib.xtc.sizeofPayload())
      {
      }
      ~BatchEntry();
    };
  };
};

using namespace XtcData;
using namespace Pds::Eb;

Batch::Batch(const Dgram& contrib, uint64_t pid) :
  _datagram(contrib)
{
  TimeStamp ts(pid, contrib.seq.stamp().control());
  _datagram.seq        = Sequence(contrib.seq.clock(), ts);
  _datagram.xtc.extent = sizeof(Xtc);

  new(((Batch1*)&this[1])->_pool) BatchEntry(_datagram);

  _datagram.xtc.extent += sizeof(_datagram);
}

Batch::~Batch()
{
}

size_t Batch::size()
{
  return sizeof(Batch) + sizeof(Batch1);
}

#include <unistd.h>

void Batch::init(GenericPoolW& pool, unsigned batchDepth, unsigned iovPoolDepth)
{
  Queue<Batch> list;                    // Revisit: Unneeded locking here

  for (unsigned i = 0; i < batchDepth; ++i)
  {
    Batch* batch = (Batch*)pool.alloc(pool.sizeofObject());
    printf("Batch init %d, batch = %p, batch1 = %p\n", i, batch, &batch[1]);
    usleep(100000);
    new((void*)&batch[1]) Batch1(i, iovPoolDepth);
    list.insert(batch);
  }
  for (unsigned i = 0; i < batchDepth; ++i)
  {
    Batch* batch = list.remove();
    printf("Returning batch %i @ %p to pool\n", i, batch);
    usleep(100000);
    pool.free(batch);
  }
}

unsigned Batch::index() const
{
  return ((Batch1*)&this[1])->_index;
}

void Batch::append(const Dgram& contrib)
{
  BatchEntry* entry = new(((Batch1*)&this[1])->_pool) BatchEntry(contrib);
  assert(entry != (void*)0);

  _datagram.xtc.alloc(entry->size());
}

struct fi_msg_rma* Batch::finalize()
{
  _rmaIov.len = (char*)_datagram.xtc.next() - (char*)&_datagram;

  _rmaMsg.desc          = ((Batch1*)&this[1])->_desc;
  _rmaMsg.msg_iov       = ((Batch1*)&this[1])->_pool.iovs();
  _rmaMsg.iov_count     = ((Batch1*)&this[1])->_pool.iovSize();
  _rmaMsg.rma_iov       = &_rmaIov;
  _rmaMsg.rma_iov_count = 1;

  return &_rmaMsg;
}
