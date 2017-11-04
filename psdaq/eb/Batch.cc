#include "Batch.hh"
#include "IovElement.hh"

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
        _pool(iovPoolDepth)
      {
      }
      ~Batch1() { };
    private:
      friend class Batch;
    private:
      const unsigned      _index;
      Fabrics::LocalIOVec _pool;
    };

    class BatchEntry : public IovElement
    {
    public:
      BatchEntry(const XtcData::Dgram& contrib) :
        IovElement(&contrib, sizeof(contrib) + contrib.xtc.sizeofPayload())
      {
      }
      ~BatchEntry();
    };
  };
};

using namespace XtcData;
using namespace Pds::Eb;
using namespace Pds::Fabrics;

Batch::Batch(const Dgram& contrib, uint64_t pid) :
  _datagram(contrib)
{
  TimeStamp ts(pid, contrib.seq.stamp().control());
  _datagram.seq        = Sequence(contrib.seq.clock(), ts);
  _datagram.xtc.extent = sizeof(Xtc);

  new (_batch1()._pool) BatchEntry(_datagram);
}

Batch::~Batch()
{
  _batch1()._pool.reset();
}

size_t Batch::size()
{
  return sizeof(Batch) + sizeof(Batch1);
}

Batch1& Batch::_batch1() const
{
  return *(Batch1*)&this[1];
}

#include <unistd.h>

void Batch::init(GenericPoolW& pool,
                 unsigned      batchDepth,
                 unsigned      iovPoolDepth,
                 MemoryRegion* mr[2])
{
  Queue<Batch> list;                    // Revisit: Unneeded locking here

  for (unsigned i = 0; i < batchDepth; ++i)
  {
    Batch*  batch = (Batch*)pool.alloc(pool.sizeofObject());
    Batch1* b1    = new((void*)&batch[1]) Batch1(i, iovPoolDepth);
    for (unsigned j = 0; j < iovPoolDepth; ++j)
    {
      b1->_pool.set_iovec_mr(j, mr[j == 0 ? 0 : 1]);
    }
    printf("Batch init %2d, batch = %p, batch1 = %p, mr->desc = %p, %p\n",
           i, batch, b1, mr[0]->desc(), mr[1]->desc());
    list.insert(batch);
  }
  for (unsigned i = 0; i < batchDepth; ++i)
  {
    Batch* batch = list.remove();
    pool.free(batch);
  }
}

unsigned Batch::index() const
{
  return _batch1()._index;
}

void Batch::append(const Dgram& contrib)
{
  //if (pool().count() > 5)  return;   // Revisit: fi_tx_attr.iov_limit = 6

  BatchEntry* entry = new (_batch1()._pool) BatchEntry(contrib);
  assert(entry != (void*)0);

  _datagram.xtc.alloc(entry->size());
}

size_t Batch::extent() const
{
  return sizeof(_datagram) + _datagram.xtc.sizeofPayload();
}

LocalIOVec& Batch::pool() const
{
  return _batch1()._pool;
}
