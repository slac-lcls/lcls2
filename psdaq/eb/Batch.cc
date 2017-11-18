#include "Batch.hh"

#include "psdaq/service/GenericPoolW.hh"

#include <assert.h>
#include <new>

using namespace XtcData;
using namespace Pds::Eb;


namespace Pds {
  namespace Eb {

    class TheSrc : public Src
    {
    public:
      TheSrc(const Src& src, unsigned index) :
        Src(src)
      {
        _log |= index << 8;
      }
    };

    class Batch1
    {
    public:
      Batch1(unsigned index, void* pool, size_t size) :
        _index(index),
        _pool(pool),
        _current(pool)
      {
      }
      ~Batch1() { };
    public:
      void* allocate(size_t size)
      {
        char* buffer = (char*)_current;
        _current = buffer + size;
        return buffer;
      }
      void reset() { _current = _pool; }
    private:
      friend class Batch;
    private:
      const unsigned _index;
      void*          _pool;
      void*          _current;
    };
  };
};


Batch::Batch(const Src& src, const Dgram& contrib, uint64_t pid) :
  _parameter(NULL)
{
  Dgram* dg = new(_batch1().allocate(sizeof(contrib))) Dgram(contrib);

  TimeStamp ts(pid, contrib.seq.stamp().control());
  dg->seq        = Sequence(contrib.seq.clock(), ts);
  dg->xtc.src    = TheSrc(src, _batch1()._index);
  dg->xtc.extent = sizeof(Xtc);
}

Batch::~Batch()
{
  _batch1().reset();
}

size_t Batch::size()
{
  return sizeof(Batch) + sizeof(Batch1);
}

Batch1& Batch::_batch1() const
{
  return *(Batch1*)&this[1];
}

unsigned Batch::index() const
{
  return _batch1()._index;
}

void* Batch::pool() const
{
  return _batch1()._pool;
}

#include <unistd.h>

void Batch::init(GenericPoolW& pool,
                 char*         buffer,
                 unsigned      depth,
                 size_t        maxSize)
{
  Queue<Batch> list;                    // Revisit: Unneeded locking here

  for (unsigned i = 0; i < depth; ++i)
  {
    Batch*  batch = (Batch*)pool.alloc(pool.sizeofObject());
    new((void*)&batch[1]) Batch1(i, buffer, maxSize);
    list.insert(batch);
    buffer += maxSize;
  }
  for (unsigned i = 0; i < depth; ++i)
  {
    Batch* batch = list.remove();
    pool.free(batch);
  }
}

void* Batch::allocate(size_t size)
{
  void* entry = _batch1().allocate(size);
  assert(entry != (void*)0);

  datagram()->xtc.alloc(size);

  return entry;
}

size_t Batch::extent() const
{
  const Dgram* dg = datagram();
  return sizeof(*dg) + dg->xtc.sizeofPayload();
}
