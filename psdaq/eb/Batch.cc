#include "Batch.hh"

#include "psdaq/service/GenericPoolW.hh"
#include "xtcdata/xtc/Dgram.hh"

#include <assert.h>
#include <new>

using namespace XtcData;
using namespace Pds::Eb;


namespace Pds {
  namespace Eb {

    class Batch1
    {
    public:
      Batch1(unsigned index, void* buffer, size_t size, const Dgram** datagrams) :
        _index    (index),
        _buffer   (buffer),
        _end      ((char*)buffer + size),
        _current  (buffer),
        _datagrams(datagrams),
        _currentDg(datagrams)
      {
      }
      ~Batch1() { };
    public:
      void* allocate(size_t size)
      {
        char* buffer  = (char*)_current;
        char* current = buffer + size;
        if (current <= _end)
        {
          _current = current;
          return buffer;
        }
        return nullptr;
      }
      void store(const Dgram* datagram)
      {
        *_currentDg++ = datagram;
      }
      void reset()
      {
        _current   = _buffer;
        _currentDg = _datagrams;
      }
    private:
      friend class Batch;
    private:
      unsigned            _index;
      void* const         _buffer;
      void* const         _end;
      void*               _current;
      const Dgram** const _datagrams;
      const Dgram**       _currentDg;
    };
  };
};


Batch::Batch(const Dgram& contrib, uint64_t pid) :
  _parameter(nullptr)
{
  Dgram* dg = new(_batch1()->allocate(sizeof(contrib))) Dgram(contrib);

  dg->seq        = Sequence(contrib.seq.clock(), TimeStamp(pid, 0));
  dg->xtc.extent = sizeof(Xtc);
}

Batch::~Batch()
{
  _batch1()->reset();
}

size_t Batch::size()
{
  return sizeof(Batch) + sizeof(Batch1);
}

Batch1* Batch::_batch1() const
{
  return (Batch1*)(&this[1]);
}

unsigned Batch::index() const
{
  return _batch1()->_index;
}

void* Batch::buffer() const
{
  return _batch1()->_buffer;
}

void Batch::init(GenericPoolW& pool,
                 char*         buffer,
                 unsigned      depth,
                 size_t        maxSize,
                 const Dgram** datagrams,
                 unsigned      maxEntries,
                 Batch**       batches)
{
  Batch** blist  = batches;

  for (unsigned i = 0; i < depth; ++i)
  {
    Batch* batch = (Batch*)pool.alloc(pool.sizeofObject());
    new((void*)&batch[1]) Batch1(i, buffer, maxSize, datagrams);
    *blist++   = batch;
    buffer    += maxSize;
    datagrams += maxEntries;
  }
  blist = batches;
  for (unsigned i = 0; i < depth; ++i)
  {
    Batch* batch = *blist++;
    pool.free(batch);
  }
}

void* Batch::allocate(size_t size)
{
  void* entry = _batch1()->allocate(size);
  assert(entry != nullptr);

  datagram()->xtc.alloc(size);

  return entry;
}

void Batch::store(const Dgram* dg)
{
  _batch1()->store(dg);
}

size_t Batch::extent() const
{
  const Dgram* dg = datagram();
  return sizeof(*dg) + dg->xtc.sizeofPayload();
}

const Dgram* Batch::datagram(unsigned i) const
{
  Batch1*       batch1    = _batch1();
  const Dgram** datagrams = batch1->_datagrams;

  return (&datagrams[i] < batch1->_currentDg) ? datagrams[i] : nullptr;
}
