#include "BatchManager.hh"

#include "psdaq/xtc/Datagram.hh"

using namespace Pds::Eb;
using namespace Pds::Fabrics;

BatchManager::BatchManager(StringList&  remote,
                           std::string& port,
                           unsigned     id,       // Revisit: Should be a Src?
                           ClockTime    duration, // = ~((1 << N) - 1) = 128 uS?
                           unsigned     batchDepth,
                           unsigned     iovPoolDepth,
                           size_t       contribSize) :
  _ep           (remote.size()),
  _mr           (remote.size()),
  _ra           (remote.size()),
  _src          (src),
  _duration     (duration),
  _durationShift(__builtin_ctzll(duration.u64())),
  _durationMask (~((1 << __builtin_ctzll(duration.u64())) - 1)),
  _numBatches   (batchDepth),
  _maxBatchSize (iovPoolDepth * contribSize),
  _pool         (sizeof(Batch), batchDepth)
{
  if (__builtin_popcountll(duration.u64()) != 1)
  {
    fprintf(stderr, "Batch duration (%016lx) must be a power of 2\n", duration);
    abort();
  }

  _batchInit(iovPoolDepth);

  _batch = new(_pool) Batch();

  char*  pool = _pool;
  size_t size = batchDepth * _maxBatchSize;
  for (unsigned i = 0; i < remote.size(); ++i)
  {
    int ret = _connect(remote[i], port, pool, size, _ep[i], _mr[i], _ra[i]);
    if (!_ep[i] || !_mr[i] || ret)
    {
      fprintf(stderr, "_connect() failed at index %u", i);
      abort();
    }
  }
}

BatchManager::~BatchManager()
{
  unsigned nDsts = _ep.size();

  for (unsigned i = 0; i < nDsts; ++i)
  {
    if (_mr[i])  delete _mr[i];
    if (_ep[i])  delete _ep[i];
  }

  for (unsigned i = 0; i < _pool.numberofObjects(); ++i)
  {
    Batch* batch = new(_pool) Batch(new IovecPool(poolDepth));
    delete batch->pool();
    delete batch;
  }
}

void BatchManager::_batchInit(unsigned poolDepth)
{
  // Revisit: Is this really the way you preinitialize pool objects?
  for (unsigned i = 0; i < _pool.numberofObjects(); ++i)
  {
    Batch* batch = new(_pool) Batch((Batch&)_pool, new IovecPool(poolDepth));
    delete batch;                       // Return it to the pool
  }
}

int BatchManager::_connect(std::string&   remote,
                           std::string&   port,
                           char*          pool,
                           size_t         size,
                           Endpoint*&     ep,
                           MemoryRegion*& mr,
                           RemoteAddress& ra)
{
  ep = new Endpoint(remote.c_str(), port.c_str());
  if (!ep || (ep->state() != EP_UP))
  {
    fprintf(stderr, "Failed to initialize fabrics endpoint %s:%s: %s\n",
            remote.c_str(), port.c_str(), ep->error());
    perror("new Endpoint");
    return ep ? ep->error_num() : -1;
  }

  Fabric* fab = ep->fabric();

  mr = fab->register_memory(pool, size);
  if (!mr)
  {
    fprintf(stderr, "Failed to register memory region @ %p, sz %zu: %s\n",
            pool, size, fab->error());
    perror("fab->register_memory");
    return fab->error_num();
  }

  unsigned tmo = 120;
  while (!ep->connect() && tmo--)  sleep (1);
  if (!tmo)
  {
    fprintf(stderr, "Failed to connect endpoint %s:%s: %s\n",
            remote.c_str(), port.c_str(), ep->error());
    perror("ep->connect()");
    return -1;
  }

  if (!ep->recv_sync(pool, sizeof(ra), mr))
  {
    fprintf(stderr, "Failed receiving remote memory specs from server: %s\n",
            ep->error());
    perror("recv RemoteAddress");
    return ep->error_num();
  }

  memcpy(&ra, pool, sizeof(ra));
  if (size > ra.extent)
  {
    fprintf(stderr, "Remote Batch pool size (%lu) is less than local pool size (%lu)\n",
            ra.extent, size);
    return -1;
  }

  return 0;
}

void BatchManager::process(const Datagram& contrib, void* arg = NULL)
{
  Batch*    batch = _batch;
  ClockTime start = _startTime(contrib);

  if (batch->expired(start))
  {
    if (batch->clock().isZero()) // Revisit: Happens only on the very first batch
    {
      delete batch;
      return;
    }

    post(batch, arg);

    _batch = new(_pool) Batch(contrib, start);
  }

  batch->append(contrib);
}

void BatchManager::postTo(const Batch* batch,
                          unsigned     dst,
                          unsigned     slot)
{
  uint64_t dstAddr = _ra[dst].addr + slot * _maxBatchSize;

  struct fi_msg_rma* msg = batch->finalize(dstAddr,
                                           _ra[dst].rkey, // Revisit: _mr's?
                                           _mr[dst]->desc(),
                                           dstAddr);

  ret = fi_writemsg(_ep[dst], msg, FI_REMOTE_CQ_DATA); // Revisit: Flags?
  if (ret)
  {
    fprintf(stderr, "%s() failed, ret = %d (%s)\n",
            "fi_writemsg", ret, fi_strerror((int)-ret));
    return;
  }

  //_ep[dst]->writeMsg(batch->iov(), batch->iovCount(), dstAddr, immData,
  //                   _ra[dst], _mr[dst]);

  // Revisit: No obvious need to wait for completion here as nothing can be done
  // with this batch or its remote instance until a result is sent
  // - This is true on the contributor side
  // - Revisit for the EB result side

  _batchList.insert(batch);
}

void BatchManager::_release(ClockTime time)
{
  Batch* batch = _batchList.forward();
  Batch* end   = _batchList.empty();
  while (batch != end)
  {
    if (batch.clock() == time)
    {
      delete batch->disconnect();
      break;
    }
    batch = batch->forward();
  }
}

//uint64_t BatchManager::_batchId(ClockTime & clk)
//{
//  return clk.u64() / _duration;         // Batch number since EPOCH
//}

uint64_t BatchManager::_batchId(ClockTime& clk)
{
  return clk.u64() >> _durationShift;   // Batch number since EPOCH
}

//uint64_t BatchManager::_startTime(ClockTime& clk)
//{
//  return _batchId(clk) * _duration;     // Current batch start time
//}

uint64_t BatchManager::_startTime(ClockTime& clk)
{
  return clk.u64() & _durationMask;
}
