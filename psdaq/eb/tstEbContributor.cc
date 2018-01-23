#include "psdaq/eb/BatchManager.hh"
#include "psdaq/eb/Endpoint.hh"

#include "psdaq/eb/EbFtClient.hh"
#include "psdaq/eb/EbFtServer.hh"

#include "psdaq/service/Routine.hh"
#include "psdaq/service/Task.hh"
#include "psdaq/service/GenericPoolW.hh"
#include "psdaq/service/Histogram.hh"
#include "xtcdata/xtc/Dgram.hh"

#ifndef _GNU_SOURCE
#  define _GNU_SOURCE
#endif
#include <sched.h>
#include <fcntl.h>
#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <new>
#include <chrono>
#include <atomic>
#include <thread>

#define HACK_FOR_A_TEST 0

using namespace XtcData;
using namespace Pds;
using namespace Pds::Fabrics;

static const int      core_drpSim      = 8;
static const int      core_inlet       = 20; //9;
static const int      core_outlet      = 9;  //10;
static const int      core_monitor     = 21; //11;
static const unsigned mon_period       = 1;          // Seconds
static const unsigned default_id       = 0;          // Contributor's ID
static const unsigned srv_port_base    = 32768 + 64; // Base port Contributor for receiving results
static const unsigned clt_port_base    = 32768;      // Base port Contributor sends to EB on
static const uint64_t batch_duration   = 0x1ul << 9; // ~512 uS; must be power of 2
static const unsigned max_batches      = 2048;       // 1 second's worth
static const unsigned max_entries      = 512;        // 512 uS worth
static const size_t   header_size      = sizeof(Dgram);
static const size_t   input_extent     = 5; // Revisit: Number of "L3" input  data words
//static const size_t   result_extent    = 2; // Revisit: Number of "L3" result data words
static const size_t   result_extent    = input_extent;
static const size_t   max_contrib_size = header_size + input_extent  * sizeof(uint32_t);
static const size_t   max_result_size  = header_size + result_extent * sizeof(uint32_t);

static       bool     lcheck           = false;
static       unsigned lverbose         = 0;
static       int      lcoreBase        = 0;

typedef std::chrono::steady_clock::time_point tp_t;

static void dumpStats(const char* header, tp_t startTime, tp_t endTime, uint64_t count)
{
  typedef std::chrono::nanoseconds ns_t;
  int64_t dt = std::chrono::duration_cast<ns_t>(endTime - startTime).count();
  double  ds = double(dt) * 1.e-9;
  printf("\n%s:\n", header);
  printf("Time diff:  %ld ns = %.0f s\n", dt, ds);
  printf("Count:      %ld\n", count);
  printf("Avg rate:   %f KHz\n", double(count) / ds / 1000.);
}

namespace Pds {
  namespace Eb {

    class TstContribOutlet;
    class TstContribInlet;

    class TheSrc : public Src
    {
    public:
      TheSrc(Level::Type level, unsigned id) :
        Src(level)
      {
        _log |= id;
      }
    };

    class Input : public Dgram
    {
    public:
      Input(const Sequence& seq_, const uint64_t env_, const Xtc& xtc_) :
        Dgram()
      {
        seq = seq_;
        env = env_;
        xtc = xtc_;
      }
    public:
      PoolDeclare;
    public:
      uint64_t id() const { return seq.pulseId().value(); }
    };

    class DrpSim
    {
    public:
      DrpSim(unsigned poolSize, size_t maxEvtSize, unsigned id, TstContribOutlet& outlet);
      ~DrpSim() { }
    public:
      void start(TstContribOutlet*);
      void shutdown();
    public:
      void process();
    private:
      const unsigned    _maxEvtSz;
      GenericPoolW      _pool;
      const TypeId      _typeId;
      const Src         _src;
      const unsigned    _id;
      uint64_t          _pid;
      TstContribOutlet& _outlet;
      std::atomic<bool> _running;
      Input*            _heldAside;
    };

    class TstContribOutlet : public BatchManager,
                             public Routine
    {
    public:
      TstContribOutlet(std::vector<std::string>& cltAddr,
                       std::vector<std::string>& cltPort,
                       unsigned                  id,
                       uint64_t                  builders,
                       uint64_t                  duration,
                       unsigned                  maxBatches,
                       unsigned                  maxEntries,
                       size_t                    maxSize);
      virtual ~TstContribOutlet();
    public:
      const char* base() const;
      int         connect();
      void        shutdown();
    public:
      uint64_t    count() const { return _eventCount; }
    public:
      void        routine();
      void        completed();
      void        post(const Dgram* input);
      void        release(uint64_t id);
      void        post(const Batch* batch);
#if HACK_FOR_A_TEST
      void*       testPend();             // Hack for a test!
      void        testPost(const Batch*); // Hack for a test!
#endif
    private:
      Batch*     _pend();
    private:
      EbFtClient              _outlet;
      const unsigned          _id;
      const unsigned          _numEbs;
      unsigned*               _destinations;
      Queue<Batch>            _inputBatchList;
      std::atomic<unsigned>   _inputBatchOcc;
      Histogram               _inputBatchHist;
#if HACK_FOR_A_TEST
      Queue<Batch>            _testList;
#endif
      std::atomic<unsigned>   _inFlightOcc;
      Histogram               _inFlightHist;
      uint64_t                _eventCount;
      Histogram               _depTimeHist;
      Histogram               _arrTimeHist;
      Histogram               _postTimeHist;
      Histogram               _postCallHist;
      std::chrono::steady_clock::time_point _postPrevTime;
      std::atomic<bool>       _running;
      Task*                   _task;
    };

    class TstContribInlet : public Routine
    {
    public:
      TstContribInlet(std::string&      srvPort,
                      unsigned          id,
                      uint64_t          builders,
                      unsigned          maxBatches,
                      unsigned          maxEntries,
                      size_t            maxSize,
                      TstContribOutlet& outlet);
      virtual ~TstContribInlet() { }
    public:
      int      connect();
      void     shutdown();
    public:
      uint64_t count() const { return _eventCount; }
    public:
      void     routine();
    private:
      int     _process(const Dgram* result, const Dgram* input);
    private:
      const unsigned    _id;
      const unsigned    _numEbs;
      const size_t      _maxBatchSize;
      EbFtServer        _inlet;
      TstContribOutlet& _outlet;
      uint64_t          _eventCount;
      Histogram         _rttHist;
      Histogram         _pendTimeHist;
      Histogram         _pendCallHist;
      std::chrono::steady_clock::time_point _pendPrevTime;
      std::atomic<bool> _running;
      Task*             _task;
    };

    class StatsMonitor : public Routine
    {
    public:
      StatsMonitor(const TstContribOutlet* outlet,
                   const TstContribInlet*  inlet,
                   unsigned                monPd);
      virtual ~StatsMonitor() { }
    public:
      void     shutdown();
    public:
      void     routine();
    private:
      const TstContribOutlet* _outlet;
      const TstContribInlet*  _inlet;
      const unsigned          _monPd;
      std::atomic<bool>       _running;
      Task*                   _task;
    };
  };
};

using namespace Pds::Eb;

static void pin_thread(const pthread_t& th, int cpu)
{
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(cpu, &cpuset);
  int rc = pthread_setaffinity_np(th, sizeof(cpu_set_t), &cpuset);
  if (rc != 0)
  {
    fprintf(stderr, "Error calling pthread_setaffinity_np: %d\n  %s\n",
            rc, strerror(rc));
  }
}

DrpSim::DrpSim(unsigned          poolSize,
               size_t            maxEvtSize,
               unsigned          id,
               TstContribOutlet& outlet) :
  _maxEvtSz (maxEvtSize),
  _pool     (sizeof(Entry) + maxEvtSize, poolSize),
  _typeId   (TypeId::Data, 0),
  _src      (TheSrc(Level::Segment, id)),
  _id       (id),
  _pid      (0x01000000000003UL),  // Something non-zero and not on a batch boundary
  _outlet   (outlet),
  _running  (true),
  _heldAside(new(&_pool) Input(Sequence(Sequence::Marker, TransitionId::L1Accept,
                                        TimeStamp(), PulseId(-1)),
                               0ul, Xtc(_typeId, _src)))
{
}

void DrpSim::shutdown()
{
  _running = false;

  delete _heldAside;  // Make sure there's at least one event in the pool for
                      // process() to get out of resource wait and test _running
}

void DrpSim::process()
{
  static unsigned cnt = 0;

  pin_thread(pthread_self(), lcoreBase + core_drpSim);

  const uint64_t env(0);
  const Xtc      xtc(_typeId, _src);

  while(_running)
  {
    //char str[BUFSIZ];
    //printf("Enter a character to take an event: ");
    //scanf("%s", str);

    // Fake datagram header
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    const Sequence seq(Sequence::Event, TransitionId::L1Accept,
                       TimeStamp(ts), PulseId(_pid));

    _pid += 1; //27; // Revisit: fi_tx_attr.iov_limit = 6 = 1 + max # events per batch
    //_pid = ts.tv_nsec / 1000;     // Revisit: Not guaranteed to be the same across systems

    Input* idg = new(&_pool) Input(seq, env, xtc);

    size_t inputSize = input_extent * sizeof(uint32_t);

    // Revisit: Here is where L3 trigger input information would be inserted into the datagram,
    //          whatever that will look like.  SmlD is nominally the right container for this.
    uint32_t* payload = (uint32_t*)idg->xtc.alloc(inputSize);
    payload[0] = ts.tv_sec;           // Revisit: Some crud for now
    payload[1] = ts.tv_nsec;          // Revisit: Some crud for now
    payload[2] = ++cnt;               // Revisit: Some crud for now
    payload[3] = _id;                 // Revisit: Some crud for now
    payload[4] = idg->seq.pulseId().value() & 0xffffffffUL; // Revisit: Some crud for now

    _outlet.post(idg);
  }

  printf("\nDrpSim Input data pool:\n");
  _pool.dump();
}

TstContribOutlet::TstContribOutlet(std::vector<std::string>& cltAddr,
                                   std::vector<std::string>& cltPort,
                                   unsigned                  id,
                                   uint64_t                  builders,
                                   uint64_t                  duration,
                                   unsigned                  maxBatches,
                                   unsigned                  maxEntries,
                                   size_t                    maxSize) :
  BatchManager   (duration, maxBatches, maxEntries, maxSize),
  _outlet        (cltAddr, cltPort, maxBatches * (sizeof(Dgram) + maxEntries * maxSize)),
  _id            (id),
  _numEbs        (__builtin_popcountl(builders)),
  _destinations  (new unsigned[_numEbs]),
  _inputBatchOcc (0),
  _inputBatchHist(__builtin_ctz(maxBatches), 1.0),
  _inFlightOcc   (0),
  _inFlightHist  (__builtin_ctz(maxBatches), 1.0),
  _eventCount    (0),
  _depTimeHist   (12, double(1 << 12)/1000.),
  _arrTimeHist   (12, double(1 << 20)/1000.),
  _postTimeHist  (12, 1.0),
  _postCallHist  (12, 1.0),
  _postPrevTime  (std::chrono::steady_clock::now()),
  _running       (true),
  _task          (new Task(TaskObject("tOutlet", 0, 0, 0, 0, 0)))
{
  for (unsigned i = 0; i < _numEbs; ++i)
  {
    unsigned bit = __builtin_ffsl(builders) - 1;
    builders &= ~(1ul << bit);
    _destinations[i] = bit;
  }
}

TstContribOutlet::~TstContribOutlet()
{
  delete [] _destinations;
}

void TstContribOutlet::shutdown()
{
  pthread_t tid = _task->parameters().taskID();

  _running = false;
  post(::new Batch(Batch::IsLast));

  int   ret    = 0;
  void* retval = nullptr;
  ret = pthread_join(tid, &retval);
  if (ret)  perror("Outlet pthread_join");
}

int TstContribOutlet::connect()
{
  const unsigned tmo(120);
  int ret = _outlet.connect(_id, tmo);
  if (ret)
  {
    fprintf(stderr, "FtClient connect() failed\n");
    return ret;
  }

  _outlet.registerMemory(batchRegion(), batchRegionSize());

  _task->call(this);

  return ret;
}

void TstContribOutlet::post(const Dgram* input)
{
  if (lverbose > 1)
  {
    printf("DRP   Posts        input          dg           @ %16p, ts %014lx, sz %3zd, src %2d\n",
           input, input->seq.pulseId().value(), sizeof(*input) + input->xtc.sizeofPayload(), input->xtc.src.log() & 0xff);
  }

  ++_eventCount;

  process(input);
}

Batch* TstContribOutlet::_pend()
{
  Batch* batch = _inputBatchList.removeW();
  _inputBatchOcc--;

  if ((lverbose > 1) && !batch->isLast())
  {
    const Dgram* bdg = batch->datagram();
    printf("ContribOutlet rcvd input          batch        @ %16p, ts %014lx, sz %3zd\n",
           batch, bdg->seq.pulseId().value(), sizeof(*bdg) + bdg->xtc.sizeofPayload());
  }

  return batch;
}

void TstContribOutlet::post(const Batch* batch)
{
  {
    const Dgram* bdg = batch->datagram();

    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    uint64_t dS(ts.tv_sec  - bdg->seq.stamp().seconds());
    uint64_t dN(ts.tv_nsec - bdg->seq.stamp().nanoseconds());
    uint64_t dT((dS * 1000000000ul + dN) >> 12);

    _depTimeHist.bump(dT);
  }

  if ((lverbose > 1) && !batch->isLast())
  {
    const Dgram* bdg = batch->datagram();
    printf("DRP   Posting      input          batch        @ %16p, ts %014lx, sz %3zd\n",
           batch, bdg->seq.pulseId().value(), sizeof(*bdg) + bdg->xtc.sizeofPayload());
  }

  _inputBatchList.insert(const_cast<Batch*>(batch));
  _inputBatchOcc++;
  _inputBatchHist.bump(_inputBatchOcc);
}

#if HACK_FOR_A_TEST
void TstContribOutlet::testPost(const Batch* batch)
{
  _testList.insert(const_cast<Batch*>(batch));
}

void* TstContribOutlet::testPend()
{
  Batch* batch = _testList.removeW();

  return batch;
}
#endif

void TstContribOutlet::completed()
{
  _inFlightOcc--;
}

void TstContribOutlet::routine()
{
  pin_thread(_task->parameters().taskID(), lcoreBase + core_outlet);

  auto startTime = std::chrono::steady_clock::now();

  while (_running)
  {
    const Batch* batch = _pend();
    if (batch->isLast())
    {
      ::delete batch;
      continue;
    }

    {
      const Dgram*    bdg = batch->datagram();
      struct timespec ts;
      clock_gettime(CLOCK_MONOTONIC, &ts);
      uint64_t dS(ts.tv_sec  - bdg->seq.stamp().seconds());
      uint64_t dN(ts.tv_nsec - bdg->seq.stamp().nanoseconds());
      uint64_t dT((dS * 1000000000ul + dN) >> 20);

      _arrTimeHist.bump(dT);
    }

    unsigned iDst = batchId(batch->id()) % _numEbs;
    unsigned dst  = _destinations[iDst];

    if (lverbose)
    {
      static unsigned cnt    = 0;
      const Dgram*    bdg    = batch->datagram();
      void*           rmtAdx = (void*)_outlet.rmtAdx(dst, batch->index() * maxBatchSize());
      printf("ContribOutlet posts %6d        batch[%2d]    @ %16p, ts %014lx, sz %3zd to   EB %d %16p\n",
             ++cnt, batch->index(), bdg, bdg->seq.pulseId().value(), batch->extent(), dst, rmtAdx);
    }

    auto t0 = std::chrono::steady_clock::now();

#if !HACK_FOR_A_TEST
    _outlet.post(batch->buffer(), batch->extent(), dst, batch->index() * maxBatchSize());
#else
    testPost(batch);
#endif
    auto t1 = std::chrono::steady_clock::now();
    _inFlightOcc++;
    _inFlightHist.bump(_inFlightOcc);

    typedef std::chrono::microseconds us_t;

    _postTimeHist.bump(std::chrono::duration_cast<us_t>(t1 - t0).count());
    _postCallHist.bump(std::chrono::duration_cast<us_t>(t0 - _postPrevTime).count());
    _postPrevTime = t0;
  }

  auto endTime = std::chrono::steady_clock::now();

  BatchManager::dump();

  dumpStats("Outlet", startTime, endTime, _eventCount);

  _outlet.shutdown();

  char fs[80];
  sprintf(fs, "inputBatchOcc_%d.hist", _id);
  printf("Dumped input batch occupancy histogram to ./%s\n", fs);
  _inputBatchHist.dump(fs);

  sprintf(fs, "inFlightOcc_%d.hist", _id);
  printf("Dumped in-flight occupancy histogram to ./%s\n", fs);
  _inFlightHist.dump(fs);

  sprintf(fs, "depTime_%d.hist", _id);
  printf("Dumped departure time histogram to ./%s\n", fs);
  _depTimeHist.dump(fs);

  sprintf(fs, "arrTime_%d.hist", _id);
  printf("Dumped arrival time histogram to ./%s\n", fs);
  _arrTimeHist.dump(fs);

  sprintf(fs, "postTime_%d.hist", _id);
  printf("Dumped post time histogram to ./%s\n", fs);
  _postTimeHist.dump(fs);

  sprintf(fs, "postCallRate_%d.hist", _id);
  printf("Dumped post call rate histogram to ./%s\n", fs);
  _postCallHist.dump(fs);

  _task->destroy();
}

TstContribInlet::TstContribInlet(std::string&      srvPort,
                                 unsigned          id,
                                 uint64_t          builders,
                                 unsigned          maxBatches,
                                 unsigned          maxEntries,
                                 size_t            maxSize,
                                 TstContribOutlet& outlet) :
  _id          (id),
  _numEbs      (__builtin_popcountl(builders)),
  _maxBatchSize(sizeof(Dgram) + maxEntries * maxSize),
  _inlet       (srvPort, _numEbs, maxBatches * _maxBatchSize, EbFtServer::PEERS_SHARE_BUFFERS),
  _outlet      (outlet),
  _eventCount  (0),
#if !HACK_FOR_A_TEST
  _rttHist     (12, double(1 << 20)/1000.),
#else
  _rttHist     (12, double(1 << 10)/1000.),
#endif
  _pendTimeHist(12, 1.0),
  _pendCallHist(12, 1.0),
  _pendPrevTime(std::chrono::steady_clock::now()),
  _running     (true),
  _task        (new Task(TaskObject("tInlet", 0, 0, 0, 0, 0)))
{
}

void TstContribInlet::shutdown()
{
  pthread_t tid = _task->parameters().taskID();

  _running = false;

#if HACK_FOR_A_TEST
  _outlet.testPost(::new Batch(Batch::IsLast));  // Foar a test!
#endif

  int   ret    = 0;
  void* retval = nullptr;
  ret = pthread_join(tid, &retval);
  if (ret)  perror("Inlet pthread_join");
}

int TstContribInlet::connect()
{
  int ret = _inlet.connect(_id);
  if (ret)
  {
    fprintf(stderr, "FtServer connect() failed\n");
    return ret;
  }

  _task->call(this);

  return ret;
}

void TstContribInlet::routine()
{
  pin_thread(_task->parameters().taskID(), lcoreBase + core_inlet);

  auto startTime = std::chrono::steady_clock::now();

  while (_running)
  {
#if !HACK_FOR_A_TEST      // Hack for a test
    // Pend for a result datagram (batch) and process it.
    uint64_t data;
    auto t0 = std::chrono::steady_clock::now();
    if (_inlet.pend(&data))  continue;
    auto t1 = std::chrono::steady_clock::now();
    const Dgram* batch = (const Dgram*)data;

    unsigned idx = ((const char*)batch - _inlet.base()) / _maxBatchSize;

    if (batch->env == _id)
#else
    auto  t0   = std::chrono::steady_clock::now();
    void* data = _outlet.testPend();
    auto  t1   = std::chrono::steady_clock::now();
    const Batch* aBatch = (const Batch*)data;
    if (aBatch->isLast())
    {
      ::delete aBatch;
      continue;
    }
    const Dgram* batch = aBatch->datagram();

    unsigned idx = ((const char*)batch - (const char*)_outlet.batchRegion()) / _maxBatchSize;
#endif
    {
      struct timespec ts;
      clock_gettime(CLOCK_MONOTONIC, &ts);
      uint64_t dS(ts.tv_sec  - batch->seq.stamp().seconds());
      uint64_t dN(ts.tv_nsec - batch->seq.stamp().nanoseconds());
#if !HACK_FOR_A_TEST      // Hack for a test
      uint64_t dT((dS * 1000000000ul + dN) >> 20);
#else
      uint64_t dT((dS * 1000000000ul + dN) >> 10);
#endif
      _rttHist.bump(dT);

      //printf("RTT = %u S, %u ns\n", dS, dN);

      typedef std::chrono::microseconds us_t;

      _pendTimeHist.bump(std::chrono::duration_cast<us_t>(t1 - t0).count());
      _pendCallHist.bump(std::chrono::duration_cast<us_t>(t0 - _pendPrevTime).count());
      _pendPrevTime = t0;
    }
    _outlet.completed();

    if (lverbose)
    {
      static unsigned cnt = 0;
      unsigned from = batch->xtc.src.log() & 0xff;
      printf("ContribInlet  rcvd        %6d result   [%2d] @ %16p, ts %014lx, sz %3zd from EB %d\n",
             ++cnt, idx, batch, batch->seq.pulseId().value(), sizeof(*batch) + batch->xtc.sizeofPayload(), from);
    }

    const Batch*  input  = _outlet.batch(idx);
    unsigned      i      = 0;
    const Dgram*  result = (const Dgram*)batch->xtc.payload();
    const Dgram*  last   = (const Dgram*)batch->xtc.next();
    while(result != last)
    {
      _process(result, input->datagram(i++));

      result = (const Dgram*)result->xtc.next();
    }
    _eventCount += i;

    delete input;
  }

  auto endTime = std::chrono::steady_clock::now();

  dumpStats("Inlet", startTime, endTime, _eventCount);

  _inlet.shutdown();

  char fs[80];
  sprintf(fs, "rtt_%d.hist", _id);
  printf("Dumped RTT histogram to ./%s\n", fs);
  _rttHist.dump(fs);

  sprintf(fs, "pendTime_%d.hist", _id);
  printf("Dumped pend time histogram to ./%s\n", fs);
  _pendTimeHist.dump(fs);

  sprintf(fs, "pendCallRate_%d.hist", _id);
  printf("Dumped pend call rate histogram to ./%s\n", fs);
  _pendCallHist.dump(fs);

  _task->destroy();
}

int TstContribInlet::_process(const Dgram* result, const Dgram* input)
{
  bool     ok  = true;
  uint64_t pid = result->seq.pulseId().value();

  if (lcheck)
  {
    uint32_t* buffer = (uint32_t*)result->xtc.payload();
    if (_numEbs == 1)
    {
      // Timestamps aren't coherent across multiple contributor instances
      //static uint64_t _ts = 0;
      //uint64_t         ts = buffer[0];
      //ts = (ts << 32) + buffer[1];
      //if (_ts > ts)
      //{
      //  ok = false;
      //  printf("Timestamp didn't increase: previous = %016lx, current = %016lx\n", _ts, ts);
      //}
      //_ts = ts;
      static unsigned _counter = 0;
      if (buffer[2] != _counter + 1)
      {
        ok = false;
        printf("Result counter didn't increase by 1: expected %08x, got %08x\n",
               _counter + 1, buffer[2]);
      }
      _counter = buffer[2];
      static uint64_t _pid = 0;
      if (_pid > pid)
      {
        ok = false;
        printf("Pulse ID didn't increase: previous = %014lx, current = %014lx\n", _pid, pid);
      }
      _pid = pid;
    }
    if ((pid & 0xffffffffUL) != buffer[4])
    {
      ok = false;
      printf("Pulse ID mismatch: expected %08lx, got %08x\n",
             pid & 0xffffffffUL, buffer[4]);
    }
    if (!ok)
    {
      printf("ContribInlet  found result   %16p, ts %014lx, sz %d, pyld:\n",
             result, pid, result->xtc.sizeofPayload());
      for (unsigned i = 0; i < result_extent; ++i)
        printf(" %08x", buffer[i]);
      printf("\n");

      abort();                          // Revisit
    }
  }
  if (input == nullptr)
  {
    fprintf(stderr, "Input datagram not found\n");
    abort();
  }
  else if (pid != input->seq.pulseId().value())
  {
    printf("Result pulse ID doesn't match Input datagram's: %014lx, %014lx\n",
           pid, input->seq.pulseId().value());
    abort();
  }

  // This method is used to handle the result datagram and to dispose of
  // the original source datagram, in other words:
  // - Possibly forward pebble to FFB
  // - Possibly forward pebble to AMI
  // - Return to pool
  // e.g.:
  //   Pebble* pebble = findPebble(result);
  //   handle(result, pebble);
  //   delete pebble;

  delete (const Input*)input;           // Revisit: Downcast okay here?

  return 0;
}

StatsMonitor::StatsMonitor(const TstContribOutlet* outlet,
                           const TstContribInlet*  inlet,
                           unsigned                 monPd) :
  _outlet (outlet),
  _inlet  (inlet),
  _monPd  (monPd),
  _running(true),
  _task   (new Task(TaskObject("tMonitor", 0, 0, 0, 0, 0)))
{
  _task->call(this);
}

void StatsMonitor::shutdown()
{
  pthread_t tid = _task->parameters().taskID();

  _running = false;

  int   ret    = 0;
  void* retval = nullptr;
  ret = pthread_join(tid, &retval);
  if (ret)  perror("StatsMon pthread_join");
}

void StatsMonitor::routine()
{
  pin_thread(pthread_self(), lcoreBase + core_monitor);

  uint64_t outCnt    = 0;
  uint64_t outCntPrv = _outlet ? _outlet->count() : 0;
  uint64_t inCnt     = 0;
  uint64_t inCntPrv  = _inlet  ? _inlet->count()  : 0;
  auto     now       = std::chrono::steady_clock::now();

  while(_running)
  {
    std::this_thread::sleep_for(std::chrono::seconds(_monPd));

    auto then = now;
    now       = std::chrono::steady_clock::now();

    outCnt    = _outlet ? _outlet->count() : outCntPrv;
    inCnt     = _inlet  ? _inlet->count()  : inCntPrv;

    auto dT   = std::chrono::duration_cast<std::chrono::microseconds>(now - then).count();
    auto dOC  = outCnt - outCntPrv;
    auto dIC  = inCnt  - inCntPrv;

    printf("Outlet / Inlet: N %016lx / %016lx, dN %7ld / %7ld, rate %7.02f / %7.02f KHz\n",
           outCnt, inCnt, dOC, dIC, double(dOC) / dT * 1.0e3, double(dIC) / dT * 1.0e3);

    outCntPrv = outCnt;
    inCntPrv  = inCnt;
  }

  _task->destroy();
}


static DrpSim*           drpSim       (nullptr);
static TstContribOutlet* contribOutlet(nullptr);
static TstContribInlet*  contribInlet (nullptr);
static StatsMonitor*     statsMonitor (nullptr);

void sigHandler(int signal)
{
  static unsigned callCount = 0;

  printf("sigHandler() called\n");

  if (callCount == 0)
  {
    if (drpSim)  drpSim->shutdown();
  }

  if (callCount++)
  {
    fprintf(stderr, "Aborting on 2nd ^C...\n");
    ::abort();
  }
}

void usage(char *name, char *desc)
{
  fprintf(stderr, "Usage:\n");
  fprintf(stderr, "  %s [OPTIONS] <builder_spec> [<builder_spec> [...]]\n", name);

  if (desc)
    fprintf(stderr, "\n%s\n", desc);

  fprintf(stderr, "\n<builder_spec> has the form '[<builder_id>:]<builder_addr>'\n");
  fprintf(stderr, "  <builder_id> must be in the range 0 - 63.\n");
  fprintf(stderr, "  If the ':' is omitted, <builder_id> will be given the\n");
  fprintf(stderr, "  spec's positional value from 0 - 63, from left to right.\n");

  fprintf(stderr, "\nOptions:\n");

  fprintf(stderr, " %-20s %s (server: %d)\n",  "-S <srv_port>",
          "Base port number for Contributors", srv_port_base);
  fprintf(stderr, " %-20s %s (client: %d)\n",  "-C <clt_port>",
          "Base port number for Builders",     clt_port_base);

  fprintf(stderr, " %-20s %s (default: %d)\n",      "-i <ID>",
          "Unique ID of this Contributor (0 - 63)", default_id);
  fprintf(stderr, " %-20s %s (default: %014lx)\n",  "-D <batch duration>",
          "Batch duration (must be power of 2)",    batch_duration);
  fprintf(stderr, " %-20s %s (default: %d)\n",      "-B <max batches>",
          "Maximum number of extant batches",       max_batches);
  fprintf(stderr, " %-20s %s (default: %d)\n",      "-E <max entries>",
          "Maximum number of entries per batch",    max_entries);
  fprintf(stderr, " %-20s %s (default: %d)\n",      "-M <seconds>",
          "Monitoring printout period",             mon_period);
  fprintf(stderr, " %-20s %s (default: %d)\n",      "-T <core>",
          "Base core number to pin threads to",     lcoreBase);

  fprintf(stderr, " %-20s %s\n", "-c", "enable result checking");
  fprintf(stderr, " %-20s %s\n", "-v", "enable debugging output (repeat for increased detail)");
  fprintf(stderr, " %-20s %s\n", "-h", "display this help output");
}

int main(int argc, char **argv)
{
  // Gather the list of EBs from an argument
  // - Somehow ensure this list is the same for all instances of the client
  // - Maybe pingpong this list around to see whether anyone disagrees with it?

  int op, ret = 0;
  unsigned id         = default_id;
  unsigned srvBase    = srv_port_base;  // Port served to builders
  unsigned cltBase    = clt_port_base;  // Port served by builders
  uint64_t duration   = batch_duration;
  unsigned maxBatches = max_batches;
  unsigned maxEntries = max_entries;
  unsigned monPeriod  = mon_period;

  while ((op = getopt(argc, argv, "h?vcS:C:i:D:B:E:M:T:")) != -1)
  {
    switch (op)
    {
      case 'S':  srvBase    = strtoul(optarg, nullptr, 0);  break;
      case 'C':  cltBase    = strtoul(optarg, nullptr, 0);  break;
      case 'i':  id         = strtoul(optarg, nullptr, 0);  break;
      case 'D':  duration   = atoll(optarg);                break;
      case 'B':  maxBatches = atoi(optarg);                 break;
      case 'E':  maxEntries = atoi(optarg);                 break;
      case 'c':  lcheck     = true;                         break;
      case 'M':  monPeriod  = atoi(optarg);                 break;
      case 'T':  lcoreBase  = atoi(optarg);                 break;
      case 'v':  ++lverbose;                                break;
      case '?':
      case 'h':
        usage(argv[0], (char*)"Test event contributor");
        return 1;
    }
  }

  if (id > 63)
  {
    fprintf(stderr, "Contributor ID is out of range 0 - 63: %d\n", id);
    return 1;
  }
  if (srvBase + id > 0x0000ffff)
  {
    fprintf(stderr, "Server port is out of range 0 - 65535: %d\n", srvBase + id);
    return 1;
  }
  char port[8];
  snprintf(port, sizeof(port), "%d", srvBase + id);
  std::string srvPort(port);

  std::vector<std::string> cltAddr;
  std::vector<std::string> cltPort;
  uint64_t builders = 0;
  if (optind < argc)
  {
    unsigned srcId = 0;
    do
    {
      char* builder = argv[optind];
      char* colon   = strchr(builder, ':');
      unsigned bid  = colon ? atoi(builder) : srcId++;
      if (bid > 63)
      {
        fprintf(stderr, "Builder ID is out of range 0 - 63: %d\n", bid);
        return 1;
      }
      if (cltBase + bid > 0x0000ffff)
      {
        fprintf(stderr, "Client port is out of range 0 - 65535: %d\n", cltBase + bid);
        return 1;
      }
      builders |= 1ul << bid;
      snprintf(port, sizeof(port), "%d", cltBase + bid);
      cltAddr.push_back(std::string(colon ? &colon[1] : builder));
      cltPort.push_back(std::string(port));
    }
    while (++optind < argc);
  }
  else
  {
    fprintf(stderr, "Builder address(s) is required\n");
    return 1;
  }

  ::signal( SIGINT, sigHandler );

  printf("Parameters of Contributor ID %d:\n", id);
  printf("  Batch duration:             %014lx = %ld uS\n", duration, duration);
  printf("  Batch pool depth:           %d\n", maxBatches);
  printf("  Max # of entries per batch: %d\n", maxEntries);
  printf("  Max contribution size:      %zd, batch size: %zd\n",
         max_contrib_size, sizeof(Dgram) + maxEntries * max_contrib_size);
  printf("  Max result       size:      %zd, batch size: %zd\n",
         max_result_size,  sizeof(Dgram) + maxEntries * max_result_size);
  printf("  Monitoring period:          %d\n", monPeriod);
  printf("  Thread base core number:    %d\n", lcoreBase);

  // The order of the connect() calls must match the Event Builder(s) side's

  pin_thread(pthread_self(), lcoreBase + core_outlet);
  TstContribOutlet outlet(cltAddr, cltPort, id, builders, duration, maxBatches, maxEntries, max_contrib_size);
  contribOutlet = &outlet;
  if ( (ret = outlet.connect()) )  return ret; // Connect to EB's inlet

  pin_thread(pthread_self(), lcoreBase + core_inlet);
  TstContribInlet  inlet (         srvPort, id, builders,           maxBatches, maxEntries, max_result_size, outlet);
  contribInlet  = &inlet;
  if ( (ret = inlet.connect())  )  return ret; // Connect to EB's outlet

  pin_thread(pthread_self(), lcoreBase + core_monitor);
  StatsMonitor statsMon(&outlet, &inlet, monPeriod);
  statsMonitor = &statsMon;

  pin_thread(pthread_self(), lcoreBase + core_drpSim);
  DrpSim sim(maxBatches * maxEntries, max_contrib_size, id, outlet);
  drpSim = &sim;

  sim.process();

  statsMon.shutdown();
  inlet.shutdown();
  outlet.shutdown();

  return ret;
}
