#include "psdaq/eb/Endpoint.hh"
#include "psdaq/eb/BatchManager.hh"

#include "psdaq/eb/EbLfServer.hh"
#include "psdaq/eb/EbLfClient.hh"

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
#include <assert.h>
#include <climits>
#include <bitset>
#include <new>
#include <chrono>
#include <atomic>
#include <thread>

using namespace XtcData;
using namespace Pds;
using namespace Pds::Fabrics;

static const int      core_base        = 6; // devXX, 8: accXX
static const int      core_offset      = 2; // Allows Ctrb and EB to run on the same machine
static const unsigned mon_period       = 1;          // Seconds
static const unsigned default_id       = 0;          // Contributor's ID (< 64)
static const unsigned max_ctrbs        = 64;         // Maximum possible number of Contributors
static const unsigned srv_port_base    = 32768 + max_ctrbs; // Base port Ctrb receives results       on
static const unsigned clt_port_base    = 32768;             // Base port Ctrb sends    contributions on
static const unsigned max_batches      = 2048;       // Maximum number of batches in circulation
static const unsigned max_entries      = 64;        // < or = to batch_duration
static const uint64_t batch_duration   = max_entries;// > or = to max_entries; power of 2; beam pulse ticks (1 uS)
static const size_t   header_size      = sizeof(Dgram);
static const size_t   input_extent     = 3; // Revisit: Number of "L3" input  data words
static const size_t   result_extent    = 3; // Revisit: Number of "L3" result data words
static const size_t   max_contrib_size = header_size + input_extent  * sizeof(uint32_t);
static const size_t   max_result_size  = header_size + result_extent * sizeof(uint32_t);
static const uint64_t nanosecond       = 1000000000ul; // Nonconfigurable constant: don't change

static       bool     lcheck           = false;
static       unsigned lverbose         = 0;
static       int      lcore1           = core_base + core_offset + 0;
static       int      lcore2           = core_base + core_offset + 12; // devXX, 0 for accXX

typedef std::chrono::steady_clock::time_point TimePoint_t;
typedef std::chrono::microseconds             us_t;
typedef std::chrono::nanoseconds              ns_t;


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
      Input() : Dgram() {}
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
      DrpSim(unsigned maxBatches, unsigned maxEntries, size_t maxEvtSize, unsigned id);
      ~DrpSim();
    public:
      void         shutdown();
    public:
      const Dgram* genEvent();
    private:
      const unsigned _maxBatches;
      const unsigned _maxEntries;
      const size_t   _maxEvtSz;
      const unsigned _id;
      const uint64_t _env;
      const Xtc      _xtc;
      uint64_t       _pid;
      GenericPoolW   _pool;
      Input*         _heldAside;
    };

    class TstContribOutlet : public BatchManager,
                             public Routine
    {
    public:
      TstContribOutlet(std::vector<std::string>& addrs,
                       std::vector<std::string>& ports,
                       unsigned                  id,
                       uint64_t                  builders,
                       uint64_t                  duration,
                       unsigned                  maxBatches,
                       unsigned                  maxEntries,
                       size_t                    maxSize);
      virtual ~TstContribOutlet();
    public:
      void        startup();
      void        shutdown();
    public:
      uint64_t    count() const { return _eventCount; }
    public:
      void        routine();
      void        completed();
      void        post(const Batch*);
    private:
      DrpSim                _drpSim;
      EbLfClient*           _transport;
      const unsigned        _id;
      const unsigned        _numEbs;
      unsigned*             _destinations;
    private:
      uint64_t              _eventCount;
      uint64_t              _batchCount;
    private:
      std::atomic<unsigned> _inFlightOcc;
      Histogram             _inFlightHist;
      Histogram             _depTimeHist;
      Histogram             _postTimeHist;
      Histogram             _postCallHist;
      TimePoint_t           _postPrevTime;
      std::atomic<bool>     _running;
      Task*                 _task;
    };

    class TstContribInlet : public Routine
    {
    private:
      static size_t _calcBatchSize(unsigned maxEntries, size_t maxSize);
      static void*  _allocBatchRegion(unsigned maxBatches, size_t maxBatchSize);
    public:
      TstContribInlet(const char*  addr,
                      std::string& port,
                      unsigned     id,
                      uint64_t     builders,
                      unsigned     maxBatches,
                      unsigned     maxEntries,
                      size_t       maxSize);
      virtual ~TstContribInlet();
    public:
      void     startup(TstContribOutlet* outlet);
      void     shutdown();
    public:
      uint64_t count()        const { return _eventCount;   }
      size_t   maxBatchSize() const { return _maxBatchSize; }
    public:
      void     routine();
    private:
      void    _process(const Dgram* result, const Dgram* input);
    private:
      const unsigned    _id;
      const unsigned    _numEbs;
      const size_t      _maxBatchSize;
      void*             _region;
      EbLfServer*       _transport;
      TstContribOutlet* _outlet;
    private:
      uint64_t          _eventCount;
    private:
      Histogram         _rttHist;
      Histogram         _pendTimeHist;
      Histogram         _pendCallHist;
      TimePoint_t       _pendPrevTime;
      std::atomic<bool> _running;
      Task*             _task;
    };

    class StatsMonitor
    {
    public:
      StatsMonitor(const TstContribOutlet* outlet,
                   const TstContribInlet*  inlet,
                   unsigned                monPd);
      virtual ~StatsMonitor() { }
    public:
      void     shutdown();
    public:
      void     process();
    private:
      const TstContribOutlet* _outlet;
      const TstContribInlet*  _inlet;
      const unsigned          _monPd;
      std::atomic<bool>       _running;
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

DrpSim::DrpSim(unsigned maxBatches,
               unsigned maxEntries,
               size_t   maxEvtSize,
               unsigned id) :
  _maxBatches(maxBatches),
  _maxEntries(maxEntries),
  _maxEvtSz  (maxEvtSize),
  _id        (id),
  _env       (0),
  _xtc       (TypeId(TypeId::Data, 0), TheSrc(Level::Segment, id)),
  _pid       (0x01000000000003UL),  // Something non-zero and not on a batch boundary
  _pool      (sizeof(Entry) + maxEvtSize, maxBatches * maxEntries),
  _heldAside (new(&_pool) Input())
{
}

DrpSim::~DrpSim()
{
  printf("\nDrpSim Input data pool:\n");
  _pool.dump();
}

void DrpSim::shutdown()
{
  delete _heldAside;  // Make sure there's at least one event in the pool for
                      // process() to get out of resource wait and test _running
}

const Dgram* DrpSim::genEvent()
{
  static unsigned cnt = 0;

  // Fake datagram header
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  const Sequence seq(Sequence::Event, TransitionId::L1Accept,
                     TimeStamp(ts), PulseId(_pid));

  _pid += 1; //27; // Revisit: fi_tx_attr.iov_limit = 6 = 1 + max # events per batch
  //_pid = ts.tv_nsec / 1000;     // Revisit: Not guaranteed to be the same across systems
  //_pid += _maxEntries;

  Input* idg = new(&_pool) Input(seq, _env, _xtc);

  size_t inputSize = input_extent * sizeof(uint32_t);

  // Revisit: Here is where L3 trigger input information would be inserted into the datagram,
  //          whatever that will look like.  SmlD is nominally the right container for this.
  uint32_t* payload = (uint32_t*)idg->xtc.alloc(inputSize);
  payload[0] = ++cnt;               // Revisit: Some crud for now
  payload[1] = _id;                 // Revisit: Some crud for now
  payload[2] = idg->seq.pulseId().value() & 0xffffffffUL; // Revisit: Some crud for now

  return idg;
}


TstContribOutlet::TstContribOutlet(std::vector<std::string>& addrs,
                                   std::vector<std::string>& ports,
                                   unsigned                  id,
                                   uint64_t                  builders,
                                   uint64_t                  duration,
                                   unsigned                  maxBatches,
                                   unsigned                  maxEntries,
                                   size_t                    maxSize) :
  BatchManager (duration, maxBatches, maxEntries, maxSize),
  _drpSim      (maxBatches, maxEntries, maxSize, id),
  _transport   (new EbLfClient(addrs, ports)),
  _id          (id),
  _numEbs      (std::bitset<64>(builders).count()),
  _destinations(new unsigned[_numEbs]),
  _eventCount  (0),
  _batchCount  (0),
  _inFlightOcc (0),
  _inFlightHist(__builtin_ctz(maxBatches), 1.0),
  _depTimeHist (12, double(1 << 16)/1000.),
  _postTimeHist(12, 1.0),
  _postCallHist(12, 1.0),
  _postPrevTime(std::chrono::steady_clock::now()),
  _running     (true),
  _task        (new Task(TaskObject("tOutlet", 0, 0, 0, 0, 0)))
{
  for (unsigned i = 0; i < _numEbs; ++i)
  {
    unsigned bit = __builtin_ffsl(builders) - 1;
    builders &= ~(1ul << bit);
    _destinations[i] = bit;
  }

  const unsigned tmo(120);              // Seconds
  if (_transport->connect(id, tmo, batchRegion(), batchRegionSize()))
  {
    fprintf(stderr, "TstContribOutlet: Transport connect failed\n");
    abort();
  }
}

TstContribOutlet::~TstContribOutlet()
{
  delete [] _destinations;

  if (_transport)  delete _transport;
}

void TstContribOutlet::startup()
{
  _task->call(this);
}

void TstContribOutlet::shutdown()
{
  pthread_t tid = _task->parameters().taskID();

  _running = false;
  _drpSim.shutdown();

  int   ret    = 0;
  void* retval = nullptr;
  ret = pthread_join(tid, &retval);
  if (ret)  perror("Outlet pthread_join");
}

void TstContribOutlet::post(const Batch* batch)
{
  unsigned     dst    = _destinations[batchId(batch->id()) % _numEbs];
  const Dgram* bdg    = batch->datagram();
  size_t       extent = batch->extent();
  uint32_t     idx    = batch->index();

  if (lverbose)
  {
    uint64_t pid    = bdg->seq.pulseId().value();
    void*    rmtAdx = (void*)_transport->rmtAdx(dst, idx * maxBatchSize());
    printf("ContribOutlet posts %6ld        batch[%4d]    @ %16p, ts %014lx, sz %3zd to   EB %d %16p\n",
           _batchCount, idx, bdg, pid, extent, dst, rmtAdx);
  }

  auto t0 = std::chrono::steady_clock::now();
  _transport->post(dst, bdg, extent, idx * maxBatchSize(), (_id << 24) + idx);
  auto t1 = std::chrono::steady_clock::now();

  ++_batchCount;

  {
    auto d = std::chrono::seconds            { bdg->seq.stamp().seconds()     } +
             std::chrono::nanoseconds        { bdg->seq.stamp().nanoseconds() };
    std::chrono::steady_clock::time_point tp { std::chrono::duration_cast<std::chrono::steady_clock::duration>(d) };
    int64_t dT(std::chrono::duration_cast<ns_t>(t0 - tp).count());
    _depTimeHist.bump(dT >> 16);

    dT = std::chrono::duration_cast<us_t>(t1 - t0).count();
    if (dT > 4095)  printf("postTime = %ld us\n", dT);
    _postTimeHist.bump(dT);
    _postCallHist.bump(std::chrono::duration_cast<us_t>(t0 - _postPrevTime).count());
    _postPrevTime = t0;
  }

  _inFlightOcc++;
  _inFlightHist.bump(_inFlightOcc);

  // For open-loop running; Also comment out call to EB::processBulk()
  //const Dgram*       bdg    = batch->datagram();
  //const Dgram*       result = (const Dgram*)bdg->xtc.payload();
  //const Dgram* const last   = (const Dgram*)bdg->xtc.next();
  //unsigned           i      = 0;
  //while(result != last)
  //{
  //  const Dgram* next = (const Dgram*)result->xtc.next();
  //
  //  delete (const Input*)(batch->datagram(i++));
  //
  //  result = next;
  //}
  //delete batch;

  //usleep(10000);
}

void TstContribOutlet::completed()
{
  _inFlightOcc--;
}

void TstContribOutlet::routine()
{
  pin_thread(_task->parameters().taskID(), lcore1);

  while (_running)
  {
    //char str[BUFSIZ];
    //printf("Enter a character to take an event: ");
    //scanf("%s", str);
    //usleep(1000000);

    const Dgram* event = _drpSim.genEvent();

    if (lverbose > 1)
    {
      printf("Batching           input          dg           @ %16p, ts %014lx, sz %3zd, src %2d\n",
             event, event->seq.pulseId().value(), sizeof(*event) + event->xtc.sizeofPayload(), event->xtc.src.log() & 0xff);
    }

    ++_eventCount;

    process(event);
  }

  // Revisit: Call post(lastBatch) here

  BatchManager::dump();

  char fs[80];
  sprintf(fs, "inFlightOcc_%d.hist", _id);
  printf("Dumped in-flight occupancy histogram to ./%s\n", fs);
  _inFlightHist.dump(fs);

  sprintf(fs, "depTime_%d.hist", _id);
  printf("Dumped departure time histogram to ./%s\n", fs);
  _depTimeHist.dump(fs);

  sprintf(fs, "postTime_%d.hist", _id);
  printf("Dumped post time histogram to ./%s\n", fs);
  _postTimeHist.dump(fs);

  sprintf(fs, "postCallRate_%d.hist", _id);
  printf("Dumped post call rate histogram to ./%s\n", fs);
  _postCallHist.dump(fs);

  _transport->shutdown();

  _task->destroy();
}


TstContribInlet::TstContribInlet(const char*  ifAddr,
                                 std::string& port,
                                 unsigned     id,
                                 uint64_t     builders,
                                 unsigned     maxBatches,
                                 unsigned     maxEntries,
                                 size_t       maxSize) :
  _id          (id),
  _numEbs      (std::bitset<64>(builders).count()),
  _maxBatchSize(_calcBatchSize(maxEntries, maxSize)),
  _region      (_allocBatchRegion(maxBatches, _maxBatchSize)),
  _transport   (new EbLfServer(ifAddr, port, _numEbs)),
  _outlet      (nullptr),
  _eventCount  (0),
  _rttHist     (12, 1.0), //double(1 << 16)/1000.),
  _pendTimeHist(12, 1.0),
  _pendCallHist(12, 1.0),
  _pendPrevTime(std::chrono::steady_clock::now()),
  _running     (true),
  _task        (new Task(TaskObject("tInlet", 0, 0, 0, 0, 0)))
{
  size_t size = maxBatches * _maxBatchSize;

  if (_region == nullptr)
  {
    fprintf(stderr, "No memory found for a result region of size %zd\n", size);
    abort();
  }

  if (_transport->connect(id, _region, size, EbLfBase::PEERS_SHARE_BUFFERS))
  {
    fprintf(stderr, "TstContribInlet: Transport connect failed\n");
    abort();
  }
}

TstContribInlet::~TstContribInlet()
{
  if (_transport)  delete _transport;
  if (_region)     free  (_region);
}

size_t TstContribInlet::_calcBatchSize(unsigned maxEntries, size_t maxSize)
{
  size_t alignment = sysconf(_SC_PAGESIZE);
  size_t size      = sizeof(Dgram) + maxEntries * maxSize;
  size             = alignment * ((size + alignment - 1) / alignment);
  return size;
}

void* TstContribInlet::_allocBatchRegion(unsigned maxBatches, size_t maxBatchSize)
{
  size_t   alignment = sysconf(_SC_PAGESIZE);
  size_t   size      = maxBatches * maxBatchSize;
  assert((size & (alignment - 1)) == 0);
  void*    region    = nullptr;
  int      ret       = posix_memalign(&region, alignment, size);
  if (ret)
  {
    perror("posix_memalign");
    return nullptr;
  }

  return region;
}

void TstContribInlet::startup(TstContribOutlet* outlet)
{
  _outlet = outlet;

  _task->call(this);
}

void TstContribInlet::shutdown()
{
  pthread_t tid = _task->parameters().taskID();

  _running = false;

  int   ret    = 0;
  void* retval = nullptr;
  ret = pthread_join(tid, &retval);
  if (ret)  perror("Inlet pthread_join");
}

void TstContribInlet::routine()
{
  pin_thread(_task->parameters().taskID(), lcore2);

  while (_running)
  {
    // Pend for a result datagram (batch) and process it.
    fi_cq_data_entry wc;
    auto t0 = std::chrono::steady_clock::now();
    if (_transport->pend(&wc))  continue;
    auto t1 = std::chrono::steady_clock::now();

    unsigned     idx   = wc.data & 0x00ffffff;
    unsigned     srcId = wc.data >> 24;
    const Dgram* bdg   = (const Dgram*)(_transport->lclAdx(srcId, idx * _maxBatchSize));

    if (lverbose)
    {
      static unsigned cnt    = 0;
      uint64_t        pid    = bdg->seq.pulseId().value();
      size_t          extent = sizeof(*bdg) + bdg->xtc.sizeofPayload();
      printf("ContribInlet  rcvd        %6d result   [%4d] @ %16p, ts %014lx, sz %3zd from EB %d\n",
             cnt++, idx, bdg, pid, extent, srcId);
    }

    if (bdg->env == _id)
    {
      auto d = std::chrono::seconds            { bdg->seq.stamp().seconds()     } +
               std::chrono::nanoseconds        { bdg->seq.stamp().nanoseconds() };
      std::chrono::steady_clock::time_point tp { std::chrono::duration_cast<std::chrono::steady_clock::duration>(d) };
      int64_t dT(std::chrono::duration_cast<us_t>(t1 - tp).count());
      _rttHist.bump(dT); // >> 16);
      //printf("In  Batch %014lx RTT  = %ld S, %ld ns\n", bdg->seq.pulseId().value(), dS, dN);

      dT = std::chrono::duration_cast<us_t>(t1 - t0).count();
      if (dT > 4095)  printf("pendTime = %ld us\n", dT);
      _pendTimeHist.bump(dT);
      _pendCallHist.bump(std::chrono::duration_cast<us_t>(t0 - _pendPrevTime).count());
      _pendPrevTime = t0;
    }

    _transport->postCompRecv(srcId);

    const Batch*       input  = _outlet->batch(idx);
    const Dgram*       result = (const Dgram*)bdg->xtc.payload();
    const Dgram* const last   = (const Dgram*)bdg->xtc.next();
    unsigned           i      = 0;
    while(result != last)
    {
      _process(result, input->datagram(i++));

      result = (const Dgram*)result->xtc.next();
    }
    _eventCount += i;

    _outlet->completed();

    delete input;
  }

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

  _transport->shutdown();

  _task->destroy();
}

void TstContribInlet::_process(const Dgram* result, const Dgram* input)
{
  bool     ok  = true;
  uint64_t pid = result->seq.pulseId().value();

  if (lcheck)
  {
    uint32_t* payload = (uint32_t*)result->xtc.payload();
    if (_numEbs == 1)
    {
      static unsigned _counter = 0;
      if (payload[0] != _counter + 1)
      {
        ok = false;
        printf("Result counter didn't increase by 1: expected %08x, got %08x\n",
               _counter + 1, payload[0]);
      }
      _counter = payload[0];
      static uint64_t _pid = 0;
      if (_pid > pid)
      {
        ok = false;
        printf("Pulse ID didn't increase: previous = %014lx, current = %014lx\n", _pid, pid);
      }
      _pid = pid;
    }
    if ((pid & 0xffffffffUL) != payload[2])
    {
      ok = false;
      printf("Pulse ID mismatch: expected %08lx, got %08x\n",
             pid & 0xffffffffUL, payload[2]);
    }
    if (!ok)
    {
      printf("ContribInlet  found result   %16p, ts %014lx, sz %d, pyld:\n",
             result, pid, result->xtc.sizeofPayload());
      for (unsigned i = 0; i < result_extent; ++i)
        printf(" %08x", payload[i]);
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
}


StatsMonitor::StatsMonitor(const TstContribOutlet* outlet,
                           const TstContribInlet*  inlet,
                           unsigned                monPd) :
  _outlet (outlet),
  _inlet  (inlet),
  _monPd  (monPd),
  _running(true)
{
}

void StatsMonitor::shutdown()
{
  _running = false;
}

void StatsMonitor::process()
{
  pin_thread(pthread_self(), lcore2);

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

    auto dT   = std::chrono::duration_cast<us_t>(now - then).count();
    auto dOC  = outCnt - outCntPrv;
    auto dIC  = inCnt  - inCntPrv;

    printf("Outlet / Inlet: N %016lx / %016lx, dN %7ld / %7ld, rate %7.02f / %7.02f KHz\n",
           outCnt, inCnt, dOC, dIC, double(dOC) / dT * 1.0e3, double(dIC) / dT * 1.0e3);

    outCntPrv = outCnt;
    inCntPrv  = inCnt;
  }
}


static StatsMonitor* statsMonitor(nullptr);

void sigHandler(int signal)
{
  static std::atomic<unsigned> callCount(0);

  printf("sigHandler() called: call count = %u\n", callCount.load());

  if (callCount == 0)
  {
    if (statsMonitor)  statsMonitor->shutdown();
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
  fprintf(stderr, "\n<builder_id> must be in the range 0 - %d.\n", max_ctrbs - 1);
  fprintf(stderr, "  If the ':' is omitted, <builder_id> will be given the\n");
  fprintf(stderr, "  spec's positional value from 0 - %d, from left to right.\n", max_ctrbs - 1);

  fprintf(stderr, "\nOptions:\n");

  fprintf(stderr, " %-20s %s (default: %s)\n",        "-A <interface_addr>",
          "IP address of the interface to use",       "libfabric's 'best' choice");
  fprintf(stderr, " %-20s %s (default: %d)\n",        "-S <port>",
          "Base port for sending contributions",      srv_port_base);
  fprintf(stderr, " %-20s %s (default: %d)\n",        "-C <port>",
          "Base port for receiving results",          clt_port_base);

  fprintf(stderr, " %-20s %s (default: %d)\n",        "-i <ID>",
          "Unique ID of this Contributor (0 - 63)",   default_id);
  fprintf(stderr, " %-20s %s (default: %014lx)\n",    "-D <batch duration>",
          "Batch duration (must be power of 2)",      batch_duration);
  fprintf(stderr, " %-20s %s (default: %d)\n",        "-B <max batches>",
          "Maximum number of extant batches",         max_batches);
  fprintf(stderr, " %-20s %s (default: %d)\n",        "-E <max entries>",
          "Maximum number of entries per batch",      max_entries);
  fprintf(stderr, " %-20s %s (default: %d)\n",        "-M <seconds>",
          "Monitoring printout period",               mon_period);
  fprintf(stderr, " %-20s %s (default: %d)\n",        "-1 <core>",
          "Core number for pinning Outlet thread to", lcore1);
  fprintf(stderr, " %-20s %s (default: %d)\n",        "-2 <core>",
          "Core number for pinning other threads to", lcore2);

  fprintf(stderr, " %-20s %s\n", "-c", "enable result checking");
  fprintf(stderr, " %-20s %s\n", "-v", "enable debugging output (repeat for increased detail)");
  fprintf(stderr, " %-20s %s\n", "-h", "display this help output");
}

int main(int argc, char **argv)
{
  // Gather the list of EBs from an argument
  // - Somehow ensure this list is the same for all instances of the client
  // - Maybe pingpong this list around to see whether anyone disagrees with it?

  int      op, ret    = 0;
  unsigned id         = default_id;
  char*    ifAddr     = nullptr;
  unsigned srvBase    = srv_port_base;  // Port served to builders
  unsigned cltBase    = clt_port_base;  // Port served by builders
  uint64_t duration   = batch_duration;
  unsigned maxBatches = max_batches;
  unsigned maxEntries = max_entries;
  unsigned monPeriod  = mon_period;

  while ((op = getopt(argc, argv, "h?vcA:S:C:i:D:B:E:M:1:2:")) != -1)
  {
    switch (op)
    {
      case 'A':  ifAddr     = optarg;         break;
      case 'S':  srvBase    = atoi(optarg);   break;
      case 'C':  cltBase    = atoi(optarg);   break;
      case 'i':  id         = atoi(optarg);   break;
      case 'D':  duration   = atoll(optarg);  break;
      case 'B':  maxBatches = atoi(optarg);   break;
      case 'E':  maxEntries = atoi(optarg);   break;
      case 'c':  lcheck     = true;           break;
      case 'M':  monPeriod  = atoi(optarg);   break;
      case '1':  lcore1     = atoi(optarg);   break;
      case '2':  lcore2     = atoi(optarg);   break;
      case 'v':  ++lverbose;                  break;
      case '?':
      case 'h':
        usage(argv[0], (char*)"Test event contributor");
        return 1;
    }
  }

  if (id >= max_ctrbs)
  {
    fprintf(stderr, "Contributor ID %d is out of range 0 - %d\n", id, max_ctrbs - 1);
    return 1;
  }
  if (srvBase + id > USHRT_MAX)
  {
    fprintf(stderr, "Server port is out of range 0 - %u: %d\n", USHRT_MAX, srvBase + id);
    return 1;
  }

  std::string srvPort(std::to_string(srvBase + id));

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
      if (bid >= max_ctrbs)
      {
        fprintf(stderr, "Builder ID %d is out of range 0 - %d\n", bid, max_ctrbs - 1);
        return 1;
      }
      if (cltBase + bid + max_ctrbs > USHRT_MAX)
      {
        fprintf(stderr, "Client port %d is out of range 0 - %d\n", cltBase + bid + max_ctrbs, USHRT_MAX);
        return 1;
      }
      builders |= 1ul << bid;
      cltAddr.push_back(std::string(colon ? &colon[1] : builder));
      cltPort.push_back(std::string(std::to_string(cltBase + bid)));
    }
    while (++optind < argc);
  }
  else
  {
    fprintf(stderr, "Builder address(es) is required\n");
    return 1;
  }

  if (maxEntries > duration)
  {
    fprintf(stderr, "More batch entries (%u) requested than definable in the "
            "batch duration (%lu); reducing to avoid wasting memory\n",
            maxEntries, duration);
    maxEntries = duration;
  }

  ::signal( SIGINT, sigHandler );

  pin_thread(pthread_self(), lcore1);
  TstContribOutlet* outlet = new TstContribOutlet(cltAddr, cltPort, id, builders, duration, maxBatches, maxEntries, max_contrib_size);

  pin_thread(pthread_self(), lcore2);
  TstContribInlet*  inlet  = new TstContribInlet (ifAddr, srvPort, id, builders,           maxBatches, maxEntries, max_result_size);

  printf("\nParameters of Contributor ID %d:\n", id);
  printf("  Batch duration:             %014lx = %ld uS\n", duration, duration);
  printf("  Batch pool depth:           %d\n", maxBatches);
  printf("  Max # of entries per batch: %d\n", maxEntries);
  const unsigned ibMtu = 4096;
  unsigned mtuCnt = (sizeof(Dgram) + maxEntries * max_contrib_size + ibMtu - 1) / ibMtu;
  unsigned mtuRem = (sizeof(Dgram) + maxEntries * max_contrib_size) % ibMtu;
  printf("  sizeof(Dgram):              %zd\n", sizeof(Dgram));
  printf("  Max contribution size:      %zd, batch size: %zd, # MTUs: %d, last: %d / %d (%f%%)\n",
         max_contrib_size, outlet->maxBatchSize(), mtuCnt, mtuRem, ibMtu, 100. * double(mtuRem) / double(ibMtu));
  mtuCnt = (sizeof(Dgram) + maxEntries * max_result_size + ibMtu - 1) / ibMtu;
  mtuRem = (sizeof(Dgram) + maxEntries * max_result_size) % ibMtu;
  printf("  Max result       size:      %zd, batch size: %zd, # MTUs: %d, last: %d / %d (%f%%)\n",
         max_result_size,  inlet->maxBatchSize(),  mtuCnt, mtuRem, ibMtu, 100. * double(mtuRem) / double(ibMtu));
  printf("  Monitoring period:          %d\n", monPeriod);
  printf("  Thread core numbers:        %d, %d\n", lcore1, lcore2);
  printf("\n");

  pin_thread(pthread_self(), lcore2);
  StatsMonitor* statsMon = new StatsMonitor(outlet, inlet, monPeriod);
  statsMonitor = statsMon;

  inlet->startup(outlet);
  outlet->startup();

  statsMon->process();

  printf("\nStatsMonitor was shut down.\n");
  printf("\nShutting down Outlet...\n");
  outlet->shutdown();
  printf("\nShutting down Inlet...\n");
  inlet->shutdown();

  delete statsMon;
  delete inlet;
  delete outlet;

  return ret;
}
