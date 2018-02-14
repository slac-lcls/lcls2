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
static const unsigned default_id       = 0;          // Contributor's ID
static const unsigned srv_port_base    = 32768 + 64; // Base port Contributor for receiving results
static const unsigned clt_port_base    = 32768;      // Base port Contributor sends to EB on
static const unsigned max_batches      = 2048;       // 1 second's worth
static const unsigned max_entries      = 512;        // < or = to batch_duration
static const uint64_t batch_duration   = max_entries;// > or = to max_entries; power of 2
static const size_t   header_size      = sizeof(Dgram);
static const size_t   input_extent     = 5; // Revisit: Number of "L3" input  data words
//static const size_t   result_extent    = 2; // Revisit: Number of "L3" result data words
static const size_t   result_extent    = input_extent;
static const size_t   max_contrib_size = header_size + input_extent  * sizeof(uint32_t);
static const size_t   max_result_size  = header_size + result_extent * sizeof(uint32_t);
static const uint64_t nanosecond       = 1000000000ul;

static       bool     lcheck           = false;
static       unsigned lverbose         = 0;
static       int      lcore1           = core_base + core_offset + 0;
static       int      lcore2           = core_base + core_offset + 12; // devXX, 0 for accXX

typedef std::chrono::steady_clock::time_point tp_t;

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
      void        startup();
      void        shutdown();
      int         connect();
    public:
      uint64_t    count() const { return _eventCount; }
    public:
      void        routine();
      void        completed();
      void        post(const Batch*);
    private:
      DrpSim                                _drpSim;
      EbFtClient                            _transport;
      const unsigned                        _id;
      const unsigned                        _numEbs;
      unsigned*                             _destinations;
    private:
      uint64_t                              _eventCount;
      uint64_t                              _batchCount;
    private:
      std::atomic<unsigned>                 _inFlightOcc;
      Histogram                             _inFlightHist;
      Histogram                             _depTimeHist;
      Histogram                             _postTimeHist;
      Histogram                             _postCallHist;
      std::chrono::steady_clock::time_point _postPrevTime;
      std::atomic<bool>                     _running;
      Task*                                 _task;
    };

    class TstContribInlet : public Routine
    {
    private:
      static size_t _calcBatchSize(unsigned maxEntries, size_t maxSize);
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
      void     startup();
      void     shutdown();
      int      connect();
    public:
      uint64_t count()        const { return _eventCount;   }
      size_t   maxBatchSize() const { return _maxBatchSize; }
    public:
      void     routine();
    private:
      void    _process(const Dgram* result, const Dgram* input);
    private:
      const unsigned                        _id;
      const unsigned                        _numEbs;
      const size_t                          _maxBatchSize;
      EbFtServer                            _transport;
      TstContribOutlet&                     _outlet;
    private:
      uint64_t                              _eventCount;
    private:
      Histogram                             _rttHist;
      Histogram                             _pendTimeHist;
      Histogram                             _pendCallHist;
      std::chrono::steady_clock::time_point _pendPrevTime;
      std::atomic<bool>                     _running;
      Task*                                 _task;
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
  payload[0] = ts.tv_sec;           // Revisit: Some crud for now
  payload[1] = ts.tv_nsec;          // Revisit: Some crud for now
  payload[2] = ++cnt;               // Revisit: Some crud for now
  payload[3] = _id;                 // Revisit: Some crud for now
  payload[4] = idg->seq.pulseId().value() & 0xffffffffUL; // Revisit: Some crud for now

  return idg;
}


TstContribOutlet::TstContribOutlet(std::vector<std::string>& cltAddr,
                                   std::vector<std::string>& cltPort,
                                   unsigned                  id,
                                   uint64_t                  builders,
                                   uint64_t                  duration,
                                   unsigned                  maxBatches,
                                   unsigned                  maxEntries,
                                   size_t                    maxSize) :
  BatchManager (duration, maxBatches, maxEntries, maxSize),
  _drpSim      (maxBatches, maxEntries, maxSize, id),
  _transport   (cltAddr, cltPort, batchRegionSize()),
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
}

TstContribOutlet::~TstContribOutlet()
{
  delete [] _destinations;
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

int TstContribOutlet::connect()
{
  const unsigned tmo(120);
  int ret = _transport.connect(_id, tmo);
  if (ret)
  {
    fprintf(stderr, "FtClient connect() failed\n");
    return ret;
  }

  _transport.registerMemory(batchRegion(), batchRegionSize());

  return ret;
}

void TstContribOutlet::post(const Batch* batch)
{
  unsigned iDst = batchId(batch->id()) % _numEbs;
  unsigned dst  = _destinations[iDst];

  if (_batchCount > 10)
  {
    const Dgram* bdg = batch->datagram();

    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    int64_t dS(ts.tv_sec  - bdg->seq.stamp().seconds());
    int64_t dN(ts.tv_nsec - bdg->seq.stamp().nanoseconds());
    int64_t dT(dS * nanosecond + dN);

    _depTimeHist.bump(dT >> 16);
  }

  if (lverbose)
  {
    static unsigned cnt    = 0;
    const Dgram*    bdg    = batch->datagram();
    void*           rmtAdx = (void*)_transport.rmtAdx(dst, batch->index() * maxBatchSize());
    printf("ContribOutlet posts %6d        batch[%2d]    @ %16p, ts %014lx, sz %3zd to   EB %d %16p\n",
           ++cnt, batch->index(), bdg, bdg->seq.pulseId().value(), batch->extent(), dst, rmtAdx);
  }

  ++_batchCount;

  auto t0 = std::chrono::steady_clock::now();

  _transport.post(batch->buffer(), batch->extent(), dst, batch->index() * maxBatchSize());

  auto t1 = std::chrono::steady_clock::now();

  //if (std::chrono::duration<double>(t0 - startTime).count() > 10.)
  if (_batchCount > 9)
  {
    typedef std::chrono::microseconds us_t;

    _postTimeHist.bump(std::chrono::duration_cast<us_t>(t1 - t0).count());
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

  //usleep(1000000);
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

  _transport.shutdown();

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
  _numEbs      (std::bitset<64>(builders).count()),
  _maxBatchSize(_calcBatchSize(maxEntries, maxSize)),
  _transport   (srvPort, _numEbs, maxBatches * _maxBatchSize, EbFtServer::PEERS_SHARE_BUFFERS),
  _outlet      (outlet),
  _eventCount  (0),
  _rttHist     (12, double(1 << 16)/1000.),
  _pendTimeHist(12, 1.0),
  _pendCallHist(12, 1.0),
  _pendPrevTime(std::chrono::steady_clock::now()),
  _running     (true),
  _task        (new Task(TaskObject("tInlet", 0, 0, 0, 0, 0)))
{
}

size_t TstContribInlet::_calcBatchSize(unsigned maxEntries, size_t maxSize)
{
  size_t alignment = sysconf(_SC_PAGESIZE);
  size_t size      = sizeof(Dgram) + maxEntries * maxSize;
  size             = alignment * ((size + alignment - 1) / alignment);
  return size;
}

void TstContribInlet::startup()
{
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

int TstContribInlet::connect()
{
  int ret = _transport.connect(_id);
  if (ret)
  {
    fprintf(stderr, "FtServer connect() failed\n");
    return ret;
  }

  return ret;
}

void TstContribInlet::routine()
{
  pin_thread(_task->parameters().taskID(), lcore2);

  auto startTime = std::chrono::steady_clock::now();

  while (_running)
  {
    // Pend for a result datagram (batch) and process it.
    uint64_t data;
    auto t0 = std::chrono::steady_clock::now();
    if (_transport.pend(&data))  continue;
    auto t1 = std::chrono::steady_clock::now();
    const Dgram* batch = (const Dgram*)data;

    unsigned idx = ((const char*)batch - _transport.base()) / _maxBatchSize;

    if ((batch->env == _id) && (std::chrono::duration<double>(t0 - startTime).count() > 2.))
    {
      struct timespec ts;
      clock_gettime(CLOCK_MONOTONIC, &ts);
      int64_t dS(ts.tv_sec  - batch->seq.stamp().seconds());
      int64_t dN(ts.tv_nsec - batch->seq.stamp().nanoseconds());
      int64_t dT(dS * nanosecond + dN);

      _rttHist.bump(dT >> 16);
      //printf("In  Batch %014lx RTT  = %ld S, %ld ns\n", batch->seq.pulseId().value(), dS, dN);

      typedef std::chrono::microseconds us_t;

      _pendTimeHist.bump(std::chrono::duration_cast<us_t>(t1 - t0).count());
      _pendCallHist.bump(std::chrono::duration_cast<us_t>(t0 - _pendPrevTime).count());
      _pendPrevTime = t0;
    }

    if (lverbose)
    {
      static unsigned cnt = 0;
      unsigned from = batch->xtc.src.log() & 0xff;
      printf("ContribInlet  rcvd        %6d result   [%2d] @ %16p, ts %014lx, sz %3zd from EB %d\n",
             ++cnt, idx, batch, batch->seq.pulseId().value(), sizeof(*batch) + batch->xtc.sizeofPayload(), from);
    }

    const Batch*       input  = _outlet.batch(idx);
    const Dgram*       result = (const Dgram*)batch->xtc.payload();
    const Dgram* const last   = (const Dgram*)batch->xtc.next();
    unsigned           i      = 0;
    while(result != last)
    {
      _process(result, input->datagram(i++));

      result = (const Dgram*)result->xtc.next();
    }
    _eventCount += i;

    _outlet.completed();

    delete input;
  }

  _transport.shutdown();

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

void TstContribInlet::_process(const Dgram* result, const Dgram* input)
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

    auto dT   = std::chrono::duration_cast<std::chrono::microseconds>(now - then).count();
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
  fprintf(stderr, "  <builder_id> must be in the range 0 - 63.\n");
  fprintf(stderr, "  If the ':' is omitted, <builder_id> will be given the\n");
  fprintf(stderr, "  spec's positional value from 0 - 63, from left to right.\n");

  fprintf(stderr, "\nOptions:\n");

  fprintf(stderr, " %-20s %s (server: %d)\n",  "-S <srv_port>",
          "Base port number for Contributors", srv_port_base);
  fprintf(stderr, " %-20s %s (client: %d)\n",  "-C <clt_port>",
          "Base port number for Builders",     clt_port_base);

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

  int op, ret = 0;
  unsigned id         = default_id;
  unsigned srvBase    = srv_port_base;  // Port served to builders
  unsigned cltBase    = clt_port_base;  // Port served by builders
  uint64_t duration   = batch_duration;
  unsigned maxBatches = max_batches;
  unsigned maxEntries = max_entries;
  unsigned monPeriod  = mon_period;

  while ((op = getopt(argc, argv, "h?vcS:C:i:D:B:E:M:1:2:")) != -1)
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
      case '1':  lcore1     = atoi(optarg);                 break;
      case '2':  lcore2     = atoi(optarg);                 break;
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
  if (srvBase + id > USHRT_MAX)
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
      if (cltBase + bid > USHRT_MAX)
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

  if (maxEntries > duration)
  {
    fprintf(stderr, "More batch entries (%u) requested than definable in the "
            "batch duration (%lu); reducing to avoid wasting memory\n",
            maxEntries, duration);
    maxEntries = duration;
  }

  ::signal( SIGINT, sigHandler );

  pin_thread(pthread_self(), lcore1);
  TstContribOutlet outlet(cltAddr, cltPort, id, builders, duration, maxBatches, maxEntries, max_contrib_size);

  pin_thread(pthread_self(), lcore2);
  TstContribInlet  inlet (         srvPort, id, builders,           maxBatches, maxEntries, max_result_size, outlet);

  printf("Parameters of Contributor ID %d:\n", id);
  printf("  Batch duration:             %014lx = %ld uS\n", duration, duration);
  printf("  Batch pool depth:           %d\n", maxBatches);
  printf("  Max # of entries per batch: %d\n", maxEntries);
  printf("  Max contribution size:      %zd, batch size: %zd\n",
         max_contrib_size, outlet.maxBatchSize());
  printf("  Max result       size:      %zd, batch size: %zd\n",
         max_result_size,  inlet.maxBatchSize());
  printf("  Monitoring period:          %d\n", monPeriod);
  printf("  Thread core numbers:        %d, %d\n", lcore1, lcore2);
  printf("\n");

  // The order of the connect() calls must match the Event Builder(s) side's
  if ( (ret = outlet.connect()) )  return ret; // Connect to EB's inlet
  if ( (ret = inlet.connect())  )  return ret; // Connect to EB's outlet

  pin_thread(pthread_self(), lcore2);
  StatsMonitor statsMon(&outlet, &inlet, monPeriod);
  statsMonitor = &statsMon;

  inlet.startup();
  outlet.startup();

  statsMon.process();

  printf("\nStatsMonitor was shut down.\n");
  printf("\nShutting down Outlet...\n");
  outlet.shutdown();
  printf("\nShutting down Inlet...\n");
  inlet.shutdown();

  return ret;
}
