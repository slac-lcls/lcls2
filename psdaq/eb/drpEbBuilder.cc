#include "psdaq/eb/BatchManager.hh"
#include "psdaq/eb/EventBuilder.hh"
#include "psdaq/eb/EbEvent.hh"

#include "psdaq/eb/EbFtServer.hh"
#include "psdaq/eb/EbFtClient.hh"

#include "psdaq/service/Routine.hh"
#include "psdaq/service/Task.hh"
#include "psdaq/service/GenericPoolW.hh"
#include "psdaq/service/Histogram.hh"
#include "xtcdata/xtc/Dgram.hh"

#ifndef _GNU_SOURCE
#  define _GNU_SOURCE
#endif
#include <sched.h>
#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <time.h>
#include <inttypes.h>
#include <chrono>
#include <atomic>
#include <chrono>

using namespace XtcData;
using namespace Pds;
using namespace Pds::Fabrics;

static const int      core_inlet       = 8;
static const int      core_outlet      = 9;
static const int      core_monitor     = 10;
static const unsigned mon_period       = 1;          // Seconds
static const unsigned default_id       = 0;          // Builder's ID
static const unsigned srv_port_base    = 32768;      // Base port Builder for receiving results
static const unsigned clt_port_base    = 32768 + 64; // Base port Builder sends to EB on
static const uint64_t batch_duration   = 1ul << 3;   //0x00000000000080UL; // ~128 uS; must be power of 2
static const unsigned max_batches      = 1000; //16; //8192;     // 1 second's worth
static const unsigned max_entries      = 10;  //128;     // 128 uS worth
static const size_t   header_size      = sizeof(Dgram);
static const size_t   input_extent     = 2; // Revisit: Number of "L3" input  data words
static const size_t   result_extent    = 2; // Revisit: Number of "L3" result data words
static const size_t   max_contrib_size = header_size + input_extent  * sizeof(uint32_t);
static const size_t   max_result_size  = header_size + result_extent * sizeof(uint32_t);

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
    class TstEbInlet;

    class TheSrc : public Src
    {
    public:
      TheSrc(Level::Type level, unsigned id) :
        Src(level)
      {
        _log |= id;
      }
    };

    class ResultDest
    {
    public:
      ResultDest() : _msk(0), _dst(_base) { }
    public:
      PoolDeclare;
    public:
      void       destination(const Src& dst)
      {
        uint64_t msk(1ul << (dst.log() & 0x00ffffff));
        if ((_msk & msk) == 0)
        {
          _msk   |= msk;
          *_dst++ = dst;
        }
      }
    public:
      size_t     nDsts()                 const { return _dst - _base; }
      const Src& destination(unsigned i) const { return _base[i];     }
    private:
      uint64_t  _msk;
      Src*      _dst;
    private:
      Src       _base[];
    };

    class TstEbOutlet : public BatchManager,
                        public Routine
    {
    public:
      TstEbOutlet(std::vector<std::string>& cltAddr,
                  std::vector<std::string>& cltPort,
                  unsigned                  id,
                  uint64_t                  duration,
                  unsigned                  maxBatches,
                  unsigned                  maxEntries,
                  size_t                    maxSize);
      virtual ~TstEbOutlet() { }
    public:
      int      connect();
      void     shutdown();
    public:
      uint64_t count() const { return _batchCount; }
    public:
      void     routine();
      void     post(const Batch* batch);
    private:
      Batch*  _pend();
    private:
      EbFtClient                            _outlet;
      const unsigned                        _id;
      const unsigned                        _maxEntries;
      Queue<Batch>                          _pending;
      uint64_t                              _batchCount;
      Histogram                             _depTimeHist;
      Histogram                             _postTimeHist;
      Histogram                             _postCallHist;
      std::chrono::steady_clock::time_point _postPrevTime;
      std::atomic<bool>                     _running;
      Task*                                 _task;
    };

    // NB: TstEbInlet can't be a Routine since EventBuilder already is one
    class TstEbInlet : public EventBuilder
    {
    public:
      TstEbInlet(std::string&  srvPort,
                 unsigned      id,
                 uint64_t      duration,
                 unsigned      maxBatches,
                 unsigned      maxEntries,
                 size_t        maxSize,
                 uint64_t      contributors,
                 BatchManager& outlet);
      virtual ~TstEbInlet();
    public:
      int      connect();
      void     shutdown()    { _running = false;   }
    public:
      uint64_t count() const { return _eventCount; }
    public:
      void     process();
    public:
      void     process(EbEvent* event);
      uint64_t contract(const Dgram* contrib) const;
      void     fixup(EbEvent* event, unsigned srcId);
    private:
      const unsigned                        _maxBatches;
      const size_t                          _maxBatchSize;
      EbFtServer                            _inlet;
      const unsigned                        _id;
      const Src                             _src;
      const TypeId                          _tag;
      const uint64_t                        _contract;
      GenericPool                           _results; // No RW as it shadows the batch pool, which has RW
      //EbDummyTC                             _dummy;   // Template for TC of dummy contributions  // Revisit: ???
      BatchManager&                         _outlet;
      uint64_t                              _eventCount;
      Histogram                             _arrTimeHist;
      Histogram                             _pendTimeHist;
      Histogram                             _pendCallHist;
      std::chrono::steady_clock::time_point _pendPrevTime;
      std::atomic<bool>                     _running;
    };

    class StatsMonitor : public Routine
    {
    public:
      StatsMonitor(const TstEbOutlet* outlet,
                   const TstEbInlet*  inlet,
                   unsigned           monPd);
      virtual ~StatsMonitor() { }
    public:
      void     shutdown();
 public:
      void     routine();
    private:
      const TstEbOutlet* _outlet;
      const TstEbInlet*  _inlet;
      const unsigned     _monPd;
      std::atomic<bool>  _running;
      Task*              _task;
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

TstEbInlet::TstEbInlet(std::string&  srvPort,
                       unsigned      id,
                       uint64_t      duration,
                       unsigned      maxBatches,
                       unsigned      maxEntries,
                       size_t        maxSize,
                       uint64_t      contributors,
                       BatchManager& outlet) :
  EventBuilder (new Task(TaskObject("tEB_BackEnd", 0, 0, 0, 0, 0)),
                maxBatches, maxEntries, __builtin_popcountl(contributors), duration),
  _maxBatches  (maxBatches),
  _maxBatchSize(sizeof(Dgram) + maxEntries * maxSize),
  _inlet       (srvPort, __builtin_popcountl(contributors), maxBatches * _maxBatchSize, EbFtServer::PER_PEER_BUFFERS),
  _id          (id),
  _src         (TheSrc(Level::Event, id)),
  _tag         (TypeId::Data, 0), //_l3SummaryType),
  _contract    (contributors),
  _results     (sizeof(ResultDest) + __builtin_popcountl(contributors) * sizeof(Src) , maxBatches * maxEntries),
  //_dummy       (Level::Fragment),
  _outlet      (outlet),
  _eventCount  (0),
  _arrTimeHist (12, double(1 << 20)/1000.),
  _pendTimeHist(12, 1.0),
  _pendCallHist(12, 1.0),
  _pendPrevTime(std::chrono::steady_clock::now()),
  _running     (true)
{
}

TstEbInlet::~TstEbInlet()
{
  backEndTask()->destroy();
}

int TstEbInlet::connect()
{
  int ret = _inlet.connect(_id);
  if (ret)
  {
    fprintf(stderr, "FtServer connect() failed\n");
  }
  return ret;
}

void TstEbInlet::process()
{
  // Event builder sorts contributions into a time ordered list
  // Then calls the user's process with complete events to build the result datagram
  // Post complete events to the outlet

  // Iterate over contributions in the batch
  // Event build them according to their trigger group

  pin_thread(pthread_self(), lcoreBase + core_inlet);

  start();                              // Start the event timeout timer

  uint64_t contribCount = 0;
  auto     startTime    = std::chrono::steady_clock::now();

  while (_running)
  {
    // Pend for an input datagram (batch) and pass its datagrams to the event builder.
    uint64_t data;
    auto t0 = std::chrono::steady_clock::now();
    if (_inlet.pend(&data))  continue;
    auto t1 = std::chrono::steady_clock::now();
    const Dgram* batch = (const Dgram*)data;

    {
      struct timespec ts;
      clock_gettime(CLOCK_MONOTONIC, &ts);
      uint64_t dS(ts.tv_sec  - batch->seq.stamp().seconds());
      uint64_t dN(ts.tv_nsec - batch->seq.stamp().nanoseconds());
      uint64_t dT((dS * 1000000000ul + dN) >> 20);

      _arrTimeHist.bump(dT);
    }

    typedef std::chrono::microseconds us_t;

    _pendTimeHist.bump(std::chrono::duration_cast<us_t>(t1 - t0).count());
    _pendCallHist.bump(std::chrono::duration_cast<us_t>(t0 - _pendPrevTime).count());
    _pendPrevTime = t0;

    if (lverbose)
    {
      static unsigned cnt = 0;
      unsigned idx  = (((char*)batch - _inlet.base()) / _maxBatchSize) % _maxBatches;
      unsigned from = batch->xtc.src.log() & 0xff;
      printf("EbInlet  rcvd  %6d        batch[%2d]    @ %16p, ts %014lx, sz %3zd from Contrib %d\n", ++cnt, idx,
             batch, batch->seq.pulseId().value(), sizeof(*batch) + batch->xtc.sizeofPayload(), from);
    }

    contribCount += processBulk(batch);
  }

  cancel();                             // Stop the event timeout timer

  auto endTime = std::chrono::steady_clock::now();

  dumpStats("Inlet per contributor rate",
            startTime, endTime, contribCount / __builtin_popcountl(_contract));

  printf("\nEvent builder dump:\n");
  EventBuilder::dump(0);

  printf("\nInlet results pool:\n");
  _results.dump();

  _inlet.shutdown();

  char fs[80];
  sprintf(fs, "arrTime_%d.hist", _id);
  printf("Dumped arrival time histogram to ./%s\n", fs);
  _arrTimeHist.dump(fs);

  sprintf(fs, "pendTime_%d.hist", _id);
  printf("Dumped pend time histogram to ./%s\n", fs);
  _pendTimeHist.dump(fs);

  sprintf(fs, "pendCallRate_%d.hist", _id);
  printf("Dumped pend call rate histogram to ./%s\n", fs);
  _pendCallHist.dump(fs);
}

void TstEbInlet::process(EbEvent* event)
{
  if (lverbose > 3)
  {
    static unsigned cnt = 0;
    printf("TstEbInlet::process event dump:\n");
    event->dump(++cnt);
  }
  ++_eventCount;

  // Iterate over the event and build a result datagram
  const EbContribution** const  last    = event->end();
  const EbContribution*  const* contrib = event->begin();
  Dgram                         cdg    (*(event->creator()));
  cdg.env                               = cdg.xtc.src.log() & 0xff;
  cdg.xtc                               = Xtc(_tag, _src);
  Batch*                        batch   = _outlet.allocate(&cdg); // This may post to the outlet
  ResultDest*                   rDest   = (ResultDest*)batch->parameter();
  if (!rDest)
  {
    rDest = new(&_results) ResultDest();
    batch->parameter(rDest);
  }
  size_t    pSize   = result_extent * sizeof(uint32_t);
  Dgram*    rdg     = new(batch->allocate(sizeof(*rdg) + pSize)) Dgram(cdg);
  uint32_t* buffer  = (uint32_t*)rdg->xtc.alloc(pSize);
  memset(buffer, 0, pSize);             // Revisit: Some crud for now
  do
  {
    const Dgram* idg = *contrib;

    rDest->destination(idg->xtc.src);

    uint32_t*    payload = (uint32_t*)idg->xtc.payload();
    *(uint64_t*)buffer = *(uint64_t*)payload == 0xdeadbeef ? 1 : 0;
  }
  while (++contrib != last);

  if (lverbose > 2)
  {
    printf("EbInlet  processed          result[%2d] @ %16p, ts %014lx, sz %3zd\n",
           rDest->destination(0).phy(), rdg, rdg->seq.pulseId().value(), sizeof(*rdg) + rdg->xtc.sizeofPayload());
  }
}

uint64_t TstEbInlet::contract(const Dgram* contrib) const
{
  // This method called when the event is created, which happens when the event
  // builder recognizes the first contribution.  This contribution contains
  // information from the L1 trigger that identifies which readout groups are
  // involved.  This routine can thus look up the expecte list of contributors
  // (the contract) to the event for each of the readout groups and logically OR
  // them together to provide the overall contract.  The list of contributors
  // participating in each readout group is provided at configuration time.

  return _contract;
}

void TstEbInlet::fixup(EbEvent* event, unsigned srcId)
{
  // Revisit: Nothing can usefully be done here since there is no way to
  //          know the buffer index to be used for the result, I think

  if (lverbose)
  {
    fprintf(stderr, "Fixup event %014lx, size %zu, for source %d\n",
            event->sequence(), event->size(), srcId);
  }

  // Revisit: What can be done here?
  //          And do we want to send a result to a contributor we haven't heard from?
  //Datagram* datagram = (Datagram*)event->data();
  //Damage    dropped(1 << Damage::DroppedContribution);
  //Src       source(srcId, 0);
  //
  //datagram->xtc.damage.increase(Damage::DroppedContribution);
  //
  //new(&datagram->xtc.tag) Xtc(_dummy, source, dropped); // Revisit: No tag, no TC
}

TstEbOutlet::TstEbOutlet(std::vector<std::string>& cltAddr,
                         std::vector<std::string>& cltPort,
                         unsigned                  id,
                         uint64_t                  duration,
                         unsigned                  maxBatches,
                         unsigned                  maxEntries,
                         size_t                    maxSize) :
  BatchManager (duration, maxBatches, maxEntries, maxSize),
  _outlet      (cltAddr, cltPort, maxBatches * (sizeof(Dgram) + maxEntries * maxSize)),
  _id          (id),
  _maxEntries  (maxEntries),
  _pending     (),
  _batchCount  (0),
  _depTimeHist (12, double(1 << 20)/1000.),
  _postTimeHist(12, 1.0),
  _postCallHist(12, 1.0),
  _postPrevTime(std::chrono::steady_clock::now()),
  _running     (true),
  _task        (new Task(TaskObject("tOutlet", 0, 0, 0, 0, 0)))
{
}

void TstEbOutlet::shutdown()
{
  pthread_t tid = _task->parameters().taskID();

  _running = false;
  post(::new Batch(Batch::IsLast));

  int   ret    = 0;
  void* retval = nullptr;
  ret = pthread_join(tid, &retval);
  if (ret)  perror("Outlet pthread_join");
}

int TstEbOutlet::connect()
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

Batch* TstEbOutlet::_pend()
{
  Batch* batch = _pending.removeW();

  if (lverbose > 2)
  {
    const Dgram* bdg  = batch->datagram();
    printf("EbOutlet read                batch[%2d] @ %16p, ts %014lx, sz %3zd\n",
           batch->index(), bdg, bdg->seq.pulseId().value(), sizeof(*bdg) + bdg->xtc.sizeofPayload());
  }

  return batch;
}

void TstEbOutlet::post(const Batch* result)
{
  // Accumulate output datagrams (result) from the event builder into a batch
  // datagram.  When the batch completes, post it to the back end.

  _pending.insert(const_cast<Batch*>(result));
}

void TstEbOutlet::routine()
{
  pin_thread(_task->parameters().taskID(), lcoreBase + core_outlet);

  auto startTime = std::chrono::steady_clock::now();

  while (_running)
  {
    // Pend for a result batch.  When it arrives, post its payload of datagrams
    // to all contributors to the batch, as given by the ResultDest structure.
    //
    // This could be done in the same thread as the that of the EB but isn't
    // so that the event builder can make progress building events from input
    // contributions while the back end is possibly stalled behind the transport
    // waiting for resources to transmit the batch.

    Batch* result = _pend();
    if (result->isLast())
    {
      ::delete result;
      continue;
    }

    ++_batchCount;

    {
      const Dgram* batch = result->datagram();

      struct timespec ts;
      clock_gettime(CLOCK_MONOTONIC, &ts);
      uint64_t dS(ts.tv_sec  - batch->seq.stamp().seconds());
      uint64_t dN(ts.tv_nsec - batch->seq.stamp().nanoseconds());
      uint64_t dT((dS * 1000000000ul + dN) >> 20);

      _depTimeHist.bump(dT);
    }

    ResultDest*  rDest  = (ResultDest*)result->parameter();
    const size_t nDsts  = rDest->nDsts();
    const Dgram* rdg    = result->datagram();
    size_t       extent = result->extent();

    auto t0 = std::chrono::steady_clock::now();

    for (unsigned iDst = 0; iDst < nDsts; ++iDst)
    {
      const Src& src = rDest->destination(iDst);
      uint32_t   dst = src.log() & 0x00ffffff;
      uint32_t   idx = src.phy();

      if (lverbose)
      {
        static unsigned cnt = 0;
        void* rmtAdx = (void*)_outlet.rmtAdx(dst, idx * maxBatchSize());
        printf("EbOutlet posts       %6d result   [%2d] @ %16p, ts %014lx, sz %3zd to   Contrib %d = %16p\n",
               ++cnt, idx, rdg, rdg->seq.pulseId().value(), extent, dst, rmtAdx);
      }

      if (_outlet.post(rdg, extent, dst, idx * maxBatchSize()))  break;
    }
    auto t1 = std::chrono::steady_clock::now();

    typedef std::chrono::microseconds us_t;

    _postTimeHist.bump(std::chrono::duration_cast<us_t>(t1 - t0).count());
    _postCallHist.bump(std::chrono::duration_cast<us_t>(t0 - _postPrevTime).count());
    _postPrevTime = t0;

    delete rDest;
    delete result;
  }

  auto endTime = std::chrono::steady_clock::now();

  BatchManager::dump();

  dumpStats("Outlet", startTime, endTime, _batchCount * _maxEntries);

  _outlet.shutdown();

  char fs[80];
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

StatsMonitor::StatsMonitor(const TstEbOutlet* outlet,
                           const TstEbInlet*  inlet,
                           unsigned           monPd) :
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
  uint64_t inCnt     = 0;
  uint64_t inCntPrv  = _inlet  ? _inlet->count()  : 0;
  uint64_t outCnt    = 0;
  uint64_t outCntPrv = _outlet ? _outlet->count() : 0;
  auto     now       = std::chrono::steady_clock::now();

  while(_running)
  {
    sleep(_monPd);

    auto then = now;
    now       = std::chrono::steady_clock::now();

    inCnt     = _inlet  ? _inlet->count()  : inCntPrv;
    outCnt    = _outlet ? _outlet->count() : outCntPrv;

    auto dT   = std::chrono::duration_cast<std::chrono::microseconds>(now - then).count();
    auto dIC  = inCnt  - inCntPrv;
    auto dOC  = outCnt - outCntPrv;

    printf("Inlet / Outlet: N %016lx / %016lx, dN %7ld / %7ld, rate %7.02f / %7.02f KHz\n",
           inCnt, outCnt, dIC, dOC, double(dIC) / dT * 1.0e3, double(dOC) / dT * 1.0e3);

    inCntPrv  = inCnt;
    outCntPrv = outCnt;
  }

  _task->destroy();
}


static TstEbInlet*   ebInlet      = nullptr;
static TstEbOutlet*  ebOutlet     = nullptr;
static StatsMonitor* statsMonitor = nullptr;

void sigHandler( int signal )
{
  static unsigned callCount = 0;

  printf("sigHandler() called\n");

  if (callCount == 0)
  {
    if (ebInlet)  ebInlet->shutdown();
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
  fprintf(stderr, "  %s [OPTIONS] <contributor_spec> [<contributor_spec> [...]]\n", name);

  if (desc)
    fprintf(stderr, "\n%s\n", desc);

  fprintf(stderr, "\n<contributor_spec> has the form '[<contributor_id>:]<contributor_addr>'\n");
  fprintf(stderr, "  <contributor_id> must be in the range 0 - 63.\n");
  fprintf(stderr, "  If the ':' is omitted, <contributor_id> will be given the\n");
  fprintf(stderr, "  spec's positional value from 0 - 63, from left to right.\n");

  fprintf(stderr, "\nOptions:\n");

  fprintf(stderr, " %-20s %s (server: %d)\n",  "-S <srv_port>",
          "Base port number for Builders",     srv_port_base);
  fprintf(stderr, " %-20s %s (client: %d)\n",  "-C <clt_port>",
          "Base port number for Contributors", clt_port_base);

  fprintf(stderr, " %-20s %s (default: %d)\n",     "-i <ID>",
          "Unique ID of this builder (0 - 63)",    default_id);
  fprintf(stderr, " %-20s %s (default: %014lx)\n", "-D <batch duration>",
          "Batch duration (must be power of 2)",   batch_duration);
  fprintf(stderr, " %-20s %s (default: %d)\n",     "-B <max batches>",
          "Maximum number of extant batches",      max_batches);
  fprintf(stderr, " %-20s %s (default: %d)\n",     "-E <max entries>",
          "Maximum number of entries per batch",   max_entries);
  fprintf(stderr, " %-20s %s (default: %d)\n",     "-M <seconds>",
          "Monitoring printout period",            mon_period);
  fprintf(stderr, " %-20s %s (default: %d)\n",     "-T <core>",
          "Base core number to pin threads to",    lcoreBase);

  fprintf(stderr, " %-20s %s\n", "-v", "enable debugging output (repeat for increased detail)");
  fprintf(stderr, " %-20s %s\n", "-h", "display this help output");
}

int main(int argc, char **argv)
{
  int op, ret = 0;
  unsigned id         = default_id;
  unsigned srvBase    = srv_port_base;  // Port served to contributors
  unsigned cltBase    = clt_port_base;  // Port served by contributors
  uint64_t duration   = batch_duration;
  unsigned maxBatches = max_batches;
  unsigned maxEntries = max_entries;
  unsigned monPeriod  = mon_period;;

  while ((op = getopt(argc, argv, "h?vS:C:i:D:B:E:M:T:")) != -1)
  {
    switch (op)
    {
      case 'S':  srvBase    = strtoul(optarg, nullptr, 0);  break;
      case 'C':  cltBase    = strtoul(optarg, nullptr, 0);  break;
      case 'i':  id         = atoi(optarg);                 break;
      case 'D':  duration   = atoll(optarg);                break;
      case 'B':  maxBatches = atoi(optarg);                 break;
      case 'E':  maxEntries = atoi(optarg);                 break;
      case 'M':  monPeriod  = atoi(optarg);                 break;
      case 'T':  lcoreBase  = atoi(optarg);                 break;
      case 'v':  ++lverbose;                                break;
      case '?':
      case 'h':
      default:
        usage(argv[0], (char*)"Test event builder");
        return 1;
    }
  }

  if (id > 63)
  {
    fprintf(stderr, "Builder ID is out of range 0 - 63: %d\n", id);
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
  uint64_t contributors = 0;
  if (optind < argc)
  {
    unsigned srcId = 0;
    do
    {
      char* contributor = argv[optind];
      char* colon       = strchr(contributor, ':');
      unsigned cid      = colon ? atoi(contributor) : srcId++;
      if (cid > 63)
      {
        fprintf(stderr, "Contributor ID is out of the range 0 - 63: %d\n", cid);
        return 1;
      }
      if (cltBase + cid > 0x0000ffff)
      {
        fprintf(stderr, "Client port is out of range 0 - 65535: %d\n", cltBase + cid);
        return 1;
      }
      contributors |= 1ul << cid;
      snprintf(port, sizeof(port), "%d", cltBase + cid);
      cltAddr.push_back(std::string(colon ? &colon[1] : contributor));
      cltPort.push_back(std::string(port));
    }
    while (++optind < argc);
  }
  else
  {
    fprintf(stderr, "Contributor address(s) is required\n");
    return 1;
  }

  ::signal( SIGINT, sigHandler );

  printf("Parameters of Builder ID %d:\n", id);
  printf("  Batch duration:             %014lx = %ld uS\n", duration, duration);
  printf("  Batch pool depth:           %d\n", maxBatches);
  printf("  Max # of entries per batch: %d\n", maxEntries);
  printf("  Max contribution size:      %zd, batch size: %zd\n",
         max_contrib_size, sizeof(Dgram) + maxEntries * max_contrib_size);
  printf("  Max result       size:      %zd, batch size: %zd\n",
         max_result_size,  sizeof(Dgram) + maxEntries * max_result_size);
  printf("  Monitoring period:          %d\n", monPeriod);
  printf("  Thread base core number:    %d\n", lcoreBase);

  // The order of the connect() calls must match the Contributor(s) side's

  pin_thread(pthread_self(), lcoreBase + core_outlet);
  TstEbOutlet outlet(cltAddr, cltPort, id, duration, maxBatches, maxEntries, max_result_size);
  ebOutlet = &outlet;

  pin_thread(pthread_self(), lcoreBase + core_inlet);
  TstEbInlet  inlet (         srvPort, id, duration, maxBatches, maxEntries, max_contrib_size, contributors, outlet);
  ebInlet  = &inlet;
  if ( (ret = inlet.connect())  )  return ret; // Connect to Contributor's outlet

  pin_thread(pthread_self(), lcoreBase + core_outlet);
  if ( (ret = outlet.connect()) )  return ret; // Connect to Contributors' inlet

  pin_thread(pthread_self(), lcoreBase + core_monitor);
  StatsMonitor statsMon(&outlet, &inlet, monPeriod);
  statsMonitor = &statsMon;

  pin_thread(pthread_self(), lcoreBase + core_inlet);

  inlet.process();

  statsMon.shutdown();
  outlet.shutdown();

  return ret;
}
