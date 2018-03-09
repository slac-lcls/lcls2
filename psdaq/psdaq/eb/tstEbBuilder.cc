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
#include <climits>
#include <bitset>
#include <chrono>
#include <atomic>
#include <thread>

using namespace XtcData;
using namespace Pds;
using namespace Pds::Fabrics;

static const int      core_base        = 6; // devXX, 8: accXX
static const int      core_offset      = 1; // Allows Ctrb and EB to run on the same machine
static const unsigned mon_period       = 1;          // Seconds
static const unsigned default_id       = 0;          // Builder's ID (< 64)
static const unsigned max_ebs          = 64;         // Maximum possible number of Builders
static const unsigned srv_port_base    = 32768;           // Base port EB receives contributions on
static const unsigned clt_port_base    = 32768 + max_ebs; // Base port EB sends    results       on
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

static       unsigned lverbose         = 0;
static       int      lcore1           = core_base + core_offset + 0;
static       int      lcore2           = core_base + core_offset + 12; // devXX, 0 for accXX

typedef std::chrono::steady_clock::time_point tp_t;
typedef std::chrono::microseconds             us_t;
typedef std::chrono::nanoseconds              ns_t;


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

    class TstEbOutlet : public BatchManager
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
      void     post(const Batch* batch);
    private:
      EbFtClient                            _transport;
      const unsigned                        _id;
      const unsigned                        _maxEntries;
    private:
      uint64_t                              _batchCount;
    private:
      Histogram                             _depTimeHist;
      Histogram                             _postTimeHist;
      Histogram                             _postCallHist;
      std::chrono::steady_clock::time_point _postPrevTime;
    };

    // NB: TstEbInlet can't be a Routine since EventBuilder already is one
    class TstEbInlet : public EventBuilder
    {
    private:
      static size_t _calcBatchSize(unsigned maxEntries, size_t maxSize);
    public:
      TstEbInlet(const char*   srvAddr,
                 std::string&  srvPort,
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
      uint64_t count()        const { return _eventCount;   }
      size_t   maxBatchSize() const { return _maxBatchSize; }
    public:
      void     process();
    public:
      void     process(EbEvent* event);
      uint64_t contract(const Dgram* contrib) const;
      void     fixup(EbEvent* event, unsigned srcId);
    private:
      const unsigned                        _maxBatches;
      const size_t                          _maxBatchSize;
      EbFtServer                            _transport;
      const unsigned                        _id;
      const Xtc                             _xtc;
      const uint64_t                        _contract;
      GenericPool                           _results; // No RW as it shadows the batch pool, which has RW
      //EbDummyTC                             _dummy;   // Template for TC of dummy contributions  // Revisit: ???
      BatchManager&                         _outlet;
    private:
      uint64_t                              _eventCount;
    private:
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

TstEbInlet::TstEbInlet(const char*   srvAddr,
                       std::string&  srvPort,
                       unsigned      id,
                       uint64_t      duration,
                       unsigned      maxBatches,
                       unsigned      maxEntries,
                       size_t        maxSize,
                       uint64_t      contributors,
                       BatchManager& outlet) :
  EventBuilder (maxBatches, maxEntries, std::bitset<64>(contributors).count(), duration),
  _maxBatches  (maxBatches),
  _maxBatchSize(_calcBatchSize(maxEntries, maxSize)),
  _transport   (srvAddr, srvPort, std::bitset<64>(contributors).count(), maxBatches * _maxBatchSize, EbFtServer::PER_PEER_BUFFERS),
  _id          (id),
  _xtc         (TypeId(TypeId::Data, 0), TheSrc(Level::Event, id)), //_l3SummaryType
  _contract    (contributors),
  _results     (sizeof(ResultDest) + std::bitset<64>(contributors).count() * sizeof(Src) , maxBatches * maxEntries),
  //_dummy       (Level::Fragment),
  _outlet      (outlet),
  _eventCount  (0),
  _arrTimeHist (12, double(1 << 16)/1000.),
  _pendTimeHist(12, double(1 <<  8)/1000.),
  _pendCallHist(12, 1.0),
  _pendPrevTime(std::chrono::steady_clock::now()),
  _running     (true)
{
}

TstEbInlet::~TstEbInlet()
{
}

size_t TstEbInlet::_calcBatchSize(unsigned maxEntries, size_t maxSize)
{
  size_t alignment = sysconf(_SC_PAGESIZE);
  size_t size      = sizeof(Dgram) + maxEntries * maxSize;
  size             = alignment * ((size + alignment - 1) / alignment);
  return size;
}

int TstEbInlet::connect()
{
  int ret = _transport.connect(_id);
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

  pin_thread(task()->parameters().taskID(), lcore2);
  pin_thread(pthread_self(),                lcore2);

  start();                              // Start the event timeout timer

  pin_thread(pthread_self(),                lcore1);

  auto startTime = std::chrono::steady_clock::now();

  while (_running)
  {
    // Pend for an input datagram (batch) and pass its datagrams to the event builder.
    uint64_t data;
    auto t0 = std::chrono::steady_clock::now();
    if (_transport.pend(&data))  continue;
    auto t1 = std::chrono::steady_clock::now();
    const Dgram* batch = (const Dgram*)data;

    //if (std::chrono::duration<double>(t0 - startTime).count() > 2.)
    {
      struct timespec ts;
      clock_gettime(CLOCK_MONOTONIC, &ts);
      int64_t dS(ts.tv_sec  - batch->seq.stamp().seconds());
      int64_t dN(ts.tv_nsec - batch->seq.stamp().nanoseconds());
      int64_t dT(dS * nanosecond + dN);

      _arrTimeHist.bump(dT >> 16);
      //printf("In  Batch  %014lx Pend = %ld S, %ld ns\n", batch->seq.pulseId().value(), dS, dN);

      dT = std::chrono::duration_cast<ns_t>(t1 - t0).count();
      if (dT > 1048576)  printf("pendTime = %ld\n", dT);
      _pendTimeHist.bump(dT >> 8);
      _pendCallHist.bump(std::chrono::duration_cast<us_t>(t0 - _pendPrevTime).count());
      _pendPrevTime = t0;
    }

    if (lverbose)
    {
      static unsigned cnt = 0;
      unsigned idx  = (((char*)batch - _transport.base()) / _maxBatchSize) % _maxBatches;
      unsigned from = batch->xtc.src.log() & 0xff;
      printf("EbInlet  rcvd  %6d        batch[%2d]    @ %16p, ts %014lx, sz %3zd from Contrib %d\n", ++cnt, idx,
             batch, batch->seq.pulseId().value(), sizeof(*batch) + batch->xtc.sizeofPayload(), from);
    }

    processBulk(batch);                 // Comment out for open-loop running
  }

  cancel();                             // Stop the event timeout timer

  printf("\nEvent builder dump:\n");
  EventBuilder::dump(0);

  printf("\nInlet results pool:\n");
  _results.dump();

  _transport.shutdown();

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
  Dgram                         cdg    (*(event->creator()));     // Initialize a new DG
  cdg.env                               = cdg.xtc.src.log() & 0xff; // Revisit: Used only for measuring RTT?
  cdg.xtc                               = _xtc;
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
    for (unsigned i = 0; i < result_extent; ++i)
    {
      buffer[i] |= payload[i];          // Revisit: Some crud for now
    }
  }
  while (++contrib != last);

  if (lverbose > 2)
  {
    printf("EbInlet  processed          result[%2d]    @ %16p, ts %014lx, sz %3zd\n",
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
  _transport   (cltAddr, cltPort),
  _id          (id),
  _maxEntries  (maxEntries),
  _batchCount  (0),
  _depTimeHist (12, double(1 << 16)/1000.),
  _postTimeHist(12, double(1 <<  8)/1000.),
  _postCallHist(12, 1.0),
  _postPrevTime(std::chrono::steady_clock::now())
{
}

void TstEbOutlet::shutdown()
{
  BatchManager::dump();

  _transport.shutdown();

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
}

int TstEbOutlet::connect()
{
  const unsigned tmo(120);
  int ret = _transport.connect(_id, tmo, batchRegionSize());
  if (ret)
  {
    fprintf(stderr, "FtClient connect() failed\n");
    return ret;
  }

  _transport.registerMemory(batchRegion(), batchRegionSize());

  return ret;
}

void TstEbOutlet::post(const Batch* batch)
{
  // Accumulate output datagrams (result) from the event builder into a batch
  // datagram.  When the batch completes, post it to the contributors.

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

  if (lverbose > 2)
  {
    const Dgram* bdg = batch->datagram();
    printf("EbOutlet read                batch[%2d] @ %16p, ts %014lx, sz %3zd\n",
           batch->index(), bdg, bdg->seq.pulseId().value(), sizeof(*bdg) + bdg->xtc.sizeofPayload());
  }

  ++_batchCount;

  ResultDest*  rDest  = (ResultDest*)batch->parameter();
  const size_t nDsts  = rDest->nDsts();
  const Dgram* bdg    = batch->datagram();
  size_t       extent = batch->extent();

  auto t0 = std::chrono::steady_clock::now();

  for (unsigned iDst = 0; iDst < nDsts; ++iDst)
  {
    const Src& src = rDest->destination(iDst);
    uint32_t   dst = src.log() & 0x00ffffff;
    uint32_t   idx = src.phy();

    if (lverbose)
    {
      static unsigned cnt = 0;
      void* rmtAdx = (void*)_transport.rmtAdx(dst, idx * maxBatchSize());
      printf("EbOutlet posts       %6d result   [%2d] @ %16p, ts %014lx, sz %3zd to   Contrib %d = %16p\n",
             ++cnt, idx, bdg, bdg->seq.pulseId().value(), extent, dst, rmtAdx);
    }

    if (_transport.post(bdg, extent, dst, idx * maxBatchSize()))  break;
  }

  auto t1 = std::chrono::steady_clock::now();

  //if (std::chrono::duration<double>(t0 - startTime).count() > 10.)
  //if (_batchCount > 9)
  {
    int64_t dT = std::chrono::duration_cast<ns_t>(t1 - t0).count();
    if (dT > 1048576)  printf("postTime = %ld\n", dT);
    _postTimeHist.bump(dT >> 8);
    _postCallHist.bump(std::chrono::duration_cast<us_t>(t0 - _postPrevTime).count());
    _postPrevTime = t0;
  }

  delete rDest;
  delete batch;
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
  pin_thread(pthread_self(), lcore2);

  uint64_t inCnt     = 0;
  uint64_t inCntPrv  = _inlet  ? _inlet->count()  : 0;
  uint64_t outCnt    = 0;
  uint64_t outCntPrv = _outlet ? _outlet->count() : 0;
  auto     now       = std::chrono::steady_clock::now();

  while(_running)
  {
    std::this_thread::sleep_for(std::chrono::seconds(_monPd));

    auto then = now;
    now       = std::chrono::steady_clock::now();

    inCnt     = _inlet  ? _inlet->count()  : inCntPrv;
    outCnt    = _outlet ? _outlet->count() : outCntPrv;

    auto dT   = std::chrono::duration_cast<us_t>(now - then).count();
    auto dIC  = inCnt  - inCntPrv;
    auto dOC  = outCnt - outCntPrv;

    printf("Inlet / Outlet: N %016lx / %016lx, dN %7ld / %7ld, rate %7.02f / %7.02f KHz\n",
           inCnt, outCnt, dIC, dOC, double(dIC) / dT * 1.0e3, double(dOC) / dT * 1.0e3);

    inCntPrv  = inCnt;
    outCntPrv = outCnt;
  }

  _task->destroy();
}


static TstEbInlet* ebInlet(nullptr);

void sigHandler( int signal )
{
  static std::atomic<unsigned> callCount(0);

  printf("sigHandler() called: call count = %u\n", callCount.load());

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

  fprintf(stderr, " %-20s %s (default: %s)\n",  "-A <interface_addr>",
          "IP address of the interface to use", "libfabric's 'best' choice");

  fprintf(stderr, " %-20s %s (server: %d)\n",  "-S <srv_port>",
          "Base port number for Builders",     srv_port_base);
  fprintf(stderr, " %-20s %s (client: %d)\n",  "-C <clt_port>",
          "Base port number for Contributors", clt_port_base);

  fprintf(stderr, " %-20s %s (default: %d)\n",        "-i <ID>",
          "Unique ID of this builder (0 - 63)",       default_id);
  fprintf(stderr, " %-20s %s (default: %014lx)\n",    "-D <batch duration>",
          "Batch duration (must be power of 2)",      batch_duration);
  fprintf(stderr, " %-20s %s (default: %d)\n",        "-B <max batches>",
          "Maximum number of extant batches",         max_batches);
  fprintf(stderr, " %-20s %s (default: %d)\n",        "-E <max entries>",
          "Maximum number of entries per batch",      max_entries);
  fprintf(stderr, " %-20s %s (default: %d)\n",        "-M <seconds>",
          "Monitoring printout period",               mon_period);
  fprintf(stderr, " %-20s %s (default: %d)\n",        "-1 <core>",
          "Core number for pinning Inlet thread to",  lcore1);
  fprintf(stderr, " %-20s %s (default: %d)\n",        "-2 <core>",
          "Core number for pinning other threads to", lcore2);

  fprintf(stderr, " %-20s %s\n", "-v", "enable debugging output (repeat for increased detail)");
  fprintf(stderr, " %-20s %s\n", "-h", "display this help output");
}

int main(int argc, char **argv)
{
  int op, ret = 0;
  unsigned id         = default_id;
  char*    srvAddr    = nullptr;
  unsigned srvBase    = srv_port_base;  // Port served to contributors
  unsigned cltBase    = clt_port_base;  // Port served by contributors
  uint64_t duration   = batch_duration;
  unsigned maxBatches = max_batches;
  unsigned maxEntries = max_entries;
  unsigned monPeriod  = mon_period;

  while ((op = getopt(argc, argv, "h?vA:S:C:i:D:B:E:M:1:2:")) != -1)
  {
    switch (op)
    {
      case 'A':  srvAddr    = optarg;                       break;
      case 'S':  srvBase    = strtoul(optarg, nullptr, 0);  break;
      case 'C':  cltBase    = strtoul(optarg, nullptr, 0);  break;
      case 'i':  id         = atoi(optarg);                 break;
      case 'D':  duration   = atoll(optarg);                break;
      case 'B':  maxBatches = atoi(optarg);                 break;
      case 'E':  maxEntries = atoi(optarg);                 break;
      case 'M':  monPeriod  = atoi(optarg);                 break;
      case '1':  lcore1     = atoi(optarg);                 break;
      case '2':  lcore2     = atoi(optarg);                 break;
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
      if (cltBase + cid > USHRT_MAX)
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

  if (maxEntries > duration)
  {
    fprintf(stderr, "More batch entries (%u) requested than definable in the "
            "batch duration (%lu); reducing to avoid wasting memory\n",
            maxEntries, duration);
    maxEntries = duration;
  }

  ::signal( SIGINT, sigHandler );

  pin_thread(pthread_self(), lcore1);
  TstEbOutlet outlet(cltAddr, cltPort, id, duration, maxBatches, maxEntries, max_result_size);

  pin_thread(pthread_self(), lcore2);
  TstEbInlet  inlet (srvAddr, srvPort, id, duration, maxBatches, maxEntries, max_contrib_size, contributors, outlet);
  ebInlet  = &inlet;

  printf("Parameters of Builder ID %d:\n", id);
  printf("  Batch duration:             %014lx = %ld uS\n", duration, duration);
  printf("  Batch pool depth:           %d\n", maxBatches);
  printf("  Max # of entries per batch: %d\n", maxEntries);
  printf("  Max contribution size:      %zd, batch size: %zd\n",
         max_contrib_size, inlet.maxBatchSize());
  printf("  Max result       size:      %zd, batch size: %zd\n",
         max_result_size,  outlet.maxBatchSize());
  printf("  Monitoring period:          %d\n", monPeriod);
  printf("  Thread core numbers:        %d, %d\n", lcore1, lcore2);
  printf("\n");

  // The order of the connect() calls must match the Contributor(s) side's
  if ( (ret = inlet.connect())  )  return ret; // Connect to Contributor's outlet
  if ( (ret = outlet.connect()) )  return ret; // Connect to Contributors' inlet

  pin_thread(pthread_self(), lcore2);
  StatsMonitor statsMon(&outlet, &inlet, monPeriod);

  inlet.process(); // Revisit: Can't make this a routine because EB has one in Timer

  printf("\nInlet was shut down.\n");
  printf("\nShutting down StatsMonitor...\n");
  statsMon.shutdown();
  printf("\nShutting down Outlet...\n");
  outlet.shutdown();

  return ret;
}
