#include "psdaq/eb/BatchManager.hh"
#include "psdaq/eb/EventBuilder.hh"
#include "psdaq/eb/EbEvent.hh"

#include "psdaq/eb/EbFtServer.hh"
#include "psdaq/eb/EbFtClient.hh"

#include "psdaq/service/Routine.hh"
#include "psdaq/service/Task.hh"
#include "psdaq/service/GenericPoolW.hh"
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
#include <condition_variable>
#include <mutex>

using namespace XtcData;
using namespace Pds;
using namespace Pds::Fabrics;

static const int      thread_base      = 0;
static const int      thread_inlet     = thread_base + 8;
static const int      thread_outlet    = thread_base + 20; //9;
static const int      thread_monitor   = thread_base + 10;
static const unsigned default_id       = 0;          // Builder's ID
static const unsigned srv_port_base    = 32768;      // Base port Builder for receiving results
static const unsigned clt_port_base    = 32768 + 64; // Base port Builder sends to EB on
static const uint64_t batch_duration   = 1ul << 9;   // ~512 uS; must be power of 2
static const unsigned max_batches      = 2048;       // 1 second's worth
static const unsigned max_entries      = 512;        // 512 uS worth
static const size_t   header_size      = sizeof(Dgram);
static const size_t   input_extent     = 5; // Revisit: Number of "L3" input  data words
//static const size_t   result_extent    = 2; // Revisit: Number of "L3" result data words
static const size_t   result_extent    = input_extent;
static const size_t   max_contrib_size = header_size + input_extent  * sizeof(uint32_t);
static const size_t   max_result_size  = header_size + result_extent * sizeof(uint32_t);

static       unsigned lverbose         = 0;

//static void dump(const Pds::Dgram* dg)
//{
//  char buff[128];
//  time_t t = dg->seq.clock().seconds();
//  strftime(buff,128,"%H:%M:%S",localtime(&t));
//  printf("%s.%09u %016lx %s extent 0x%x ctns %s damage %x\n",
//         buff,
//         dg->seq.clock().nanoseconds(),
//         dg->seq.pulseId().value(),
//         Pds::TransitionId::name(dg->seq.service()),
//         dg->xtc.extent,
//         Pds::TypeId::name(dg->xtc.contains.id()),
//         dg->xtc.damage.value());
//}

static void dumpStats(const char* header, timespec startTime, timespec endTime, uint64_t count)
{
  uint64_t ts = startTime.tv_sec;
  ts = (ts * 1000000000ul) + startTime.tv_nsec;
  uint64_t te = endTime.tv_sec;
  te = (te * 1000000000ul) + endTime.tv_nsec;
  int64_t  dt = te - ts;
  double   ds = double(dt) * 1.e-9;
  printf("\n%s:\n", header);
  printf("Start time: %ld s, %ld ns\n", startTime.tv_sec, startTime.tv_nsec);
  printf("End   time: %ld s, %ld ns\n", endTime.tv_sec,   endTime.tv_nsec);
  printf("Time diff:  %ld = %.0f s\n", dt, ds);
  printf("Count:      %ld\n", count);
  printf("Avg rate:   %.0f Hz\n", double(count) / ds);
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
      void     post(Batch* batch);
    private:
      Batch*  _pend();
    private:
      EbFtClient              _outlet;
      unsigned                _id;
      unsigned                _maxEntries;
      std::mutex              _mutex;
      std::condition_variable _resultCv;
      Queue<Batch>            _pending;
      uint64_t                _batchCount;
      volatile bool           _running;
      Task*                   _task;
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
      unsigned      _maxBatches;
      size_t        _maxBatchSize;
      EbFtServer    _inlet;
      unsigned      _id;
      Src           _src;
      const TypeId  _tag;
      uint64_t      _contract;
      GenericPool   _results;           // No RW as it shadows the batch pool, which has RW
      //EbDummyTC     _dummy;           // Template for TC of dummy contributions  // Revisit: ???
      BatchManager& _outlet;
      uint64_t      _eventCount;
      volatile bool _running;
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
      unsigned           _monPd;
      volatile bool      _running;
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
    printf("Error calling pthread_setaffinity_np: %d\n", rc);
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
  EventBuilder (new Task(TaskObject("tEB_BackEnd")),
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

  pin_thread(pthread_self(), thread_inlet);

  start();                              // Start the event timeout timer

  uint64_t contribCount = 0;
  timespec startTime;
  clock_gettime(CLOCK_REALTIME, &startTime);

  while (_running)
  {
    // Pend for an input datagram (batch) and pass its datagrams to the event builder.
    uint64_t data;
    if (_inlet.pend(&data))  continue;
    const Dgram* batch = (const Dgram*)data;

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

  timespec endTime;
  clock_gettime(CLOCK_REALTIME, &endTime);

  dumpStats("Inlet per contributor rate",
            startTime, endTime, contribCount / __builtin_popcountl(_contract));

  printf("\nEvent builder dump:\n");
  EventBuilder::dump(0);

  printf("\nInlet results pool:\n");
  _results.dump();

  _inlet.shutdown();
}

void TstEbInlet::process(EbEvent* event)
{
  if (lverbose > 3)
  {
    static unsigned cnt = 0;
    //printf("Event ts %014lx, contract %016lx, size = %3zd\n",
    //       event->sequence(), event->contract(), event->size());
    printf("TstEbInlet::process event dump:\n");
    event->dump(++cnt);
  }
  ++_eventCount;

  // Iterate over the event and build a result datagram
  const EbContribution** const  last    = event->end();
  const EbContribution*  const* contrib = event->begin();
  const Dgram*                  cdg     = *contrib;
  Batch*                        batch   = _outlet.allocate(cdg); // This may post to the outlet
  ResultDest*                   rDest   = (ResultDest*)batch->parameter();
  if (!rDest)
  {
    rDest = new(&_results) ResultDest();
    batch->parameter(rDest);
  }
  size_t    pSize   = result_extent * sizeof(uint32_t);
  Dgram*    rdg     = new(batch->allocate(sizeof(*rdg) + pSize)) Dgram(*cdg);
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
  BatchManager(duration, maxBatches, maxEntries, maxSize),
  _outlet     (cltAddr, cltPort, maxBatches * (sizeof(Dgram) + maxEntries * maxSize)),
  _id         (id),
  _maxEntries (maxEntries),
  _mutex      (),
  _resultCv   (),
  _pending    (),
  _batchCount (0),
  _running    (true),
  _task       (new Task(TaskObject("tOutlet", 0, 0, 0, 0, 0)))
{
}

void TstEbOutlet::shutdown()
{
  pthread_t tid = _task->parameters().taskID();

  _running = false;
  //std::unique_lock<std::mutex> lk(_mutex);
  _resultCv.notify_one();

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
  Batch* batch;
  while ((batch = _pending.remove()) == _pending.empty())
  {
    std::unique_lock<std::mutex> lk(_mutex);
    _resultCv.wait(lk);

    if (!_running)  return nullptr;
  }

  if (lverbose > 2)
  {
    const Dgram* bdg  = batch->datagram();
    printf("EbOutlet read                batch[%2d] @ %16p, ts %014lx, sz %3zd\n",
           batch->index(), bdg, bdg->seq.pulseId().value(), sizeof(*bdg) + bdg->xtc.sizeofPayload());
  }

  return batch;
}

void TstEbOutlet::post(Batch* result)
{
  // Accumulate output datagrams (result) from the event builder into a batch
  // datagram.  When the batch completes, post it to the back end.

  _pending.insert(result);
  std::unique_lock<std::mutex> lk(_mutex);
  _resultCv.notify_one();
}

void TstEbOutlet::routine()
{
  pin_thread(_task->parameters().taskID(), thread_outlet);

  timespec startTime;
  clock_gettime(CLOCK_REALTIME, &startTime);

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
    if (!result)
    {
      if (_running)  printf("Outlet: Wakeup with no pending result batch\n");
      continue;
    }

    ++_batchCount;

    ResultDest*  rDest  = (ResultDest*)result->parameter();
    const size_t nDsts  = rDest->nDsts();
    const Dgram* rdg    = result->datagram();
    size_t       extent = result->extent();

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

    delete rDest;
    delete result;
  }

  timespec endTime;
  clock_gettime(CLOCK_REALTIME, &endTime);

  BatchManager::dump();

  dumpStats("Outlet", startTime, endTime, _batchCount * _maxEntries);

  _outlet.shutdown();

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
  fprintf(stderr, " %-20s %s (default: %d\n",      "-B <max batches>",
          "Maximum number of extant batches",      max_batches);
  fprintf(stderr, " %-20s %s (default: %d\n",      "-E <max entries>",
          "Maximum number of entries per batch",   max_entries);
  fprintf(stderr, " %-20s %s\n",                    "-M <seconds>",
          "Monitoring printout period");

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
  unsigned monPeriod  = 1;

  while ((op = getopt(argc, argv, "h?vS:C:i:D:B:E:M:")) != -1)
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

  printf("Parameters:\n");
  printf("  Batch duration:             %014lx = %ld uS\n", duration, duration);
  printf("  Batch pool depth:           %d\n",  maxBatches);
  printf("  Max # of entries per batch: %d\n",  maxEntries);
  printf("  Max contribution size:      %zd, batch size: %zd\n",
         max_contrib_size, sizeof(Dgram) + maxEntries * max_contrib_size);
  printf("  Max result       size:      %zd, batch size: %zd\n",
         max_result_size,  sizeof(Dgram) + maxEntries * max_result_size);

  // The order of the connect() calls must match the Contributor(s) side's

  pin_thread(pthread_self(), thread_outlet);
  TstEbOutlet outlet(cltAddr, cltPort, id, duration, maxBatches, maxEntries, max_result_size);
  ebOutlet = &outlet;

  pin_thread(pthread_self(), thread_inlet);
  TstEbInlet  inlet (         srvPort, id, duration, maxBatches, maxEntries, max_contrib_size, contributors, outlet);
  ebInlet  = &inlet;
  if ( (ret = inlet.connect())  )  return ret; // Connect to Contributor's outlet

  pin_thread(pthread_self(), thread_outlet);
  if ( (ret = outlet.connect()) )  return ret; // Connect to Contributors' inlet

  pin_thread(pthread_self(), thread_monitor);
  StatsMonitor statsMon(&outlet, &inlet, monPeriod);
  statsMonitor = &statsMon;

  pin_thread(pthread_self(), thread_inlet);

  inlet.process();

  statsMon.shutdown();
  outlet.shutdown();

  return ret;
}
