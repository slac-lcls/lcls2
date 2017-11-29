#include "psdaq/eb/BatchManager.hh"
#include "psdaq/eb/EventBuilder.hh"
#include "psdaq/eb/EbEvent.hh"
#include "psdaq/eb/EbContribution.hh"
#include "psdaq/eb/Endpoint.hh"

#include "psdaq/eb/EbFtServer.hh"
#include "psdaq/eb/EbFtClient.hh"

#include "psdaq/service/Routine.hh"
#include "psdaq/service/Task.hh"
#include "psdaq/service/GenericPoolW.hh"
#include "xtcdata/xtc/Dgram.hh"

#include <unistd.h>
#include <stdio.h>
#include <time.h>
#include <signal.h>
#include <stdlib.h>
#include <inttypes.h>

using namespace XtcData;
using namespace Pds;
using namespace Pds::Fabrics;


static const unsigned default_id       = 0;          // Builder's ID
static const unsigned srv_port_base    = 32768;      // Base port Builder for receiving results
static const unsigned clt_port_base    = 32768 + 64; // Base port Builder sends to EB on
static const uint64_t batch_duration   = 1ul << 9;   //0x00000000000080UL; // ~128 uS; must be power of 2
static const unsigned max_batches      = 2048; //16; //8192;     // 1 second's worth
static const unsigned max_entries      = 512;  //128;     // 128 uS worth
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
//         dg->seq.stamp().pulseID(),
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

#define NDST  64                        // Revisit: NDST

    class ResultDest
    {
    public:
      ResultDest() :
        _msk(0),
        _dst(_dest)
      {
      }
    public:
      PoolDeclare;
    public:
      void destination(unsigned dst, unsigned idx)
      {
        uint64_t msk = 1ul << dst;

        if ((_msk & msk) == 0)
        {
          _msk       |= msk;
          _index[dst] = idx;
          *_dst++     = dst;
        }
      }
    public:
      size_t   nDsts()         const { return _dst - _dest;   }
      unsigned dst(unsigned i) const { return _dest[i];       }
      unsigned idx(unsigned i) const { return _index[dst(i)]; }
      void     destination(unsigned i, unsigned* dst_, unsigned* idx_) const
      {
        *dst_ = dst(i);
        *idx_ = idx(i);
      }
    private:
      uint64_t  _msk;
      unsigned* _dst;
    private:
      unsigned  _dest[NDST];
      unsigned  _index[NDST];
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
      ~TstEbOutlet() { }
    public:
      void shutdown();
    public:                             // Routine interface
      void routine();
      void post(Batch* batch);
    private:
      Batch* _pend();
    private:
      EbFtClient    _outlet;
      unsigned      _maxEntries;
      volatile bool _running;
      Task*         _task;
      Semaphore     _resultSem;
      Queue<Batch>  _pending;
    };

    class TstEbInlet : public EventBuilder
    {
    public:
      TstEbInlet(std::string& srvPort,
                 unsigned     id,
                 uint64_t     duration,
                 unsigned     maxBatches,
                 unsigned     maxEntries,
                 size_t       maxSize,
                 uint64_t     contributors);
    virtual   ~TstEbInlet() { }
    public:
      void     shutdown();
    public:
      void     process(TstEbOutlet& outlet);
    public:
      void     process(EbEvent* event);
      uint64_t contract(const Dgram* contrib) const;
      void     fixup(EbEvent* event, unsigned srcId);
    private:
      unsigned      _maxBatches;
      size_t        _maxBatchSize;
      EbFtServer    _inlet;
      Src           _src;
      const TypeId  _tag;
      uint64_t      _contract;
      GenericPool   _results;           // No RW as it shadows the batch pool, which has RW
      //EbDummyTC     _dummy;           // Template for TC of dummy contributions  // Revisit: ???
      TstEbOutlet*  _outlet;
      volatile bool _running;
    };
  };
};


using namespace Pds::Eb;

TstEbInlet::TstEbInlet(std::string& srvPort,
                       unsigned     id,
                       uint64_t     duration,
                       unsigned     maxBatches,
                       unsigned     maxEntries,
                       size_t       maxSize,
                       uint64_t     contributors) :
  EventBuilder(maxBatches, maxEntries, __builtin_popcountl(contributors), duration),
  _maxBatches(maxBatches),
  _maxBatchSize(sizeof(Dgram) + maxEntries * maxSize),
  _inlet(srvPort, __builtin_popcountl(contributors), maxBatches * _maxBatchSize, EbFtServer::PER_PEER_BUFFERS),
  _src(TheSrc(Level::Event, id)),
  _tag(TypeId::Data, 0), //_l3SummaryType),
  _contract(contributors),
  _results(sizeof(ResultDest), maxBatches * maxEntries),
  //_dummy(Level::Fragment),
  _outlet(NULL),
  _running(true)
{
  int ret = _inlet.connect(id);
  if (ret)
  {
    fprintf(stderr, "FtServer connect() failed\n");
    abort();
  }
}

void TstEbInlet::shutdown()
{
  cancel();                             // Stop the event timeout timer

  _running = false;
}

void TstEbInlet::process(TstEbOutlet& outlet)
{
  // Event builder sorts contributions into a time ordered list
  // Then calls the user's process with complete events to build the result datagram
  // Post complete events to the outlet

  // Iterate over contributions in the batch
  // Event build them according to their trigger group

  _outlet = &outlet;

  start();                              // Start the event timeout timer

  uint64_t contribCount = 0;
  timespec startTime;
  clock_gettime(CLOCK_REALTIME, &startTime);

  while (_running)
  {
    // Pend for an input datagram (batch) and pass its datagrams to the event builder.
    const Dgram* batch = (const Dgram*)_inlet.pend();
    if (!batch)  break;  //continue;              // Revisit: This causes a race when quitting

    unsigned idx = (((char*)batch - _inlet.base()) / _maxBatchSize) % _maxBatches;

    if (lverbose)
    {
      static unsigned cnt = 0;
      unsigned from = batch->xtc.src.log() & 0xff;
      printf("EbInlet  rcvd  %6d        batch[%2d]    @ %16p, ts %014lx, sz %3zd from Contrib %d\n", ++cnt, idx,
             batch, batch->seq.stamp().pulseId(), sizeof(*batch) + batch->xtc.sizeofPayload(), from);
    }

    const Dgram* entry = (const Dgram*)batch->xtc.payload();
    const Dgram* last  = (const Dgram*)batch->xtc.next();
    while(entry != last)
    {
      if (lverbose > 1)
      {
        unsigned from = entry->xtc.src.log() & 0xff;
        printf("EbInlet  found              a  contrib @ %16p, ts %014lx, sz %3zd from Contrib %d\n",
               entry, entry->seq.stamp().pulseId(), sizeof(*entry) + entry->xtc.sizeofPayload(), from);
      }

      EventBuilder::process(entry, idx);

      //if (entry->seq.stamp().control() == CNTRL_QUIT)  _running = false;

      ++contribCount;

      entry = (const Dgram*)entry->xtc.next();
    }
  }

  timespec endTime;
  clock_gettime(CLOCK_REALTIME, &endTime);

  //dumpStats("Inlet per contributor rate",
  //          startTime, endTime, contribCount / __builtin_popcount(_contract));
  dumpStats("Inlet contribution rate", startTime, endTime, contribCount);

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

  // Iterate over the event and build a result datagram
  EbContribution* last    = event->empty();
  EbContribution* contrib = event->creator();
  const Dgram*    cdg     = contrib->datagram();
  Batch*          batch   = _outlet->allocate(cdg); // This may post to the outlet
  ResultDest*     rDest   = (ResultDest*)batch->parameter();
  if (!rDest)
  {
    rDest = new(&_results) ResultDest();
    batch->parameter(rDest);
  }
  size_t          pSize   = result_extent * sizeof(uint32_t);
  Dgram*          rdg     = new(batch->allocate(sizeof(*rdg) + pSize)) Dgram(*cdg);
  uint32_t*       buffer  = (uint32_t*)rdg->xtc.alloc(pSize);
  memset(buffer, 0, pSize);             // Revisit: Some crud for now
  do
  {
    rDest->destination(contrib->number(), contrib->data());

    const Dgram* idg     = contrib->datagram();
    uint32_t*    payload = (uint32_t*)idg->xtc.payload();
    for (unsigned i = 0; i < result_extent; ++i)
    {
      buffer[i] |= payload[i];          // Revisit: Some crud for now
    }
  }
  while ((contrib = contrib->forward()) != last);

  if (lverbose > 2)
  {
    printf("EbInlet  processed          result[%2d] @ %16p, ts %014lx, sz %3zd\n",
           rDest->idx(0), rdg, rdg->seq.stamp().pulseId(), sizeof(*rdg) + rdg->xtc.sizeofPayload());
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
  BatchManager(TheSrc(Level::Event, id), duration, maxBatches, maxEntries, maxSize),
  _outlet(cltAddr, cltPort, maxBatches * (sizeof(Dgram) + maxEntries * maxSize)),
  _maxEntries(maxEntries),
  _running(true),
  _task (new Task(TaskObject("tOutlet"))),
  _resultSem(Semaphore::EMPTY),
  _pending()
{
  const unsigned tmo(120);
  int ret = _outlet.connect(id, tmo);
  if (ret)
  {
    fprintf(stderr, "FtClient connect() failed\n");
    abort();
  }

  _outlet.registerMemory(batchRegion(), batchRegionSize());

  _task->call(this);
}

void TstEbOutlet::shutdown()
{
  _running = false;

  BatchManager::dump();

 _task->destroy();
}

Batch* TstEbOutlet::_pend()
{
  _resultSem.take();
  Batch* batch = _pending.remove();

  if (lverbose > 2)
  {
    const Dgram* bdg  = batch->datagram();
    printf("EbOutlet read                batch[%2d] @ %16p, ts %014lx, sz %3zd\n",
           batch->index(), bdg, bdg->seq.stamp().pulseId(), sizeof(*bdg) + bdg->xtc.sizeofPayload());
  }

  return batch;
}

void TstEbOutlet::routine()
{
  uint64_t batchCount = 0;
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

    ++batchCount;

    ResultDest*    rDest  = (ResultDest*)result->parameter();
    const unsigned nDsts  = rDest->nDsts();
    const Dgram*   rdg    = result->datagram();
    size_t         extent = result->extent();

    for (unsigned iDst = 0; iDst < nDsts; ++iDst)
    {
      unsigned dst, idx;
      rDest->destination(iDst, &dst, &idx);

      if (lverbose)
      {
        static unsigned cnt = 0;
        void* rmtAdx = (void*)_outlet.rmtAdx(dst, idx * maxBatchSize());
        printf("EbOutlet posts       %6d result   [%2d] @ %16p, ts %014lx, sz %3zd to   Contrib %d = %16p\n",
               ++cnt, idx, rdg, rdg->seq.stamp().pulseId(), extent, dst, rmtAdx);
      }

      _outlet.post(rdg, extent, dst, idx * maxBatchSize());
    }

    delete rDest;
    delete result;
  }

  timespec endTime;
  clock_gettime(CLOCK_REALTIME, &endTime);

  dumpStats("Outlet", startTime, endTime, batchCount * _maxEntries);

  _outlet.shutdown();
}

void TstEbOutlet::post(Batch* result)
{
  // Accumulate output datagrams (result) from the event builder into a batch
  // datagram.  When the batch completes, post it to the back end.

  _pending.insert(result);
  _resultSem.give();
}


static TstEbInlet*  ebInlet   = NULL;
static TstEbOutlet* ebOutlet  = NULL;

void sigHandler( int signal )
{
  static unsigned callCount = 0;

  printf("sigHandler() called\n");

  if (ebInlet  && (callCount == 0))  ebInlet->shutdown();
  if (ebOutlet && (callCount == 0))  ebOutlet->shutdown();

  if (callCount++)  ::abort();
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

  while ((op = getopt(argc, argv, "h?vS:C:i:D:B:E:")) != -1)
  {
    switch (op)
    {
      case 'S':  srvBase    = strtoul(optarg, NULL, 0);  break;
      case 'C':  cltBase    = strtoul(optarg, NULL, 0);  break;
      case 'i':  id         = atoi(optarg);              break;
      case 'D':  duration   = atoll(optarg);             break;
      case 'B':  maxBatches = atoi(optarg);              break;
      case 'E':  maxEntries = atoi(optarg);              break;
      case 'v':  ++lverbose;                             break;
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

  TstEbInlet  inlet (         srvPort, id, duration, maxBatches, maxEntries, max_contrib_size, contributors);
  TstEbOutlet outlet(cltAddr, cltPort, id, duration, maxBatches, maxEntries, max_result_size);
  ebInlet  = &inlet;
  ebOutlet = &outlet;

  inlet.process(outlet);

  return ret;
}
