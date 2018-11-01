#include "psdaq/eb/EbAppBase.hh"

#include "psdaq/eb/BatchManager.hh"
#include "psdaq/eb/EbEvent.hh"

#include "psdaq/eb/EbLfClient.hh"
#include "psdaq/eb/EbLfServer.hh"

#include "psdaq/eb/utilities.hh"
#include "psdaq/eb/StatsMonitor.hh"

#include "psdaq/service/Histogram.hh"
#include "psdaq/service/Collection.hh"
#include "xtcdata/xtc/Dgram.hh"

#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>                     // For getopt()
#include <climits>
#include <bitset>
#include <chrono>
#include <atomic>
#include <vector>
#include <unordered_map>
#include <cassert>
#include <iostream>


using namespace XtcData;
using namespace Pds;
using namespace Pds::Fabrics;

static const int      core_base        = 8; // 6: devXX, 8: accXX
static const int      core_offset      = 1; // Allows Ctrb and EB to run on the same machine
static const unsigned rtMon_period     = 1; // Seconds
static const unsigned default_id       = 0; // Builder's ID (< 64)
static const size_t   header_size      = sizeof(Dgram);
static const size_t   input_extent     = 2; // Revisit: Number of "L3" input  data words
static const size_t   result_extent    = 2; // Revisit: Number of "L3" result data words
static const size_t   max_contrib_size = header_size + input_extent  * sizeof(uint32_t);
static const size_t   max_result_size  = header_size + result_extent * sizeof(uint32_t);
static const char*    dflt_partition   = "Test";
static const char*    dflt_rtMon_host  = "psdev7b";
static const char*    dflt_coll_host   = "drp-tst-acc06";

static       unsigned lverbose         = 0;
static       int      lcore1           = core_base + core_offset + 0;
static       int      lcore2           = core_base + core_offset + 0; // 12: devXX, 0: accXX

using TimePoint_t = std::chrono::steady_clock::time_point;
using Duration_t  = std::chrono::steady_clock::duration;
using us_t        = std::chrono::microseconds;
using ns_t        = std::chrono::nanoseconds;


namespace Pds {
  namespace Eb {

    using EbLfLinkMap = std::unordered_map<unsigned, Pds::Eb::EbLfLink*>;

    class TheSrc : public Src
    {
    public:
      TheSrc(Level::Type level, unsigned id) :
        Src(level)
      {
        _log |= id;
      }
    };

    class L3EbApp : public EbAppBase,
                    public BatchManager
    {
    public:
      L3EbApp(const char*                     ifAddr,
              const std::string&              port,
              const std::vector<std::string>& addrs,
              const std::vector<std::string>& ports,
              const std::string&              mrqPort,
              unsigned                        id,
              uint64_t                        duration,
              unsigned                        maxBatches,
              unsigned                        maxEntries,
              size_t                          maxInputSize,
              size_t                          maxResultSize,
              uint64_t                        contributors,
              unsigned                        numMrqs,
              StatsMonitor&                   statsMon);
      virtual ~L3EbApp();
    public:
      void         shutdown();
    public:
      uint64_t     eventCount()   const { return _eventCount; }
      uint64_t     batchCount()   const { return _batchCount; }
    public:
      void         process();
    public:                             // For BatchManager
      virtual
      void         post(const Batch* batch);
      virtual
      void         post(const Dgram*);
    public:                             // For EventBuilder
      virtual
      void         process(EbEvent* event);
    public:                             // Ultimately loaded from a shareable
      void         handle(const Dgram* ctrb, uint32_t* result, unsigned length);
    private:
      void        _updateHists(TimePoint_t      t0,
                               TimePoint_t      t1,
                               const TimeStamp& stamp);
    private:
      EbLfClient*            _l3Transport;
      EbLfLinkMap            _l3Links;
      EbLfServer*            _mrqTransport;
      EbLfLinkMap            _mrqLinks;
      const unsigned         _id;
      const Xtc              _xtc;
    private:
      std::vector<uint32_t>  _dstList;
    private:
      uint64_t               _eventCount;
      uint64_t               _batchCount;
      uint64_t               _freeBatchCnt;
      uint64_t               _freeEpochCnt;
      uint64_t               _freeEventCnt;
    private:
      Histogram              _depTimeHist;
      Histogram              _postTimeHist;
      Histogram              _postCallHist;
      TimePoint_t            _postPrevTime;
    private:
      std::atomic<bool>      _running;
    };
  };
};


using namespace Pds::Eb;

L3EbApp::L3EbApp(const char*                     ifAddr,
                 const std::string&              port,
                 const std::vector<std::string>& addrs,
                 const std::vector<std::string>& ports,
                 const std::string&              mrqPort,
                 unsigned                        id,
                 uint64_t                        duration,
                 unsigned                        maxBatches,
                 unsigned                        maxEntries,
                 size_t                          maxInputSize,
                 size_t                          maxResultSize,
                 uint64_t                        contributors,
                 unsigned                        numMrqs,
                 StatsMonitor&                   smon) :
  EbAppBase    (ifAddr, port, id, duration, maxBatches, maxEntries, maxInputSize,
                maxInputSize, sizeof(Dgram), contributors),
  BatchManager (duration, maxBatches, maxEntries, maxResultSize),
  _l3Transport (new EbLfClient()),
  _l3Links     (),
  _mrqTransport(new EbLfServer(ifAddr, mrqPort.c_str())),
  _mrqLinks    (),
  _id          (id),
  _xtc         (TypeId(TypeId::Data, 0), TheSrc(Level::Event, id)), //_l3SummaryType
  _dstList     (std::bitset<64>(contributors).count()),
  _eventCount  (0),
  _batchCount  (0),
  _freeBatchCnt(freeBatchCount()),
  _freeEpochCnt(freeEpochCount()),
  _freeEventCnt(freeEventCount()),
  _depTimeHist (12, double(1 << 16)/1000.),
  _postTimeHist(12, 1.0),
  _postCallHist(12, 1.0),
  _postPrevTime(std::chrono::steady_clock::now()),
  _running     (true)
{
  for (unsigned i = 0; i < addrs.size(); ++i)
  {
    const char*    addr = addrs[i].c_str();
    const char*    port = ports[i].c_str();
    EbLfLink*      link;
    const unsigned tmo(120000);         // Milliseconds
    if (_l3Transport->connect(addr, port, tmo, &link))
    {
      fprintf(stderr, "%s: Error connecting to Trigger EbLfServer at %s:%s\n",
              __func__, addr, port);
      abort();
    }
    if (link->preparePoster(batchRegion(), batchRegionSize(), i, id, lverbose))
    {
      fprintf(stderr, "%s: Failed to prepare Trigger link to %s:%s\n",
              __func__, addr, port);
      abort();
    }
    _l3Links[link->id()] = link;

    printf("Trigger EbLfServer ID %d connected\n", link->id());
  }

  for (unsigned i = 0; i < numMrqs; ++i)
  {
    EbLfLink*      link;
    const unsigned tmo(120000);         // Milliseconds
    if (_mrqTransport->connect(&link, tmo))
    {
      fprintf(stderr, "%s: Error connecting to Mon EbLfClient[%d]\n", __func__, i);
      abort();
    }

    if (link->preparePender(i, id, lverbose))
    {
      fprintf(stderr, "%s: Failed to prepare Mon link[%d]\n", __func__, i);
      abort();
    }
    _mrqLinks[link->id()] = link;

    printf("Mon EbLfClient ID %d connected\n", link->id());
  }

  smon.registerIt("TEB.EvtRt",  _eventCount,   StatsMonitor::RATE);
  smon.registerIt("TEB.EvtCt",  _eventCount,   StatsMonitor::SCALAR);
  smon.registerIt("TEB.BatCt",  _batchCount,   StatsMonitor::SCALAR);
  smon.registerIt("TEB.FrBtCt", _freeBatchCnt, StatsMonitor::SCALAR);
  smon.registerIt("TEB.FrEpCt", _freeEpochCnt, StatsMonitor::SCALAR);
  smon.registerIt("TEB.FrEvCt", _freeEventCnt, StatsMonitor::SCALAR);
}

L3EbApp::~L3EbApp()
{
  if (_mrqTransport)  delete _mrqTransport;
  if (_l3Transport)   delete _l3Transport;
}

void L3EbApp::shutdown()
{
  _running = false;
}

void L3EbApp::process()
{
  //pinThread(task()->parameters().taskID(), lcore2);
  //pinThread(pthread_self(),                lcore2);

  //start();                              // Start the event timeout timer

  pinThread(pthread_self(),                lcore1);

  while (_running)
  {
    EbAppBase::process();

    _freeBatchCnt = freeBatchCount();
    _freeEpochCnt = freeEpochCount();
    _freeEventCnt = freeEventCount();
  }

  //cancel();                             // Stop the event timeout timer

  if (_mrqTransport)
  {
    for (EbLfLinkMap::iterator it  = _mrqLinks.begin();
                               it != _mrqLinks.end(); ++it)
    {
      _mrqTransport->shutdown(it->second);
    }
    _mrqLinks.clear();
  }
  if (_l3Transport)
  {
    for (EbLfLinkMap::iterator it  = _l3Links.begin();
                               it != _l3Links.end(); ++it)
    {
      _l3Transport->shutdown(it->second);
    }
    _l3Links.clear();
  }

  EbAppBase::shutdown();

  BatchManager::dump();

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

void L3EbApp::process(EbEvent* event)
{
  // Accumulate output datagrams (result) from the event builder into a batch
  // datagram.  When the batch completes, post it to the contributors.

  if (lverbose > 3)
  {
    static unsigned cnt = 0;
    printf("L3EbApp::process event dump:\n");
    event->dump(++cnt);
  }
  ++_eventCount;

  const Dgram* dg = event->creator();
  if (dg->seq.isEvent() || !inTrSpace(dg))
  {
    Dgram     cdg(*dg);
    cdg.xtc = _xtc;
    Batch*    batch  = allocate(&cdg);    // This may post a batch
    assert(batch);
    size_t    pSize  = result_extent * sizeof(uint32_t);
    Dgram*    rdg    = new(batch->allocate(sizeof(*rdg) + pSize)) Dgram(cdg);
    uint32_t* result = reinterpret_cast<uint32_t*>(rdg->xtc.alloc(pSize));

    // Iterate over the event and build a result datagram
    const EbContribution** const  last    = event->end();
    const EbContribution*  const* ctrb    = event->begin();
    uint32_t*                     dstList = _dstList.data();
    do
    {
      // Due to the possibility of multiple readout groups, a contribution being
      // the first one in its batch doesn't necessarily imply that the other
      // contributions of the event are also first in their corresponding batches
      if ((*ctrb)->seq.isFirst())        // For the 1st ctrb in the batch...
      {
        // Revisit: Commented out alternate method, EbAppBase::process(), EbCtrbBase::post()
        const Dgram* bdg = (*ctrb) - 1;  // ... find the Input batch datagram
        *dstList++ = bdg->xtc.src.phy(); // ... and save the return address info
        //*dstList++ = (*ctrb)->xtc.src.phy(); // ... save the return address info
      }

      handle(*ctrb, result, result_extent);
    }
    while (++ctrb != last);

    if (rdg->seq.isEvent())
    {
      if (result[1])
      {
        uint64_t data;
        int      rc = _mrqTransport->poll(&data);
        result[1] = rc ? 0 : data;
        if (!rc)  _mrqLinks[ImmData::src(data)]->postCompRecv();
      }
    }
    else                                // Non-event
    {
      post(batch);
      flush();
    }

    if ((lverbose > 2)) // || result[1])
    {
      printf("Processed                  result  [%4d] @ "
             "%16p, pid %014lx, sz %4zd: [0] = %08x, [1] = %08x\n",
             batch->index(), rdg, rdg->seq.pulseId().value(),
             sizeof(*rdg) + rdg->xtc.sizeofPayload(), result[0], result[1]);
    }
  }
  else
  {
    // Iterate over the event and build a result datagram
    const EbContribution** const  last = event->end();
    const EbContribution*  const* ctrb = event->begin();
    do
    {
      handle(*ctrb, nullptr, 0);
    }
    while (++ctrb != last);

    if (lverbose > 2)
    {
      printf("Processed               non-event         @ "
             "%16p, pid %014lx, sz %4zd, src %08x, %08x\n",
             dg, dg->seq.pulseId().value(), sizeof(*dg) + dg->xtc.sizeofPayload(),
             dg->xtc.src.log(), dg->xtc.src.phy());
    }
  }
}

// Ultimately, this method is provided by the users and is loaded from a
// sharable library loaded during a configure transition.
void L3EbApp::handle(const Dgram* ctrb, uint32_t* result, unsigned length)
{
  if (result)
  {
    const uint32_t* input = (uint32_t*)ctrb->xtc.payload();
    result[0] |= input[0] == 0xdeadbeef ? 1 : 0;
    result[1] |= input[1] == 0x12345678 ? 1 : 0;
  }
}

void L3EbApp::post(const Batch* batch)
{
  const Dgram* bdg    = batch->datagram();
  size_t       extent = batch->extent();

  auto t0(std::chrono::steady_clock::now());
  for (std::vector<unsigned>::iterator it  = _dstList.begin();
                                       it != _dstList.end(); ++it)
  {
    unsigned  dst    = ImmData::src(*it);
    unsigned  idx    = ImmData::idx(*it);
    unsigned  offset = idx * maxBatchSize();
    EbLfLink* link   = _l3Links[dst];
    uint64_t  data   = ImmData::buffer(_id, idx);

    if (lverbose)
    {
      uint64_t pid    = bdg->seq.pulseId().value();
      void*    rmtAdx = (void*)link->rmtAdx(idx * maxBatchSize());
      printf("L3EbApp posts       %6ld result  [%4d] @ "
             "%16p, pid %014lx, sz %4zd to   Ctrb %2d @ %16p\n",
             _batchCount, idx, bdg, pid, extent, link->id(), rmtAdx);
    }

    //Dgram const* result = (Dgram const*)bdg->xtc.payload();
    //const Batch* inputs = batch;
    //printf("        data %08lx           idx %4d rPid %014lx iPid %014lx          dst %2d id %2d svc %d\n",
    //       data, idx, result->seq.pulseId().value(), inputs->datagram()->seq.pulseId().value(),
    //       dst, link->id(), result->seq.service());

    if (link->post(bdg, extent, offset, data) < 0)  break;
  }
  auto t1(std::chrono::steady_clock::now());

  ++_batchCount;

  _updateHists(t0, t1, bdg->seq.stamp());

  // Revisit: The following deallocation constitutes a race with the posts to
  // the transport above as the batch's memory cannot be allowed to be reused
  // for a subsequent batch before the transmit completes.  Waiting for
  // completion here would impact performance.  Since there are many batches,
  // and only one is active at a time, a previous batch will have completed
  // transmitting before the next one starts (or the subsequent transmit won't
  // start), thus making it safe to "pre-delete" it here.
  deallocate(batch);
}

void L3EbApp::post(const Dgram* nonEvent)
{
  assert(false);                        // Unused virtual method
}

void L3EbApp::_updateHists(TimePoint_t      t0,
                           TimePoint_t      t1,
                           const TimeStamp& stamp)
{
  auto        d  = std::chrono::seconds     { stamp.seconds()     } +
                   std::chrono::nanoseconds { stamp.nanoseconds() };
  TimePoint_t tp { std::chrono::duration_cast<Duration_t>(d) };
  int64_t     dT ( std::chrono::duration_cast<ns_t>(t0 - tp).count() );

  _depTimeHist.bump(dT >> 16);

  dT = std::chrono::duration_cast<us_t>(t1 - t0).count();
  //if (dT > 4095)  printf("postTime = %ld us\n", dT);
  _postTimeHist.bump(dT);
  _postCallHist.bump(std::chrono::duration_cast<us_t>(t0 - _postPrevTime).count());
  _postPrevTime = t0;
}


static StatsMonitor* lstatsMon(nullptr);
static L3EbApp*      lApp     (nullptr);

void sigHandler( int signal )
{
  static std::atomic<unsigned> callCount(0);

  if (callCount == 0)
  {
    printf("\nShutting down StatsMonitor...\n");
    if (lstatsMon)  lstatsMon->shutdown();

    if (lApp)  lApp->L3EbApp::shutdown();
  }

  if (callCount++)
  {
    fprintf(stderr, "Aborting on 2nd ^C...\n");
    ::abort();
  }
}

static void usage(char *name, char *desc)
{
  fprintf(stderr, "Usage:\n");
  fprintf(stderr, "  %s [OPTIONS] <contributor_spec> [<contributor_spec> [...]]\n", name);

  if (desc)
    fprintf(stderr, "\n%s\n", desc);

  fprintf(stderr, "\n<contributor_spec> has the form '<id>:<addr>:<port>'\n");
  fprintf(stderr, "<id> must be in the range 0 - %d.\n", MAX_TEBS - 1);
  fprintf(stderr, "Low numbered <port> values are treated as offsets into the corresponding ranges:\n");
  fprintf(stderr, "  Trigger EB:       %d - %d\n", TEB_PORT_BASE, TEB_PORT_BASE + MAX_TEBS - 1);
  fprintf(stderr, "  Trigger result:   %d - %d\n", DRP_PORT_BASE, DRP_PORT_BASE + MAX_DRPS - 1);
  fprintf(stderr, "  Monitor requests: %d - %d\n", MRQ_PORT_BASE, MRQ_PORT_BASE + MAX_MEBS - 1);

  fprintf(stderr, "\nOptions:\n");

  fprintf(stderr, " %-20s %s (default: %s)\n",        "-A <interface_addr>",
          "IP address of the interface to use",       "libfabric's 'best' choice");
  fprintf(stderr, " %-20s %s (default: %d)\n",        "-T <port>",
          "Base port for receiving contributions",    TEB_PORT_BASE);
  fprintf(stderr, " %-20s %s (default: %d)\n",        "-M <port>",
          "Base port for receiving monitor requests", MRQ_PORT_BASE);

  fprintf(stderr, " %-20s %s (default: %d)\n",        "-i <ID>",
          "Unique ID of this builder (0 - 63)",       default_id);
  fprintf(stderr, " %-20s %s (default: %014lx)\n",    "-d <batch duration>",
          "Batch duration (must be power of 2)",      BATCH_DURATION);
  fprintf(stderr, " %-20s %s (default: %d)\n",        "-b <max batches>",
          "Maximum number of extant batches",         MAX_BATCHES);
  fprintf(stderr, " %-20s %s (default: %d)\n",        "-e <max entries>",
          "Maximum number of entries per batch",      MAX_ENTRIES);
  fprintf(stderr, " %-20s %s (default: %d)\n",        "-r <count>",
          "Number of Mon Requester EB nodes",         0);
  fprintf(stderr, " %-20s %s (default: %d)\n",        "-m <seconds>",
          "Run-time monitoring printout period",      rtMon_period);
  fprintf(stderr, " %-20s %s (default: %s)\n",        "-Z <address>",
          "Run-time monitoring ZMQ server host",      dflt_rtMon_host);
  fprintf(stderr, " %-20s %s (default: %s)\n",        "-P <partition name>",
          "Partition tag",                            dflt_partition);
  fprintf(stderr, " %-20s %s (default: %d)\n",        "-p <partition number>",
          "Partition number",                         0);
  fprintf(stderr, " %-20s %s (default: %s)\n",        "-C <address>",
          "Collection server",                        dflt_coll_host);
  fprintf(stderr, " %-20s %s (default: %d)\n",        "-1 <core>",
          "Core number for pinning App thread to",    lcore1);
  fprintf(stderr, " %-20s %s (default: %d)\n",        "-2 <core>",
          "Core number for pinning other threads to", lcore2);

  fprintf(stderr, " %-20s %s\n", "-v", "enable debugging output (repeat for increased detail)");
  fprintf(stderr, " %-20s %s\n", "-h", "display this help output");
}

static
void joinCollection(std::string&              server,
                    unsigned                  partition,
                    unsigned                  portBase,
                    uint64_t&                 contributors,
                    std::vector<std::string>& addrs,
                    std::vector<std::string>& ports,
                    unsigned&                 tebId,
                    unsigned&                 numMrqs)
{
  Collection collection(server, partition, "teb");
  collection.connect();
  std::cout << "cmstate:\n" << collection.cmstate.dump(4) << std::endl;

  std::string id = std::to_string(collection.id());
  tebId = collection.cmstate["teb"][id]["teb_id"];
  //std::cout << "TEB: ID " << tebId << std::endl;

  for (auto it : collection.cmstate["drp"].items())
  {
    unsigned    ctrbId  = it.value()["drp_id"];
    std::string address = it.value()["connect_info"]["infiniband"];
    //std::cout << "DRP: ID " << ctrbId << "  " << address << std::endl;
    contributors |= 1ul << ctrbId;
    addrs.push_back(address);
    ports.push_back(std::string(std::to_string(portBase + ctrbId)));
    //printf("DRP Clt[%d] port = %d\n", ctrbId, portBase + ctrbId);
  }

  numMrqs = 0;
  if (collection.cmstate.find("meb") != collection.cmstate.end())
  {
    for (auto it : collection.cmstate["meb"].items())
    {
      ++numMrqs;
    }
  }
}

int main(int argc, char **argv)
{
  const unsigned NO_PARTITION = unsigned(-1UL);
  int            op, ret      = 0;
  unsigned       partition    = NO_PARTITION;
  std::string    partitionTag  (dflt_partition);
  unsigned       id           = default_id;
  char*          ifAddr       = nullptr;
  unsigned       tebPortNo    = 0;      // Port served to contributors
  unsigned       mrqPortNo    = 0;      // Port served to monitors
  uint64_t       duration     = BATCH_DURATION;
  unsigned       maxBatches   = MAX_BATCHES;
  unsigned       maxEntries   = MAX_ENTRIES;
  unsigned       rtMonPeriod  = rtMon_period;
  unsigned       rtMonVerbose = 0;
  const char*    rtMonHost    = dflt_rtMon_host;
  unsigned       numMrqs      = 0;
  std::string    collSrv      = dflt_coll_host;

  while ((op = getopt(argc, argv, "h?vVA:T:M:i:d:b:e:r:m:Z:P:p:C:1:2:")) != -1)
  {
    switch (op)
    {
      case 'A':  ifAddr       = optarg;             break;
      case 'T':  tebPortNo    = atoi(optarg);       break;
      case 'M':  mrqPortNo    = atoi(optarg);       break;
      case 'i':  id           = atoi(optarg);       break;
      case 'd':  duration     = atoll(optarg);      break;
      case 'b':  maxBatches   = atoi(optarg);       break;
      case 'e':  maxEntries   = atoi(optarg);       break;
      case 'r':  numMrqs      = atoi(optarg);       break;
      case 'm':  rtMonPeriod  = atoi(optarg);       break;
      case 'Z':  rtMonHost    = optarg;             break;
      case 'P':  partitionTag = optarg;             break;
      case 'p':  partition    = std::stoi(optarg);  break;
      case 'C':  collSrv      = optarg;             break;
      case '1':  lcore1       = atoi(optarg);       break;
      case '2':  lcore2       = atoi(optarg);       break;
      case 'v':  ++lverbose;                        break;
      case 'V':  ++rtMonVerbose;                    break;
      case '?':
      case 'h':
      default:
        usage(argv[0], (char*)"Trigger Event Builder application");
        return 1;
    }
  }

  if (partition == NO_PARTITION)
  {
    fprintf(stderr, "Partition number must be specified\n");
    return 1;
  }

  const unsigned numPorts    = MAX_DRPS + MAX_TEBS + MAX_MEBS + MAX_MEBS;
  const unsigned tebPortBase = TEB_PORT_BASE + numPorts * partition;
  const unsigned drpPortBase = DRP_PORT_BASE + numPorts * partition;
  const unsigned mrqPortBase = MRQ_PORT_BASE + numPorts * partition;

  std::vector<std::string> drpAddrs;
  std::vector<std::string> drpPorts;
  uint64_t contributors = 0;
  if (optind < argc)
  {
    do
    {
      char* contributor = argv[optind];
      char* colon1      = strchr(contributor, ':');
      char* colon2      = strrchr(contributor, ':');
      if (!colon1 || (colon1 == colon2))
      {
        fprintf(stderr, "DRP input '%s' is not of the form <ID>:<IP>:<port>\n", contributor);
        return 1;
      }
      unsigned cid  = atoi(contributor);
      unsigned port = atoi(&colon2[1]);
      if (cid >= MAX_DRPS)
      {
        fprintf(stderr, "DRP ID %d is out of range 0 - %d\n", cid, MAX_DRPS - 1);
        return 1;
      }
      if (port < MAX_DRPS)  port += drpPortBase;
      if ((port < drpPortBase) || (port >= drpPortBase + MAX_DRPS))
      {
        fprintf(stderr, "DRP client port %d is out of range %d - %d\n",
                drpPortBase, drpPortBase + MAX_DRPS, port);
        return 1;
      }
      contributors |= 1ul << cid;
      drpAddrs.push_back(std::string(&colon1[1]).substr(0, colon2 - &colon1[1]));
      drpPorts.push_back(std::string(std::to_string(port)));
    }
    while (++optind < argc);
  }
  else
  {
    joinCollection(collSrv, partition, drpPortBase, contributors, drpAddrs, drpPorts, id, numMrqs);
  }
  if (id >= MAX_TEBS)
  {
    fprintf(stderr, "TEB ID %d is out of range 0 - %d\n", id, MAX_TEBS - 1);
    return 1;
  }
  if ((drpAddrs.size() == 0) || (drpPorts.size() == 0))
  {
    fprintf(stderr, "Missing required DRP address(es)\n");
    return 1;
  }

  if  (tebPortNo < MAX_TEBS)  tebPortNo += tebPortBase;
  if ((tebPortNo < tebPortBase) || (tebPortNo >= tebPortBase + MAX_TEBS))
  {
    fprintf(stderr, "TEB Server port %d is out of range %d - %d\n",
            tebPortNo, tebPortBase, tebPortBase + MAX_TEBS);
    return 1;
  }
  std::string tebPort(std::to_string(tebPortNo + id));
  //printf("TEB Srv port = %s\n", tebPort.c_str());

  if  (mrqPortNo < MAX_MEBS)  mrqPortNo += mrqPortBase;
  if ((mrqPortNo < mrqPortBase) || (mrqPortNo >= mrqPortBase + MAX_TEBS))
  {
    fprintf(stderr, "MRQ Server port %d is out of range %d - %d\n",
            mrqPortNo, mrqPortBase, mrqPortBase + MAX_TEBS);
    return 1;
  }
  std::string mrqPort(std::to_string(mrqPortNo + id));
  //printf("MRQ Srv port = %s\n", mrqPort.c_str());

  if (maxEntries > duration)
  {
    fprintf(stderr, "Warning: More batch entries (%u) requested than definable "
            "in the batch duration (%lu); reducing to avoid wasting memory\n",
            maxEntries, duration);
    maxEntries = duration;
  }

  ::signal( SIGINT, sigHandler );

  EbAppBase::lverbose = lverbose;

  printf("\nParameters of Trigger Event Builder ID %d:\n",  id);
  printf("  Thread core numbers:        %d, %d\n",          lcore1, lcore2);
  printf("  Partition:                  %d: '%s'\n",        partition, partitionTag.c_str());
  printf("  Run-time monitoring period: %d\n",              rtMonPeriod);
  printf("  Run-time monitoring host:   %s\n",              rtMonHost);
  printf("  Number of Monitor EBs:      %d\n",              numMrqs);
  printf("  Batch duration:             %014lx = %ld uS\n", duration, duration);
  printf("  Batch pool depth:           %d\n",              maxBatches);
  printf("  Max # of entries per batch: %d\n",              maxEntries);
  printf("\n");

  pinThread(pthread_self(), lcore2);
  StatsMonitor* smon = new StatsMonitor(rtMonHost,
                                        partition,
                                        partitionTag,
                                        rtMonPeriod,
                                        rtMonVerbose);
  lstatsMon = smon;

  pinThread(pthread_self(), lcore1);
  L3EbApp* app = new L3EbApp(ifAddr,   tebPort,
                             drpAddrs, drpPorts,
                             mrqPort,
                             id,
                             duration,
                             maxBatches,
                             maxEntries,
                             max_contrib_size,
                             max_result_size,
                             contributors,
                             numMrqs,
                             *smon);
  lApp = app;

  const unsigned ibMtu = 4096;
  unsigned mtuCnt = (sizeof(Dgram) + maxEntries * max_contrib_size + ibMtu - 1) / ibMtu;
  unsigned mtuRem = (sizeof(Dgram) + maxEntries * max_contrib_size) % ibMtu;
  printf("\n");
  printf("  Max contribution size:      %zd, batch size: %zd, # MTUs: %d, last: %d / %d (%f%%)\n",
         max_contrib_size, app->maxInputSize(), mtuCnt, mtuRem, ibMtu, 100. * double(mtuRem) / double(ibMtu));
  mtuCnt = (sizeof(Dgram) + maxEntries * max_result_size + ibMtu - 1) / ibMtu;
  mtuRem = (sizeof(Dgram) + maxEntries * max_result_size) % ibMtu;
  printf("  Max result       size:      %zd, batch size: %zd, # MTUs: %d, last: %d / %d (%f%%)\n",
         max_result_size,  app->maxBatchSize(),  mtuCnt, mtuRem, ibMtu, 100. * double(mtuRem) / double(ibMtu));
  printf("\n");

  // Event builder sorts contributions into a time ordered list
  // Then calls the user's process() with complete events to build the result datagram
  // Post completed result batches to the outlet

  // Iterate over contributions in the batch
  // Event build them according to their trigger group

  // Wait a bit to allow other components of the system to establish connections
  sleep(1);

  app->process();

  printf("\nL3EbApp was shut down.\n");

  delete smon;
  delete app;

  return ret;
}
