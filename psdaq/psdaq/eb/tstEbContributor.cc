#include "psdaq/eb/MebContributor.hh"
#include "psdaq/eb/TebContributor.hh"
#include "psdaq/eb/EbCtrbInBase.hh"

#include "psdaq/eb/utilities.hh"
#include "psdaq/eb/StatsMonitor.hh"
#include "psdaq/eb/EbLfClient.hh"
#include "psdaq/eb/utilities.hh"

#include "psdaq/service/GenericPoolW.hh"
#include "psdaq/service/Collection.hh"
#include "xtcdata/xtc/Dgram.hh"

#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <csignal>
#include <cstdint>
#include <string>
#include <vector>
#include <atomic>
#include <thread>
#include <iostream>

//#define SINGLE_EVENTS                   // Define to take one event at a time by hitting <CR>

using namespace XtcData;
using namespace Pds;
using namespace Pds::Fabrics;

static const int      core_base        = 6;          // devXX: 6, accXX: 8
static const int      core_offset      = 2;          // Allows Ctrb and EB to run on the same machine
static const unsigned rtMon_period     = 1;          // Seconds
static const unsigned default_id       = 0;          // Contributor's ID (< 64)
static const size_t   header_size      = sizeof(Dgram);
static const size_t   input_extent     = 2;    // Revisit: Number of "L3" input  data words
static const size_t   result_extent    = 2;    // Revisit: Number of "L3" result data words
static const size_t   max_contrib_size = header_size + input_extent  * sizeof(uint32_t);
static const size_t   max_result_size  = header_size + result_extent * sizeof(uint32_t);
static const unsigned mon_buf_cnt      = 8;    // Revisit: Corresponds to monReqServer::numberofEvBuffers
static const size_t   mon_buf_size     = 1024; // Revisit: Corresponds to monReqServer:sizeofBuffers option
static const size_t   mon_trSize       = 1024; // Revisit: Corresponds to monReqServer:??? option


namespace Pds {
  namespace Eb {

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
      Input(const Sequence& seq_, const Xtc& xtc_) :
        Dgram()
      {
        seq = seq_;
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
      DrpSim(unsigned maxBatches,
             unsigned maxEntries,
             size_t   maxEvtSize,
             unsigned id);
      ~DrpSim();
    public:
      void         shutdown();
    public:
      const Dgram* genInput();
    public:
      const uint64_t& allocPending() const { return _allocPending; }
    private:
      const unsigned _maxBatches;
      const unsigned _maxEntries;
      const size_t   _maxEvtSz;
      const unsigned _id;
      const Xtc      _xtc;
      uint64_t       _pid;
      GenericPoolW   _pool;
    private:
      TransitionId::Value _trId;
    private:
      uint64_t       _allocPending;
    };

    class EbCtrbIn : public EbCtrbInBase
    {
    public:
      EbCtrbIn(const TebCtrbParams& prms,
               MebContributor*      mon,
               StatsMonitor&        smon,
               const char*          outDir);
      virtual ~EbCtrbIn();
    public:                             // For EbCtrbInBase
      virtual void process(const Dgram* result, const void* input);
    private:
      MebContributor* _mon;
      FILE*           _xtcFile;
    private:
      uint64_t        _eventCount;
    };

    class EbCtrbApp : public TebContributor
    {
    public:
      EbCtrbApp(const TebCtrbParams& prms,
                StatsMonitor&        smon);
      virtual ~EbCtrbApp();
    public:
      void     shutdown();
      void     process(EbCtrbIn&);
    private:
      DrpSim               _drpSim;
      const TebCtrbParams& _prms;
    private:
      uint64_t             _eventCount;
      uint64_t             _inFlightCnt;
    private:
      std::atomic<bool>    _running;
      std::thread*         _outThread;
    };
  };
};


using namespace Pds::Eb;

DrpSim::DrpSim(unsigned maxBatches,
               unsigned maxEntries,
               size_t   maxEvtSize,
               unsigned id) :
  _maxBatches  (maxBatches),
  _maxEntries  (maxEntries),
  _maxEvtSz    (maxEvtSize),
  _id          (id),
  _xtc         (TypeId(TypeId::Data, 0), TheSrc(Level::Segment, id)),
  _pid         (0x01000000000003ul),  // Something non-zero and not on a batch boundary
  _pool        (sizeof(Entry) + maxEvtSize, maxBatches * maxEntries),
  _trId        (TransitionId::Unknown),
  _allocPending(0)
{
}

DrpSim::~DrpSim()
{
  printf("\nDrpSim Input data pool:\n");
  _pool.dump();
}

void DrpSim::shutdown()
{
  _pool.stop();
}

const Dgram* DrpSim::genInput()
{
  const uint32_t bounceRate = 16 * 1024 * 1024 - 1;

  if       (_trId == TransitionId::Disable)   _trId = TransitionId::Enable;
  else if  (_trId != TransitionId::L1Accept)  _trId = (TransitionId::Value)(_trId + 2);
  else if ((_pid % bounceRate) == (0x01000000000003ul % bounceRate))
                                              _trId =  TransitionId::Disable;

  // Fake datagram header
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  const Sequence seq(Sequence::Event, _trId, TimeStamp(ts), PulseId(_pid));

  ++_allocPending;
  void* buffer = _pool.alloc(sizeof(Input));
  --_allocPending;
  if (!buffer)  return (Dgram*)buffer;
  Input* idg = ::new(buffer) Input(seq, _xtc);

  size_t inputSize = input_extent * sizeof(uint32_t);

  // Revisit: Here is where L3 trigger input information would be inserted into the datagram,
  //          whatever that will look like
  uint32_t* payload = (uint32_t*)idg->xtc.alloc(inputSize);
  payload[0] = _pid %   3 == 0 ? 0xdeadbeef : 0xabadcafe;
  payload[1] = _pid % 119 == 0 ? 0x12345678 : 0;

#ifdef SINGLE_EVENTS
  payload[1] = 0x12345678;

  _pid += _maxEntries;
#else
  _pid += 1; //27; // Revisit: fi_tx_attr.iov_limit = 6 = 1 + max # events per batch
  //_pid = ts.tv_nsec / 1000;     // Revisit: Not guaranteed to be the same across systems
#endif

  return idg;
}


EbCtrbIn::EbCtrbIn(const TebCtrbParams& prms,
                   MebContributor*      mon,
                   StatsMonitor&        smon,
                   const char*          outDir) :
  EbCtrbInBase(prms),
  _mon        (mon),
  _xtcFile    (nullptr),
  _eventCount (0)
{
  if (outDir)
  {
    char filename[PATH_MAX];
    snprintf(filename, PATH_MAX, "%s/data-%02d.xtc", outDir, prms.id);
    FILE* xtcFile = fopen(filename, "w");
    if (!xtcFile)
    {
      fprintf(stderr, "%s: Error opening output xtc file '%s'.\n",
              __func__, filename);
      abort();
    }
    _xtcFile = xtcFile;
  }

  smon.registerIt("CtbI.EvtCt",  _eventCount,  StatsMonitor::SCALAR);
  smon.registerIt("CtbI.RxPdg",   rxPending(), StatsMonitor::SCALAR);
  if (_mon)
    smon.registerIt("MCtbO.EvtCt", _mon->eventCount(), StatsMonitor::SCALAR);
}

EbCtrbIn::~EbCtrbIn()
{
}

void EbCtrbIn::process(const Dgram* result, const void* appPrm)
{
  const Input* input = (const Input*)appPrm;
  uint64_t     pid   = result->seq.pulseId().value();

  if (pid != input->seq.pulseId().value())
  {
    fprintf(stderr, "Result, Input pulse ID mismatch - got: %014lx, expected: %014lx\n",
           pid, input->seq.pulseId().value());
    abort();
  }

  if (result->seq.isEvent())            // L1Accept
  {
    uint32_t* response = (uint32_t*)result->xtc.payload();

    if (response[0] && _xtcFile)        // Persist the data
    {
      if (fwrite(input, sizeof(*input) + input->xtc.sizeofPayload(), 1, _xtcFile) != 1)
        fprintf(stderr, "Error writing to output xtc file.\n");
    }
    if (response[1] && _mon)  _mon->post(input, response[1]);
  }
  else                                  // Other Transition
  {
    //printf("Non-event rcvd: pid = %014lx, service = %d\n", pid, result->seq.service());

    if (_mon)  _mon->post(input);
  }

  // Revisit: Race condition
  // There's a race when the input datagram is posted to the monitoring node(s).
  // It is nominally safe to delete the DG only when the post has completed.
  delete input;

  ++_eventCount;
}


EbCtrbApp::EbCtrbApp(const TebCtrbParams& prms,
                     StatsMonitor&        smon) :
  TebContributor(prms),
  _drpSim      (prms.maxBatches, prms.maxEntries, prms.maxInputSize, prms.id),
  _prms        (prms),
  _eventCount  (0),
  _inFlightCnt (0),
  _running     (true),
  _outThread   (nullptr)
{
  smon.registerIt("CtbO.EvtRt",  _eventCount,            StatsMonitor::RATE);
  smon.registerIt("CtbO.EvtCt",  _eventCount,            StatsMonitor::SCALAR);
  smon.registerIt("CtbO.BatCt",   batchCount(),          StatsMonitor::SCALAR);
  smon.registerIt("CtbO.BtAlCt",  batchAllocCnt(),       StatsMonitor::SCALAR);
  smon.registerIt("CtbO.BtFrCt",  batchFreeCnt(),        StatsMonitor::SCALAR);
  smon.registerIt("CtbO.BtWtg",   batchWaiting(),        StatsMonitor::SCALAR);
  smon.registerIt("CtbO.AlPdg",  _drpSim.allocPending(), StatsMonitor::SCALAR);
  smon.registerIt("CtbO.TxPdg",   txPending(),           StatsMonitor::SCALAR);
  smon.registerIt("CtbO.InFlt",  _inFlightCnt,           StatsMonitor::SCALAR);
}

EbCtrbApp::~EbCtrbApp()
{
}

void EbCtrbApp::shutdown()
{
  _running = false;

  _drpSim.shutdown();

  if (_outThread)  _outThread->join();
}

#ifdef SINGLE_EVENTS
#  include <iostream>                   // std::cin, std::cout
#endif

void EbCtrbApp::process(EbCtrbIn& in)
{
  TebContributor::startup(in);

  pinThread(pthread_self(), _prms.core[0]);

#ifdef SINGLE_EVENTS
  printf("Hit <return> for an event\n");
#endif

  while (_running)
  {
#ifdef SINGLE_EVENTS
    std::string blank;
    std::getline(std::cin, blank);
#endif
    //usleep(100000);

    const Dgram* input = _drpSim.genInput();
    if (!input)  break;

    if (_prms.verbose > 1)
    {
      const char* svc = TransitionId::name(input->seq.service());
      uint32_t*   inp = (uint32_t*)input->xtc.payload();
      printf("Batching  %15s  dg             @ "
             "%16p, pid %014lx, sz %4zd, src %2d, [0] = %08x, [1] = %08x\n",
             svc, input, input->seq.pulseId().value(),
             sizeof(*input) + input->xtc.sizeofPayload(),
             input->xtc.src.value(), inp[0], inp[1]);
    }

    if (TebContributor::process(input, (const void*)input)) // 2nd arg is returned with the result
      ++_eventCount;

    _inFlightCnt  = inFlightCnt();
  }

  TebContributor::shutdown();
}


static StatsMonitor* lstatsMon(nullptr);
static EbCtrbApp*    lApp     (nullptr);

void sigHandler(int signal)
{
  static std::atomic<unsigned> callCount(0);

  if (callCount == 0)
  {
    printf("\nShutting down StatsMonitor...\n");
    if (lstatsMon)  lstatsMon->shutdown();

    if (lApp)  lApp->EbCtrbApp::shutdown();
  }

  if (callCount++)
  {
    fprintf(stderr, "Aborting on 2nd ^C...\n");
    ::abort();
  }
}

static
void usage(char *name, char *desc, TebCtrbParams& prms)
{
  fprintf(stderr, "Usage:\n");
  fprintf(stderr, "  %s [OPTIONS]\n", name);

  if (desc)
    fprintf(stderr, "\n%s\n", desc);

  fprintf(stderr, "\n<spec>, below, has the form '<id>:<addr>:<port>', with\n");
  fprintf(stderr, "<id> typically in the range 0 - 63.\n");
  fprintf(stderr, "Low numbered <port> values are treated as offsets into the corresponding ranges:\n");
  fprintf(stderr, "  Trigger EB:     %d - %d\n", TEB_PORT_BASE, TEB_PORT_BASE + MAX_TEBS - 1);
  fprintf(stderr, "  Trigger result: %d - %d\n", DRP_PORT_BASE, DRP_PORT_BASE + MAX_DRPS - 1);
  fprintf(stderr, "  Monitor EB:     %d - %d\n", MEB_PORT_BASE, MEB_PORT_BASE + MAX_MEBS - 1);

  fprintf(stderr, "\nOptions:\n");

  fprintf(stderr, " %-20s %s (default: %s)\n",        "-A <interface_addr>",
          "IP address of the interface to use",       "libfabric's 'best' choice");
  fprintf(stderr, " %-20s %s (default: %s)\n",        "-T \"<spec>[, <spec>[, ...]]\"",
          "TEB to send contributions to",             "None, required or Collection is used");
  fprintf(stderr, " %-20s %s (default: %s)\n",        "-D <port>",
          "Base port for receiving results",          prms.port.c_str());
  fprintf(stderr, " %-20s %s (default: %s)\n",        "-M \"<spec>[, <spec>[, ...]]\"",
          "MEB to send contributions to",             "None, required or Collection is used");

  fprintf(stderr, " %-20s %s (default: %s)\n",        "-o <output dir>",
          "Output directory for data-<ID>.xtc files", "None");
  fprintf(stderr, " %-20s %s (default: %d)\n",        "-i <ID>",
          "Unique ID of this Contributor (0 - 63)",   prms.id);
  fprintf(stderr, " %-20s %s (default: %014lx)\n",    "-d <batch duration>",
          "Batch duration (must be power of 2)",      prms.duration);
  fprintf(stderr, " %-20s %s (default: %d)\n",        "-b <max batches>",
          "Maximum number of extant batches",         prms.maxBatches);
  fprintf(stderr, " %-20s %s (default: %d)\n",        "-e <max entries>",
          "Maximum number of entries per batch",      prms.maxEntries);
  fprintf(stderr, " %-20s %s (default: %d)\n",        "-n <num buffers>",
          "Number of Mon buffers",                    mon_buf_cnt);
  fprintf(stderr, " %-20s %s (default: %zd)\n",       "-s <size>",
          "Mon buffer size",                          mon_buf_size);
  fprintf(stderr, " %-20s %s (default: %d)\n",        "-m <seconds>",
          "Run-time monitoring printout period",      rtMon_period);
  fprintf(stderr, " %-20s %s (default: %s)\n",        "-Z <address>",
          "Run-time monitoring ZMQ server host",      RTMON_HOST);
  fprintf(stderr, " %-20s %s (default: %d)\n",        "-R <port>",
          "Run-time monitoring ZMQ server port",      RTMON_PORT_BASE);
  fprintf(stderr, " %-20s %s (default: %s)\n",        "-P <partition name>",
          "Partition tag",                            PARTITION);
  fprintf(stderr, " %-20s %s (default: %d)\n",        "-p <partition number>",
          "Partition number",                         0);
  fprintf(stderr, " %-20s %s (default: %s)\n",        "-C <address>",
          "Collection server",                        COLL_HOST);
  fprintf(stderr, " %-20s %s (default: %d)\n",        "-1 <core>",
          "Core number for pinning App thread to",    prms.core[0]);
  fprintf(stderr, " %-20s %s (default: %d)\n",        "-2 <core>",
          "Core number for pinning other threads to", prms.core[1]);

  fprintf(stderr, " %-20s %s\n", "-v", "enable debugging output (repeat for increased detail)");
  fprintf(stderr, " %-20s %s\n", "-h", "display this help output");
}

static
int parseSpec(const char*               who,
              char*                     spec,
              unsigned                  maxId,
              unsigned                  portBase,
              std::vector<std::string>& addrs,
              std::vector<std::string>& ports,
              uint64_t*                 bits)
{
  unsigned portMin = portBase;
  unsigned portMax = portBase + maxId;
  do
  {
    char* target = strsep(&spec, ", ");
    if (!*target)  continue;
    char* colon1 = strchr(target, ':');
    char* colon2 = strrchr(target, ':');
    if (!colon1 || (colon1 == colon2))
    {
      fprintf(stderr, "%s input '%s' is not of the form <ID>:<IP>:<port>\n",
              who, target);
      return 1;
    }
    unsigned id   = atoi(target);
    unsigned port = atoi(&colon2[1]);
    if (id > maxId)
    {
      fprintf(stderr, "%s ID %d is out of range 0 - %d\n", who, id, maxId);
      return 1;
    }
    if  (port < portMax - portMin)  port += portMin;
    if ((port < portMin) || (port >= portMax))
    {
      fprintf(stderr, "%s client port %d is out of range %d - %d\n",
              who, port, portMin, portMax);
      return 1;
    }
    *bits |= 1ul << id;
    addrs.push_back(std::string(&colon1[1]).substr(0, colon2 - &colon1[1]));
    ports.push_back(std::to_string(port));
  }
  while (spec);

  return 0;
}

static
void joinCollection(std::string&   server,
                    unsigned       partition,
                    unsigned       tebPortBase,
                    unsigned       mebPortBase,
                    TebCtrbParams& tebPrms,
                    MebCtrbParams& mebPrms)
{
  Collection collection(server, partition, "drp");
  collection.connect();
  std::cout << "cmstate:\n" << collection.cmstate.dump(4) << std::endl;

  std::string id = std::to_string(collection.id());
  tebPrms.id = collection.cmstate["drp"][id]["drp_id"];
  mebPrms.id = tebPrms.id;
  //std::cout << "DRP: ID " << tebPrms.id << std::endl;

  uint64_t builders = 0;
  for (auto it : collection.cmstate["teb"].items())
  {
    unsigned    tebId   = it.value()["teb_id"];
    std::string address = it.value()["connect_info"]["infiniband"];
    //std::cout << "TEB: ID " << tebId << "  " << address << std::endl;
    builders |= 1ul << tebId;
    tebPrms.addrs.push_back(address);
    tebPrms.ports.push_back(std::string(std::to_string(tebPortBase + tebId)));
    //printf("TEB Clt[%d] port = %d\n", tebId, tebPortBase + tebId);
  }
  tebPrms.builders = builders;

  if (collection.cmstate.find("meb") != collection.cmstate.end())
  {
    for (auto it : collection.cmstate["meb"].items())
    {
      unsigned    mebId   = it.value()["meb_id"];
      std::string address = it.value()["connect_info"]["infiniband"];
      //std::cout << "MEB: ID " << mebId << "  " << address << std::endl;
      mebPrms.addrs.push_back(address);
      mebPrms.ports.push_back(std::string(std::to_string(mebPortBase + mebId)));
      //printf("MEB Clt[%d] port = %d\n", mebId, mebPortBase + mebId);
    }
  }
}

int main(int argc, char **argv)
{
  // Gather the list of EBs from an argument
  // - Somehow ensure this list is the same for all instances of the client
  // - Maybe circulate this list around to see whether anyone disagrees with it?

  const unsigned NO_PARTITION = unsigned(-1UL);
  int            op, ret      = 0;
  unsigned       partition    = NO_PARTITION;
  std::string    partitionTag  (PARTITION);
  const char*    rtMonHost    = RTMON_HOST;
  unsigned       rtMonPort    = RTMON_PORT_BASE;
  unsigned       rtMonPeriod  = rtMon_period;
  unsigned       rtMonVerbose = 0;
  unsigned       drpPortNo    = 0;      // Port served to TEBs
  char*          tebSpec      = nullptr;
  char*          mebSpec      = nullptr;
  char*          outDir       = nullptr;
  std::string    collSrv        (COLL_HOST);
  TebCtrbParams  tebPrms { /* .addrs         = */ { },
                           /* .ports         = */ { },
                           /* .ifAddr        = */ nullptr,
                           /* .port          = */ std::to_string(drpPortNo),
                           /* .id            = */ default_id,
                           /* .builders      = */ 0,
                           /* .duration      = */ BATCH_DURATION,
                           /* .maxBatches    = */ MAX_BATCHES,
                           /* .maxEntries    = */ MAX_ENTRIES,
                           /* .maxInputSize  = */ max_contrib_size,
                           /* .maxResultSize = */ max_result_size,
                           /* .core          = */ { core_base + core_offset + 0,
                                                    core_base + core_offset + 12 },
                           /* .verbose       = */ 0 };
  MebCtrbParams  mebPrms { /* .addrs         = */ { },
                           /* .ports         = */ { },
                           /* .id            = */ default_id,
                           /* .maxEvents     = */ mon_buf_cnt,
                           /* .maxEvSize     = */ mon_buf_size,
                           /* .maxTrSize     = */ mon_trSize,
                           /* .verbose       = */ 0 };

  while ((op = getopt(argc, argv, "h?vVA:T:D:M:o:i:d:b:e:m:Z:R:P:p:n:s:C:1:2:")) != -1)
  {
    switch (op)
    {
      case 'A':  tebPrms.ifAddr     = optarg;             break;
      case 'T':  tebSpec            = optarg;             break;
      case 'D':  drpPortNo          = atoi(optarg);       break;
      case 'M':  mebSpec            = optarg;             break;
      case 'o':  outDir             = optarg;             break;
      case 'i':  tebPrms.id         = atoi(optarg);
                 mebPrms.id         = tebPrms.id;         break;
      case 'd':  tebPrms.duration   = atoll(optarg);      break;
      case 'b':  tebPrms.maxBatches = atoi(optarg);       break;
      case 'e':  tebPrms.maxEntries = atoi(optarg);       break;
      case 'm':  rtMonPeriod        = atoi(optarg);       break;
      case 'Z':  rtMonHost          = optarg;             break;
      case 'R':  rtMonPort          = atoi(optarg);       break;
      case 'P':  partitionTag       = optarg;             break;
      case 'p':  partition          = std::stoi(optarg);  break;
      case 'n':  mebPrms.maxEvents  = atoi(optarg);       break;
      case 's':  mebPrms.maxEvSize  = atoi(optarg);       break;
      case 'C':  collSrv            = optarg;             break;
      case '1':  tebPrms.core[0]    = atoi(optarg);       break;
      case '2':  tebPrms.core[1]    = atoi(optarg);       break;
      case 'v':  ++tebPrms.verbose;
                 ++mebPrms.verbose;                       break;
      case 'V':  ++rtMonVerbose;                          break;
      case '?':
      case 'h':
        usage(argv[0], (char*)"Test DRP event contributor", tebPrms);
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
  const unsigned mebPortBase = MEB_PORT_BASE + numPorts * partition;

  if (tebSpec)
  {
    int rc = parseSpec("TEB",
                       tebSpec,
                       MAX_TEBS - 1,
                       tebPortBase,
                       tebPrms.addrs,
                       tebPrms.ports,
                       &tebPrms.builders);
    if (rc)  return rc;
  }

  std::vector<std::string> mebAddrs;
  std::vector<std::string> mebPorts;
  uint64_t                 unused;
  if (mebSpec)
  {
    int rc = parseSpec("MEB",
                       mebSpec,
                       MAX_MEBS - 1,
                       mebPortBase,
                       mebPrms.addrs,
                       mebPrms.ports,
                       &unused);
    if (rc)  return rc;
  }

  if (!tebSpec && !mebSpec)
  {
    joinCollection(collSrv, partition, tebPortBase, mebPortBase, tebPrms, mebPrms);
  }
  if (tebPrms.id >= MAX_DRPS)
  {
    fprintf(stderr, "DRP ID %d is out of range 0 - %d\n",
            tebPrms.id, MAX_DRPS - 1);
    return 1;
  }
  if ((tebPrms.addrs.size() == 0) || (tebPrms.ports.size() == 0))
  {
    fprintf(stderr, "Missing required TEB address(es)\n");
    return 1;
  }

  if  (drpPortNo < MAX_DRPS)  drpPortNo += drpPortBase;
  if ((drpPortNo < drpPortBase) || (drpPortNo >= drpPortBase + MAX_DRPS))
  {
    fprintf(stderr, "DRP Server port %d is out of range %d - %d\n",
            drpPortNo, drpPortBase, drpPortBase + MAX_DRPS);
    return 1;
  }
  tebPrms.port = std::to_string(drpPortNo + tebPrms.id);
  //printf("DRP Srv port = %s\n", tebPrms.port.c_str());

  // Revisit: Fix maxBatches to what will fit in the ImmData idx field?
  if (tebPrms.maxBatches - 1 > ImmData::MaxIdx)
  {
    fprintf(stderr, "Batch index ([0, %d]) can exceed available range ([0, %d])\n",
            tebPrms.maxBatches - 1, ImmData::MaxIdx);
    abort();
  }

  // Revisit: Fix maxEntries to equal duration?
  if (tebPrms.maxEntries > tebPrms.duration)
  {
    fprintf(stderr, "More batch entries (%u) requested than definable "
            "in the batch duration (%lu)\n",
            tebPrms.maxEntries, tebPrms.duration);
    abort();
  }

  ::signal( SIGINT, sigHandler );

  printf("\nParameters of Contributor ID %d:\n",            tebPrms.id);
  printf("  Thread core numbers:        %d, %d\n",          tebPrms.core[0], tebPrms.core[1]);
  printf("  Partition:                  %d: '%s'\n",        partition, partitionTag.c_str());
  printf("  Run-time monitoring period: %d\n",              rtMonPeriod);
  printf("  Run-time monitoring host:   %s\n",              rtMonHost);
  printf("  Number of Monitor EBs:      %zd\n",             mebPrms.addrs.size());
  printf("  Batch duration:             %014lx = %ld uS\n", tebPrms.duration, tebPrms.duration);
  printf("  Batch pool depth:           %d\n",              tebPrms.maxBatches);
  printf("  Max # of entries per batch: %d\n",              tebPrms.maxEntries);
  printf("\n");
  printf("  TEB port range: %d - %d\n", tebPortBase, tebPortBase + MAX_TEBS);
  printf("  DRP port range: %d - %d\n", drpPortBase, drpPortBase + MAX_DRPS);
  printf("  MEB port range: %d - %d\n", mebPortBase, mebPortBase + MAX_MEBS);
  printf("\n");

  pinThread(pthread_self(), tebPrms.core[1]);
  StatsMonitor* smon = new StatsMonitor(rtMonHost,
                                        rtMonPort,
                                        partition,
                                        partitionTag,
                                        rtMonPeriod,
                                        rtMonVerbose);
  lstatsMon = smon;

  pinThread(pthread_self(), tebPrms.core[0]);
  EbCtrbApp*      app = new EbCtrbApp(tebPrms, *smon);
  MebContributor* meb = (mebPrms.addrs.size() != 0) ? new MebContributor(mebPrms) : nullptr;
  EbCtrbIn*       in  = new EbCtrbIn (tebPrms, meb, *smon, outDir);
  lApp = app;

  const unsigned ibMtu = 4096;
  unsigned mtuCnt = (tebPrms.maxEntries * tebPrms.maxInputSize  + ibMtu - 1) / ibMtu;
  unsigned mtuRem = (tebPrms.maxEntries * tebPrms.maxInputSize) % ibMtu;
  printf("\n");
  printf("  sizeof(Dgram):              %zd\n", sizeof(Dgram));
  printf("  Max contribution size:      %zd, batch size: %zd, # MTUs: %d, last: %d / %d (%f%%)\n",
         tebPrms.maxInputSize,  app->maxBatchSize(), mtuCnt, mtuRem, ibMtu, 100. * double(mtuRem) / double(ibMtu));
  mtuCnt = (tebPrms.maxEntries * tebPrms.maxResultSize  + ibMtu - 1) / ibMtu;
  mtuRem = (tebPrms.maxEntries * tebPrms.maxResultSize) % ibMtu;
  printf("  Max result       size:      %zd, batch size: %zd, # MTUs: %d, last: %d / %d (%f%%)\n",
         tebPrms.maxResultSize, in->maxBatchSize(),  mtuCnt, mtuRem, ibMtu, 100. * double(mtuRem) / double(ibMtu));
  printf("\n");

  // Wait a bit to allow other components of the system to establish connections
  sleep(1);                             // Revisit: Should be coordinated with the rest of the system

  app->process(*in);

  printf("\nEbCtrbApp was shut down.\n");

  if (in)    delete in;
  if (meb)   delete meb;
  if (app)   delete app;
  if (smon)  delete smon;

  return ret;
}
