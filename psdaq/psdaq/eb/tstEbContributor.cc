#include "psdaq/eb/MonContributor.hh"
#include "psdaq/eb/EbContributor.hh"
#include "psdaq/eb/EbCtrbInBase.hh"

#include "psdaq/eb/utilities.hh"
#include "psdaq/eb/StatsMonitor.hh"
#include "psdaq/eb/EbLfClient.hh"
#include "psdaq/eb/utilities.hh"

#include "psdaq/service/GenericPoolW.hh"
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
#include <unordered_map>
#include <atomic>
#include <thread>

//#define SINGLE_EVENTS                   // Define to take one event at a time by hitting <CR>

using namespace XtcData;
using namespace Pds;
using namespace Pds::Fabrics;

static const int      core_base        = 6;          // devXX, 8: accXX
static const int      core_offset      = 2;          // Allows Ctrb and EB to run on the same machine
static const unsigned rtMon_period     = 1;          // Seconds
static const unsigned default_id       = 0;          // Contributor's ID (< 64)
static const unsigned max_ctrbs        = 64;         // Maximum possible number of Contributors
static const unsigned max_ebs          = 64;         // Maximum possible number of Event Builders
static const unsigned max_mons         = 64;         // Maximum possible number of Monitors
                                                     // Base ports for:
static const unsigned l3i_port_base    = 32768;                    // L3  EB to receive L3 contributions
static const unsigned l3r_port_base    = l3i_port_base + max_ebs;  // L3  EB to send    results
static const unsigned mrq_port_base    = l3r_port_base + max_ebs;  // L3  EB to receive monitor requests
static const unsigned meb_port_base    = mrq_port_base + max_mons; // Mon EB to receive data contributions
static const unsigned max_batches      = 8192;       // Maximum number of batches in circulation
static const unsigned max_entries      = 64;          // < or = to batch_duration
static const uint64_t batch_duration   = max_entries;// > or = to max_entries; power of 2; beam pulse ticks (1 uS)
static const size_t   header_size      = sizeof(Dgram);
static const size_t   input_extent     = 2;    // Revisit: Number of "L3" input  data words
static const size_t   result_extent    = 2;    // Revisit: Number of "L3" result data words
static const size_t   max_contrib_size = header_size + input_extent  * sizeof(uint32_t);
static const size_t   max_result_size  = header_size + result_extent * sizeof(uint32_t);
static const char*    dflt_partition   = "Test";
static const char*    dflt_rtMon_addr  = "tcp://psdev7b:55561";
static const unsigned mon_epochs       = 1;    // Revisit: Corresponds to monReqServer::numberof_epochs
static const unsigned mon_transitions  = 18;   // Revisit: Corresponds to monReqServer::numberof_trBuffers
static const unsigned mon_buf_cnt      = 8;    // Revisit: Corresponds to monReqServer:numberofEvBuffers
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
    };

    class EbCtrbIn : public EbCtrbInBase
    {
    public:
      EbCtrbIn(const EbCtrbParams& prms,
               MonContributor*     mon,
               StatsMonitor&       smon,
               const char*         outDir);
      virtual ~EbCtrbIn();
    public:                             // For EbCtrbInBase
      virtual void process(const Dgram* result, const void* input);
    private:
      MonContributor* _mon;
      FILE*           _xtcFile;
    private:
      uint64_t        _eventCount;
    };

    class EbCtrbApp : public EbContributor
    {
    public:
      EbCtrbApp(const EbCtrbParams& prms,
                StatsMonitor&       smon);
      virtual ~EbCtrbApp();
    public:
      void     shutdown();
      void     process(EbCtrbIn&);
    private:
      DrpSim              _drpSim;
      const EbCtrbParams& _prms;
    private:
      uint64_t            _eventCount;
      uint64_t            _freeBatchCnt;
      uint64_t            _inFlightCnt;
    private:
      std::atomic<bool>   _running;
      std::thread*        _outThread;
    };
  };
};


using namespace Pds::Eb;

DrpSim::DrpSim(unsigned maxBatches,
               unsigned maxEntries,
               size_t   maxEvtSize,
               unsigned id) :
  _maxBatches(maxBatches),
  _maxEntries(maxEntries),
  _maxEvtSz  (maxEvtSize),
  _id        (id),
  _xtc       (TypeId(TypeId::Data, 0), TheSrc(Level::Segment, id)),
  _pid       (0x01000000000003ul),  // Something non-zero and not on a batch boundary
  _pool      (sizeof(Entry) + maxEvtSize, maxBatches * maxEntries),
  _trId      (TransitionId::Unknown)
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

  void* buffer = _pool.alloc(sizeof(Input));
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


EbCtrbIn::EbCtrbIn(const EbCtrbParams& prms,
                   MonContributor*     mon,
                   StatsMonitor&       smon,
                   const char*         outDir) :
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

  smon.registerIt("CtbI.EvtCt",  _eventCount,        StatsMonitor::SCALAR);
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
    fprintf(stderr, "Result pulse ID doesn't match Input datagram's: %014lx, %014lx\n",
           pid, input->seq.pulseId().value());
    abort();
  }

  if (result->seq.isEvent())            // L1Accept
  {
    uint32_t* response = (uint32_t*)result->xtc.payload();

    if (response[0] && _xtcFile)         // Persist the data
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


EbCtrbApp::EbCtrbApp(const EbCtrbParams& prms,
                     StatsMonitor&       smon) :
  EbContributor(prms),
  _drpSim      (prms.maxBatches, prms.maxEntries, prms.maxInputSize, prms.id),
  _prms        (prms),
  _eventCount  (0),
  _freeBatchCnt(freeBatchCount()),
  _inFlightCnt (0),
  _running     (true),
  _outThread   (nullptr)
{
  smon.registerIt("CtbO.EvtRt",  _eventCount,   StatsMonitor::RATE);
  smon.registerIt("CtbO.EvtCt",  _eventCount,   StatsMonitor::SCALAR);
  smon.registerIt("CtbO.BatCt",   batchCount(), StatsMonitor::SCALAR);
  smon.registerIt("CtbO.FrBtCt", _freeBatchCnt, StatsMonitor::SCALAR);
  smon.registerIt("CtbO.InFlt",  _inFlightCnt,  StatsMonitor::SCALAR);
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
  // Wait a bit to allow other components of the system to establish connections
  sleep(1);

  EbContributor::startup(in);

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

    const Dgram* input = _drpSim.genInput();
    if (!input)  break;

    if (_prms.verbose > 1)
    {
      const char* svc = TransitionId::name(input->seq.service());
      printf("Batching  %15s  dg             @ "
             "%16p, pid %014lx, sz %4zd, src %2d\n",
             svc, input, input->seq.pulseId().value(),
             sizeof(*input) + input->xtc.sizeofPayload(),
             input->xtc.src.log() & 0xff);
    }

    if (EbContributor::process(input, (const void*)input)) // 2nd arg is returned with the result
      ++_eventCount;

    _freeBatchCnt = freeBatchCount();
    _inFlightCnt  = inFlightCnt();
  }

  EbContributor::shutdown();
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

void usage(char *name, char *desc, EbCtrbParams& prms)
{
  fprintf(stderr, "Usage:\n");
  fprintf(stderr, "  %s [OPTIONS]\n", name);

  if (desc)
    fprintf(stderr, "\n%s\n", desc);

  fprintf(stderr, "\n<spec>, below, has the form '<id>:<addr>:<port>', with\n");
  fprintf(stderr, "<id> typically in the range 0 - 63.\n");
  fprintf(stderr, "Low numbered <port> values are treated as offsets into the corresponding ranges:\n");
  fprintf(stderr, "  L3  EB:     %d - %d\n", l3i_port_base, l3i_port_base + max_ebs - 1);
  fprintf(stderr, "  L3  result: %d - %d\n", l3r_port_base, l3r_port_base + max_ebs - 1);
  fprintf(stderr, "  Mon EB:     %d - %d\n", meb_port_base, meb_port_base + max_ebs - 1);

  fprintf(stderr, "\nOptions:\n");

  fprintf(stderr, " %-20s %s (default: %s)\n",          "-A <interface_addr>",
          "IP address of the interface to use",         "libfabric's 'best' choice");
  fprintf(stderr, " %-20s %s (default: %s)\n",          "-L \"<spec>[, <spec>[, ...]]\"",
          "Event Builder to send L3 contributions to",  "None, required");
  fprintf(stderr, " %-20s %s (default: %s)\n",          "-S <port>",
          "Base port for receiving results",            prms.port.c_str());
  fprintf(stderr, " %-20s %s (default: %s)\n",          "-M \"<spec>[, <spec>[, ...]]\"",
          "Event Builder to send Mon contributions to", "None, required");

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
          "Run-time monitoring ZMQ server address",   dflt_rtMon_addr);
  fprintf(stderr, " %-20s %s (default: %s)\n",        "-P <partition>",
          "Partition tag",                            dflt_partition);
  fprintf(stderr, " %-20s %s (default: %d)\n",        "-1 <core>",
          "Core number for pinning App thread to",    prms.core[0]);
  fprintf(stderr, " %-20s %s (default: %d)\n",        "-2 <core>",
          "Core number for pinning other threads to", prms.core[1]);

  fprintf(stderr, " %-20s %s\n", "-v", "enable debugging output (repeat for increased detail)");
  fprintf(stderr, " %-20s %s\n", "-h", "display this help output");
}

static int parseSpec(const char*               who,
                     char*                     spec,
                     unsigned                  maxId,
                     unsigned                  portMin,
                     unsigned                  portMax,
                     std::vector<std::string>& addrs,
                     std::vector<std::string>& ports,
                     uint64_t*                 bits)
{
  do
  {
    char* target = strsep(&spec, ", ");
    if (!*target)  continue;
    char* colon1 = strchr(target, ':');
    char* colon2 = strrchr(target, ':');
    if (!colon1 || (colon1 == colon2))
    {
      fprintf(stderr, "%s: Input '%s' is not of the form <ID>:<IP>:<port>\n",
              who, target);
      return 1;
    }
    unsigned id   = atoi(target);
    unsigned port = atoi(&colon2[1]);
    if (id > maxId)
    {
      fprintf(stderr, "%s: ID %d is out of range 0 - %d\n", who, id, maxId);
      return 1;
    }
    if  (port < portMax - portMin + 1)  port += portMin;
    if ((port < portMin) || (port > portMax))
    {
      fprintf(stderr, "%s: Port %d is out of range %d - %d\n",
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

int main(int argc, char **argv)
{
  // Gather the list of EBs from an argument
  // - Somehow ensure this list is the same for all instances of the client
  // - Maybe pingpong this list around to see whether anyone disagrees with it?

  int           op, ret      = 0;
  const char*   rtMonAddr    = dflt_rtMon_addr;
  unsigned      rtMonPeriod  = rtMon_period;
  unsigned      rtMonVerbose = 0;
  unsigned      l3rPortNo    = l3r_port_base;  // Port served to L3  EBs
  char*         l3iSpec      = nullptr;
  char*         mebSpec      = nullptr;
  char*         outDir       = nullptr;
  std::string   partitionTag  (dflt_partition);
  EbCtrbParams  tPrms { /* .addrs         = */ { },
                        /* .ports         = */ { },
                        /* .ifAddr        = */ nullptr,
                        /* .port          = */ std::to_string(l3rPortNo),
                        /* .id            = */ default_id,
                        /* .builders      = */ 0,
                        /* .duration      = */ batch_duration,
                        /* .maxBatches    = */ max_batches,
                        /* .maxEntries    = */ max_entries,
                        /* .maxInputSize  = */ max_contrib_size,
                        /* .maxResultSize = */ max_result_size,
                        /* .core          = */ { core_base + core_offset + 0,
                                                 core_base + core_offset + 12 },
                        /* .verbose       = */ 0 };
  MonCtrbParams mPrms { /* .addrs         = */ { },
                        /* .ports         = */ { },
                        /* .id            = */ default_id,
                        /* .maxEvents     = */ mon_buf_cnt,
                        /* .maxEvSize     = */ mon_buf_size,
                        /* .maxTrSize     = */ mon_trSize,
                        /* .verbose       = */ 0 };

  while ((op = getopt(argc, argv, "h?vVA:L:R:M:o:i:d:b:e:m:Z:P:n:s:1:2:")) != -1)
  {
    switch (op)
    {
      case 'A':  tPrms.ifAddr     = optarg;               break;
      case 'L':  l3iSpec          = optarg;               break;
      case 'R':  l3rPortNo        = atoi(optarg);         break;
      case 'M':  mebSpec          = optarg;               break;
      case 'o':  outDir           = optarg;               break;
      case 'i':  tPrms.id         = atoi(optarg);
                 mPrms.id         = tPrms.id;             break;
      case 'd':  tPrms.duration   = atoll(optarg);        break;
      case 'b':  tPrms.maxBatches = atoi(optarg);         break;
      case 'e':  tPrms.maxEntries = atoi(optarg);         break;
      case 'm':  rtMonPeriod      = atoi(optarg);         break;
      case 'Z':  rtMonAddr        = optarg;               break;
      case 'P':  partitionTag     = std::string(optarg);  break;
      case 'n':  mPrms.maxEvents  = atoi(optarg);         break;
      case 's':  mPrms.maxEvSize  = atoi(optarg);         break;
      case '1':  tPrms.core[0]    = atoi(optarg);         break;
      case '2':  tPrms.core[1]    = atoi(optarg);         break;
      case 'v':  ++tPrms.verbose;
                 ++mPrms.verbose;                         break;
      case 'V':  ++rtMonVerbose;                          break;
      case '?':
      case 'h':
        usage(argv[0], (char*)"Test DRP event contributor", tPrms);
        return 1;
    }
  }

  if (tPrms.id >= max_ctrbs)
  {
    fprintf(stderr, "Contributor ID %d is out of range 0 - %d\n",
            tPrms.id, max_ctrbs - 1);
    return 1;
  }

  if ((l3rPortNo < l3r_port_base) || (l3rPortNo > l3r_port_base + max_ctrbs))
  {
    fprintf(stderr, "Server port %d is out of range %d - %d\n",
            l3rPortNo, l3r_port_base, l3r_port_base + max_ctrbs);
    return 1;
  }
  tPrms.port = std::to_string(l3rPortNo);

  if (l3iSpec)
  {
    int rc = parseSpec("l3i",
                       l3iSpec,
                       max_ebs - 1,
                       l3i_port_base,
                       l3i_port_base + max_ebs - 1,
                       tPrms.addrs,
                       tPrms.ports,
                       &tPrms.builders);
    if (rc)  return rc;
  }
  else
  {
    fprintf(stderr, "Missing required L3 EB address(es)\n");
    return 1;
  }

  std::vector<std::string> mebAddrs;
  std::vector<std::string> mebPorts;
  uint64_t                 unused;
  if (mebSpec)
  {
    int rc = parseSpec("meb",
                       mebSpec,
                       max_ebs - 1,
                       meb_port_base,
                       meb_port_base + max_ebs - 1,
                       mPrms.addrs,
                       mPrms.ports,
                       &unused);
    if (rc)  return rc;
  }

  if (tPrms.maxEntries > tPrms.duration)
  {
    fprintf(stderr, "More batch entries (%u) requested than definable in the "
            "batch duration (%lu); reducing to avoid wasting memory\n",
            tPrms.maxEntries, tPrms.duration);
    tPrms.maxEntries = tPrms.duration;
  }

  ::signal( SIGINT, sigHandler );

  printf("\nParameters of Contributor ID %d:\n",            tPrms.id);
  printf("  Thread core numbers:        %d, %d\n",          tPrms.core[0], tPrms.core[1]);
  printf("  Partition tag:             '%s'\n",             partitionTag.c_str());
  printf("  Run-time monitoring period: %d\n",              rtMonPeriod);
  printf("  Run-time monitoring server: %s\n",              rtMonAddr);
  printf("  Batch duration:             %014lx = %ld uS\n", tPrms.duration, tPrms.duration);
  printf("  Batch pool depth:           %d\n",              tPrms.maxBatches);
  printf("  Max # of entries per batch: %d\n",              tPrms.maxEntries);
  printf("\n");

  pinThread(pthread_self(), tPrms.core[1]);
  StatsMonitor* smon = new StatsMonitor(rtMonAddr, partitionTag, rtMonPeriod, rtMonVerbose);
  lstatsMon = smon;

  pinThread(pthread_self(), tPrms.core[0]);
  EbCtrbApp*      app = new EbCtrbApp(tPrms, *smon);
  MonContributor* meb = mebSpec ? new MonContributor(mPrms) : nullptr;
  EbCtrbIn*       in  = new EbCtrbIn (tPrms, meb, *smon, outDir);
  lApp = app;

  const unsigned ibMtu = 4096;
  unsigned mtuCnt = (sizeof(Dgram) + tPrms.maxEntries * tPrms.maxInputSize  + ibMtu - 1) / ibMtu;
  unsigned mtuRem = (sizeof(Dgram) + tPrms.maxEntries * tPrms.maxInputSize) % ibMtu;
  printf("\n");
  printf("  sizeof(Dgram):              %zd\n", sizeof(Dgram));
  printf("  Max contribution size:      %zd, batch size: %zd, # MTUs: %d, last: %d / %d (%f%%)\n",
         tPrms.maxInputSize,  app->maxBatchSize(), mtuCnt, mtuRem, ibMtu, 100. * double(mtuRem) / double(ibMtu));
  mtuCnt = (sizeof(Dgram) + tPrms.maxEntries * tPrms.maxResultSize  + ibMtu - 1) / ibMtu;
  mtuRem = (sizeof(Dgram) + tPrms.maxEntries * tPrms.maxResultSize) % ibMtu;
  printf("  Max result       size:      %zd, batch size: %zd, # MTUs: %d, last: %d / %d (%f%%)\n",
         tPrms.maxResultSize, in->maxBatchSize(),  mtuCnt, mtuRem, ibMtu, 100. * double(mtuRem) / double(ibMtu));
  printf("\n");

  app->process(*in);

  printf("\nEbCtrbApp was shut down.\n");

  if (in)    delete in;
  if (meb)   delete meb;
  if (app)   delete app;
  if (smon)  delete smon;

  return ret;
}
