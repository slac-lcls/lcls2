#include "psdaq/eb/BatchManager.hh"
#include "psdaq/eb/Endpoint.hh"

#include "psdaq/eb/EbFtClient.hh"
#include "psdaq/eb/EbFtServer.hh"

#include "psdaq/service/Routine.hh"
#include "psdaq/service/Task.hh"
#include "psdaq/service/Semaphore.hh"
#include "psdaq/service/GenericPoolW.hh"
#include "xtcdata/xtc/Dgram.hh"

#include <new>
#include <fcntl.h>
#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

using namespace XtcData;
using namespace Pds;
using namespace Pds::Fabrics;

static const unsigned default_id       = 0;          // Contributor's ID
static const unsigned srv_port_base    = 32768 + 64; // Base port Contributor for receiving results
static const unsigned clt_port_base    = 32768;      // Base port Contributor sends to EB on
static const uint64_t batch_duration   = 0x1ul << 9; //0x00000000000080UL; // ~128 uS; must be power of 2
static const unsigned max_batches      = 2048; //16; //8192;     // 1 second's worth
static const unsigned max_entries      = 512;  //128;     // 128 uS worth
static const size_t   header_size      = sizeof(Dgram);
static const size_t   input_extent     = 5; // Revisit: Number of "L3" input  data words
//static const size_t   result_extent    = 2; // Revisit: Number of "L3" result data words
static const size_t   result_extent    = input_extent;
static const size_t   max_contrib_size = header_size + input_extent  * sizeof(uint32_t);
static const size_t   max_result_size  = header_size + result_extent * sizeof(uint32_t);

static       bool     lcheck           = false;
static       unsigned lverbose         = 0;


//static void dump(const Dgram* dg)
//{
//  char buff[128];
//  time_t t = dg->seq.clock().seconds();
//  strftime(buff,128,"%H:%M:%S",localtime(&t));
//  printf("[%16p] %s.%09u %016lx %s extent 0x%x damage %x\n",
//         dg,
//         buff,
//         dg->seq.clock().nanoseconds(),
//         dg->seq.stamp().pulseID(),
//         Pds::TransitionId::name(Pds::TransitionId::Value((dg->seq.stamp().fiducials()>>24)&0xff)),
//         dg->xtc.extent, dg->xtc.damage.value());
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

    class Input : public Entry
    {
    public:
      Input(Sequence& seq, Env& env, Xtc& xtc) :
        Entry(),
        _datagram()
      {
        _datagram.seq = seq;
        _datagram.env = env;
        _datagram.xtc = xtc;
      }
    public:
      PoolDeclare;
    public:
      Dgram* datagram() { return &_datagram; }
    public:
      uint64_t id() const { return _datagram.seq.stamp().pulseId(); }
    private:
      Dgram _datagram;
    };

    class DrpSim : public Routine
    {
    public:
      DrpSim(unsigned poolSize, unsigned id);
      ~DrpSim()
      {
      }
    public:
      void start(TstContribOutlet* outlet);
      void shutdown() { _running = false; }
    public:
      void routine();
    public:
      void release(Dgram* result);
    private:
      unsigned          _maxEvtSz;
      GenericPoolW      _pool;
      TypeId            _typeId;
      Src               _src;
      unsigned          _id;
      uint64_t          _pid;
      Queue<Input>      _inFlightList;
      TstContribOutlet* _outlet;
      volatile bool     _running;
      Task*             _task;
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
      ~TstContribOutlet();
    public:
      void shutdown();
    public:
      void routine();
      void post(Dgram* input);
      void release(uint64_t id);
      void post(Batch* batch);
    private:
      Batch* _pend();
    private:
      EbFtClient    _outlet;
      unsigned      _numEbs;
      unsigned*     _destinations;
      BatchList     _inFlightList;      // Listhead of batches in flight
      Queue<Batch>  _inputBatchList;
      Semaphore     _inputBatchSem;
      uint64_t      _eventCount;
      volatile bool _running;
      Task*         _task;
    };

    class TstContribInlet
    {
    public:
      TstContribInlet(std::string&      srvPort,
                      unsigned          id,
                      uint64_t          builders,
                      unsigned          maxBatches,
                      unsigned          maxEntries,
                      size_t            maxSize,
                      TstContribOutlet& outlet,
                      DrpSim&           sim);
      ~TstContribInlet() { }
    public:
      void shutdown() { _running = false; }
    public:
      void process();
      int _process(Dgram* result);
    private:
      EbFtServer        _inlet;
      TstContribOutlet& _outlet;
      DrpSim&           _sim;
      volatile bool     _running;
    };
  };
};

using namespace Pds::Eb;

DrpSim::DrpSim(unsigned poolSize, unsigned id) :
  _maxEvtSz    (max_contrib_size),
  _pool        (sizeof(Entry) + _maxEvtSz, poolSize),
  _typeId      (TypeId::Data, 0),
  _src         (TheSrc(Level::Segment, id)),
  _id          (id),
  _pid         (0x01000000000003UL),  // Something non-zero and not on a batch boundary
  _inFlightList(),
  _outlet      (NULL),
  _running     (true),
  _task        (new Task(TaskObject("drp")))
{
}

void DrpSim::start(TstContribOutlet* outlet)
{
  _outlet = outlet;

  _task->call(this);
}

void DrpSim::routine()
{
  static unsigned cnt = 0;

  while(_running)
  {
    // Fake datagram header
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    Sequence seq(Sequence::Event, TransitionId::L1Accept, ClockTime(ts), TimeStamp(_pid));
    Env      env(0);
    Xtc      xtc(_typeId, _src);

    _pid += 1; //27; // Revisit: fi_tx_attr.iov_limit = 6 = 1 + max # events per batch
    //_pid = ts.tv_nsec / 1000;     // Revisit: Not guaranteed to be the same across systems

    // Make proxy to data (small data)
    Input* input = new(&_pool) Input(seq, env, xtc);
    Dgram* pdg   = input->datagram();

    size_t inputSize = input_extent*sizeof(uint32_t);

    // Revisit: Here is where L3 trigger input information would be inserted into the datagram,
    //          whatever that will look like.  SmlD is nominally the right container for this.
    uint32_t* payload = (uint32_t*)pdg->xtc.alloc(inputSize);
    payload[0] = ts.tv_sec;           // Revisit: Some crud for now
    payload[1] = ts.tv_nsec;          // Revisit: Some crud for now
    payload[2] = ++cnt;               // Revisit: Some crud for now
    payload[3] = 1 << _id;            // Revisit: Some crud for now
    payload[4] = pdg->seq.stamp().pulseId() & 0xffffffffUL; // Revisit: Some crud for now

    _inFlightList.insert(input);
    _outlet->post(pdg);

    //usleep(1);
  }

  printf("\nDrpSim Input data pool:\n");
  _pool.dump();
}

void DrpSim::release(Dgram* result)
{
  uint64_t id    = result->seq.stamp().pulseId();
  Input*   input = _inFlightList.atHead();
  Input*   last  = _inFlightList.empty();
  while (input != last)
  {
    Entry* next = input->next();

    if (id == input->id())
    {
      delete (Input*)_inFlightList.remove(input); // Revisit: Replace with atomic list
      break;
    }

    input = (Input*)next;
  }
}

TstContribOutlet::TstContribOutlet(std::vector<std::string>& cltAddr,
                                   std::vector<std::string>& cltPort,
                                   unsigned                  id,
                                   uint64_t                  builders,
                                   uint64_t                  duration,
                                   unsigned                  maxBatches,
                                   unsigned                  maxEntries,
                                   size_t                    maxSize) :
  BatchManager   (TheSrc(Level::Event, id), duration, maxBatches, maxEntries, maxSize),
  _outlet        (cltAddr, cltPort, maxBatches * (sizeof(Dgram) + maxEntries * maxSize)),
  _numEbs        (__builtin_popcountl(builders)),
  _destinations  (new unsigned[_numEbs]),
  _inFlightList  (),
  _inputBatchList(),
  _inputBatchSem (Semaphore::EMPTY),
  _eventCount    (0),
  _running       (true),
  _task          (new Task(TaskObject("tInlet")))
{
  const unsigned tmo(120);
  int ret = _outlet.connect(id, tmo);
  if (ret)
  {
    fprintf(stderr, "FtClient connect() failed\n");
    abort();
  }

  _outlet.registerMemory(batchRegion(), batchRegionSize());

  for (unsigned i = 0; i < _numEbs; ++i)
  {
    unsigned bit = __builtin_ffsl(builders) - 1;
    builders &= ~(1ul << bit);
    _destinations[i] = bit;
  }

  _task->call(this);
}

TstContribOutlet::~TstContribOutlet()
{
  delete [] _destinations;
}

void TstContribOutlet::shutdown()
{
  _running = false;

  _task->destroy();
}

void TstContribOutlet::routine()
{
  timespec startTime;
  clock_gettime(CLOCK_REALTIME, &startTime);

  while (_running)
  {
    const Batch* batch = _pend();

    unsigned iDst = batchId(batch->id()) % _numEbs;
    unsigned dst  = _destinations[iDst];

    if (lverbose)
    {
      static unsigned cnt    = 0;
      const Dgram*    bdg    = batch->datagram();
      void*           rmtAdx = (void*)_outlet.rmtAdx(dst, batch->index() * maxBatchSize());
      printf("ContribOutlet posts %6d        batch[%2d]    @ %16p, ts %014lx, sz %3zd to   EB %d %16p\n",
             ++cnt, batch->index(), bdg, bdg->seq.stamp().pulseId(), batch->extent(), dst, rmtAdx);
    }

    _inFlightList.insert(const_cast<Batch*>(batch));    // Revisit: Replace with atomic list

    _outlet.post(batch->pool(), batch->extent(), dst, batch->index() * maxBatchSize());
  }

  BatchManager::dump();

  timespec endTime;
  clock_gettime(CLOCK_REALTIME, &endTime);

  dumpStats("Outlet", startTime, endTime, _eventCount);

  _outlet.shutdown();
}

void TstContribOutlet::post(Dgram* input)
{
  if (lverbose > 1)
  {
    printf("DRP   Posts        input          dg           @ %16p, ts %014lx, sz %3zd, src %2d\n",
           input, input->seq.stamp().pulseId(), sizeof(*input) + input->xtc.sizeofPayload(), input->xtc.src.log() & 0xff);
  }

  ++_eventCount;

  process(input);
}

void TstContribOutlet::release(uint64_t id)
{
  Batch* batch = _inFlightList.atHead();
  Batch* last  = _inFlightList.empty();
  while (batch != last)
  {
    Entry* next = batch->next();

    if (id == batch->id())
    {
      delete (Batch*)_inFlightList.remove(batch); // Revisit: Replace with atomic list
      break;
    }

    batch = (Batch*)next;
  }
}

void TstContribOutlet::post(Batch* batch)
{
  if (lverbose > 1)
  {
    const Dgram* bdg = batch->datagram();
    printf("DRP   Posting      input          batch        @ %16p, ts %014lx, sz %3zd\n",
           batch, bdg->seq.stamp().pulseId(), sizeof(*bdg) + bdg->xtc.sizeofPayload());
  }

  _inputBatchList.insert(batch);
  _inputBatchSem.give();
}

Batch* TstContribOutlet::_pend()
{
  _inputBatchSem.take();
  Batch* batch = _inputBatchList.remove();

  if (lverbose > 1)
  {
    const Dgram* bdg = batch->datagram();
    printf("ContribOutlet rcvd input          batch        @ %16p, ts %014lx, sz %3zd\n",
           batch, bdg->seq.stamp().pulseId(), sizeof(*bdg) + bdg->xtc.sizeofPayload());
  }

  return batch;
}

TstContribInlet::TstContribInlet(std::string&      srvPort,
                                 unsigned          id,
                                 uint64_t          builders,
                                 unsigned          maxBatches,
                                 unsigned          maxEntries,
                                 size_t            maxSize,
                                 TstContribOutlet& outlet,
                                 DrpSim&           sim) :
  _inlet  (srvPort, __builtin_popcountl(builders), maxBatches * (sizeof(Dgram) + maxEntries * maxSize), EbFtServer::PEERS_SHARE_BUFFERS),
  _outlet (outlet),
  _sim    (sim),
  _running(true)
{
  int ret = _inlet.connect(id);
  if (ret)
  {
    fprintf(stderr, "FtServer connect() failed\n");
    abort();
  }
}

void TstContribInlet::process()
{
  uint64_t eventCount = 0;
  timespec startTime;
  clock_gettime(CLOCK_REALTIME, &startTime);

  while (_running)
  {
    // Pend for a result datagram (batch) and process it.
    Dgram* batch = (Dgram*)_inlet.pend();
    if (!batch)  break;  //continue;              // Revisit: This may cause a race when quitting

    if (lverbose)
    {
      static unsigned cnt = 0;
      unsigned idx  = ((char*)batch - _inlet.base()) / (sizeof(Dgram) + max_entries * max_result_size);
      unsigned from = batch->xtc.src.log() & 0xff;
      printf("ContribInlet  rcvd        %6d result   [%2d] @ %16p, ts %014lx, sz %3zd from EB %d\n",
             ++cnt, idx, batch, batch->seq.stamp().pulseId(), sizeof(*batch) + batch->xtc.sizeofPayload(), from);
    }

    Dgram* result = (Dgram*)batch->xtc.payload();
    Dgram* last   = (Dgram*)batch->xtc.next();
    while(result != last)
    {
      _process(result);

      ++eventCount;

      result = (Dgram*)result->xtc.next();
    }

    _outlet.release(batch->seq.stamp().pulseId());
  }

  timespec endTime;
  clock_gettime(CLOCK_REALTIME, &endTime);

  dumpStats("Inlet", startTime, endTime, eventCount);

  _inlet.shutdown();
}

int TstContribInlet::_process(Dgram* result)
{
  if (lcheck)
  {
    bool ok = true;
    uint32_t* buffer = (uint32_t*)result->xtc.payload();
    // Timestamp isn't coherent when multiple contributor processes generate it
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
    uint64_t         pid = result->seq.stamp().pulseId();
    if (_pid > pid)
    {
      ok = false;
      printf("Pulse ID didn't increase: previous = %014lx, current = %014lx\n", _pid, pid);
    }
    _pid = pid;
    if ((pid & 0xffffffffUL) != buffer[4])
    {
      ok = false;
      printf("Pulse ID mismatch: expected %08lx, got %08x\n",
             pid & 0xffffffffUL, buffer[4]);
    }
    if (!ok)
    {
      printf("ContribInlet  found result   %16p, ts %014lx, sz %d, pyld %08x %08x %08x %08x %08x\n",
             result, pid, result->xtc.sizeofPayload(),
             buffer[0], buffer[1], buffer[2], buffer[3], buffer[4]);
      abort();                          // Revisit
    }
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

  _sim.release(result);                 // Revisit: Replace this dependency with something more generic

  return 0;
}

static TstContribOutlet* contribOutlet = NULL;
static TstContribInlet*  contribInlet  = NULL;

void sigHandler(int signal)
{
  static unsigned callCount = 0;

  printf("sigHandler() called\n");

  if (contribInlet  && (callCount == 0))  contribInlet->shutdown();
  if (contribOutlet && (callCount == 0))  contribOutlet->shutdown();

  if (callCount++)  ::abort();
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

  fprintf(stderr, " %-20s %s (server: %d)\n", "-B <srv_port>",
          "Base port number for Contributors", srv_port_base);
  fprintf(stderr, " %-20s %s (client: %d)\n", "-P <clt_port>",
          "Base port number for Builders", clt_port_base);

  fprintf(stderr, " %-20s %s (default: %d)\n", "-i <ID>",
          "Unique ID this Contributor will assume (0 - 63)", default_id);

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
  unsigned id = default_id;
  unsigned srvBase = srv_port_base;     // Port served to builders
  unsigned cltBase = clt_port_base;     // Port served by builders

  while ((op = getopt(argc, argv, "hvcB:P:i:")) != -1)
  {
    switch (op)
    {
      case 'B':  srvBase = strtoul(optarg, NULL, 0);  break;
      case 'P':  cltBase = strtoul(optarg, NULL, 0);  break;
      case 'i':  id      = strtoul(optarg, NULL, 0);  break;
      case 'c':  lcheck  = true;                      break;
      case 'v':  ++lverbose;                          break;
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

  printf("Parameters:\n");
  printf("  Batch duration:             %014lx = %ld uS\n", batch_duration, batch_duration);
  printf("  Batch pool depth:           %d\n", max_batches);
  printf("  Max # of entries per batch: %d\n", max_entries);
  printf("  Max contribution size:      %zd\n", max_contrib_size);
  printf("  Max result       size:      %zd\n", max_result_size);

  DrpSim sim(max_batches * max_entries, id);

  TstContribOutlet outlet(cltAddr, cltPort, id, builders, batch_duration, max_batches, max_entries, max_contrib_size);
  TstContribInlet  inlet (         srvPort, id, builders,                 max_batches, max_entries, max_result_size, outlet, sim);
  contribInlet  = &inlet;
  contribOutlet = &outlet;

  sim.start(&outlet);

  inlet.process();

  sim.shutdown();

  return ret;
}
