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
static const uint64_t batch_duration   = 0x00000000000080UL; // ~128 uS; must be power of 2
static const unsigned max_batches      = 16; //8192;     // 1 second's worth
static const unsigned max_entries      = 128;     // 128 uS worth
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
//  printf("[%p] %s.%09u %016lx %s extent 0x%x damage %x\n",
//         dg,
//         buff,
//         dg->seq.clock().nanoseconds(),
//         dg->seq.stamp().pulseID(),
//         Pds::TransitionId::name(Pds::TransitionId::Value((dg->seq.stamp().fiducials()>>24)&0xff)),
//         dg->xtc.extent, dg->xtc.damage.value());
//}

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
      DrpSim(unsigned poolSize, unsigned id) :
        _maxEvtSz(max_contrib_size),
        _pool    (sizeof(Entry) + _maxEvtSz, poolSize),
        _typeId  (TypeId::Data, 0),
        _src     (TheSrc(Level::Segment, id)),
        _id      (id),
        _pid     (0x01000000000003UL),  // Something non-zero and not on a batch boundary
        _task    (new Task(TaskObject("drp"))),
        _sem     (Semaphore::EMPTY)
      {
        _task->call(this);
      }
      ~DrpSim() { }
    public:
      Dgram* get()
      {
        _sem.take();

        Input* input = _inputList.remove();
        _inFlightList.insert(input);
        return input->datagram();
      }
      void release(uint64_t id)
      {
        Input* input = _inFlightList.atHead();
        Input* end   = _inFlightList.empty();
        while (input != end)
        {
          Entry* next = input->next();

          if (id > input->id())
          {
            _inFlightList.remove(input);      // Revisit: Replace with atomic list
            delete input;
          }
          input = (Input*)next;
        }
      }
    public:
      void*  pool()     { return _pool.buffer(); }
      size_t poolSize() { return _pool.size();   }
    public:
      void routine()
      {
        static unsigned cnt = 0;

        while(1)
        {
          //char* buffer = (char*)_pool->alloc(_maxEvtSz);

          // Fake datagram header
          struct timespec ts;
          clock_gettime(CLOCK_REALTIME, &ts);
          Sequence seq(Sequence::Event, TransitionId::L1Accept, ClockTime(ts), TimeStamp(_pid));
          Env      env(0);
          Xtc      xtc(_typeId, _src);

          _pid += 27; // Revisit: fi_tx_attr.iov_limit = 6 = 1 + max # events per batch
          //_pid = ts.tv_nsec / 1000;     // Revisit: Not guaranteed to be the same across systems

          // Make proxy to data (small data)
          //Dgram* pdg = (Dgram*)buffer;
          //pdg->seq = seq;
          //pdg->env = env;
          //pdg->xtc = xtc;

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

          _inputList.insert(input);
          _sem.give();

          if (lverbose > 1)
            printf("DrpSim wrote input dg %p (ts %014lx, ext %zd)\n",
                   pdg, pdg->seq.stamp().pulseId(), sizeof(*pdg) + pdg->xtc.sizeofPayload());

          usleep(100);
        }
      }
    private:
      unsigned     _maxEvtSz;
      GenericPoolW _pool;
      TypeId       _typeId;
      Src          _src;
      unsigned     _id;
      uint64_t     _pid;
      Task*        _task;
      Semaphore    _sem;
      Queue<Input> _inputList;
      Queue<Input> _inFlightList;
    };

    class TstContribOutlet : public BatchManager,
                             public Routine
    {
    public:
      TstContribOutlet(EbFtBase& outlet,
                       unsigned  id,
                       uint64_t  builders,
                       uint64_t  duration,
                       unsigned  maxBatches,
                       unsigned  maxEntries,
                       size_t    maxSize) :
        BatchManager(TheSrc(Level::Event, id), duration, maxBatches, maxEntries, maxSize),
        _outlet(outlet),
        _numEbs(__builtin_popcountl(builders)),
        _destinations(new unsigned[_numEbs]),
        _sim(maxBatches * maxEntries, id), // Simulate DRP in a separate thread
        _inFlightList(),
        _running(true),
        _task (new Task(TaskObject("tInlet")))
      {
        for (unsigned i = 0; i < _numEbs; ++i)
        {
          unsigned bit = __builtin_ffsl(builders) - 1;
          builders &= ~(1 << bit);
          _destinations[i] = bit;
        }

        MemoryRegion* mr[2];

        mr[0] = outlet.registerMemory(batchPool(), batchPoolSize());
        mr[1] = outlet.registerMemory(_sim.pool(), _sim.poolSize());

        start(maxBatches, maxEntries, mr);

        _task->call(this);
      }
      ~TstContribOutlet() { delete [] _destinations; }
    public:
      void shutdown()
      {
        _running = false;

        _task->destroy();
      }
    public:
      void routine()
      {
        while (_running)
        {
          const Dgram* input = _pend();
          if (!input)  continue;

          process(input);
        }
      }
      void post(Batch* batch)
      {
        _inFlightList.insert(batch);    // Revisit: Replace with atomic list

        unsigned iDst = batchId(batch->id()) % _numEbs;
        unsigned dst  = _destinations[iDst];

        if (lverbose)
        {
          static unsigned cnt = 0;
          void*    rmtAdx = (void*)_outlet.rmtAdx(dst, batch->index() * maxBatchSize());
          printf("ContribOutlet posts %6d        batch[%2d] @ %p, ts %014lx, sz %zd to EB %d %p\n", ++cnt, batch->index(),
                 batch->datagram(), batch->datagram()->seq.stamp().pulseId(), batch->extent(), dst, rmtAdx);
        }

        _outlet.post(batch->pool(), batch->extent(), dst, batch->index() * maxBatchSize(), NULL);
      }
      void release(uint64_t id)
      {
        Batch* batch = _inFlightList.atHead();
        Batch* end   = _inFlightList.empty();
        while (batch != end)
        {
          if (id < batch->id())  break;

          Entry* next = batch->next();

          delete (Batch*)_inFlightList.remove(batch); // Revisit: Replace with atomic list

          batch = (Batch*)next;
        }
      }
      void freeInput(Dgram* result)
      {
        _sim.release(result->seq.stamp().pulseId());
      }
    private:
      Dgram* _pend()
      {
        Dgram* pdg = _sim.get();

        if (lverbose > 1)
          printf("ContribOutlet read dg %p (ts %014lx, ext %zd)\n",
                 pdg, pdg->seq.stamp().pulseId(), sizeof(*pdg) + pdg->xtc.sizeofPayload());

        return pdg;
      }
    private:
      EbFtBase&     _outlet;
      unsigned      _numEbs;
      unsigned*     _destinations;
      DrpSim        _sim;
      BatchList     _inFlightList;      // Listhead of batches in flight
      volatile bool _running;
      Task*         _task;
    };

    class TstContribInlet
    {
    public:
      TstContribInlet(EbFtBase&         inlet,
                      const char*       base,
                      TstContribOutlet& outlet) :
        _inlet(inlet),
        _base(base),
        _outlet(outlet),
        _running(true)
      {
      }
      ~TstContribInlet() { }
    public:
      void shutdown() { _running = false; }
    public:
      void process()
      {
        while (_running)
        {
          // Pend for a result datagram (batch) and handle it.
          Dgram* batch = (Dgram*)_inlet.pend();
          if (!batch)  break;

          if (lverbose)
          {
            static unsigned cnt = 0;
            unsigned idx = ((char*)batch - _base) / (max_entries * max_result_size);
            printf("ContribInlet  rcvd        %6d result[%2d] @ %p, ts %014lx, sz %zd\n", ++cnt, idx,
                   batch, batch->seq.stamp().pulseId(), sizeof(*batch) + batch->xtc.sizeofPayload());
          }

          Dgram* entry = (Dgram*)batch->xtc.payload();
          Dgram* end   = (Dgram*)batch->xtc.next();
          while(entry != end)
          {
            _process(entry);

            entry = (Dgram*)entry->xtc.next();
          }

          _outlet.release(batch->seq.stamp().pulseId());
        }
      }
      int _process(Dgram* result)
      {
        if (lcheck)
        {
          bool ok = true;
          uint32_t* buffer = (uint32_t*)result->xtc.payload();
          static uint64_t _ts = 0;
          uint64_t         ts = buffer[0];
          ts = (ts << 32) | buffer[1];
          if (ts <= _ts)
          {
            ok = false;
            printf("Timestamp didn't increase: previous = %016lx, current = %016lx\n", _ts, ts);
          }
          _ts = ts;
          static unsigned _counter = 0;
          if (buffer[2] != _counter + 1)
          {
            ok = false;
            printf("Result counter didn't increase by 1: expected %08x, got %08x\n",
                   _counter + 1, buffer[2]);
          }
          _counter = buffer[2];
          if ((result->seq.stamp().pulseId() & 0xffffffffUL) != buffer[4])
          {
            ok = false;
            printf("Pulse ID mismatch: expected %08lx, got %08x\n",
                   result->seq.stamp().pulseId() & 0xffffffffUL, buffer[4]);
          }
          if (!ok)
          {
            printf("ContribInlet  found result   %p, ts %014lx, sz %d, pyld %08x %08x %08x %08x %08x\n",
                   result, result->seq.stamp().pulseId(), result->xtc.sizeofPayload(),
                   buffer[0], buffer[1], buffer[2], buffer[3], buffer[4]);
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

        _outlet.freeInput(result);
        return 0;
      }
    private:
      EbFtBase&         _inlet;
      const char*       _base;
      TstContribOutlet& _outlet;
      volatile bool     _running;
    };
  };
};


using namespace Pds::Eb;

static TstContribInlet* contribInlet = NULL;

void sigHandler(int signal)
{
  static unsigned callCount = 0;

  printf("sigHandler() called\n");

  if (contribInlet)  contribInlet->shutdown();

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
      builders |= 1 << bid;
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

  unsigned numEbs      = __builtin_popcountl(builders);
  size_t   contribSize = max_batches * (sizeof(Dgram) + max_entries * max_contrib_size);
  size_t   resultSize  = max_batches * (sizeof(Dgram) + max_entries * max_result_size);

  EbFtClient ftc(cltAddr, cltPort, contribSize);
  unsigned   tmo(120);                  // Seconds
  ret = ftc.connect(id, tmo);
  if (ret)
  {
    fprintf(stderr, "FtClient connect() failed\n");
    return ret;
  }

  EbFtServer fts(srvPort, numEbs, resultSize, EbFtServer::PEERS_SHARE_BUFFERS);
  ret = fts.connect(id);
  if (ret)
  {
    fprintf(stderr, "FtServer connect() failed\n");
    return ret;
  }

  TstContribOutlet outlet(ftc, id, builders, batch_duration, max_batches, max_entries, max_contrib_size);
  TstContribInlet  inlet (fts, fts.base(), outlet);
  contribInlet = &inlet;

  inlet.process();

  outlet.shutdown();
  fts.shutdown();
  ftc.shutdown();

  return ret;
}
