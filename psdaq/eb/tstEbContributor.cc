#include "psdaq/eb/BatchManager.hh"
#include "psdaq/eb/EbFtClient.hh"

#include "psdaq/service/Routine.hh"
#include "psdaq/service/Task.hh"
#include "psdaq/service/Semaphore.hh"
#include "psdaq/service/GenericPoolW.hh"
#include "xtcdata/xtc/Dgram.hh"
#include "xtcdata/xtc/XtcIterator.hh"

#include <new>
#include <poll.h>
#include <fcntl.h>
#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

using namespace XtcData;
using namespace Pds;

static const uint64_t batch_duration   = 0x00000000000080UL; // ~128 uS; must be power of 2
static const unsigned max_batches      = 16; //8192;     // 1 second's worth
static const unsigned max_entries      = 128;     // 128 uS worth
static const size_t   header_size      = sizeof(Dgram);
static const size_t   input_extent     = 5; // Revisit: Number of "L3" input  data words
static const size_t   result_extent    = 2; // Revisit: Number of "L3" result data words
static const size_t   max_contrib_size = header_size + input_extent  * sizeof(uint32_t);
static const size_t   max_result_size  = header_size + result_extent * sizeof(uint32_t);

static       bool     lverbose         = false;


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

    class DrpSim : public Routine
    {
    public:
      DrpSim(unsigned poolSize, unsigned id) :
        _maxEvtSz(max_contrib_size),
        _pool    (new GenericPoolW(_maxEvtSz, poolSize)),
        _typeId  (TypeId::Data, 0),
        _src     (Level::Segment),
        _pid     (0),
        _task    (new Task(TaskObject("drp"))),
        _sem     (Semaphore::FULL)
      {
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        _pid = ((uint64_t)ts.tv_sec) * 1000000 + ts.tv_nsec / 1000;

        if (pipe(_fd)) perror("pipe");
        _task->call(this);
      }
      ~DrpSim() { delete _pool; }
    public:
      // poll interface
      int   fd() const { return _fd[0]; }
      void  go()       { _sem.give();   }
    public:
      void routine()
      {
        while(1)
        {
          char* buffer = (char*)_pool->alloc(_maxEvtSz);

          // Fake datagram header
          struct timespec ts;
          clock_gettime(CLOCK_REALTIME, &ts);
          Sequence seq(Sequence::Event, TransitionId::L1Accept, ClockTime(ts), TimeStamp(_pid));
          Env      env(0);
          Xtc      xtc(_typeId, _src);

          _pid += 16;                   // Avoid too many events per batch for now
          //_pid = ts.tv_nsec / 1000;     // Revisit: Not guaranteed to be the same across systems

          // Make proxy to data (small data)
          Dgram* pdg = (Dgram*)buffer;
          pdg->seq = seq;
          pdg->env = env;
          pdg->xtc = xtc;

          size_t inputSize = input_extent*sizeof(uint32_t);

          // Revisit: Here is where L3 trigger input information would be inserted into the datagram,
          //          whatever that will look like.  SmlD is nominally the right container for this.
          uint32_t* payload = (uint32_t*)pdg->xtc.alloc(inputSize);
          for (unsigned i = 0; i < input_extent; ++i)
          {
            payload[i] = i;             // Revisit: Some crud for now
          }

          ::write(_fd[1], &pdg, sizeof(pdg)); // Write just the _pointer_
          //if (lverbose)
          //  printf("DrpSim wrote event %p to fd %d\n", pdg, _fd[1]);

          _sem.take();                  // Revisit: Kludge to avoid flooding
        }
      }
    private:
      unsigned      _maxEvtSz;
      GenericPoolW* _pool;
      TypeId        _typeId;
      Src           _src;
      uint64_t      _pid;
      int           _fd[2];
      Task*         _task;
      Semaphore     _sem;
    };

    class TstContribOutlet : public BatchManager,
                             public Routine
    {
    public:
      TstContribOutlet(EbFtBase& outlet,
                       unsigned  id,
                       unsigned  numEbs,
                       uint64_t  duration,
                       unsigned  maxBatches,
                       unsigned  maxEntries,
                       size_t    maxSize) :
        BatchManager(outlet, id, duration, maxBatches, maxEntries, maxSize),
        _numEbs(numEbs),
        _sim(maxBatches * maxEntries, id), // Simulate DRP in a separate thread
        _running(true),
        _task (new Task(TaskObject("tInlet")))
      {
        _task->call(this);
      }
      ~TstContribOutlet() { }
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
          const Dgram* contrib = _pend();
          if (!contrib)  continue;

          process(contrib);
        }
      }
      void post(Batch* batch, void* arg)
      {
        unsigned dst = batchId(batch->id()) % _numEbs;

        printf("ContribOutlet posts batch    %p, ts %014lx, sz %d\n",
               batch->datagram(), batch->datagram()->seq.stamp().pulseId(), batch->datagram()->xtc.extent);

        postTo(batch, dst, batch->index());
      }
    private:
      Dgram* _pend()
      {
        char* p;
        if (::read(_sim.fd(), &p, sizeof(p)) < 0)
        {
          perror("read");
          return NULL;
        }

        _sim.go();        // Revisit: Kludge to allow preparation of next event

        Dgram* pdg = reinterpret_cast<Dgram*>(p);

        //if (lverbose)
        //  printf("ContribOutlet read dg %p from fd %d\n", pdg, _sim.fd());

        return pdg;
      }
    private:
      unsigned      _numEbs;
      DrpSim        _sim;
      volatile bool _running;
      Task*         _task;
    };

    class TstContribInlet : public XtcIterator
    {
    public:
      TstContribInlet(EbFtBase& inlet) :
        XtcIterator(),
        _inlet(inlet),
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

          if (!batch)  continue;
          printf("ContribInlet  rcvd  batch    %p, ts %014lx, sz %d\n",
                 batch, batch->seq.stamp().pulseId(), batch->xtc.extent);

          iterate(&batch->xtc);               // Calls process(xtc) below
        }
      }
      int process(Xtc* xtc)
      {
        Dgram* result = (Dgram*)xtc->payload();

        printf("ContribInlet  found result   %p, ts %014lx, sz %d\n",
               result, result->seq.stamp().pulseId(), result->xtc.extent);

        // This method is used to handle the result datagram and to dispose of
        // the original source datagram, in other words:
        // - Possibly forward pebble to FFB
        // - Possibly forward pebble to AMI
        // - Return to pool
        // e.g.:
        //   Pebble* pebble = findPebble(result);
        //   handle(result, pebble);
        //   delete pebble;

        return 0;
      }
    private:
      EbFtBase&     _inlet;
      volatile bool _running;
    };
  };
};


using namespace Pds::Eb;

static TstContribInlet* contribInlet = NULL;

void sigHandler(int signal)
{
  if (contribInlet)  contribInlet->shutdown();

  ::exit(signal);
}


void usage(char *name, char *desc)
{
  fprintf(stderr, "Usage:\n");
  fprintf(stderr, "  %s [OPTIONS] <srv_addr> [<srv_addr> [...]]\n", name);

  if (desc)
    fprintf(stderr, "\n%s\n", desc);

  fprintf(stderr, "\nOptions:\n");

  fprintf(stderr, " %-20s %s\n", "-P <dst_port>",
          "Destination control port number (client: 32768)");

  fprintf(stderr, " %-20s %s\n", "-i <ID>",
          "Unique ID this instance should assume (0 - 63) (default: 0)");

  fprintf(stderr, " %-20s %s\n", "-h", "display this help output");
  fprintf(stderr, " %-20s %s\n", "-v", "enable debugging output");
}

int main(int argc, char **argv)
{
  int op, ret = 0;
  unsigned id = 0;
  std::string dst_port("32768");        // Connects with server port
  std::vector<std::string> dst_addr;

  while ((op = getopt(argc, argv, "hvP:i:")) != -1)
  {
    switch (op)
    {
      case 'P':  dst_port = std::string(optarg);       break;
      case 'i':  id       = strtoul(optarg, NULL, 0);  break;
      case 'v':  lverbose = true;                      break;
      case '?':
      case 'h':
        usage(argv[0], (char*)"Test event contributor");
        return 1;
    }
  }

  // Gather the list of EBs from an argument
  // - Somehow ensure this list is the same for all instances of the client
  // - Maybe pingpong this list around to see whether anyone disagrees with it?

  if (optind < argc)
  {
    do { dst_addr.push_back(std::string(argv[optind])); } while (++optind < argc);
  }
  else
  {
    fprintf(stderr, "Destination address(s) is required\n");
    return 1;
  }

  ::signal( SIGINT, sigHandler );

  unsigned numEbs      = dst_addr.size();
  size_t   contribSize = max_batches * (sizeof(Dgram) + max_entries * max_contrib_size);
  size_t   resultSize  = max_batches * (sizeof(Dgram) + max_entries * max_result_size);

  EbFtClient ftc(dst_addr, dst_port, resultSize, contribSize);
  unsigned   tmo(120);                  // Seconds
  ret = ftc.connect(tmo);
  if (ret)
  {
    fprintf(stderr, "FtClient connect() failed\n");
    return ret;
  }

  TstContribOutlet outlet(ftc, id, numEbs, batch_duration, max_batches, max_entries, max_contrib_size);
  TstContribInlet  inlet (ftc);
  contribInlet = &inlet;

  inlet.process();

  outlet.shutdown();
  ftc.shutdown();

  return ret;
}
