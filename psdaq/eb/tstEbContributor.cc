#include "psdaq/eb/BatchManager.hh"

#include "psdaq/eb/FtInlet.hh"
#include "psdaq/eb/FtOutlet.hh"

#include "psdaq/service/Routine.hh"
#include "psdaq/service/Task.hh"
#include "psdaq/service/Semaphore.hh"
#include "psdaq/service/GenericPoolW.hh"
#include "psdaq/xtc/Datagram.hh"
#include "psdaq/xtc/XtcType.hh"
#include "xtcdata/xtc/XtcIterator.hh"
#include "xtcdata/xtc/DetInfo.hh"
//#include "xtcdata/psddl/smldata.ddl.h"

#include <new>
#include <poll.h>
#include <fcntl.h>
#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

using namespace Pds;

//typedef Pds::SmlData::ProxyV1 SmlD;

static const uint64_t  batch_duration   = 0x20000; // 128 uS; should be power of 2
static const unsigned  max_batches      = 16; //8192;     // 1 second's worth
static const unsigned  max_entries      = 4;  //128;      // 128 uS worth
static const size_t    header_size      = /*sizeof(SmlD) + 2 **/ sizeof(Xtc) + sizeof(Datagram);
static const size_t    input_extent     = 5; // Revisit: Number of "L3" input data words
static const size_t    result_extent    = 2;
static const size_t    max_contrib_size = header_size + input_extent  * sizeof(uint32_t);
static const size_t    max_result_size  = header_size + result_extent * sizeof(uint32_t);

static       bool      lverbose         = false;


//static void dump(const Pds::Datagram* dg)
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

//static const TypeId  smlType((TypeId::Type)SmlD::TypeId, SmlD::Version);

namespace Pds {
  namespace Eb {

    class DrpSim : public Routine
    {
    public:
      DrpSim(unsigned poolSize, unsigned id) :
        _maxEvtSz(/*sizeof(SmlD) + 2 **/ sizeof(Xtc) + sizeof(Datagram)),
        _pool    (new GenericPoolW(_maxEvtSz, poolSize)),
        _info    (0, DetInfo::XppGon, 0, DetInfo::Wave8, id),
        _pid     (0),
        _task    (new Task(TaskObject("drp"))),
        _sem     (Semaphore::FULL)
      {
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
          clock_gettime(CLOCK_REALTIME, &ts); // Revisit: Clock won't be synchronous across machines
          Dgram dg;
          dg.seq = Sequence(Sequence::Event, TransitionId::L1Accept, ClockTime(ts), TimeStamp(0, _pid++, 0));

          // Make proxy to data (small data)
          Datagram* pdg = new (buffer) Datagram(dg, _l3SummaryType /*smlType*/, _info);
          size_t inputSize = input_extent*sizeof(uint32_t);
          //SmlD* smlD = new (pdg->xtc.next()) SmlD((char*)pdg->xtc.next()-(char*)0+sizeof(Datagram),
          //                                        _l3SummaryType, // Revisit: Replace this
          //                                        inputSize);     // Revisit: Replace this
          //pdg->xtc.alloc(smlD->_sizeof());

          // Revisit: Here is where L3 trigger input information would be inserted into the datagram,
          //          whatever that will look like.  SmlD is nominally the right container for this.
          uint32_t* payload = (uint32_t*)pdg->xtc.alloc(inputSize);
          for (unsigned i = 0; i < input_extent; ++i)
          {
            payload[i] = i;             // Revisit: Some crud for now
          }

          ::write(_fd[1], &pdg, sizeof(pdg)); // Write just the _pointer_
          if (lverbose)
            printf("DrpSim wrote event %p to fd %d\n", pdg, _fd[1]);

          _sem.take();                  // Revisit: Kludge to avoid flooding
        }
      }
    private:
      unsigned      _maxEvtSz;
      GenericPoolW* _pool;
      DetInfo       _info;
      uint64_t      _pid;
      int           _fd[2];
      Task*         _task;
      Semaphore     _sem;
    };

    class TstContribOutlet : public BatchManager,
                             public Routine
    {
    public:
      TstContribOutlet(FtOutlet& outlet,
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
          const Datagram* contrib = _pend();
          if (!contrib)  continue;

          process(contrib);
        }
      }
      void post(Batch* batch, void* arg)
      {
        unsigned dst = batchId(batch->clock()) % _numEbs;

        postTo(batch, dst, batch->index());
      }
    private:
      Datagram* _pend()
      {
        char* p;
        if (::read(_sim.fd(), &p, sizeof(p)) < 0)
        {
          perror("read");
          return NULL;
        }

        _sim.go();        // Revisit: Kludge to allow preparation of next event

        Datagram* pdg = reinterpret_cast<Datagram*>(p);

        if (lverbose)
          printf("ContribOutlet read dg %p from fd %d\n", pdg, _sim.fd());

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
      TstContribInlet(FtInlet& inlet) :
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
          // Pend for a result Datagram (batch) and handle it.
          Datagram* batch = (Datagram*)_inlet.pend();

          iterate(&batch->xtc);               // Calls process(xtc) below
        }
      }
      int process(Xtc* xtc)
      {
        //Datagram* result = (Datagram*)xtc->payload();

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
      FtInlet&      _inlet;
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

  fprintf(stderr, " %-20s %s\n", "-S <src_port>",
          "Source control port number (server: 32769)");
  fprintf(stderr, " %-20s %s\n", "-D <dst_port>",
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
  std::string src_port("32769");        // Port served to client
  std::string dst_port("32768");        // Connects with server port
  std::vector<std::string> dst_addr;

  while ((op = getopt(argc, argv, "hvS:D:i:")) != -1)
  {
    switch (op)
    {
      case 'S':  src_port = std::string(optarg);       break;
      case 'D':  dst_port = std::string(optarg);       break;
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
  size_t   contribSize = max_batches * max_entries * max_contrib_size;
  FtOutlet fto(dst_addr, dst_port);
  unsigned tmo(120);                    // Seconds
  ret = fto.connect(contribSize, tmo);
  if (ret)
  {
    fprintf(stderr, "FtOutlet connect() failed\n");
    return ret;
  }
  size_t   resultSize = max_batches * max_entries * max_result_size;
  char*    resultPool = new char[resultSize];
  FtInlet  fti(src_port);
  bool     shared(true); // Revisit: Whether peer's buffers are shared with all contributors
  ret = fti.connect(resultPool, numEbs, resultSize, shared);
  if (ret)
  {
    fprintf(stderr, "FtInlet connect() failed\n");
    delete resultPool;
    return ret;
  }

  TstContribOutlet outlet(fto, id, numEbs, batch_duration, max_batches, max_entries, max_contrib_size);
  TstContribInlet  inlet (fti);
  contribInlet = &inlet;

  inlet.process();

  outlet.shutdown();

  fto.shutdown();
  fti.shutdown();
  delete resultPool;
  return ret;
}
