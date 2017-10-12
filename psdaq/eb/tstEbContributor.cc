#include "psdaq/eb/BatchManager.hh"
#include "psdaq/eb/BatchHandler.hh"

#include "psdaq/xtc/Datagram.hh"
#include "psdaq/service/Routine.hh"
#include "psdaq/service/Task.hh"
#include "pdsdata/xtc/DetInfo.hh"
#include "pdsdata/psddl/smldata.ddl.h"

#include <new>
#include <poll.h>
#include <fcntl.h>
#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

typedef Pds::SmlData::ProxyV1 SmlD;

static const ClockTime batch_duration(0x0, 0x20000); // 128 uS; should be power of 2
static const unsigned  max_batches       = 8192;     // 1 second's worth
static const unsigned  max_batch_entries = 128;      // 128 uS worth
static const size_t    max_contrib_size  = sizeof(SmlD) + 2 * sizeof(Xtc) + sizeof(Datagram);
static const size_t    max_result_size   = max_contrib_size + 2 * sizeof(uint32_t);


static void dump(const Pds::Datagram* dg)
{
  char buff[128];
  time_t t = dg->seq.clock().seconds();
  strftime(buff,128,"%H:%M:%S",localtime(&t));
  printf("[%p] %s.%09u %016lx %s extent 0x%x damage %x\n",
         dg,
         buff,
         dg->seq.clock().nanoseconds(),
         dg->seq.stamp().pulseID(),
         Pds::TransitionId::name(Pds::TransitionId::Value((dg->seq.stamp().fiducials()>>24)&0xff)),
         dg->xtc.extent, dg->xtc.damage.value());
}

static const TypeId  smlType((TypeId::Type)SmlD::TypeId, SmlD::Version);

namespace Pds {
  namespace Eb {

    class DrpSim : public Routine
    {
    public:
      DrpSim(unsigned poolSize, unsigned id) :
        _maxEvtSz(sizeof(SmlD) + 2 * sizeof(Xtc) + sizeof(Datagram)),
        _pool    (new RingPool(poolSize, _maxEvtSz)),
        _info    (0, DetInfo::XppGon, 0, DetInfo::Wave8, id),
        _pid     (0),
        _task    (new Task(TaskObject("drp")))
      {
        if (pipe(_fd)) perror("pipe");
        _task->call(this);
      }
      ~DrpSim() { delete _pool; }
    public:
      // poll interface
      int   fd() const { return _fd[0]; }
    public:
      //  data interface
      void* alloc(unsigned minSize) {  return _pool->alloc(minSize); }
    public:
      void routine()
      {
        const unsigned extent = 5;      // Revisit: Number of "L3" input data words

        while(1)
        {
          char* buffer = (char*)alloc(_maxEvtSz);
          if (buffer==0)  continue;

          // Fake datagram header
          struct timespec ts;
          clock_gettime(CLOCK_REALTIME, &ts); // Revisit: Clock won't be synchronous across machines
          Dgram dg;
          dg.seq = Sequence(Sequence::Event, TransitionId::L1Accept, ClockTime(ts), TimeStamp(_pid++));

          // Make proxy to data (small data)
          Datagram* pdg = new (buffer) Datagram(dg, smlType, _info);
          SmlD* smlD = new (pdg->xtc.next()) SmlD((char*)pdg->xtc.next()-(char*)0+sizeof(Datagram),
                                                  genType,                  // Revisit: Replace this
                                                  extent*sizeof(uint32_t)); // Revisit: Replace this
          pdg->xtc.alloc(smlD->_sizeof());

          // Revisit: Here is where L3 trigger input information would be inserted into the datagram,
          //          whatever that will look like.  SmlD is nominally the right container for this.
          uint32_t* payload = pdg->xtc.alloc(extent*sizeof(uint32_t));
          for (unsigned i = 0; i < extent; ++i)
          {
            payload[i] = i;             // Revisit: Some crud for now
          }

          ::write(_fd[1], &pdg, sizeof(pdg)); // Write just the _pointer_
          if (lverbose)
            printf("DrpSim wrote event %p to fd %d\n", pdg, _fd[1]);
        }
      }
    private:
      unsigned             _maxEvtSz;
      RingPool*            _pool;
      DetInfo              _info;
      uint64_t             _pid;
      int                  _fd[2];
      Task*                _task;
    };

    class TstContribOutlet : public BatchManager
    {
    public:
      TstContribOutlet(std::vector<std::string>& remote,
                       std::string&              port,
                       unsigned                  id) :
        BatchManager(remote, port, id, batch_duration, max_batches, max_batch_entries),
        _sim(id),                       // Simulate DRP in a separate thread
        _running(true)
      {
      }
      ~TstContribOutlet() { }
    public:
      void shutdown()  { _running = false; }
    public:
      void process()
      {
        while (_running)
        {
          Datagram* contrib = _pend();
          if (!contrib)  continue;

          process(contrib);
        }
      }
      void post(cont Batch* batch)
      {
        unsigned dst = _batchId(batch->clock()) % _numEbs;

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
        else
        {
          Datagram* pdg = reinterpret_cast<Datagram*>(p);

          if (lverbose)
            printf("ContribOutlet read dg %p from fd %d\n", pdg, _sim.fd());

          return pdg;
        }
      }
    private:
      DrpSim        _sim;
      volatile bool _running;
    };

    class TstContribInlet : public BatchHandler,
                            public Routine
    {
    public:
      TstContribInlet(std::vector<std::string>& remote,
                      std::string&              port) :
        BatchHandler(port, remote.size(), 1, max_batches, max_result_size),
        _running(true),
        _task (new Task(TaskObject("tInlet")))
      {
        _task->call(this);
      }
      ~TstContribInlet() { }
    public:
      void shutdown()  { _running = false; }
    public:
      void routine()
      {
        while (_running)
        {
          // Pend for a result Datagram (batch) and handle it.
          pend();
        }

        BatchHandler::shutdown();
      }
      void process(Datagram* result)
      {
        // This method is used to iterate over the results batch to determine how to dispose of the
        // original datagram, e.g.:
        // - Possibly forward to FFB
        // - Possibly forward to AMI
        // - Return to pool
        // i.e.:
        // class AnXtcIterator : public XtcIterator
        // {
        // public:
        //   AnXtcIterator() : XtcIterator() {}
        //   virtual ~AnXtcIterator() {}
        // public:
        //   virtual int process(Xtc* xtc)
        //   {
        //     Datagram* eventResult = (Datagram*)xtc->payload();
        //     _handle(eventResult);
        //     delete findPebble(eventResult);
        //     return 0;
        //   }
        // };
        // AnXtcIterator xtc();
        // xtc.iterate(result->xtc);

        release(result);
      }
    private:
      volatile bool _running;
      Task*         _task;
    };
  };
};

using namespace Pds::Eb;

static bool              lverbose      = false;
static TstContribOutlet* contribOutlet = NULL;

void sigHandler(int signal)
{
  if (contribOutlet)  contribOutlet->shutdown();

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
          "Destination control port number (client: 47592)");

  fprintf(stderr, " %-20s %s\n", "-i <ID>",
          "Unique ID this instance should assume (0 - 63) (default: 0)");

  fprintf(stderr, " %-20s %s\n", "-h", "display this help output");
  fprintf(stderr, " %-20s %s\n", "-v", "enable debugging output");
}

int main(int argc, char **argv)
{
  int op, ret = 0;
  unsigned id = 0;
  std::string dst_port;
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

  TstContribInlet  inlet (dst_addr, dst_port);
  TstContribOutlet outlet(dst_addr, dst_port, id);
  contribOutlet = &outlet;

  outlet.process();

  inlet.shutdown();
  inlet.join();                          // Revisit...
  return -ret;
}
