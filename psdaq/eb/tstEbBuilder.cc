#include "psdaq/eb/EventBuilder.hh"

#include "psdaq/xtc/Datagram.hh"
#include "psdaq/xtc/XtcType.hh"
#include "psdaq/service/Routine.hh"
#include "psdaq/service/Task.hh"
#include "pdsdata/psddl/smldata.ddl.h"

#include <unistd.h>
#include <stdio.h>
#include <time.h>
#include <signal.h>
#include <stdlib.h>
#include <inttypes.h>

typedef Pds::SmlData::ProxyV1 SmlD;

static const ClockTime batch_duration(0x0, 0x20000); // 128 uS; must be power of 2
static const unsigned  max_batches       = 8192;     // 1 second's worth
static const unsigned  max_entries       = 128;      // 128 uS worth
static const size_t    max_contrib_size  = sizeof(SmlD) + 2 * sizeof(Xtc) + sizeof(Datagram);
static const size_t    max_result_size   = max_contrib_size + 2 * sizeof(uint32_t);

static void dump(const Pds::Datagram* dg)
{
  char buff[128];
  time_t t = dg->seq.clock().seconds();
  strftime(buff,128,"%H:%M:%S",localtime(&t));
  printf("%s.%09u %016lx %s extent 0x%x ctns %s damage %x\n",
         buff,
         dg->seq.clock().nanoseconds(),
         dg->seq.stamp().pulseID(),
         Pds::TransitionId::name(dg->seq.service()),
         dg->xtc.extent,
         Pds::TypeId::name(dg->xtc.contains.id()),
         dg->xtc.damage.value());
}

static const TypeId  smlType((TypeId::Type)SmlD::TypeId, SmlD::Version);

namespace Pds {
  namespace Eb {

    class Result : public Datagram
    {
    public:
      Result(const Datagram& dg, const TypeId& ctn, const Src& src) :
        Datagram(dg, ctn, src),
        _dst(_dest),
        _idx(_index)
      {
      }
    public:
      void destination(unsigned dst, unsigned idx)
      {
        _dst++ = dst;
        _idx++ = idx;
      }
    public:
      size_t nDsts() const { return _dst - _dest; }
    private:
      // Revisit: reserve space for the payload here, I think
      Xtc       _reserved_0;
      SmlD      _reserved_1;
      uint32_t  _reserved_2[2];         // Revisit: buffer
    private:
      unsigned* _dst;
      unsigned* _idx;
    private:
#define NDST  64                        // Revisit: NDEST
      unsigned  _dest[NDST];
      unsigned  _index[NDST];
    };

    class TstEventBuilder : public EventBuilder
    {
    public:
      TstEventBuilder(const Src&   id,
                      TstEbOutlet& outlet,
                      unsigned     epochs,
                      unsigned     entries,
                      uint64_t     epochMask,
                      uint64_t     contributors) :
        EventBuilder(epochs, entries, epochMask),
        _id(id),
        _tag(_l3SummaryType),
        _contract(contributors),
        _results(sizeof(Result), entries),
        _outlet(outlet),
        _dummy(Level::Fragment)
      {
      }
      virtual ~TstEventBuilder() { }
    public:
      void process(EbEvent* event)
      {
        const unsigned recThresh = 10; // Revisit: Some crud for now
        const unsigned monThresh = 5;  // Revisit: Some crud for now
        const unsigned rExtent   = 2;  // Revisit: Number of "L3" result data words

        Result* rdg  = _allocate(event);
        SmlD*   smlD = new(rdg->xtc.next()) SmlD((char*)rdg->xtc.next()-(char*)0+sizeof(Datagram),
                                                 genType,                   // Revisit: Replace this
                                                 rExtent*sizeof(uint32_t)); // Revisit: Define this
        rdg->xtc.alloc(smlD->_sizeof());
        uint32_t* buffer = rdg->xtc.alloc(rExtent * sizeof(uint32_t));

        // Iterate over the event and build a result datagram
        EbContribution*  creator = event->creator();
        EbContribution** contrib = event->_tail;
        EbContribution** end     = event->_head;
        unsigned         sum     = 0;
        do
        {
          EbContribution* idg = *contrib;

          // The source of the contribution is one of the dsts
          rdg->destination(idg->number(), idg - creator);

          SmlD*     input   = (SmlD*)idg->xtc.payload(); // Contains the input data
          uint32_t* payload = (uint32_t*)idg->xtc.next();
          unsigned  iExtent = input->extent();
          for (unsigned i = 0; i < iExtent; ++i)
          {
            sum += payload[i];          // Revisit: Some crud for now
          }
        }
        while (++contrib != end);

        buffer[0] = sum > recThresh ? 0 : 1; // Revisit: Some crud for now
        buffer[1] = sum > monThresh ? 0 : 1; // Revisit: Some crud for now

        _outlet.post(rdg);
      }
      uint64_t contract(EbContribution* contrib)
      {
        // Determine the read-out group to which contrib belongs
        // Look up the expected list of contributors -> contract

        return _contract;
      }
      void fixup(EbEvent* event, unsigned srcId)
      {
        // Revisit: I think there is nothing that can usefully be done here
        //          since there is no way to know the buffer index to be
        //          used for the result

        if (lverbose)
        {
          fprintf(stderr, "%s: Called for source %d\n", __PRETTY_FUNCTION__, srcId);
        }

        //Datagram* datagram = (Datagram*)event->data();
        //Damage    dropped(1 << Damage::DroppedContribution);
        //Src       source(srcId, 0);
        //
        //datagram->xtc.damage.increase(Damage::DroppedContribution);
        //
        //new(&datagram->xtc.tag) Xtc(_dummy, source, dropped); // Revisit: No tag, no TC
      }
    private:
      Result* _allocate(EbEvent* event)
      {
        EbContributor* contrib = event->creator();
        Datagram*      src     = contrib->datagram();

        // Due to the resource-wait memory allocator used here, zero is never returned
        Result* result = new(&_results) Result(src, _tag, _id);
        return result;
      }
    private:
      Src          _id;
      TypeId&      _tag;
      uint64_t     _contract;
      GenericPoolW _results;
      TstEbOutlet& _outlet;
      //EbDummyTC    _dummy;            // Template for TC of dummy contributions  // Revisit: ???
    };

    class TstEbInlet : public BatchHandler
    {
    public:
      TstEbInlet(std::vector<std::string>& remote,
                 std::string&              port,
                 const Src&                id,
                 uint64_t                  clients,
                 TstEbOutlet&              outlet) :
        BatchHandler(port, remote.size(), remote.size(), num_batches, max_contrib_size),
        _eventBuilder(id, outlet, max_batches, max_entries, batch_duration, clients),
        _xtc(_eventBuilder),
        _running(true)
      {
      }
      virtual ~TstEbInlet() { }
    public:
      void start()     { _eventBuilder.start(); } // Start the event timeout timer
      void shutdown()  { _running = false; }
    public:
      void process()
      {
        while (_running)
        {
          // Pend for an input Datagram (batch) and pass it to the event builder.
          // Here is where the Datagram could be posted to an EB running in another
          // thread.  Instead, it is handled by the EB in the current thread.
          pend();
        }

        BatchHandler::shutdown();
      }
      int process(Datagram* contribution)
      {
        _eventBuilder.process(contribution);

        return 0;
      }
    private:
      TstEventBuilder _eventBuilder;
      volatile bool   _running;
    };

    class TstEbOutlet : public BatchManager,
                        public Routine
    {
    public:
      TstEbOutlet(std::vector<std::string>& remote,
                  std::string&              port,
                  unsigned                  id) :
        BatchManager(remote, port, id, batch_duration, max_batches, max_entries),
        _running(true),
        _task (new Task(TaskObject("tOutlet")))
      {
        if (pipe(_fd))
          perror("pipe");

        _task->call(this);
      }
      ~TstEbOutlet() { }
    public:
      void shutdown()  { _running = false; }
    public:
      void post(Result* result)
      {
        if (::write(_fd[1], &result, sizeof(result)))
          perror("write");
      }
    public:                             // Routine interface
      void routine()
      {
        while (_running)
        {
          // Pend for an output datagram (result) from the event builder.  When
          // it arrives, process it into a Datagram (batch).  When the datagram
          // is complete, post it to all contributors.
          // This could be done in the same thread as the that of the EB.
          Result* result = _pend();
          process(result->datagram(), result);
        }
      }
      void post(const Batch* batch, void* arg)
      {
        Result*  result = (Result*)arg;
        unsigned nDsts  = result->nDsts();

        for (unsigned dst = 0; dst < nDsts; ++dst)
        {
          postTo(batch, result->dst(dst), result->idx(dst));
        }
      }
    private:
      Result* _pend()
      {
        Result* result;

        if (::read(_fd[0], &result, sizeof(result)))
          perror("read");

        return result;
      }
    private:
      volatile bool _running;
      int           _fd[2];
      Task*         _task;
    };
  };
};

using namespace Pds::Eb;

static bool        lverbose = false;
static TstEbInlet* ebInlet  = NULL;

void sigHandler( int signal )
{
  if (ebInlet)  ebInlet->shutdown();

  ::exit(signal);
}

void usage(char *name, char *desc)
{
  fprintf(stderr, "Usage:\n");
  fprintf(stderr, "  %s [OPTIONS] <clt_addr> [<clt_addr> [...]]\n", name);

  if (desc)
    fprintf(stderr, "\n%s\n", desc);

  fprintf(stderr, "\nOptions:\n");

  fprintf(stderr, " %-20s %s\n", "-P <dst_port>",
          "destination control port number (client: 47592)");

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
      case 'P':  dst_port = std::string(optarg);  break;
      case 'i':  id       = atoi(optarg);         break;
      case 'v':  lverbose = true;                 break;
      case '?':
      case 'h':
      default:
        usage(argv[0], (char*)"Test event builder");
        return 1;
    }
  }

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

  TstEbOutlet outlet(dst_addr, dst_port, id);
  TstEbInlet  inlet (dst_addr, dst_port, outlet);
  ebOutlet = &outlet;
  inlet.start();                        // Start event timeout timer

  outlet.process();

  inlet.shutdown();
  inlet.join();                         // Revisit...
  return -ret;
}
