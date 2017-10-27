#include "psdaq/eb/BatchManager.hh"
#include "psdaq/eb/EventBuilder.hh"
#include "psdaq/eb/EbEvent.hh"
#include "psdaq/eb/EbContribution.hh"

#include "psdaq/eb/EbFtServer.hh"

#include "psdaq/service/Routine.hh"
#include "psdaq/service/Task.hh"
#include "psdaq/service/GenericPoolW.hh"
#include "xtcdata/xtc/Dgram.hh"
#include "xtcdata/xtc/XtcIterator.hh"

#include <unistd.h>
#include <stdio.h>
#include <time.h>
#include <signal.h>
#include <stdlib.h>
#include <inttypes.h>

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

namespace Pds {
  namespace Eb {
    class TheSrc : public Src
    {
    public:
      TheSrc(unsigned id) :
        Src()
      {
        _log = id;
      }
    };

    class Result : public Dgram
    {
    public:
      Result(const Dgram& dg) : //, const TypeId& ctn, unsigned src) :
        Dgram(dg), // Revisit: , ctn, TheSrc(src)), // Revisit: Yuck!
        _dst(_dest),
        _idx(_index)
      {
      }
    public:
      void destination(unsigned dst, unsigned idx)
      {
        *_dst++ = dst;
        *_idx++ = idx;
      }
    public:
      size_t   nDsts()         const { return _dst - _dest; }
      unsigned dst(unsigned i) const { return _dest[i];     }
      unsigned idx(unsigned i) const { return _index[i];    }
    private:
      uint32_t  _reserved_0[result_extent]; // Revisit: buffer
    private:
      unsigned* _dst;
      unsigned* _idx;
    private:
#define NDST  64                        // Revisit: NDST
      unsigned  _dest[NDST];
      unsigned  _index[NDST];
    };

    class TstEbOutlet : public BatchManager,
                        public Routine
    {
    public:
      TstEbOutlet(EbFtBase& outlet,
                  unsigned  id,
                  uint64_t  duration,
                  unsigned  maxBatches,
                  unsigned  maxEntries,
                  size_t    maxSize);
      ~TstEbOutlet() { }
    public:
      void shutdown()
      {
        _running = false;

        _task->destroy();
      }
    public:
      void post(Result* result)
      {
        if (::write(_fd[1], &result, sizeof(result)))
          perror("write");
      }
    public:                             // Routine interface
      void routine();
      void post(Batch* batch, void* arg);
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

    class TstEbInlet : public EventBuilder,
                       public XtcIterator
    {
    public:
      TstEbInlet(EbFtBase&    inlet,
                 unsigned     id,
                 uint64_t     duration,
                 unsigned     maxBatches,
                 unsigned     maxEntries,
                 uint64_t     contributors,
                 TstEbOutlet& outlet);
    virtual   ~TstEbInlet() { }
    public:
      void     shutdown()  { _running = false; }
    public:
      void     process();
      int      process(Xtc* xtc);
    public:
      void     init(TstEbOutlet* outlet);
      void     process(EbEvent* event);
      uint64_t contract(Dgram* contrib) const;
      void     fixup(EbEvent* event, unsigned srcId);
    private:
      Result* _allocate(EbEvent* event);
    private:
      EbFtBase&     _inlet;
      unsigned      _id;
      const TypeId  _tag;
      uint64_t      _contract;
      GenericPoolW  _results;
      //EbDummyTC     _dummy;           // Template for TC of dummy contributions  // Revisit: ???
      TstEbOutlet&  _outlet;
      volatile bool _running;
    };

  };
};


using namespace Pds::Eb;

TstEbInlet::TstEbInlet(EbFtBase&    inlet,
                       unsigned     id,
                       uint64_t     duration,
                       unsigned     maxBatches,
                       unsigned     maxEntries,
                       uint64_t     contributors,
                       TstEbOutlet& outlet) :
  EventBuilder(maxBatches, maxEntries, duration),
  XtcIterator(),
  _inlet(inlet),
  _id(id),
  _tag(TypeId::Data, 0), //_l3SummaryType),
  _contract(contributors),
  _results(sizeof(Result), maxEntries),
  //_dummy(Level::Fragment),
  _outlet(outlet),
  _running(true)
{
  start();                              // Start the event timeout timer
}

void TstEbInlet::process()
{
  while (_running)
  {
    // Pend for an input datagram (batch) and pass it to the event builder.
    Dgram* batch = (Dgram*)_inlet.pend();

    iterate(&batch->xtc);               // Calls process(xtc) below

    _outlet.release(batch->seq.stamp().pulseId());
  }
}

int TstEbInlet::process(Xtc* xtc)
{
  Dgram* contribution = (Dgram*)xtc->payload();

  EventBuilder::process(contribution);

  return 0;
}

void TstEbInlet::process(EbEvent* event)
{
  const unsigned recThresh = 10;       // Revisit: Some crud for now
  const unsigned monThresh = 5;        // Revisit: Some crud for now

  Result*   rdg    = _allocate(event);
  uint32_t* buffer = (uint32_t*)rdg->xtc.alloc(result_extent * sizeof(uint32_t));

  // Iterate over the event and build a result datagram
  EbContribution*  creator = event->creator();
  EbContribution** contrib = event->start();
  EbContribution** end     = event->end();
  unsigned         sum     = 0;
  do
  {
    EbContribution* idg = *contrib;

    // The source of the contribution is one of the dsts
    rdg->destination(idg->number(), idg - creator);

    uint32_t* payload = (uint32_t*)idg->xtc.next();
    for (unsigned i = 0; i < result_extent; ++i)
    {
      sum += payload[i];               // Revisit: Some crud for now
    }
  }
  while (++contrib != end);

  buffer[0] = sum > recThresh ? 0 : 1; // Revisit: Some crud for now
  buffer[1] = sum > monThresh ? 0 : 1; // Revisit: Some crud for now

  _outlet.post(rdg);
}

uint64_t TstEbInlet::contract(Dgram* contrib) const
{
  // Determine the read-out group to which contrib belongs
  // Look up the expected list of contributors -> contract

  return _contract;
}

void TstEbInlet::fixup(EbEvent* event, unsigned srcId)
{
  // Revisit: Nothing can usefully be done here since there is no way to
  //          know the buffer index to be used for the result, I think

  if (lverbose)
  {
    fprintf(stderr, "%s: Called for source %d\n", __PRETTY_FUNCTION__, srcId);
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

Result* TstEbInlet::_allocate(EbEvent* event)
{
  EbContribution* contrib = event->creator();
  Dgram*          src     = contrib->datagram();

  // Due to the resource-wait memory allocator used here, zero is never returned
  Result* result = new(&_results) Result(*src); // Revisit: , _tag, _id);
  return result;
}

TstEbOutlet::TstEbOutlet(EbFtBase& outlet,
                         unsigned  id,
                         uint64_t  duration,
                         unsigned  maxBatches,
                         unsigned  maxEntries,
                         size_t    maxSize) :
  BatchManager(outlet, id, duration, maxBatches, maxEntries, maxSize),
  _running(true),
  _task (new Task(TaskObject("tOutlet")))
{
  if (pipe(_fd))
    perror("pipe");

  _task->call(this);
}

void TstEbOutlet::routine()
{
  while (_running)
  {
    // Pend for an output datagram (result) from the event builder.  When
    // it arrives, process it into a datagram (batch).  When the datagram
    // is complete, post it to all contributors.
    // This could be done in the same thread as the that of the EB.
    Result* result = _pend();
    process(result, result);            // Revisit: 2nd arg
  }
}

void TstEbOutlet::post(Batch* batch, void* arg)
{
  Result*  result = (Result*)arg;
  unsigned nDsts  = result->nDsts();

  for (unsigned dst = 0; dst < nDsts; ++dst)
  {
    postTo(batch, result->dst(dst), result->idx(dst));
  }
}


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

  fprintf(stderr, " %-20s %s\n", "-P <src_port>",
          "Source control port number (server: 32768)");

  fprintf(stderr, " %-20s %s\n", "-i <ID>",
          "Unique ID this instance should assume (0 - 63) (default: 0)");

  fprintf(stderr, " %-20s %s\n", "-h", "display this help output");
  fprintf(stderr, " %-20s %s\n", "-v", "enable debugging output");
}

int main(int argc, char **argv)
{
  int op, ret = 0;
  unsigned id = 0;
  std::string src_port("32768");        // Port served to client
  std::vector<std::string> dst_addr;

  while ((op = getopt(argc, argv, "hvP:i:")) != -1)
  {
    switch (op)
    {
      case 'P':  src_port = std::string(optarg);  break;
      case 'i':  id       = atoi(optarg);         break;
      case 'v':  lverbose = true;                 break;
      case '?':
      case 'h':
      default:
        usage(argv[0], (char*)"Test event builder");
        return 1;
    }
  }

  uint64_t clients = 0;
  unsigned srcId   = 0;
  if (optind < argc)
  {
    do
    {
      dst_addr.push_back(std::string(argv[optind]));
      clients |= 1 << srcId++;          // Revisit srcId value
    }
    while (++optind < argc);
  }
  else
  {
    fprintf(stderr, "Destination address(s) is required\n");
    return 1;
  }

  ::signal( SIGINT, sigHandler );

  unsigned numContribs = dst_addr.size();
  size_t   contribSize = max_batches * (sizeof(Dgram) + max_entries * max_contrib_size);
  size_t   resultSize  = max_batches * (sizeof(Dgram) + max_entries * max_result_size);

  EbFtServer fts(src_port, numContribs, contribSize, resultSize);
  ret = fts.connect();
  if (ret)
  {
    fprintf(stderr, "FtServer connect() failed\n");
    return ret;
  }

  TstEbOutlet outlet(fts, id, batch_duration, max_batches, max_entries, max_result_size);
  TstEbInlet  inlet (fts, id, batch_duration, max_batches, max_entries, clients, outlet);
  ebInlet = &inlet;

  inlet.process();

  outlet.shutdown();
  fts.shutdown();

  return ret;
}
