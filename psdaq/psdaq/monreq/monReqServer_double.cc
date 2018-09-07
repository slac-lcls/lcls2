#include "xtcdata/app/XtcMonitorServer.hh"

#include <errno.h>
#include <unistd.h>
#include <sys/types.h>
#include <time.h>
#include <stdlib.h>

#include "psdaq/eb/EbAppBase.hh"

#include "psdaq/eb/MonLfClient.hh"

#include "psdaq/service/Task.hh"
#include <vector>

#include <time.h>
#include <math.h>

static const unsigned epoch_duration = 1;
static const unsigned sizeof_ctrbs   = 10*1024*1024;


using namespace XtcData;
using namespace Pds::Eb;

namespace Pds {
  class MyXtcMonitorServer : public XtcMonitorServer,
                             public EbAppBase {
  public:
    MyXtcMonitorServer(const char*  tag,
                       unsigned     sizeofBuffers,
                       unsigned     numberofEvBuffers,
                       unsigned     numberofEvQueues,
                       const char*  ifAddr,
                       std::string& port,
                       unsigned     id,
                       uint64_t     duration,
                       unsigned     maxBatches,
                       uint64_t     contributors) :
      XtcMonitorServer(tag,
                       sizeofBuffers,
                       numberofEvBuffers,
                       numberofEvQueues),
      EbAppBase(ifAddr, port,           // EbLfServer params to Contributor link
                id,                     // This monitor's ID
                epoch_duration,         // Minimal, I think
                maxBatches,             // 1 per Mon client, I think
                numberofEvBuffers,      // Number of events to hold
                sizeof_ctrbs,           // Size of data contributions
                contributors),          // Same value as L3 EBs
      _nL3Eb(0),
      _l3Link(nullptr),
      _bufFreeList(),
      _pool(sizeof(Dgram) + __builtin_popcountl(contributors) * sizeof(Dgram*),
            nBuffers)                   // Revisit: nBuffers
    {
      _init();
    }
    ~MyXtcMonitorServer()
    {
      if (_l3Link)  delete _l3Link;
    }
  public:

    void run(std::vector<std::string>& addrs,
             std::vector<std::string>& ports)
    {
      //
      //  Set up connections to the L3 Event Builders
      //
      _nL3Eb  = addrs.size();
      _iL3Eb  = 0;
      _l3Link = new MonLfClient(addrs, ports);

      const unsigned tmo(120);              // Seconds
      if (_l3Link->connect(tmo))
      {
        fprintf(stderr, "monReqServer: L3 client connect failed\n");
        abort();
      }

      for (unsigned i = 0; i < _numberofEvBuffers; ++i)
        _bufFreeList.push(i);

      //
      //  Poll for events (forever)
      //
      EbAppBase::process();
    }

    void process(EbEvent* event)
    {
      void* buffer = _pool.alloc(sizeof(Dgram));
      if (!buffer)
      {
        fprintf(stderr, "%s: Dgram pool allocation failed", __PRETTY_FUNCTION__);
        _pool.dump();
        abort();
      }
      Dgram*  dg  = new(buffer) Dgram(*(event->creator()));
      size_t  cnt = event->end() - event->begin();
      Dgram** buf = dg->xtc.alloc(cnt * sizeof(*(event->begin())));
      memcpy(buf, event->begin(), cnt * sizeof(*(event->begin())));

      // Revisit:
      //if( !dg->seq.isEvent() && j != 1 ) {
      //  delete[] p;
      //}
      //else {
      //  if (XtcMonitorServer::events((Dgram*)p) == XtcMonitorServer::Deferred) {
      //    continue;
      //  }
      //  else {
      //    delete[] p;
      //  }
      //}

      if (XtcMonitorServer::events(dg) == XtcMonitorServer::Handled)
        delete dg;
    }

    void shutdown()
    {
      if (_monLink)  _monLink->shutdown();

      EbAppBase::shutdown();
    }

  private:
    void _copyDatagram(/*const*/ Dgram* dg, char* buf)
    {
      Dgram* odg = new((void*)buf) Dgram(*dg);

      // Iterate over the event and build a result datagram
      const Dgram** const  last = (const Dgram**)dg->xtc.next();
      const Dgram*  const* ctrb = (const Dgram**)dg->xtc.payload();
      do
      {
        const Dgram* idg = *ctrb;

        buf = odg->xtc.alloc(idg->xtc.extent);

        if (sizeof(*odg) + odg->xtc.sizeofPayload() > _sizeOfBuffers)
          exit(1);            // The memcpy would blow by the buffer size limit

        memcpy(buf, &idg->xtc, idg->xtc.extent);
      }
      while (++contrib != last);

      _bufFreeList.push(dg->xtc.src.phy() & ((1 << 24) - 1));  // Revisit: Thread safety
    }

    void _deleteDatagram(Dgram* dg)
    {
      delete dg;
    }

    void _requestDatagram()
    {
      //printf("_requestDatagram\n");

      if (_bufFreeList.size() == 0)
      {
        fprintf(stderr, "%s: No free buffers available\n", __PRETTY_FUNCTION__);
        return;
      }

      unsigned idx = _bufFreeList.pop();  // Revisit: Thread safety

      int rc = -1;
      for (unsigned i = 0; i < _nL3Eb; ++i)
      {
        unsigned iL3Eb = _iL3Eb++;
        if (_iL3Eb == _nL3Eb)  _iL3Eb = 0;

        rc = _l3Link->post(iL3Eb, (_id << 24) | idx);
        if (rc == 0)  break;
      }
      if (rc)
      {
        fprintf(stderr, "%s:Unable to post request to L3 EB(s)\n", __PRETTY_FUNCTION__);
      }
    }

  private:
    unsigned             _nL3Eb;
    MonLfClient*         _l3Link;
    GenericPool          _pool;
    std::queue<unsigned> _bufFreeList;
  private:
    unsigned             _iL3Eb;
  };
};

using namespace Pds;


void usage(char* progname) {
  printf("Usage: %s -p <platform> -P <partition> -i <node mask> -n <numb shm buffers> -s <shm buffer size> [-q <# event queues>] [-t <tag name>] [-d] [-c] [-g <max groups>] [-h]\n", progname);
}

int main(int argc, char** argv) {

  const unsigned NO_PLATFORM = unsigned(-1UL);
  unsigned platform=NO_PLATFORM;
  const char* partition = 0;
  const char* tag = 0;
  int numberOfBuffers = 0;
  unsigned sizeOfBuffers = 0;
  unsigned nevqueues = 1;
  unsigned node =  0xffff;
  unsigned nodes = 6;
  bool ldist = false;

  int c;
  while ((c = getopt(argc, argv, "p:i:g:n:P:s:q:t:dch")) != -1) {
    errno = 0;
    char* endPtr;
    switch (c) {
      case 'p':
        platform = strtoul(optarg, &endPtr, 0);
        if (errno != 0 || endPtr == optarg) platform = NO_PLATFORM;
        break;
      case 'i':
        node = strtoul(optarg, &endPtr, 0);
        break;
      case 'g':
        nodes = strtoul(optarg, &endPtr, 0);
        break;
      case 'n':
        sscanf(optarg, "%d", &numberOfBuffers);
        break;
      case 'P':
        partition = optarg;
        break;
      case 't':
        tag = optarg;
        break;
      case 'q':
        nevqueues = strtoul(optarg, NULL, 0);
        break;
      case 's':
        sizeOfBuffers = (unsigned) strtoul(optarg, NULL, 0);
        break;
      case 'd':
        ldist = true;
        break;
      case 'h':
        // help
        usage(argv[0]);
        return 0;
        break;
      default:
        printf("Unrecogized parameter\n");
        usage(argv[0]);
        break;
    }
  }

  if (!numberOfBuffers || !sizeOfBuffers || platform == NO_PLATFORM || !partition || node == 0xffff) {
    fprintf(stderr, "Missing parameters!\n");
    usage(argv[0]);
    return 1;
  }

  if (numberOfBuffers<8) numberOfBuffers=8;

  if (!tag) tag=partition;

  printf("\nPartition Tag:%s\n", tag);

  MyXtcMonitorServer* apps = new MyXtcMonitorServer(tag,
                                                    sizeOfBuffers,
                                                    numberOfBuffers,
                                                    nevqueues);
  apps->distribute(ldist);

  apps->run(node, platform);              //pass variable node and platform

  apps->shutdown();

  return 0;
}
