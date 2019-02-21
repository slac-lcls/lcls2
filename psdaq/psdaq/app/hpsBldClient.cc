//
//  Merge the multicast BLD with the timing stream from a TPR
//
#include <stdio.h>
#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <strings.h>
#include <pthread.h>
#include <string>
#include <signal.h>

#include "psdaq/tpr/Client.hh"
#include "psdaq/tpr/Frame.hh"
#include "psdaq/bld/Client.hh"
#include "psdaq/bld/TestType.hh"
#include "xtcdata/xtc/Dgram.hh"
#include "AppUtils.hh"

void usage(const char* p) {
  printf("Usage: %s [options]\n",p);
  printf("Options: -a <ip mcast address, dotted notation>\n");
  printf("         -i <ip interface, name or dotted notation>\n");
  printf("         -p <UDP port>\n");
}

static int      event = 0;
static int64_t  bytes = 0;
static unsigned misses = 0;

static Pds::Tpr::Client* tpr = 0;

void* countThread(void* args)
{
  timespec tv;
  clock_gettime(CLOCK_REALTIME,&tv);
  unsigned oevent = event;
  int64_t  obytes = bytes;
  unsigned omisses = misses;
  while(1) {
    usleep(1000000);
    timespec otv = tv;
    clock_gettime(CLOCK_REALTIME,&tv);
    unsigned nevent = event;
    int64_t  nbytes = bytes;
    unsigned nmisses = misses;

    double dt     = double( tv.tv_sec - otv.tv_sec) + 1.e-9*(double(tv.tv_nsec)-double(otv.tv_nsec));
    double erate  = double(nevent-oevent)/dt;
    double mrate  = double(nmisses-omisses)/dt;
    double dbytes = double(nbytes-obytes)/dt;
    double ebytes = dbytes/erate;
    unsigned dbsc = 0, ersc=0, mrsc=0, ebsc=0;
    
    static const char scchar[] = { ' ', 'k', 'M' };

    if (erate > 1.e6) {
      ersc     = 2;
      erate   *= 1.e-6;
    }
    else if (erate > 1.e3) {
      ersc     = 1;
      erate   *= 1.e-3;
    }

    if (mrate > 1.e6) {
      mrsc     = 2;
      mrate   *= 1.e-6;
    }
    else if (mrate > 1.e3) {
      mrsc     = 1;
      mrate   *= 1.e-3;
    }

    if (dbytes > 1.e6) {
      dbsc    = 2;
      dbytes *= 1.e-6;
    }
    else if (dbytes > 1.e3) {
      dbsc    = 1;
      dbytes *= 1.e-3;
    }

    if (ebytes > 1.e6) {
      ebsc    = 2;
      ebytes *= 1.e-6;
    }
    else if (ebytes > 1.e3) {
      ebsc    = 1;
      ebytes *= 1.e-3;
    }
      
    printf("Events %7.2f %cHz [%u]:  Size %7.2f %cBps (%7.2f %cB/evt): Misses %7.2f %cHz\n", 
           erate , scchar[ersc], nevent, 
           dbytes, scchar[dbsc], 
           ebytes, scchar[ebsc],
           mrate , scchar[mrsc]);

    oevent = nevent;
    obytes = nbytes;
    omisses = nmisses;
  }
  return 0;
}

static void sigHandler(int signal)
{
  psignal(signal, "bld_client received signal");
  tpr->stop();
  ::exit(signal);
}

int main(int argc, char* argv[])
{
  extern char* optarg;
  char c;

  unsigned addr = 0;
  unsigned intf = 0;
  unsigned short port = 8197;
  unsigned partn = 0;
  bool lverbose = false;
  unsigned nprint = 10;
  
  while ( (c=getopt( argc, argv, "a:i:p:P:v")) != EOF ) {
    switch(c) {
    case 'a':
      addr = Psdaq::AppUtils::parse_ip       (optarg);
      break;
    case 'i':
      intf = Psdaq::AppUtils::parse_interface(optarg);
      break;
    case 'p':
      port = strtoul(optarg,NULL,0);
      break;
    case 'P':
      partn = strtoul(optarg,NULL,0);
      break;
    case 'v':
      lverbose = true;
      break;
    default:
      usage(argv[0]);
      return 0;
    }
  }

  if (!addr || !intf) {
    usage(argv[0]);
    return -1;
  }

  //
  //  Open the timing receiver
  //
  tpr = new Pds::Tpr::Client("/dev/tpra");
  //
  //  Open the bld receiver
  //
  Pds::Bld::Client  bld(intf, addr, port);

  pthread_attr_t tattr;
  pthread_attr_init(&tattr);
  pthread_t thr;
  if (pthread_create(&thr, &tattr, &countThread, 0)) {
    perror("Error creating read thread");
    return -1;
  }

  struct sigaction sa;
  sa.sa_handler = sigHandler;
  sa.sa_flags = SA_RESETHAND;

  sigaction(SIGINT ,&sa,NULL);
  sigaction(SIGABRT,&sa,NULL);
  sigaction(SIGKILL,&sa,NULL);
  sigaction(SIGSEGV,&sa,NULL);

  tpr->start(partn);
  //  tpr->start();

  char* eventb = new char[ sizeof(XtcData::Dgram)+sizeof(Pds::Bld::TestType) ];
  uint64_t ppulseId=0;

  while(1) {

    XtcData::Dgram* dgram = reinterpret_cast<XtcData::Dgram*>(eventb);
    XtcData::Xtc&   xtc   = *new((char*)&dgram->xtc) 
      XtcData::Xtc(XtcData::TypeId(XtcData::TypeId::Data, 0));
    
    //  First, fetch BLD component
    uint64_t pulseId = bld.fetch((char*)xtc.alloc(sizeof(Pds::Bld::TestType)), 
                                 sizeof(Pds::Bld::TestType));
    
    //  Second, fetch header (should already be waiting)
    const Pds::Tpr::Frame* frame = tpr->advance(pulseId);

    if (frame) {
      ppulseId = pulseId;
      new (&dgram->seq) XtcData::Sequence(XtcData::Sequence::Event, 
                                          XtcData::TransitionId::L1Accept,
                                          *reinterpret_cast<const XtcData::TimeStamp*>(&frame->timeStamp),
                                          XtcData::PulseId(frame->pulseId));
      event++;
      bytes += sizeof(XtcData::Dgram)+sizeof(Pds::Bld::TestType);
      if (lverbose)
        printf(" %9u.%09u %016lx extent 0x%x payload %08x %08x...\n",
               dgram->seq.stamp().seconds(),
               dgram->seq.stamp().nanoseconds(),
               dgram->seq.pulseId().value(),
               dgram->xtc.extent,
               reinterpret_cast<uint32_t*>(dgram->xtc.payload())[0],
               reinterpret_cast<uint32_t*>(dgram->xtc.payload())[1]);
    }
    else {
      misses++;
      if (nprint) {
        printf("Miss: %016lx  prev %016lx\n",
               pulseId, ppulseId);
      }
    }
  }

  pthread_join(thr,NULL);

  return 0;
}

