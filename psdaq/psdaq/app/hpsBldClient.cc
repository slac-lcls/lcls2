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

static int      count = 0;
static int      event = 0;
static int64_t  bytes = 0;
static unsigned lanes = 0;

static Pds::Tpr::Client* tpr = 0;

void* countThread(void* args)
{
  timespec tv;
  clock_gettime(CLOCK_REALTIME,&tv);
  unsigned ocount = count;
  unsigned oevent = event;
  int64_t  obytes = bytes;
  while(1) {
    usleep(1000000);
    timespec otv = tv;
    clock_gettime(CLOCK_REALTIME,&tv);
    unsigned ncount = count;
    unsigned nevent = event;
    int64_t  nbytes = bytes;

    double dt     = double( tv.tv_sec - otv.tv_sec) + 1.e-9*(double(tv.tv_nsec)-double(otv.tv_nsec));
    double rate   = double(ncount-ocount)/dt;
    double erate  = double(nevent-oevent)/dt;
    double dbytes = double(nbytes-obytes)/dt;
    double tbytes = dbytes/rate;
    unsigned dbsc = 0, rsc=0, ersc=0, tbsc=0;
    
    if (count < 0) break;

    static const char scchar[] = { ' ', 'k', 'M' };
    if (rate > 1.e6) {
      rsc     = 2;
      rate   *= 1.e-6;
    }
    else if (rate > 1.e3) {
      rsc     = 1;
      rate   *= 1.e-3;
    }

    if (erate > 1.e6) {
      ersc     = 2;
      erate   *= 1.e-6;
    }
    else if (erate > 1.e3) {
      ersc     = 1;
      erate   *= 1.e-3;
    }

    if (dbytes > 1.e6) {
      dbsc    = 2;
      dbytes *= 1.e-6;
    }
    else if (dbytes > 1.e3) {
      dbsc    = 1;
      dbytes *= 1.e-3;
    }
    
    if (tbytes > 1.e6) {
      tbsc    = 2;
      tbytes *= 1.e-6;
    }
    else if (tbytes > 1.e3) {
      tbsc    = 1;
      tbytes *= 1.e-3;
    }
    
    printf("Packets %7.2f %cHz [%u]:  Size %7.2f %cBps [%lld B] (%7.2f %cB/evt): Events %7.2f %cHz [%u]:  valid %02x\n", 
           rate  , scchar[rsc ], ncount, 
           dbytes, scchar[dbsc], (long long)nbytes, 
           tbytes, scchar[tbsc], 
           erate , scchar[ersc], nevent, 
           lanes);
    lanes = 0;

    ocount = ncount;
    oevent = nevent;
    obytes = nbytes;
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

  while ( (c=getopt( argc, argv, "a:i:p:P:")) != EOF ) {
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

  char* event = new char[ sizeof(XtcData::Dgram)+sizeof(Pds::Bld::TestType) ];

  while(1) {

    //  First, fetch BLD component
    uint64_t pulseId = bld.fetch(event+sizeof(XtcData::Dgram), sizeof(Pds::Bld::TestType));
    
    //  Second, fetch header (should already be waiting)
    const Pds::Tpr::Frame* frame = tpr->advance(pulseId);

    if (frame) {
      new (event) XtcData::Sequence(XtcData::Sequence::Event, 
                                    XtcData::TransitionId::L1Accept,
                                    *reinterpret_cast<const XtcData::TimeStamp*>(&frame->timeStamp),
                                    XtcData::PulseId(frame->pulseId));
      count++;
      event++;
      bytes += sizeof(XtcData::Dgram)+sizeof(Pds::Bld::TestType);
    }
  }

  pthread_join(thr,NULL);

  return 0;
}

