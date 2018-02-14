#include <stdio.h>
#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <arpa/inet.h>
#include <signal.h>

#include <string>
#include <new>
#include <vector>

#include <cpsw_api_user.h>
#include <cpsw_yaml_keydefs.h>
#include <cpsw_yaml.h>

#include "psdaq/cphw/BldControl.hh"
using Pds::Cphw::BldControl;

#include "psdaq/bld/Header.hh"
#include "psdaq/bld/HpsEvent.hh"
#include "psdaq/bld/HpsEventIterator.hh"
#include "psdaq/bld/Server.hh"
#include "psdaq/bld/TestType.hh"

void usage(const char* p) {
  printf("Usage: %s [options]\n",p);
  printf("Options: -a <ip address, dotted notation>\n");
  printf("         -m <channel mask>\n");
  printf("         -p <words> (max packet size)\n");
}

using namespace Bld;

static int      count = 0;
static int      event = 0;
static int64_t  bytes = 0;
static unsigned lanes = 0;
static BldControl* cntl = 0;

static void sigHandler( int signal ) 
{
  psignal(signal, "bld_control received signal");

  cntl->disable();
  
  printf("BLD disabled\n");
  ::exit(signal);
}

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

int main(int argc, char* argv[])
{
  extern char* optarg;
  char c;

  const char* ip   = "192.168.2.10";
  unsigned short port = 8197;
  unsigned mask = 1;
  unsigned psize(0x3c0);

  while ( (c=getopt( argc, argv, "a:m:p:")) != EOF ) {
    switch(c) {
    case 'a':
      ip = optarg;
      break;
    case 'm':
      mask = strtoul(optarg,NULL,0);
      break;
    case 'p':
      psize = strtoul(optarg,NULL,0);
      break;
    default:
      usage(argv[0]);
      return 0;
    }
  }

  struct sigaction sa;
  sa.sa_handler = sigHandler;
  sa.sa_flags = SA_RESETHAND;

  sigaction(SIGINT ,&sa,NULL);
  sigaction(SIGABRT,&sa,NULL);
  sigaction(SIGKILL,&sa,NULL);
  sigaction(SIGSEGV,&sa,NULL);

  int fd = ::socket(AF_INET, SOCK_DGRAM, 0);
  if (fd < 0) {
    perror("Open socket");
    return -1;
  }

  sockaddr_in saddr;
  saddr.sin_family = PF_INET;

  if (inet_aton(ip,&saddr.sin_addr) < 0) {
    perror("Converting IP");
    return -1;
  }
    
  saddr.sin_port = htons(port);

  if (connect(fd, (sockaddr*)&saddr, sizeof(saddr)) < 0) {
    perror("Error connecting UDP socket");
    return -1;
  }

  sockaddr_in haddr;
  socklen_t   haddr_len = sizeof(haddr);
  if (getsockname(fd, (sockaddr*)&haddr, &haddr_len) < 0) {
    perror("Error retrieving local address");
    return -1;
  }

  pthread_attr_t tattr;
  pthread_attr_init(&tattr);
  pthread_t thr;
  if (pthread_create(&thr, &tattr, &countThread, &fd)) {
    perror("Error creating read thread");
    return -1;
  }

  //  Set the target address
  Pds::Cphw::Reg::set(ip, 8193, 0);

  cntl = BldControl::locate();

  printf("Writing mask\n");
  cntl->channelMask = mask;
  printf("Mask written\n");
  cntl->channelSevr = (1ULL<<60)-1;
  cntl->setMaxSize(psize);

  printf("WordCount : %u words\n",cntl->wordsLeft());
    
  cntl->enable(haddr);

  int fd_mc;
  {
    fd_mc = ::socket(AF_INET, SOCK_DGRAM, 0);
    if (fd_mc < 0) {
      perror("Open mcast socket");
      return -1;
    }

    sockaddr_in saddr_mc;
    saddr_mc.sin_family      = PF_INET;
    saddr_mc.sin_addr.s_addr = htonl(Pds::Bld::TestType::IP);
    saddr_mc.sin_port        = htons(Pds::Bld::TestType::Port);
    
   int y=1;
    if(setsockopt(fd_mc, SOL_SOCKET, SO_BROADCAST, (char*)&y, sizeof(y)) == -1) {
      perror("set broadcast");
      return -1;
    }

    sockaddr_in sa;
    sa.sin_family = AF_INET;
    sa.sin_addr.s_addr = haddr.sin_addr.s_addr;
    sa.sin_port = htons(11001);
    printf("Binding to %x.%u\n", ntohl(sa.sin_addr.s_addr),ntohs(sa.sin_port));
    if (::bind(fd_mc, (sockaddr*)&sa, sizeof(sa)) < 0) {
      perror("bind");
      return -1;
    }

    if (connect(fd_mc, (sockaddr*)&saddr_mc, sizeof(saddr_mc)) < 0) {
      perror("Error connecting UDP mcast socket");
      return -1;
    }
  }

  const unsigned buffsize=Pds::Bld::Header::MTU;
  char* buff = new char[buffsize];

  Pds::Bld::Server bldServer(fd_mc, Pds::Bld::TestType::Src);

  do {
    ssize_t ret = read(fd,buff,buffsize);
    if (ret < 0) break;
    count++;
    bytes += ret;
    HpsEventIterator it(buff,ret);
    do {
      HpsEvent ev = *it;
      lanes |= ev.valid;
      event++;

#ifdef PID_CHECK
      uint64_t ndpid = ev.pulseId - opid;
      if (opid!=0 && (ndpid!=dpid)) {
        printf("Delta PID change: %u -> %u\n", 
               unsigned( dpid&0xffffffff),
               unsigned(ndpid&0xffffffff));
        dpid = ndpid;
        opid = ev.pulseId;
      }
      else
        opid = ev.pulseId;
#endif

      //  Generate BLD out
      Pds::Bld::TestType contents(ev.channels, ev.valid);
      bldServer.publish( ev.pulseId, ev.timeStamp, 
                         (char*)&contents, sizeof(contents) );
      
    } while(it.next());
  } while(1);

  pthread_join(thr,NULL);
  free(buff);

  return 0;
}

