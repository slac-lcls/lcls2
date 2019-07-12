//
//  Use PVA to retrieve the list of fields from the HPS Diagnostic Bus
//
// ( Advertise the BLD service fields back to PVA )
//
//  PV names are $PLATFORM:$BLDNAME:HPS:FIELDNAMES - full list of names on diagn bus
//               $PLATFORM:$BLDNAME:HPS:FIELDTYPES - full list of types on diagn bus
//               $PLATFORM:$BLDNAME:HPS:FIELDMASK  - bit mask of active channels
//               $PLATFORM:$BLDNAME:PAYLOAD        - structure of payload in BLD mcast
//               $PLATFORM:$BLDNAME:ADDR           - IP addr of BLD mcast (larger registry)
//               $PLATFORM:$BLDNAME:PORT           - port of BLD mcast    (larger registry)
//
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

#include "psdaq/app/AppUtils.hh"

#include "psdaq/cphw/BldControl.hh"
using Pds::Cphw::BldControl;

#include "psdaq/bld/Header.hh"
#include "psdaq/bld/HpsEvent.hh"
#include "psdaq/bld/HpsEventIterator.hh"
#include "psdaq/bld/Server.hh"

#include "psdaq/epicstools/PVBase.hh"
using Pds_Epics::EpicsPVA;
using Pds_Epics::PVBase;

void usage(const char* p) {
  printf("Usage: %s [options]\n",p);
  printf("Options: -a <ip address, dotted notation> (HPS interface)\n");
  printf("         -i <ip_address, dotted notation> ( MC interface)\n");
  printf("         -p <words> (max packet size)\n");
}

namespace Bld {
  class HpsBldControlMonitor : public Pds_Epics::PVBase {
  public:
    HpsBldControlMonitor(int         fd, 
                         sockaddr_in haddr,
                         BldControl* cntl,
                         const char* channelName) :
      Pds_Epics::PVBase(channelName),
      _fd(fd), _haddr(haddr), _cntl(cntl)
    {
    }
    virtual ~HpsBldControlMonitor() {}
  public:
    void updated() {
      _cntl->disable();
      unsigned v = getScalarAs<unsigned>();
      if (v) {
        _cntl->channelMask = v;
        _cntl->enable(_fd, _haddr);
      }
    }
  private:
    int         _fd;
    sockaddr_in _haddr;
    BldControl* _cntl;
  };
};

using namespace Bld;

static int      count = 0;
static int      event = 0;
static int64_t  bytes = 0;
static unsigned lanes = 0;
static BldControl* cntl = 0;

static int setup_mc(unsigned addr, unsigned port, unsigned interface);
static void handle_data(void*);


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
    
    printf("Packets %7.2f %cHz:  Size %7.2f %cBps (%7.2f %cB/pkt): Events %7.2f %cHz:  valid %08x\n", 
           rate  , scchar[rsc ],
           dbytes, scchar[dbsc], 
           tbytes, scchar[tbsc], 
           erate , scchar[ersc],
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

  const char* hpsip_s = 0;
  unsigned hpsip = 0;
  unsigned bldip = 0;
  unsigned short port = 8197;
  unsigned psize(0x3c0);
  const char* bldname = 0; // "DAQ:LAB2:BLD:HPSEXAMPLE";

  while ( (c=getopt( argc, argv, "a:i:p:n:")) != EOF ) {
    switch(c) {
    case 'a':
      hpsip = Psdaq::AppUtils::parse_interface(hpsip_s=optarg);
      break;
    case 'i':
      bldip = Psdaq::AppUtils::parse_interface(optarg);
      break;
    case 'p':
      psize = strtoul(optarg,NULL,0);
      break;
    case 'n':
      bldname = optarg;
      break;
    default:
      usage(argv[0]);
      return 0;
    }
  }

  if (!hpsip || !bldip) {
    printf("Missing required parameters: HPS interface and BLD interface\n");
    usage(argv[0]);
    return 0;
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
  saddr.sin_family      = PF_INET;
  saddr.sin_addr.s_addr = htonl(hpsip);
  saddr.sin_port        = htons(port);

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
  Pds::Cphw::Reg::set(hpsip_s, 8193, 0);

  cntl = BldControl::locate();
  cntl->setMaxSize(psize);

  int fd_mc;

  if (bldname) {
    //
    //  Fetch channel field names from PVA
    //
    HpsBldControlMonitor* pvaCntl = new HpsBldControlMonitor(fd, haddr, cntl, 
                                                             (std::string(bldname)+":HPS:FIELDMASK").c_str());
    //    EpicsPVA*          pvaPayl = new EpicsPVA((std::string(bldname)+":PAYLOAD").c_str());
    PVBase*            pvaAddr = new PVBase((std::string(bldname)+":ADDR").c_str());
    PVBase*            pvaPort = new PVBase((std::string(bldname)+":PORT").c_str());
  
    while(1) {
      if (pvaCntl      ->connected() &&
          //          pvaPayl      ->connected() &&
          pvaAddr      ->connected() &&
          pvaPort      ->connected())
        break;
      usleep(100000);
    }

    fd_mc = setup_mc(pvaAddr->getScalarAs<unsigned>(),
                     pvaPort->getScalarAs<unsigned>(),
                     bldip);
  }
  else {
    cntl->disable();
    cntl->channelMask = 0x18;
    cntl->enable(fd, haddr);
    fd_mc = setup_mc(0xefff8001,
                     11001,
                     bldip);
  }
  
  int iargs[] = { fd, fd_mc };
  handle_data(iargs);

  pthread_join(thr,NULL);

  return 0;
}


int setup_mc(unsigned addr, unsigned port, unsigned interface)
{
  printf("setup_mc %x/%u %x\n",addr,port,interface);

  int fd_mc;

  fd_mc = ::socket(AF_INET, SOCK_DGRAM, 0);
  if (fd_mc < 0) {
    perror("Open mcast socket");
    return -1;
  }

  sockaddr_in saddr_mc;
  saddr_mc.sin_family      = PF_INET;
  saddr_mc.sin_addr.s_addr = htonl(addr);
  saddr_mc.sin_port        = htons(port);
    
  int y=1;
  if(setsockopt(fd_mc, SOL_SOCKET, SO_BROADCAST, (char*)&y, sizeof(y)) == -1) {
    perror("set broadcast");
    return -1;
  }

  sockaddr_in sa;
  sa.sin_family = AF_INET;
  sa.sin_addr.s_addr = htonl(interface);
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

  { in_addr addr;
    addr.s_addr = htonl(interface);
    if (setsockopt(fd_mc, IPPROTO_IP, IP_MULTICAST_IF, (char*)&addr,
                   sizeof(in_addr)) < 0) {
      perror("set ip_mc_if");
      return -1;
    }
  }

  return fd_mc;
}

void handle_data(void* args)
{
  int* iargs = (int*)args;
  int fd = iargs[0], fd_mc = iargs[1];

  //  Program the crossbar to pull timing off the backplane
  //  cntl->_timing.xbar.setOut( Pds::Cphw::XBar::FPGA, Pds::Cphw::XBar::BP );

  const unsigned buffsize=Pds::Bld::Header::MTU;
  char* buff = new char[buffsize];

  Pds::Bld::Server bldServer(fd_mc);

  do {
    ssize_t ret = read(fd,buff,buffsize);
    if (ret < 0) break;
    count++;
    bytes += ret;

    HpsEventIterator it(buff,ret);
    if (it.valid()) {
      //  update the payload format, if necessary
      if (bldServer.id() != it.id()) {
        bldServer.setID(it.id());
      }

      do {
        const HpsEvent& ev = *it;
        lanes |= ev.valid;
        event++;

        //  Generate BLD out
        bldServer.publish( ev.pulseId, ev.timeStamp, 
                           (char*)&ev.valid, sizeof(uint32_t)*(1+it.nchannels()) );
      
      } while(it.next());

      //  Force BLD out
      bldServer.flush();
    }
  } while(1);

  free(buff);
}
