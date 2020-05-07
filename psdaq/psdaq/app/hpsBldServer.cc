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
#define __STDC_FORMAT_MACROS 1

#include <stdio.h>
#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <arpa/inet.h>
#include <signal.h>
#include <inttypes.h>
#include <Python.h>

#include <string>
#include <new>
#include <vector>

#include "psdaq/app/AppUtils.hh"

#include "psdaq/bld/Header.hh"
#include "psdaq/bld/HpsEvent.hh"
#include "psdaq/bld/HpsEventIterator.hh"
#include "psdaq/bld/Server.hh"

void usage(const char* p) {
  printf("Usage: %s [options]\n",p);
  printf("Options: -a <ip address, dotted notation> (HPS interface)\n");
  printf("         -i <ip_address, dotted notation> ( MC interface)\n");
  printf("         -p <words> (max packet size)\n");
}

using namespace Bld;

static int      count = 0;
static int      event = 0;
static int64_t  bytes = 0;
static unsigned lanes = 0;

static int setup_mc(unsigned addr, unsigned port, unsigned interface);
static void handle_data(void*);

static PyObject* _check(PyObject* obj) {
  if (!obj) {
    PyErr_Print();
    throw "**** python error\n";
  }
  return obj;
}

static void sigHandler( int signal ) 
{
  psignal(signal, "bld_control received signal");

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

#define HANDLE_ERR(str) {                       \
  perror(str);                                  \
  throw std::string(str); }

  int fd = ::socket(AF_INET, SOCK_DGRAM, 0);
  if (fd < 0)
    HANDLE_ERR("Open socket");

  { unsigned skbSize = 0x1000000;
    if (setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &skbSize, sizeof(skbSize)) == -1) 
      HANDLE_ERR("set so_rcvbuf");

    socklen_t sz = sizeof(skbSize);
    unsigned skbRead;
    if (getsockopt(fd, SOL_SOCKET, SO_RCVBUF, &skbRead, &sz) == -1) 
      HANDLE_ERR("get so_rcvbuf");

    printf("rcvbuf = 0x%x (0x%x)",skbRead,skbSize);
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

  //  Initialize the PVA monitor
  //  Fetch the multicast addr and port

  Py_Initialize();

  char module_name[64];
  sprintf(module_name,"psdaq.pyhpsbld.pyhpsbld");

   // returns new reference
  PyObject* m_module = _check(PyImport_ImportModule(module_name));

  PyObject* pDict = _check(PyModule_GetDict(m_module));
  PyObject* dev;
  {
    PyObject* pFunc = _check(PyDict_GetItemString(pDict, (char*)"hps_init"));

    //  Get a handle to the rogue control
    // returns new reference
    dev = _check(PyObject_CallFunction(pFunc,"ssI",bldname,hpsip_s,psize));
  }

  // "connect" to the sending socket
  char buf[4];
  send(fd,buf,sizeof(buf),0);

  unsigned mcaddr, mcport;
  {
    PyObject* pFunc = _check(PyDict_GetItemString(pDict, (char*)"hps_connect"));

    // returns new reference
    PyObject* mbytes = _check(PyObject_CallFunction(pFunc,"O",dev));

    mcaddr = PyLong_AsLong(PyDict_GetItemString(mbytes, "addr"));
    mcport = PyLong_AsLong(PyDict_GetItemString(mbytes, "port"));
        
    Py_DECREF(mbytes);
  }

  PyThreadState* m_pysave = PyEval_SaveThread(); // Py_BEGIN_ALLOW_THREADS
  
  int fd_mc = setup_mc(mcaddr,
                       mcport,
                       bldip);
  
  int iargs[] = { fd, fd_mc };
  handle_data(iargs);

  pthread_join(thr,NULL);

  PyEval_RestoreThread(m_pysave);  // Py_END_ALLOW_THREADS

  Py_DECREF(dev);
  Py_DECREF(m_module);
  Py_Finalize();

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

  uint64_t opid = 0;
  unsigned nprint = 20;

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

        if (opid && (opid+1) != ev.pulseId && nprint) {
          printf("PulseId jump 0x%" PRIx64 " to 0x%" PRIx64 "\n",
                 opid, ev.pulseId);
        }
        opid = ev.pulseId;
          
      } while(it.next());

      //  Force BLD out
      bldServer.flush();
    }
  } while(1);

  free(buff);
}

