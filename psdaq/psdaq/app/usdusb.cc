#include "psdaq/tpr/Client.hh"
#include "psdaq/tpr/Frame.hh"
#include "psdaq/app/AppUtils.hh"

#include <libusdusb4.h>
#include <libusb-1.0/libusb.h>

#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <signal.h>
#include <semaphore.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <arpa/inet.h>
#include <vector>

static bool lverbose;

using namespace Pds::Tpr;

//#define DBUG
enum { NCHANNELS = 4 };
enum Quad_Mode  { CLOCK_DIR, X1, X2, X4 };
enum Count_Mode { WRAP_FULL, LIMIT, HALT, WRAP_PRESET };

static const unsigned payloadWords = sizeof(USB4_FIFOBufferRecord)/4;

static void events(unsigned          dev,
                   Pds::Tpr::Client& client,
                   bool              lcls2,
                   unsigned          mcfd,
                   unsigned*         payload) {

  enum { MAX_RECORDS = 1024 };
  USB4_FIFOBufferRecord* _records = new USB4_FIFOBufferRecord[MAX_RECORDS];
  uint64_t _tpr_ts;
  unsigned _usd_ts;

  bool first = true;

  while(1) {
    long int nRecords = MAX_RECORDS;
    const int tmo = 100; // milliseconds
    int status = USB4_ReadFIFOBufferStruct(dev, 
                                           &nRecords,
                                           _records,
                                           tmo);
    if (status != USB4_SUCCESS) {
      printf("ReadFIFO result %d\n",status);
    }
    else {
      if (lverbose)
        printf("ReadFIFO %ld records\n",nRecords);

      for(int i=0; i<nRecords; i++) {
        const USB4_FIFOBufferRecord& record = _records[i];
        
        const Pds::Tpr::Frame* fr = client.advance();
        
        if (first) {
          first   = false;
          if (lcls2)
            _tpr_ts = fr->timeStamp;
          else
            _tpr_ts = fr->pulseId & 0x1ffff;
        }
        else {

          unsigned usd_ts = record.Time;
          //          const double tolerance = 0.03;  // AC line rate jitter
          //          const unsigned maxdfid = 20000; // 48MHz timestamp rollover

          double dusd = double(usd_ts - _usd_ts)/48e6;
          _usd_ts = usd_ts;

          double dtpr;
          if (lcls2) {
            dtpr = double(fr->timeStamp - _tpr_ts);
            _tpr_ts = fr->timeStamp;
          }
          else {
            int pid  = fr->pulseId & 0x1ffff;
            int nfid = pid - _tpr_ts;
            _tpr_ts = pid;
            if (nfid < 0)
              nfid += 0x1ffe0;
            dtpr = double(nfid) / 357.;
          }

//        double fdelta = dusd - dtpr;
        }

        payload[0] = fr->timeStamp & 0xffffffff;
        payload[1] = fr->timeStamp >> 32;
        payload[2] = fr->pulseId & 0xffffffff;
        payload[3] = fr->pulseId >> 32;
        ::send(mcfd, payload, 96+4*payloadWords, 0);
      }
    }
  }
}


static void configure(unsigned                dev,
                      std::vector<Quad_Mode>  qmode,
                      std::vector<Count_Mode> cmode) {

  unsigned _nerror = 0;

  {

#ifdef DBUG
#define USBDBUG1(func,arg0) {                   \
      printf("%s: %d\n",#func,arg0);            \
      USB4_##func(arg0); }

#define USBDBUG2(func,arg0,arg1) {              \
      printf("%s: %d %d\n",#func,arg0,arg1);    \
      USB4_##func(arg0,arg1); }

#define USBDBUG3(func,arg0,arg1,arg2) {                         \
      printf("%s: %d %d %d\n",#func,arg0,arg1,arg2);            \
      _nerror += USB4_##func(arg0,arg1,arg2)!=USB4_SUCCESS; }
#else
#define USBDBUG1(func,arg0) USB4_##func(arg0)
#define USBDBUG2(func,arg0,arg1) USB4_##func(arg0,arg1)
#define USBDBUG3(func,arg0,arg1,arg2) _nerror += USB4_##func(arg0,arg1,arg2)!=USB4_SUCCESS
#endif

    for(unsigned i=0; i<NCHANNELS; i++) {
      USBDBUG3( SetMultiplier , dev, i, (int)qmode[i]);
      USBDBUG3( SetCounterMode, dev, i, (int)cmode[i]);
      USBDBUG3( SetForward    , dev, i, 0); // A/B assignment (normal)
      USBDBUG3( SetCaptureEnabled, dev, i, 1);
      USBDBUG3( SetCounterEnabled, dev, i, 1);
    }

    // Clear the FIFO buffer
    USBDBUG1(ClearFIFOBuffer, dev);

    // Enable the FIFO buffer
    USBDBUG1(EnableFIFOBuffer, dev);

    // Clear the captured status register
    USBDBUG2(ClearCapturedStatus, dev, 0);

    static int DIN_CFG[] = { 0, 0, 0, 0, 0, 0, 0, 1 };
    USB4_SetDigitalInputTriggerConfig( dev, DIN_CFG, DIN_CFG);

    USBDBUG1(ClearDigitalInputTriggerStatus, dev);
    
    printf("Configuration Done\n");

    unsigned long count;
    USBDBUG3( GetCount, dev, 0, &count );
    printf("Read count %lu\n", count);
  }
}

extern int optind;

static int reset_usb()
{
  int n = 0;

  libusb_context* pctx;

  libusb_init(&pctx);

  const int vid = 0x09c9;
  const int pid = 0x0044;

  libusb_device_handle* phdl = libusb_open_device_with_vid_pid(pctx, vid, pid);
  if (phdl) {
    libusb_reset_device(phdl);
    libusb_close(phdl);
    n = 1;
  }

  libusb_exit(pctx);

  return n;
}

static void close_usb(int isig)
{
  printf("close_usb %d\n",isig);
  USB4_Shutdown();
  const char* nsem = "Usb4-0000";
  printf("Unlinking semaphore %s\n",nsem);
  if (sem_unlink(nsem))
    perror("Error unlinking usb4 semaphore");
  exit(0);
}

using namespace Pds;

static void usage(const char* p)
{
  printf("Usage: %s [OPTIONS]\n",p);
  printf("          -a <addr>      : MC address   [default: 0xefff1801]\n");
  printf("          -i <addr/name> : MC interface [no default] \n");
  printf("          -t             : disable timestamp check\n");
  printf("          -z             : zero encoder\n");
  printf("          -T <trigger device file,output,delay_ticks,width_ticks>\n");
  printf("                         : [default: /dev/tpra0,0,0,1]\n");
  printf("          -1 <event code>: LCLS1 event select [default: false]\n");
  printf("          -2 <rate>      : LCLS2 event select [default: 5 (10Hz)]\n");
  printf("          -v             : verbose mode\n");
}

int main(int argc, char** argv) {

  // parse the command line for our boot parameters
  bool lUsage = false;
  struct {
    const char* devname = "/dev/tpra";
    unsigned channel = 0;  // tpr channel
    unsigned output = 0;   // tpr ttl output 
    unsigned delay = 0;
    unsigned width = 1;
    unsigned rate  = 5;    // lcls2 event select
    int evcode     = -1;   // lcls1 event select
  } tpr_config;
  unsigned mcintf  = 0;
  unsigned mcaddr  = 0;
  const unsigned port = 10148;
  unsigned bldInfo = 0;
  const unsigned typeId  = 62; // UsdUsbData

  bool lzero = false;
  //  tsc = true;
  lverbose = false;

  std::vector<Count_Mode> count_mode(NCHANNELS, WRAP_FULL);
  std::vector<Quad_Mode>  quad_mode (NCHANNELS, X4);

  extern char* optarg;
  int c;
  while ( (c=getopt( argc, argv, "a:i:tzT:1:2:vh?")) != EOF ) {
    switch(c) {
    case 'a':
      mcaddr = Psdaq::AppUtils::parse_interface(optarg);
      bldInfo = mcaddr & 0xff;
      break;
    case 'i':
      mcintf = Psdaq::AppUtils::parse_interface(optarg);
      break;
    case '1':
      tpr_config.evcode = strtoul(optarg,NULL,0);
      break;
    case '2':
      tpr_config.rate = strtoul(optarg,NULL,0);
      break;
    case 'T':
      tpr_config.devname = optarg;
      { char* delim   = strchr(  optarg,',');
        char* ndelim  = strchr( delim+1,',');
        char* nndelim = strchr(ndelim+1,',');
        if (delim && ndelim && ndelim) {
          *delim = 0;
          tpr_config.channel = strtoul(delim-1,NULL,16);
          *(delim-1)= 0;
          *ndelim   = 0;
          *nndelim  = 0;
          tpr_config.output = strtoul(  delim+1,NULL,0);
          tpr_config.delay  = strtoul( ndelim+1,NULL,0);
          tpr_config.width  = strtoul(nndelim+1,NULL,0);
        }
        else {
          lUsage = true;
          printf("Error parsing trigger config [%s]\n",optarg);
        }
      }

    case 't':
      //      tsc = false;
      break;
    case 'z':
      lzero = true;
      break;
    case 'v':
      lverbose = true;
      break;
    case 'h':
      usage(argv[0]);
      exit(0);
    case '?':
    default:
      lUsage = true;
      break;
    }
  }

  if (optind < argc) {
    printf("%s: invalid argument -- %s\n",argv[0], argv[optind]);
    lUsage = true;
  }

  if (lUsage) {
    usage(argv[0]);
    return 1;
  }

  //  printf("UsdUsb is %sabling testing time step check\n", tsc ? "en" : "dis");

  //
  //  There must be a way to detect multiple instruments, but I don't know it yet
  //
  reset_usb();

  short deviceCount = 0;
  printf("Initializing device\n");
  int result = USB4_Initialize(&deviceCount);
  if (result != USB4_SUCCESS) {
    printf("Failed to initialize USB4 driver (%d)\n",result);
    close_usb(0);
    return 1;
  }

  //
  //  Need to shutdown the USB driver properly
  //
  struct sigaction int_action;

  int_action.sa_handler = close_usb;
  sigemptyset(&int_action.sa_mask);
  int_action.sa_flags = 0;
  int_action.sa_flags |= SA_RESTART;

  if (sigaction(SIGINT, &int_action, 0) > 0)
    printf("Couldn't set up SIGINT handler\n");
  if (sigaction(SIGKILL, &int_action, 0) > 0)
    printf("Couldn't set up SIGKILL handler\n");
  if (sigaction(SIGSEGV, &int_action, 0) > 0)
    printf("Couldn't set up SIGSEGV handler\n");
  if (sigaction(SIGABRT, &int_action, 0) > 0)
    printf("Couldn't set up SIGABRT handler\n");
  if (sigaction(SIGTERM, &int_action, 0) > 0)
    printf("Couldn't set up SIGTERM handler\n");

  printf("Found %d devices\n", deviceCount);

  if (lzero) {
    for(unsigned i=0; i<NCHANNELS; i++) {
      if ((result = USB4_SetPresetValue(deviceCount, i, 0)) != USB4_SUCCESS)
	printf("Failed to set preset value for channel %d : %d\n",i, result);
      if ((result = USB4_ResetCount(deviceCount, i)) != USB4_SUCCESS)
	printf("Failed to set preset value for channel %d : %d\n",i, result);
    }
    close_usb(0);
    return 1;
  }

  configure(0,quad_mode,count_mode);

  int mcfd = socket(AF_INET, SOCK_DGRAM, 0);
  if (mcfd < 0) {
    perror("Open socket");
    throw std::string("Open socket");
  }

  { struct sockaddr_in saddr;
    saddr.sin_family = AF_INET;
    saddr.sin_addr.s_addr = htonl(mcintf|0x3ff);
    saddr.sin_port = htons(port);
    memset(saddr.sin_zero, 0, sizeof(saddr.sin_zero));
    if (bind(mcfd, (sockaddr*)&saddr, sizeof(saddr)) < 0) {
      perror("bind");
      throw std::string("bind");
    }
  }

  { in_addr addr;
    addr.s_addr = htonl(mcintf);
    if (setsockopt(mcfd, IPPROTO_IP, IP_MULTICAST_IF, (char*)&addr,
                   sizeof(in_addr)) < 0) {
      perror("set ip_mc_if");
      return -1;
    }
  }

  { sockaddr_in saddr;
    saddr.sin_family      = PF_INET;
    saddr.sin_addr.s_addr = htonl(mcaddr);
    saddr.sin_port        = htons(port);

    if (connect(mcfd, (sockaddr*)&saddr, sizeof(saddr)) < 0) {
      perror("Error connecting UDP socket");
      return -1;
    }
  }

  unsigned pid = getpid();
  uint32_t* payload = new uint32_t[14+payloadWords];
  // clocktime
  // timestamp
  payload[ 4] = 0; // env
  payload[ 5] = 0; // damage
  payload[ 6] = (6<<24) | (pid&0xffffff);
  payload[ 7] = bldInfo;
  payload[ 8] = typeId;
  payload[ 9] = 4*payloadWords+20; // extent
  payload[10] = 0;   // damage
  payload[11] = (6<<24) | (pid&0xffffff);
  payload[12] = bldInfo;
  payload[13] = typeId;
  payload[14] = 4*payloadWords+20; // extent
  for(unsigned i=0; i<payloadWords; i++)
    payload[15+i] = i;

  printf("Using %s (%x)\n",tpr_config.devname,tpr_config.channel);
  Client client(tpr_config.devname,tpr_config.channel,tpr_config.evcode<0);

  client.setup(tpr_config.output, 
               tpr_config.delay, 
               tpr_config.width);
  if (tpr_config.evcode<0)
    client.start(TprBase::FixedRate(tpr_config.rate));
  else
    client.start(TprBase::EventCode(tpr_config.evcode));

  client.release();

  events(0,client,tpr_config.evcode<0,mcfd,payload);

  return 0;
}
