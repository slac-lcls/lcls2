
#include <stdio.h>
#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <time.h>
#include <arpa/inet.h>

#include "psdaq/aes-stream-drivers/AxisDriver.h"
#include "psdaq/aes-stream-drivers/DataDriver.h"

#include <string>

extern int optind;

static void usage(const char* p) {
  printf("Usage: %s [options]\n",p);
  printf("          -d <dev>       :              [default: /dev/datadev_0]\n");
  printf("          -l <lane>      :              [default: 0]\n");
  printf("          -b <type>      : BLD type     [default: ebeam]\n");
  printf("          -i <addr/name> : MC interface [no default] \n");
  printf("          -g <group>     : timing group [default: 6]\n");
}

int main(int argc, char** argv) {

  extern char* optarg;
  const char* dev = "/dev/datadev_0";

  int c;
  bool lUsage = false;
  const char* bldType = 0;
  unsigned mcintf = 0;
  unsigned group = 6;
  unsigned lane  = 0;
  bool     lverbose = false;

  //char* endptr;

  while ( (c=getopt( argc, argv, "d:b:i:l:vh?")) != EOF ) {
    switch(c) {
    case 'b':
      bldType = optarg;
      break;
    case 'd':
      dev = optarg;
      break;
    case 'i':
      mcintf = Psdaq::AppUtils::parse_interface(optarg);
      break;
    case 'g':
      group = strtoul(optarg,NULL,0);
      break;
    case 'l':
      lane = strtoul(optarg,NULL,0);
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

  if (!mcintf) {
    printf("%s: MC interface not set\n",argv[0]);
    lUsage = true;
  }

  unsigned mcaddr;
  unsigned short port = 12148;
  unsigned payloadWords;
  unsigned typeId, bldInfo;

  if (!bldType)
    lUsage = true;
  else if (strcmp(bldType,"ebeam")==0) {
    mcaddr       = 0xefff1900;
    typeId       = (7<<16) | 15; // typeid = EBeam v7
    bldInfo      = 0;
    payloadWords = 41;
  }
  else if (strcmp(bldType,"pcav")==0) {
    mcaddr       = 0xefff1901;
    typeId       = (0<<16) | 17; // typeid = PhaseCavity v0
    bldInfo      = 1;
    payloadWords = 8;
  }
  else if (strcmp(bldType,"gmd")==0) {
    mcaddr       = 0xefff1902;
    typeId       = (2<<16) | 64; // typeid = GMD v2
    bldInfo      = 2;
    payloadWords = 12;
  }
  else if (strcmp(bldType,"xgmd")==0) {
    mcaddr       = 0xefff1903;
    typeId       = (0<<16) | 64; // typeid = XGMD v0
    bldInfo      = 3;
    payloadWords = 12;
  }

  if (lUsage) {
    usage(argv[0]);
    exit(1);
  }

  {
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
    uint32_t* payload = new uint32_t[100];
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

    //  Configure the KCU to receive timing
    Parameters  para;
    para.partition = group;
    para.laneMask  = 1<<lane;
    para.device    = std::string(dev);
    para.maxTrSize = 8 * 1024 * 1024;
    para.kwargs["sim_length"] = "0";

    MemPool pool(para);
    XpmDetector fw(&para, &pool);

    json cjson = json({"body":{"drp":{"collId":{"det_info":{"readout":group}}}}});
    std::string collectionId("collId");
    fw.connect(cjson, collectionId);

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


    static const int MAX_RET_CNT_C = 1024;
    int32_t dmaRet    [MAX_RET_CNT_C];
    uint32_t dmaIndex [MAX_RET_CNT_C];
    uint32_t dest     [MAX_RET_CNT_C];
    uint32_t dmaFlags [MAX_RET_CNT_C];
    uint32_t dmaErrors[MAX_RET_CNT_C];

    while(1) {

        if (m_terminate.load(std::memory_order_relaxed)) {
            break;
        }
        int32_t ret = dmaReadBulkIndex(pool.fd(), MAX_RET_CNT_C, dmaRet, dmaIndex, dmaFlags, dmaErrors, dest);
        for (int b=0; b < ret; b++) {
            uint32_t size = dmaRet[b];
            uint32_t index = dmaIndex[b];
            uint32_t lane = (dest[b] >> 8) & 7;
            dmaSize = size;
            bytes += size;
            if (size > pool.dmaSize()) {
                logging::critical("DMA overflowed buffer: %u vs %u", size, pool.dmaSize());
                throw "DMA overflowed buffer";
            }

            uint32_t flag = dmaFlags[b];
            uint32_t err  = dmaErrors[b];
            if (err) {
                logging::error("DMA with error 0x%x  flag 0x%x",err,flag);
                //  How do I return this buffer?
                nevents++;
                continue;
            }

            const Pds::TimingHeader* timingHeader = static_cast<Pds::TimingHeader*>(pool.dmaBuffers[index]);

            XtcData::TransitionId::Value transitionId = timingHeader->service();
            if (transitionId == XtcData::TransitionId::L1Accept) {
                //  Add to BLD batch
                send(mcfd, payload, 96+4*payloadWords, 0);
            }
            dmaRetIndex(pool.fd(), index);
        }
    }
    fw.disconnect();
  }

  return 0;
}
