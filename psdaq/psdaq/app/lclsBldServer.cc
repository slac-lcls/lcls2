
#include <stdio.h>
#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <time.h>
#include <arpa/inet.h>

#include "psdaq/tpr/Client.hh"
#include "psdaq/tpr/Frame.hh"
#include "psdaq/app/AppUtils.hh"

#include <string>

using namespace Pds::Tpr;

extern int optind;

static void usage(const char* p) {
  printf("Usage: %s [options]\n",p);
  printf("          -d <dev>       : <tpr a/b>    [default: a]\n");
  printf("          -b <type>      : BLD type     [default: ebeam]\n");
  printf("          -a <addr>      : MC address   [default: 0xefff1801]\n");
  printf("          -i <addr/name> : MC interface [no default] \n");
  printf("          -r <rate>      : update rate  [default: 5 = 10Hz]\n");
  printf("          -e <eventcode> : update evcode\n");
}

int main(int argc, char** argv) {

  extern char* optarg;
  char tprid='a';

  int c;
  bool lUsage = false;
  const char* bldType = 0;
  unsigned mcintf = 0;
  unsigned rate = 5;
  int      evcode = -1;
  bool     lverbose = false;

  //char* endptr;

  while ( (c=getopt( argc, argv, "d:b:e:i:r:vh?")) != EOF ) {
    switch(c) {
    case 'b':
      bldType = optarg;
      break;
    case 'd':
      tprid  = optarg[0];
      if (strlen(optarg) != 1) {
        printf("%s: option `-r' parsing error\n", argv[0]);
        lUsage = true;
      }
      break;
    case 'i':
      mcintf = Psdaq::AppUtils::parse_interface(optarg);
      break;
    case 'e':
      evcode = strtoul(optarg,NULL,0);
      break;
    case 'r':
      rate = strtoul(optarg,NULL,0);
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

    char evrdev[16];
    sprintf(evrdev,"/dev/tpr%c",tprid);

    Client client(evrdev,1,evcode<0);

    client.setup(0, 0, 1);
    if (evcode<0)
      client.start(TprBase::FixedRate(rate));
    else
      client.start(TprBase::EventCode(evcode));

    client.release();

    while(1) {
      const Frame* fr = client.advance();
      if (!fr) continue;

      payload[0] = fr->timeStamp & 0xffffffff;
      payload[1] = fr->timeStamp >> 32;
      payload[2] = fr->pulseId & 0xffffffff;
      payload[3] = fr->pulseId >> 32;
      send(mcfd, payload, 96+4*payloadWords, 0);

      if (lverbose)
        printf("Timestamp %lu.%09u\n",fr->timeStamp>>32,unsigned(fr->timeStamp&0xffffffff));
    }
  }

  return 0;
}
