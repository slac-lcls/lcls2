
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

#include <string>

using namespace Pds::Tpr;

static bool     verbose = false;

extern int optind;

static void usage(const char* p) {
  printf("Usage: %s [options]\n",p);
  printf("          -d <dev>  : <tpr a/b>\n");
  printf("          -b <type> : simulate BLD type\n");
  printf("          -a <addr> : MC address\n");
}

int main(int argc, char** argv) {

  extern char* optarg;
  char tprid='a';

  int c;
  bool lUsage = false;
  const char* bldType = 0;
  unsigned mcaddr = 0xefff1800;
  unsigned mcintf = 0;
  unsigned short port = 12148;
  unsigned rate = 5;

  char* endptr;

  while ( (c=getopt( argc, argv, "d:a:b:i:r:h?")) != EOF ) {
    switch(c) {
    case 'd':
      tprid  = optarg[0];
      if (strlen(optarg) != 1) {
        printf("%s: option `-r' parsing error\n", argv[0]);
        lUsage = true;
      }
      break;
    case 'b':
      bldType = optarg;
      break;
    case 'a':
      mcaddr = strtoul(optarg,NULL,0);
      break;
    case 'i':
      mcintf = strtoul(optarg,NULL,0);
      break;
    case 'r':
      rate = strtoul(optarg,NULL,0);
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
    payload[ 6] = (1<<16) | 1; // typeid = Xtc v1
    payload[ 7] = (6<<24) | (pid&0xffffff);
    payload[ 8] = 0; // ebeam
    payload[ 9] = 204;
    payload[10] = 0; // damage
    payload[11] = (7<<16) | 15; // typeid = EBeam v7
    payload[12] = (6<<24) | (pid&0xffffff);
    payload[13] = 0; // ebeam
    payload[14] = 184;
    for(unsigned i=0; i<41; i++)
      payload[15+i] = i;

    char evrdev[16];
    sprintf(evrdev,"/dev/tpr%c",tprid);

    Client client(evrdev,1);

    client.start(TprBase::FixedRate(rate));
    client.release();

    while(1) {
      const Frame* fr = client.advance();
      if (!fr) continue;

      payload[0] = fr->timeStamp & 0xffffffff;
      payload[1] = fr->timeStamp >> 32;
      payload[2] = fr->pulseId & 0xffffffff;
      payload[3] = fr->pulseId >> 32;
      send(mcfd, payload, 220, 0);
    }
  }

  return 0;
}
