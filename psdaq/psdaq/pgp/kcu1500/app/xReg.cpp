
#include <sys/types.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <stdio.h>
#include <termios.h>
#include <fcntl.h>
#include <sstream>
#include <string>
#include <iomanip>
#include <iostream>
#include <string.h>
#include <stdlib.h>
#include <linux/types.h>

#include "DataDriver.h"

using namespace std;

static void usage(const char* p) {
  printf("Usage: %p -d <device> -a <addr> [-v <value]\n",p);
}

int main (int argc, char **argv) {

  int          fd = -1;
  unsigned     addr = 0, value;
  bool         lWrite = false;
  int          c;

  while( (c = getopt(argc,argv,"d:a:v:"))!=-1 ) {
    switch(c) {
    case 'd':
      { if ( (fd = open(optarg, O_RDWR)) <= 0 ) {
          perror("Error opening device file");
          return 1;
        } } break;
    case 'a':
      addr = strtoul(optarg, NULL, 0);
      break;
    case 'v':
      lWrite = true;
      value = strtoul(optarg, NULL, 0);
      break;
    default:
      usage(argv[0]);
      return 1;
    }
  }

  if (fd < 0) {
    printf("-d argument required\n");
    return 1;
  }

  if (addr == 0) {
    printf("-a argument required\n");
    return 1;
  }

  { AxiVersion vsn;
    if (axiVersionGet(fd, &vsn)>=0) {
      printf("-- Core Axi Version --\n");
      printf("firmwareVersion : %x\n", vsn.firmwareVersion);
      printf("upTimeCount     : %u\n", vsn.upTimeCount);
      printf("deviceId        : %x\n", vsn.deviceId);
      printf("buildString     : %s\n", vsn.buildString); 
    }
  }

  if (lWrite) {
    if (dmaWriteRegister(fd, addr, value)<0) {
      perror("Error writing register");
      return -1;
    }
  }
  else {
    if (dmaReadRegister(fd, addr, &value)<0) {
      perror("Error reading register");
      return -1;
    }
    printf("[%08x] = %08x\n", addr,value);
  }

  return 1;
}
