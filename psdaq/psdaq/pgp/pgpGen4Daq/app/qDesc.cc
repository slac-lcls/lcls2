
#include <sys/types.h>
#include <unistd.h>
#include <stdio.h>
#include <termios.h>
#include <fcntl.h>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <string.h>
#include <stdlib.h>
#include <signal.h>
#include <time.h>
#include <stdint.h>
#include <new>

#include "../include/DmaDriver.h"
#include "PgpDaq.hh"

void printUsage(char* name) {
  printf( "Usage: %s [-h]  -P <deviceName> [options]\n"
          "    -h         Show usage\n"
          "    -P         Set pgpcard device name\n"
          "    -c <client>\n"
          "    -d <descs> \n",
      name
  );
}

int main (int argc, char **argv) {
  int           fd;
  const char*         dev = "/dev/pgpdaq0";
  unsigned            client              = 0;
  unsigned            ndescs              = 32;

  //  char*               endptr;
  extern char*        optarg;
  int c;
  while( ( c = getopt( argc, argv, "hP:c:d:" ) ) != EOF ) {
    switch(c) {
    case 'P':
      dev = optarg;
      break;
    case 'c':
      client = strtoul(optarg,NULL,0);
      break;
    case 'd':
      ndescs = strtoul(optarg,NULL,0);
      break;
    case 'h':
      printUsage(argv[0]);
      return 0;
      break;
    default:
      printf("Error: Option could not be parsed, or is not supported yet!\n");
      printUsage(argv[0]);
      return 0;
      break;
    }
  }

  if ( (fd = open(dev, O_RDWR)) <= 0 ) {
    perror(dev);
    return(1);
  }

  // Allocate a buffer
  uint32_t* data  = new uint32_t[0x80000];
  struct DmaReadData rd;
  rd.data  = reinterpret_cast<uintptr_t>(data);

  // DMA Read
  while(1) {
    rd.index = (1<<30) | client;
    ssize_t ret = read(fd, &rd, sizeof(rd));
    if (ret < 0) {
      perror("Reading buffer");
      break;
    }

    printf("--\n");
    for(unsigned i=0; i<ndescs; i++)
      printf("%08x.%08x%c", data[2*i], data[2*i+1], (i%8)==7?'\n':' ');
    sleep(1);
  }

  free(data);
  return 0;
}
