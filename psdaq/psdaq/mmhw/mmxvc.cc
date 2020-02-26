#include <stdio.h>
#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <fcntl.h>

#include <new>
#include <vector>

#include "psdaq/mmhw/Xvc.hh"

void usage(const char* p) {
  printf("Usage: %s [options]\n",p);
  printf("Options: -d <device filename>\n");
  printf("         -a <address>\n");
  printf("         -v\n");
}

int main(int argc, char* argv[])
{
  extern char* optarg;
  char c;
  char* endptr;

  const char* dev = 0;
  unsigned addr = 0;
  unsigned msize = 0x100000;
  bool lverbose = false;

  while ( (c=getopt( argc, argv, "d:a:hv")) != EOF ) {
    switch(c) {
    case 'd':
      dev = optarg;
      break;
    case 'a':
      addr = strtoul(optarg,&endptr,0);
      break;
    case 'v':
      lverbose = true;
      break;
    default:
      usage(argv[0]);
      return 0;
    }
  }

  if (!dev) {
    printf("-d <device filename> required\n");
    return -1;
  }

  int fd = open(dev,O_RDWR);
  if (fd<0) {
    perror("Open device failed");
    return -1;
  }

  char* p = (char*)mmap(0, msize, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
  if (p == MAP_FAILED) {
    perror("Failed to map");
    return -1;
  }

  Pds::Mmhw::Jtag* jtag = reinterpret_cast<Pds::Mmhw::Jtag*>(p+addr);
  Pds::Mmhw::Xvc::launch( jtag, lverbose );

  return 0;
}
