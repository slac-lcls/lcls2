#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <fcntl.h>
#include <stdint.h>
#include <sys/mman.h>

#include "psdaq/mmhw/Reg.hh"
using Pds::Mmhw::Reg;

void usage(const char* p) {
  printf("Usage: %s [args]\n",p);
  printf("Args:    -a <address> \n"
         "         -d <device>  \n"
         "         -n <bytes>   \n");
}

int main(int argc, char** argv)
{
  extern char* optarg;

  int c;
  off_t    adx    = 0;
  unsigned nbytes = 1;
  const char* dev = 0;

  while ( (c=getopt( argc, argv, "a:d:n:")) != EOF ) {
    switch(c) {
    case 'a':
      adx    = strtoull(optarg, NULL, 0);
      break;
    case 'd':
      dev = optarg;
      break;
    case 'n':
      nbytes = strtoul(optarg, NULL, 0);
      break;
    default:
      usage(argv[0]);
      exit(1);
    }
  }
  
  int fd = open(dev, O_RDWR);
  if (fd<0) {
    perror("Open device failed");
    return -1;
  }

  Reg::set(fd);

  Reg* r = (Reg*)(adx);
  printf("[%p]:\n",r);
  for(unsigned i=0; i<nbytes/4; i++) {
      printf("%08x%c", unsigned(r[i]), (i%4)==3 ? '\n':'.');
  }
  if ((nbytes%16)!=0)
      printf("\n");

  close(fd);

  return 0;
}
