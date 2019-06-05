#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include "psdaq/hsd/Module.hh"

using Pds::HSD::Module;

void usage(const char* p) {
  printf("Usage: %s -d <dev> -f <file>\n",p);
}

int main(int argc, char** argv) {
  extern char* optarg;

  const char* devName = 0;
  const char* fname = 0;
  int c;
  while ( (c=getopt( argc, argv, "d:f:")) != EOF ) {
    switch(c) {
    case 'd': devName = optarg; break;
    case 'f': fname = optarg; break;
    }
  }

  if (!devName || !fname) {
    printf("Missing required arguments\n");
    usage(argv[0]);
    return 0;
  }

  int fd = open(devName, O_RDWR);
  if (fd<0) {
    perror("Open device failed");
    return -1;
  }

  Module::create(fd)->flash_write(fname);

  return 0;
}
