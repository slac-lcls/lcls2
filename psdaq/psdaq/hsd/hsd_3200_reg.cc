
#include <stdio.h>
#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <time.h>

extern int optind;

void usage(const char* p) {
  printf("Usage: %s [options]\n",p);
  printf("Options: -d <dev id>\n");
  printf("\t-a <register address>\n");
  printf("\t-w <value>\n");
}

int main(int argc, char** argv) {

  extern char* optarg;

  char qadc='a';
  int c;
  bool lUsage = false;
  bool lWrite = false;
  unsigned addr = 0;
  unsigned wval = 0;

  while ( (c=getopt( argc, argv, "d:a:w:")) != EOF ) {
    switch(c) {
    case 'd': qadc = optarg[0]; break;
    case 'a': addr = strtoul(optarg,NULL,0); break;
    case 'w': wval = strtoul(optarg,NULL,0); lWrite = true; break;
    case '?':
    default:
      lUsage = true;
      break;
    }
  }

  if (lUsage) {
    usage(argv[0]);
    exit(1);
  }

  char devname[16];
  sprintf(devname,"/dev/qadc%c",qadc);
  int fd = open(devname, O_RDWR);
  if (fd<0) {
    perror("Open device failed");
    return -1;
  }

  void* ptr = mmap(0, 0x100000, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
  if (ptr == MAP_FAILED) {
    perror("Failed to map");
    return 0;
  }

  uint32_t* p = reinterpret_cast<uint32_t*>((char*)ptr + addr);

  
  if (lWrite) {
    printf("Write %08x @ %08x\n", wval, addr);
    *p = wval;
  }
  printf("Read %08x @ %08x\n", *p, addr);

  return 1;
}
