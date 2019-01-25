#include <stdio.h>
#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <fcntl.h>

#include <new>
#include <vector>

void usage(const char* p) {
  printf("Usage: %s [options]\n",p);
  printf("Options: -d <device filename>\n");
  printf("         -r <address>[,<value>]\n");
}

int main(int argc, char* argv[])
{
  extern char* optarg;
  char c;
  char* endptr;

  const char* dev = 0;
  ssize_t     msize = 1<<20;
  std::vector<uint32_t> read_addrs;
  std::vector<uint32_t> write_addrs;
  std::vector<uint32_t> write_values;

  while ( (c=getopt( argc, argv, "d:m:r:h")) != EOF ) {
    switch(c) {
    case 'd':
      dev = optarg;
      break;
    case 'm':
      msize = strtoull(optarg,NULL,0);
      break;
    case 'r':
      { unsigned addr = strtoul(optarg,&endptr,0);
        if (*endptr==',') {
          unsigned v = strtoul(endptr+1,NULL,0);
          write_addrs .push_back(addr);
          write_values.push_back(v);
        }
        else
          read_addrs  .push_back(addr);
      }
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

  void* p = mmap(0, msize, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
  if (p == MAP_FAILED) {
    perror("Failed to map");
    return -1;
  }

  for(unsigned i=0; i<write_addrs.size(); i++) {
    *reinterpret_cast<uint32_t*>(p+write_addrs[i]) = write_values[i];
  }

  for(unsigned i=0; i<read_addrs.size(); i++) {
    printf("[%08x] = %08x\n",read_addrs[i],*reinterpret_cast<uint32_t*>(p+read_addrs[i]));
  }

  return 0;
}
