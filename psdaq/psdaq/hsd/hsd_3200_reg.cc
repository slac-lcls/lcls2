
#include <stdio.h>
#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <time.h>

#include <vector>

#include "psdaq/hsd/Fmc134Cpld.hh"
using Pds::HSD::Fmc134Cpld;

extern int optind;

void usage(const char* p) {
  printf("Usage: %s [options]\n",p);
  printf("Options: -d <dev id>\n");
  printf("         -r <address>[,<value>]\n");
  printf("         -x <address>[,<value>]\n");
}

int main(int argc, char** argv) {

  extern char* optarg;
  char* endptr;

  const char* devname = "/dev/pcie_adc_86";
  int c;
  bool lUsage = false;
  std::vector<uint32_t> read_addrs;
  std::vector<uint32_t> write_addrs;
  std::vector<uint32_t> write_values;

  std::vector<uint32_t> xread_addrs;
  std::vector<uint32_t> xwrite_addrs;
  std::vector<uint32_t> xwrite_values;

  while ( (c=getopt( argc, argv, "d:r:x:")) != EOF ) {
    switch(c) {
    case 'd': devname = optarg; break;
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
    case 'x':
      { unsigned addr = strtoul(optarg,&endptr,0);
        if (*endptr==',') {
          unsigned v = strtoul(endptr+1,NULL,0);
          xwrite_addrs .push_back(addr);
          xwrite_values.push_back(v);
        }
        else
          xread_addrs  .push_back(addr);
      }
      break;
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

  for(unsigned i=0; i<write_addrs.size(); i++) {
    uint32_t* r = reinterpret_cast<uint32_t*>((char*)ptr + write_addrs[i]);
    *r = write_values[i];
  }

  for(unsigned i=0; i<read_addrs.size(); i++) {
    uint32_t* r = reinterpret_cast<uint32_t*>((char*)ptr + read_addrs[i]);
    printf("[%08x] = %08x\n",read_addrs[i],unsigned(*r));
  }

  if (xwrite_addrs.size() || xread_addrs.size()) {
    Fmc134Cpld* cpld = reinterpret_cast<Fmc134Cpld*>(ptr+0x12800);

    for(unsigned i=0; i<xwrite_addrs.size(); i++)
      cpld->writeRegister(Fmc134Cpld::LMX, xwrite_addrs[i], xwrite_values[i]);

    for(unsigned i=0; i<xread_addrs.size(); i++)
      printf("[%08x] = %08x\n",xread_addrs[i], cpld->readRegister(Fmc134Cpld::LMX, xread_addrs[i]));
  }

  return 1;
}
