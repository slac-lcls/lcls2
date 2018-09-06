#include "psdaq/cphw/Reg.hh"

#include <stdio.h>
#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>

#include <new>
#include <vector>

using namespace Pds::Cphw;

void usage(const char* p) {
  printf("Usage: %s [options]\n",p);
  printf("Options: -a <ip address, dotted notation>\n");
  printf("         -r <address>[,<value>]\n");
}

int main(int argc, char* argv[])
{
  extern char* optarg;
  char c;
  char* endptr;

  const char* ip = "192.168.2.10";
  unsigned short port = 8192;
  std::vector<uint32_t> read_addrs;
  std::vector<uint32_t> write_addrs;
  std::vector<uint32_t> write_values;

  while ( (c=getopt( argc, argv, "a:r:h")) != EOF ) {
    switch(c) {
    case 'a':
      ip = optarg;
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

  Pds::Cphw::Reg::set(ip, port, 0);

  for(unsigned i=0; i<write_addrs.size(); i++) {
    Pds::Cphw::Reg* r = reinterpret_cast<Pds::Cphw::Reg*>(write_addrs[i]);
    *r = write_values[i];
  }

  for(unsigned i=0; i<read_addrs.size(); i++) {
    Pds::Cphw::Reg* r = reinterpret_cast<Pds::Cphw::Reg*>(read_addrs[i]);
    printf("[%08x] = %08x\n",read_addrs[i],unsigned(*r));
  }

  return 0;
}
