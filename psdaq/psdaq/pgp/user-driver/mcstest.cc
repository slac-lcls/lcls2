#include <sys/stat.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <linux/types.h>

#include <fcntl.h>
#include <sstream>
#include <string>
#include <iomanip>
#include <iostream>
#include <string.h>
#include <stdlib.h>
#include <argp.h>

#include "mcsfile.hh"

using namespace std;

static void usage(const char* p)
{
  printf("Usage: %s [options]\n",p);
  printf("Options: -f <mcs file>\n");
  printf("For multiple PROMs, use multiple -f <mcs file> arguments\n");
}

int main (int argc, char **argv) 
{
   std::vector<std::string> mcs_files;
   int c;

  while((c=getopt(argc,argv,"f:")) != EOF) {
    switch(c) {
    case 'f': mcs_files.push_back(std::string(optarg)); break;
    default: usage(argv[0]); return 0;
    }
  }

  for(unsigned i=0; i<mcs_files.size(); i++) {
    printf("%s\n", mcs_files[i].c_str());
    McsFile mcs(mcs_files[i].c_str());
    printf("startAddr : 0x%x\n", mcs.startAddr());
    printf("endAddr   : 0x%x\n", mcs.endAddr());
    printf("size      : 0x%x\n", mcs.read_size());
  }

  return 0;
}
