#include <chrono>
#include <unistd.h>
#include <cstring>
#include <fstream>
#include <pthread.h>
#include <poll.h>
#include "psdaq/hsd/Validator.hh"
#include "xtcdata/xtc/Dgram.hh"

static FILE* f = 0;
static bool     _lverbose = false;

static void show_usage(const char* p)
{
  printf("Usage: %s [options]\n",p);
  printf("Options: -f <input file>\n");
  printf("         -v (verbose)\n");
}

int main(int argc, char* argv[])
{
    
  int c;

  unsigned raw_start=  4, raw_rows = 20;
  unsigned fex_start=  4, fex_rows = 20;
  unsigned fex_thrlo=508, fex_thrhi=516;
  const char* ifile = 0;

  while((c = getopt(argc, argv, "f:v")) != EOF) {
    switch(c) {
    case 'f':
      ifile = optarg;
      break;
    case 'v':
      Validator::set_verbose(_lverbose = true);
      break;
    default:
      show_usage(argv[0]);
      return 0;
    }
  }

  if (!ifile) {
    printf("-f argument required\n");
    return -1;
  }

  Configuration cfg(raw_start, raw_rows,
                    fex_start, fex_rows,
                    fex_thrlo, fex_thrhi);

  f = fopen(ifile,"r");
  if (!f) {
    perror("Opening input file");
    exit(1);
  }

  size_t linesz = 0x10000;
  char* line = new char[linesz];
  ssize_t sz;
  uint32_t* event = new uint32_t[linesz/8];

  Fmc126Validator val(cfg);

  // Read
  do {

    if ((sz = getline(&line, &linesz, f))<=0)
      break;
    if (_lverbose)
      printf("Readline %zd [%32.32s]\n",sz, line);

    char* p = line;
    unsigned i;
    for(i=0; p<line+sz; i++, p++)
      event[i] = strtoul(p, &p, 16);
    int ret = i*4;

    char* data = (char*)event;

    XtcData::Transition* event_header = reinterpret_cast<XtcData::Transition*>(data);
    XtcData::TransitionId::Value transition_id = event_header->seq.service();

    if (_lverbose) {
      const uint32_t* pu32 = reinterpret_cast<const uint32_t*>(data);
      const uint64_t* pu64 = reinterpret_cast<const uint64_t*>(data);
      printf("Data: %016lx %016lx %08x %08x %08x %08x\n", pu64[0], pu64[1], pu32[4], pu32[5], pu32[6], pu32[7]);
      printf("Size %u B | Transition id %d | pulse id %lx | event counter %x\n",
             ret, transition_id, event_header->seq.pulseId().value(), *reinterpret_cast<uint32_t*>(event_header+1));
    }
      
    val.validate(data,ret);

  } while (1);
  
  Validator::dump_totals();

  return 0;
} 
