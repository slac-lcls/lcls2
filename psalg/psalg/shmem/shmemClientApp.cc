// The following is needed to get PRIu16 and friends defined in c++ files
#define __STDC_FORMAT_MACROS
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <inttypes.h>

#include "ShmemClient.hh"
#include "xtcdata/xtc/Dgram.hh"
#include "xtcdata/xtc/TransitionId.hh"
#include "psalg/utils/MacTimeFix.hh"

using namespace XtcData;
using namespace psalg::shmem;

class MyShmemClient : public ShmemClient {
public:
  MyShmemClient(int rate, int verbose) : _rate(rate), _verbose(verbose) {}
  int processDgram(Dgram* dg) {
    if (_rate) {
      timespec ts;
      ts.tv_sec  = 0;
      ts.tv_nsec = 1000000000/_rate;
      nanosleep(&ts,0);
    }
    if(_verbose)
      printf("%-15s transition: time 0x%016" PRIx64 " = %u.%09u, damage %04x, payloadSize 0x%x\n",
             TransitionId::name(dg->service()),
             dg->time.value(),
             dg->time.seconds(), dg->time.nanoseconds(),
             dg->xtc.damage.value(),
             dg->xtc.sizeofPayload());
    return 0;
  }
private:
  int _rate;
  int _verbose;
};

void usage(char* progname) {
  fprintf(stderr,"Usage: %s "
          "[-p <partitionTag>] "
          "[-i <index>] "
          "[-r <rate>] "
          "[-t] dgram timing"
          "[-R] make reconnect attempts"
          "[-v] "
          "[-V] "
          "[-h]\n", progname);
}

static void printrate(const char* name,
                      const char* units,
                      double      rate)
{
  static const char scchar[] = { ' ', 'k', 'M' };
  unsigned rsc = 0;

  if (rate > 1.e6) {
    rsc     = 2;
    rate   *= 1.e-6;
  }
  else if (rate > 1.e3) {
    rsc     = 1;
    rate   *= 1.e-3;
  }

  printf("%s %7.2f %c%s", name, rate, scchar[rsc], units);
}

static timespec tsdiff(timespec start, timespec end)
{
	timespec temp;
	if ((end.tv_nsec-start.tv_nsec)<0) {
		temp.tv_sec = end.tv_sec-start.tv_sec-1;
		temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
	} else {
		temp.tv_sec = end.tv_sec-start.tv_sec;
		temp.tv_nsec = end.tv_nsec-start.tv_nsec;
	}
	return temp;
}

int main(int argc, char* argv[]) {
  int c;
  const char* partitionTag = 0;
  unsigned index = 0;
  int rate = 0;
  int events = 0;
  int bytes = 0;
  bool timing = false;
  bool verbose = false;
  bool veryverbose = false;
  bool accept = false;
  bool reconnect = false;
  timespec ptv,tv;

  while ((c = getopt(argc, argv, "?hvVti:p:r:R")) != -1) {
    switch (c) {
    case '?':
    case 'h':
      usage(argv[0]);
      exit(0);
    case 'i':
      index = strtoul(optarg,NULL,0);
      break;
    case 'r':
      rate = strtoul(optarg,NULL,0);
      break;
    case 'p':
      partitionTag = optarg;
      break;
    case 't':
      timing = true;
      break;
    case 'R':
      reconnect = true;
      break;
    case 'V':
      veryverbose = true;
    case 'v':
      verbose = true;
      break;
    default:
      usage(argv[0]);
    }
  }

  while(1)
    {
    MyShmemClient myClient(rate,verbose);
    myClient.connect(partitionTag,index);
    while(1)
      {
      int ev_index;
      size_t buf_size;
      Dgram *dgram = (Dgram*)myClient.get(ev_index, buf_size);
      if(!dgram) break;
      if(veryverbose)
        printf("shmemClient dgram trId %d index %d size %lu\n",dgram->service(),ev_index,buf_size);
      if(!timing)
        myClient.processDgram(dgram);
      if(dgram->service() == TransitionId::L1Accept)
        {
        if(!accept && timing)
          {
          accept = true;
          clock_gettime(CLOCK_REALTIME, &ptv);
          }
        ++events;
        bytes+=dgram->xtc.sizeofPayload();
        }
      myClient.free(ev_index,buf_size);
      }
    if(timing || !reconnect)
      break;
    printf("shmemClient's server appears to have disconnected"
           " - attempting to reconnect\n");
    }
  if(timing)
    clock_gettime(CLOCK_REALTIME, &tv);

  printf("shmemClient received %d L1 dgrams %d payload bytes",events,bytes);

  if(timing)
    {
    timespec df = tsdiff(ptv,tv);
    double dt = df.tv_sec+(1.e-9*df.tv_nsec);
    double revents = double(events)/dt;
    double rbytes  = double(bytes)/dt;
    printrate(" at:\n ", "Hz", revents);
    printrate("\t ", "B/s", rbytes);
    }

  printf("\n");

  return 0;
}
