#include <time.h>

#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <netinet/ip.h>
#include <netinet/udp.h>

#include <vector>
#include <thread>

#include "FileWriter.hh"
#include "psdaq/service/Fifo.hh"
#include "psdaq/service/Histogram.hh"
#include "xtcdata/xtc/TimeStamp.hh"

using Drp::BufferedFileWriter;
using Drp::BufferedFileWriterMT;
using Pds::FifoW;
using Pds::Histogram;

//#define PERFMON
#define MTWRITER

static void usage(const char* p)
{
  printf("Usage: %s [options]\n",p);
  printf("\t-f\tOutput file\n");
  printf("\t-r\tEvent rate\n");
  printf("\t-z\tEvent size\n");
}

class EventFifo {
public:
  EventFifo(unsigned evsz) : _evsz(evsz), _fifo(256*1024) {}
public:
  void push(const XtcData::TimeStamp& ts) { _fifo.push(ts); }
  void pop (XtcData::TimeStamp& ts) { _fifo.pend(); _fifo.pop(ts); }
  unsigned evsize() const { return _evsz; }
private:
  unsigned                  _evsz;
  FifoW<XtcData::TimeStamp> _fifo;
};

void write_thread(const char* ofile, EventFifo& fifo)
{
}

int main(int argc, char **argv) 
{
  const char* ofile = NULL;
  unsigned evrate   = 1;
  ssize_t  evsz     = 0x2000;
  unsigned sleep_rate = 10;

  char c;
  while ( (c=getopt( argc, argv, "f:r:s:z:")) != EOF ) {
    switch(c) {
    case 'f':
      ofile = optarg;
      break;
    case 'r':
      evrate = strtoul(optarg,NULL,0);
      break;
    case 's':
      sleep_rate = strtoul(optarg,NULL,0);
      break;
    case 'z':
      evsz   = strtoul(optarg,NULL,0);
      break;
    default:
      usage(argv[0]);
      return 1;
    }
  }

  // sleep every 1 ms
  unsigned evrate_ms = evrate/sleep_rate;
  unsigned nevt=0,nevt_ms=0;
  unsigned ncycle=0;

  printf("evrate_ms = %d\n",evrate_ms);

  uint32_t* payload = new uint32_t[evsz>>2];
  for(unsigned i=0; i< (evsz>>2); i++)
    payload[i] = i;

#ifdef MTWRITER
  BufferedFileWriterMT m_fileWriter(4194304);
#else
  BufferedFileWriter m_fileWriter(4194304);
#endif
  if (m_fileWriter.open(ofile) != 0) 
    throw std::string("Error opening output file");

#ifdef PERFMON
  Histogram hp(7,8.);
  Histogram hm(7,64.);
#endif

  printf("delay %u us every %u events\n",1000000/sleep_rate,evrate_ms);

  printf("\tAVG\tLAST\tLAST\tTIME\n");

  timespec tv_begin;
  clock_gettime(CLOCK_REALTIME,&tv_begin);
  timespec tv_ms = tv_begin;
  double t = 0;

  uint64_t nbytes = 0, tbytes = 0;

  while(1) {
    XtcData::TimeStamp ts(ncycle,nevt++);
    payload[0] = ts.nanoseconds();
    m_fileWriter.writeEvent(payload, evsz, ts);

    tbytes += evsz;

    if (nevt_ms++ == evrate_ms) {
      timespec tv;
      clock_gettime(CLOCK_REALTIME,&tv);
      double dt = double(tv.tv_sec - tv_ms.tv_sec) +
        1.e-9*(double(tv.tv_nsec)-double(tv_ms.tv_nsec));
      int usec = 1000000/sleep_rate - dt*1000000;

      printf("%d.%u %f %u\n", unsigned(tv.tv_sec), unsigned(tv.tv_nsec),dt,usec);

      if (usec>0) {
#ifdef PERFMON
        hp.bump((usec>>3)&0x7f);
#endif
        usleep(usec);
      }
#ifdef PERFMON
      else
        hm.bump((usec>>6)&0x7f);
#endif
      //      clock_gettime(CLOCK_REALTIME,&tv_ms);
      tv_ms = tv;
      nevt_ms = 0;
    }

    if (nevt==evrate) {
      timespec tv;
      clock_gettime(CLOCK_REALTIME,&tv);
      double dt = double(tv.tv_sec - tv_begin.tv_sec) +
        1.e-9*(double(tv.tv_nsec)-double(tv_begin.tv_nsec));

      t += dt;
      nbytes += tbytes;
      printf("\t%uMB/s\t%uMB/s\t%dMB\t%d.%09d\n",
	     unsigned(double(nbytes)/ t*1.e-6),
	     unsigned(double(tbytes)/dt*1.e-6),
             unsigned(double(tbytes)*1.e-6),
             unsigned(tv.tv_sec), unsigned(tv.tv_nsec));
                      
      tv_begin = tv;
      tbytes = 0;

      ncycle++;
      nevt=0;
#ifdef PERFMON
      if ((ncycle&7)==0) {
        hp.dump();
        hm.dump();
      }
#endif
    }
  }

  return 0;
}
