#include "psdaq/epicstools/EpicsPVA.hh"
using Pds_Epics::EpicsPVA;

#include "psalg/digitizer/Stream.hh"

#include <sys/types.h>
#include <unistd.h>
#include <stdio.h>
#include <termios.h>
#include <fcntl.h>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <string.h>
#include <stdlib.h>
#include <signal.h>
#include <time.h>
#include <stdint.h>
#include <new>

FILE*               writeFile           = 0;
FILE*               summaryFile         = 0;

void sigHandler( int signal ) {
  psignal( signal, "Signal received by pgpWidget");
  if (writeFile  ) fclose(writeFile);
  if (summaryFile) fclose(summaryFile);
  printf("Signal handler pulling the plug\n");
  ::exit(signal);
}


#include "psdaq/pgp/pgpGen4Daq/include/DmaDriver.h"
#include "psdaq/pgp/pgpGen4Daq/app/PgpDaq.hh"

using Pds::HSD::EventHeader;
using Pds::HSD::StreamHeader;
using Pds::HSD::RawStream;
using Pds::HSD::ThrStream;

class IlvBuilder {
public:
  IlvBuilder(unsigned blen) : _buffer(new char[4*blen]),
                              _blen  (blen),
                              _extent(0),
                              _lanes (0) {}
public:
  void next(const char* p, unsigned len, unsigned lane) {
    if (_lanes && len != _extent) {
      printf("IlvBuilder wrong length  len %u  lane %u  extent %u\n",
             len, lane, _extent);
    }
    _extent = len;

    unsigned blen = (_blen < len) ? _blen : len;
    for(unsigned i=0; i<blen; i++) {
      _buffer[i*4+lane] = p[i];
    }
    _lanes |= (1<<lane);
    if (_lanes == 0xf) {
      _lanes = 0;
      uint32_t* q = reinterpret_cast<uint32_t*>(_buffer);
      printf("--ILV--\n");
      for(unsigned i=0; i<blen; i++)
        printf("%08x%c",q[i],(i&7)==7 ? '\n':' ');
    }
  }
private:
  char*    _buffer;
  unsigned _blen;
  unsigned _extent;
  unsigned _lanes;
};

void printUsage(char* name) {
  printf( "Usage: %s [-h]  -P <deviceName> [options]\n"
      "    -h         Show usage\n"
      "    -P         Set pgpcard device name\n"
      "    -L <lanes> Mask of lanes\n"
      "    -c         number of times to read\n"
      "    -o         Print out up to maxPrint words when reading data\n"
      "    -f <file>  Record to file\n"
      "    -d <nsec>  Delay given number of nanoseconds per event\n"
      "    -D         Set debug value           [Default: 0]\n"
      "                 bit 00          print out progress\n"
      "    -N         Exit after N events\n"
      "    -r         Report rate\n"
      "    -v <mask>  Validate each event\n"
      "    -E <str>   Push 1Hz waveforms to record <str>\n"
      "    -I <len>   Interleaved\n",
      name
  );
}

void* countThread(void*);

static int      count = 0;
static int64_t  bytes = 0;
static unsigned lanes = 0;
static unsigned buffs = 0;
static unsigned errs  = 0;
static unsigned polls = 0;

int main (int argc, char **argv) {
  int           fd;
  int           numb;
  bool          print = false;
  const char*         dev = "/dev/pgpdaq0";
  unsigned            client              = 0;
  unsigned            maxPrint            = 1024;
  unsigned            debug               = 0;
  unsigned            nevents             = unsigned(-1);
  unsigned            delay               = 0;
  unsigned            lvalidate           = 0;
  bool                reportRate          = false;
  unsigned            lanem               = 0;
  const char*         pv                  = 0;
  IlvBuilder*         ilv                 = 0;
  ::signal( SIGINT, sigHandler );

  //  char*               endptr;
  extern char*        optarg;
  int c;
  while( ( c = getopt( argc, argv, "hI:P:L:d:D:c:f:F:N:o:rv:E:" ) ) != EOF ) {
    switch(c) {
    case 'I':
      ilv = new IlvBuilder(strtoul(optarg,NULL,0));
      break;
    case 'P':
      dev = optarg;
      break;
    case 'L':
      lanem = strtoul(optarg,NULL,0);
      break;
    case 'N':
      nevents = strtoul(optarg,NULL, 0);
      break;
    case 'D':
      debug = strtoul(optarg, NULL, 0);
      if (debug & 1) print = true;
      break;
    case 'd':
      delay = strtoul(optarg, NULL, 0);
      break;
    case 'c':
      numb = strtoul(optarg  ,NULL,0);
      break;
    case 'f':
      if (!(writeFile = fopen(optarg,"w"))) {
        perror("Opening save file");
        return -1;
      }
      break;
    case 'F':
      if (!(summaryFile = fopen(optarg,"w"))) {
        perror("Opening summary file");
        return -1;
      }
      break;
    case 'o':
      maxPrint = strtoul(optarg, NULL, 0);
      print = true;
      break;
    case 'r':
      reportRate = true;
      break;
    case 'v':
      lvalidate = strtoul(optarg, NULL, 0);
      break;
    case 'E':
      pv = optarg;
      break;
    case 'h':
      printUsage(argv[0]);
      return 0;
      break;
    default:
      printf("Error: Option could not be parsed, or is not supported yet!\n");
      printUsage(argv[0]);
      return 0;
      break;
    }
  }

  char cdev[64];
  sprintf(cdev,"%s_%u",dev,client);
  if ( (fd = open(cdev, O_RDWR)) <= 0 ) {
    std::cout << "Error opening " << cdev << std::endl;
    return(1);
  }

  //
  //  Map the lanes to this reader
  //
  {
    PgpDaq::PgpCard* p = (PgpDaq::PgpCard*)mmap(NULL, sizeof(PgpDaq::PgpCard), (PROT_READ|PROT_WRITE), (MAP_SHARED|MAP_LOCKED), fd, 0);
    uint32_t MAX_LANES = p->nlanes();
    for(unsigned i=0; i<MAX_LANES; i++)
      if (lanem & (1<<i)) {
        p->dmaLane[i].client = client;
        p->dmaLane[i].blocksPause = 32<<8;
        p->pgpLane[i].axil.txControl = 1;  // disable flow control
      }
  }


  EpicsPVA* pvraw=0;
  EpicsPVA* pvfex=0;
  if (pv) {
    std::string pvbase(pv);
    pvraw = new EpicsPVA((pvbase+":RAWDATA").c_str());
    pvfex = new EpicsPVA((pvbase+":FEXDATA").c_str());
  }

  // Allocate a buffer
  uint32_t* data  = new uint32_t[0x80000];
  struct DmaReadData rd;
  rd.data  = reinterpret_cast<uintptr_t>(data);

  pthread_attr_t tattr;
  pthread_attr_init(&tattr);
  pthread_t thr;
  if (reportRate) {
    if (pthread_create(&thr, &tattr, &countThread, 0))
      perror("Error creating RDMA status thread");
  }

  RawStream::verbose( (lvalidate>>28)&7 );
  const Pds::HSD::EventHeader* event = reinterpret_cast<const Pds::HSD::EventHeader*>(data);
  RawStream* raw = 0;

  unsigned nextCount[8];
  uint64_t ppulseId =0, dpulseId =0;
  memset(nextCount,0,sizeof(nextCount));

  unsigned tsec=0;

  // DMA Read
  while(1) {
    bool lerr = false;

    rd.index = 0;
    ssize_t ret = read(fd, &rd, sizeof(rd));
    if (ret < 0) {
      perror("Reading buffer");
      break;
    }

    polls++;

    if (!rd.size) {
      continue;
    }

    if (nevents-- == 0)
      break;

    unsigned lane   = (rd.dest>>5)&7;

    if (print || event->eventType()) {

      event->dump();

      for (unsigned x=0; x<maxPrint; x++) {
        printf("%08x%c", data[x], (x%8)==7 ? '\n':' ');
      }
      if (maxPrint%8)
        printf("\n");

      if (count >= numb)
        print = false;
    }
    bytes += rd.size;

    if (lvalidate) {
      if (lvalidate&1) {
        //  Check that pulseId increments by a constant
        uint64_t pulseId = (uint64_t(data[1])<<32) | data[0];
        if (ppulseId) {
          if (dpulseId > 100 && pulseId != (ppulseId+dpulseId))
            printf("\tPulseId = %016llx [%016llx, %016llx]\n",
                   (unsigned long long)pulseId,
                   (unsigned long long)(pulseId+dpulseId),
                   (unsigned long long)(pulseId-ppulseId));
          dpulseId = pulseId - ppulseId;
        }
        ppulseId = pulseId;
      }
      if (lvalidate&2) {
        //  Check that analysis count increments by one
        //        unsigned count = data[5];
        unsigned count = data[4];
        if (nextCount[lane] && (count != nextCount[lane])) {
          lerr = true;
          if (errs < 100)
            printf("\tanalysisCount = %08x [%08x] lane %u  delta %d\n",
                   count, nextCount[lane], lane, count-nextCount[lane]);
        }
        nextCount[lane] = (count+1)&0x00ffffff;
      }

      if (event->eventType()==0) {
        const StreamHeader& rhdr = *reinterpret_cast<const StreamHeader*>(event+1);
        if (rhdr.strmtype()==0 && (lvalidate&4)) {
          //  Check that the raw payload for the test pattern is in lock step
          if (!raw)
            raw = new RawStream(*event, rhdr);
          else
            lerr |= !raw->validate(*event, rhdr);
        }
        if (rhdr.strmtype()==0 && (lvalidate&8)) {
          //  Check that the fex payload matches the raw payload
          const StreamHeader* thdr = reinterpret_cast<const StreamHeader*>(event+1);
          for(unsigned i=1; i<event->streams(); i++) {
            if (thdr->strmtype()==1) {
              ThrStream tstr(*thdr);
              lerr |= !tstr.validate(rhdr);
              break;
            }
          }
        }
      }
    }

    lanes |= 1<<lane;

    { unsigned buff = data[9]>>16;
      buffs |= (1<<buff); }

    //  Check for pgp errors
    lerr |= rd.error;

    ++count;
    if (lerr) {
      if (++errs > 20) {
        RawStream::verbose(0);
      }
      if (lvalidate&(1<<31))
        event->dump();
    }

    if (ilv)
      ilv->next(reinterpret_cast<const char*>(data),rd.size,lane);

    if (writeFile) {
      data[6] |= (lane<<20);  // write the lane into the event header
      fwrite(data,rd.size,1,writeFile);
    }

    if (summaryFile) {
      fwrite(event,sizeof(*event),1,summaryFile);
    }

    if (pv && tsec != data[3] && lane==0) {
      tsec = data[3];

      Pds::HSD::EventHeader& evhdr = *reinterpret_cast<Pds::HSD::EventHeader*>(data);

      Pds::HSD::StreamHeader& rawhdr = *new(&evhdr+1) Pds::HSD::StreamHeader;

      const uint16_t* raw = reinterpret_cast<const uint16_t*>(&rawhdr+1) + rawhdr.boffs();
      pvd::shared_vector<const unsigned> pvrawvecin;
      pvraw->getVectorAs(pvrawvecin);
      pvd::shared_vector<unsigned> pvrawvecout(thaw(pvrawvecin));
      for(unsigned i=0; i<rawhdr.samples() && i<pvraw->nelem(); i++)
        pvrawvecout[i] = raw[i];
      pvraw->putFromVector(freeze(pvrawvecout));

      void* next = const_cast<uint16_t*>(&raw[rawhdr.samples()]);

      Pds::HSD::StreamHeader& fexhdr = *new(next) Pds::HSD::StreamHeader;

      const uint16_t* fex = reinterpret_cast<const uint16_t*>(&fexhdr+1) + fexhdr.boffs();
      int nskip=0;
      pvd::shared_vector<const unsigned> pvfexvecin;
      pvfex->getVectorAs(pvfexvecin);
      pvd::shared_vector<unsigned> pvfexvecout(thaw(pvfexvecin));
      for(unsigned i=0; i<fexhdr.samples() && i<pvfex->nelem(); i++) {
        if (nskip) {
          pvfexvecout[i] = 0x200;
          nskip--;
        }
        else {
          if (fex[i]&0x8000) {
            nskip = fex[i]&0x7fff;
            pvfexvecout[i] = 0x200;
          }
          else
            pvfexvecout[i] = fex[i];
        }
      }
      pvfex->putFromVector(freeze(pvfexvecout));
    }

    if (delay) {
      timespec tv = { .tv_sec=0, .tv_nsec=delay };
      while( nanosleep(&tv, &tv) )
        ;
    }
  }
  count = -1;

  if (reportRate)
    pthread_join(thr,NULL);
  free(data);
  //  sleep(5);
  //  close(fd);
  return 0;
}

void* countThread(void* args)
{
  timespec tv;
  clock_gettime(CLOCK_REALTIME,&tv);
  unsigned opolls = polls;
  unsigned ocount = count;
  int64_t  obytes = bytes;
  while(1) {
    usleep(1000000);
    timespec otv = tv;
    clock_gettime(CLOCK_REALTIME,&tv);
    unsigned npolls = polls;
    unsigned ncount = count;
    int64_t  nbytes = bytes;

    double dt     = double( tv.tv_sec - otv.tv_sec) + 1.e-9*(double(tv.tv_nsec)-double(otv.tv_nsec));
    double prate  = double(npolls-opolls)/dt;
    double rate   = double(ncount-ocount)/dt;
    double dbytes = double(nbytes-obytes)/dt;
    unsigned dbsc = 0, rsc=0, prsc=0;

    if (count < 0) break;

    static const char scchar[] = { ' ', 'k', 'M' };

    if (prate > 1.e6) {
      prsc     = 2;
      prate   *= 1.e-6;
    }
    else if (prate > 1.e3) {
      prsc     = 1;
      prate   *= 1.e-3;
    }

    if (rate > 1.e6) {
      rsc     = 2;
      rate   *= 1.e-6;
    }
    else if (rate > 1.e3) {
      rsc     = 1;
      rate   *= 1.e-3;
    }

    if (dbytes > 1.e6) {
      dbsc    = 2;
      dbytes *= 1.e-6;
    }
    else if (dbytes > 1.e3) {
      dbsc    = 1;
      dbytes *= 1.e-3;
    }

    printf("Rate %7.2f %cHz [%u]:  Size %7.2f %cBps [%lld B]  lanes %02x  buffs %04x  errs %04x : polls %7.2f %cHz\n",
           rate  , scchar[rsc ], ncount,
           dbytes, scchar[dbsc], (long long)nbytes, lanes, buffs, errs,
           prate , scchar[prsc]);
    lanes = 0;
    buffs = 0;

    opolls = npolls;
    ocount = ncount;
    obytes = nbytes;
  }
  return 0;
}
