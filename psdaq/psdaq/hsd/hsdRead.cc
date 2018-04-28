#include "psdaq/epicstools/PVWriter.hh"
using Pds_Epics::PVWriter;

#include "psdaq/hsd/hsd.hh"

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

void sigHandler( int signal ) {
  psignal( signal, "Signal received by pgpWidget");
  if (writeFile) fclose(writeFile);
  printf("Signal handler pulling the plug\n");
  ::exit(signal);
}


#include "psdaq/pgp/pgpGen4Daq/include/DmaDriver.h"
#include "psdaq/pgp/pgpGen4Daq/app/PgpDaq.hh"

using namespace std;

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
      "    -E <str>   Push 1Hz waveforms to record <str>\n",
      name
  );
}

void* countThread(void*);

static int      count = 0;
static int64_t  bytes = 0;
static unsigned lanes = 0;
static unsigned buffs = 0;
static unsigned errs  = 0;

int main (int argc, char **argv) {
  int           fd;
  int           numb;
  bool          print = false;
  const char*         dev = "/dev/pgpdaq0";
  unsigned            client              = 0;
  int                 maxPrint            = 1024;
  unsigned            debug               = 0;
  unsigned            nevents             = unsigned(-1);
  unsigned            delay               = 0;
  unsigned            lvalidate           = 0;
  unsigned            payloadBuffers      = 0;
  bool                reportRate          = false;
  unsigned            lanem               = 0;
  const char*         pv                  = 0;
  ::signal( SIGINT, sigHandler );

  //  char*               endptr;
  extern char*        optarg;
  int c;
  while( ( c = getopt( argc, argv, "hP:L:d:D:c:f:N:o:rv:V:E:" ) ) != EOF ) {
    switch(c) {
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
    case 'V':
      payloadBuffers = strtoul(optarg, NULL, 0);
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
    cout << "Error opening " << cdev << endl;
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
        p->pgpLane[i].axil.txControl = 1;  // disable flow control
      }
  }


  PVWriter* pvraw=0;
  PVWriter* pvfex=0;
  if (pv) {
    printf("Initializing context\n");
    SEVCHK ( ca_context_create(ca_enable_preemptive_callback ),
             "Calling ca_context_create" );

    std::string pvbase(pv);
    pvraw = new PVWriter((pvbase+":RAWDATA").c_str());
    pvfex = new PVWriter((pvbase+":FEXDATA").c_str());
    ca_pend_io(0);
  }

  // Allocate a buffer
  uint32_t* data = new uint32_t[0x80000];
  struct DmaReadData rd;
  rd.data  = reinterpret_cast<uintptr_t>(data);

  pthread_attr_t tattr;
  pthread_attr_init(&tattr);
  pthread_t thr;
  if (reportRate) {
    if (pthread_create(&thr, &tattr, &countThread, 0))
      perror("Error creating RDMA status thread");
  }

  unsigned nextCount[8], nextPword=0;
  uint64_t ppulseId =0, dpulseId =0;
  memset(nextCount,0,sizeof(nextCount));

  unsigned tsec=0;

  struct tm tm_epoch;
  memset(&tm_epoch, 0, sizeof(tm_epoch));
  tm_epoch.tm_mday = 1;
  tm_epoch.tm_mon  = 0;
  tm_epoch.tm_year = 90;
  tm_epoch.tm_isdst = 0;

  time_t t_epoch = mktime(&tm_epoch);

  // DMA Read
  while(1) {
    rd.index = 0;
    ssize_t ret = read(fd, &rd, sizeof(rd));
    if (ret < 0) {
      perror("Reading buffer");
      break;
    }
    unsigned nwords = ret>>2;
    unsigned lane   = (rd.dest>>5)&7;

    if (!rd.size) {
      continue;
    }

    if (nevents-- == 0)
      break;

    if (print) {

      cout << "Ret=" << dec << ret;
      cout << ", pgpLane  =" << dec << lane;
      cout << ", pgpVc    =" << dec << ((rd.dest>>0)&0x1f);
      cout << ", EOFE     =" << dec << rd.error;
      cout << ", FifoErr  =" << dec << 0;
      cout << ", LengthErr=" << dec << 0;
      cout << endl << "   ";

      for (unsigned x=0; x<nwords && x<maxPrint; x++) {
        cout << " 0x" << setw(8) << setfill('0') << hex << data[x];
        if ( ((x+1)%10) == 0 ) cout << endl << "   ";
      }
      cout << endl;

      cout << "PID: " << hex << data[1] << setw(9) << setfill('0') << data[0] << endl;
      cout << "TS : " << dec << data[3] << "." << setw(9) << setfill('0') << data[2] << endl;
      time_t t = t_epoch + data[3];
      cout << asctime( localtime(&t) ) << endl;

      if (count >= numb)
        print = false;
    }
    if (ret>0)
      bytes += ret;

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
        if (nextCount[lane] && count != nextCount[lane])
          printf("\tanalysisCount = %08x [%08x] lane %u\n", 
                 count, nextCount[lane], lane);
        nextCount[lane] = count+1;
      }
      if (lvalidate&4) {
        //  Check that the first payload word increments by one
        if (payloadBuffers) {
          if (nextPword && data[8] != nextPword)
            printf("\tpayloadWord = %08x [%08x]\n", data[8], nextPword);
          nextPword = (data[8]+1)%payloadBuffers;
        }
      }
    }

    lanes |= 1<<lane;

    { unsigned buff = data[9]>>16;
      buffs |= (1<<buff); }

    //  Check for pgp errors
    if (rd.error)
      ++errs;

    //  Check for length errors
    { unsigned ext = 8;
      while( ext < nwords )
        ext += data[ext]/2 + 4;
      if (ext != nwords) errs++;
    }
      
    ++count;

    if (writeFile) {
      data[6] |= (lane<<20);  // write the lane into the event header
      fwrite(data,ret,1,writeFile);
    }

    if (pv && tsec != data[3] && lane==0) {
      tsec = data[3];

      Pds::HSD::EventHeader& evhdr = *reinterpret_cast<Pds::HSD::EventHeader*>(data); 

      Pds::HSD::StreamHeader& rawhdr = *new(&evhdr+1) Pds::HSD::StreamHeader;

      const uint16_t* raw = reinterpret_cast<const uint16_t*>(&rawhdr+1) + rawhdr.boffs();
      for(unsigned i=0; i<rawhdr.samples() && i<pvraw->nelem(); i++)
        reinterpret_cast<unsigned*>(pvraw->data())[i] = raw[i];
      pvraw->put();

      void* next = const_cast<uint16_t*>(&raw[rawhdr.samples()]);

      Pds::HSD::StreamHeader& fexhdr = *new(next) Pds::HSD::StreamHeader;

      const uint16_t* fex = reinterpret_cast<const uint16_t*>(&fexhdr+1) + fexhdr.boffs();
      int nskip=0;
      for(unsigned i=0; i<fexhdr.samples() && i<pvfex->nelem(); i++) {
        if (nskip) {
          reinterpret_cast<unsigned*>(pvfex->data())[i] = 0x200;
          nskip--;
        }
        else {
          if (fex[i]&0x8000) {
            nskip = fex[i]&0x7fff;
            reinterpret_cast<unsigned*>(pvfex->data())[i] = 0x200;
          }
          else
            reinterpret_cast<unsigned*>(pvfex->data())[i] = fex[i];
        }
      }
      pvfex->put();

      ca_flush_io();
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
  unsigned ocount = count;
  int64_t  obytes = bytes;
  while(1) {
    usleep(1000000);
    timespec otv = tv;
    clock_gettime(CLOCK_REALTIME,&tv);
    unsigned ncount = count;
    int64_t  nbytes = bytes;

    double dt     = double( tv.tv_sec - otv.tv_sec) + 1.e-9*(double(tv.tv_nsec)-double(otv.tv_nsec));
    double rate   = double(ncount-ocount)/dt;
    double dbytes = double(nbytes-obytes)/dt;
    unsigned dbsc = 0, rsc=0;
    
    if (count < 0) break;

    static const char scchar[] = { ' ', 'k', 'M' };
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
    
    printf("Rate %7.2f %cHz [%u]:  Size %7.2f %cBps [%lld B]  lanes %02x  buffs %04x  errs %04x\n", 
           rate  , scchar[rsc ], ncount, 
           dbytes, scchar[dbsc], (long long)nbytes, lanes, buffs, errs);
    lanes = 0;
    buffs = 0;

    ocount = ncount;
    obytes = nbytes;
  }
  return 0;
}
