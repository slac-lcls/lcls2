
#include <sys/types.h>
#include <linux/types.h>
#include <unistd.h>
#include <stdio.h>
#include <termios.h>
#include <fcntl.h>
#include <sstream>
#include <string>
#include <iomanip>
#include <iostream>
#include <string.h>
#include <stdlib.h>
#include <poll.h>

#include "DataDriver.h"

using namespace std;

void printUsage(char* name) {
  printf( "Usage: %s [-h]  -P <deviceName> [options]\n"
      "    -h        Show usage\n"
      "    -P        Set pgpcard device name\n"
      "    -l <lane> Lane to write\n"
      "    -v <vc>   Virtual channel to write\n"
      "    -s <words> Size to write\n"
      "    -n        number of times to write\n"
      "    -r        Read from device\n"
      "    -q        Quiet.  Only report rates\n"
      "    -o        Print out up to maxPrint words when reading data\n"
      "    -D        Set debug value           [Default: 0]\n"
      "                bit 00          print out progress\n"
      "    -N        Exit after N events\n",
      name
  );
}

static uint     nevents = 1;
static bool     lquiet  = false;
static uint     gcount  = 0;
static uint64_t gbytes  = 0;

void* readThread (void*);
void* countThread(void*);

int main (int argc, char **argv) {
  int           fd;
  uint          x;
  int           ret;
  time_t        t;
  uint          debug               = 0;
  uint          lane;
  uint          vc                  = 0;
  uint          size;
  uint*         data;
  const char*   dev                 = "/dev/datadev_0";
  bool          lRead               = false;

  extern char*        optarg;
  int c;
  while( ( c = getopt( argc, argv, "hP:l:D:c:n:s:o:rqv" ) ) != EOF ) {
    switch(c) {
    case 'P':
      dev = optarg;
      break;
    case 'l':
      lane = strtoul(optarg,NULL,0);
      printf("Asking for lane %x\n", lane);
      break;
    case 'n':
      nevents = strtoul(optarg,NULL, 0);
      break;
    case 'D':
      debug = strtoul(optarg, NULL, 0);
      break;
    case 'v':
      vc = strtoul(optarg, NULL, 0);
      break;
    case 's':
      size = strtoul(optarg, NULL, 0);
      break;
    case 'r':
      lRead = true;
      break;
    case 'q':
      lquiet = true;
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

  // Check ranges
  if ( size == 0 || lane > 7 || vc > 3 ) {
    printf("Invalid size, lane or vc value : %u, %u or %u\n", size, lane, vc);
    return(1);
  }

  if ( (fd = open(dev, O_RDWR)) <= 0 ) {
    cout << "Error opening " << dev << endl;
    perror(dev);
    return(1);
  }

  ioctl(fd, DMA_Set_Debug, debug);

  uint32_t dest = dmaDest(lane,vc);

  uint8_t mask[DMA_MASK_SIZE];
  dmaInitMaskBytes(mask);
  dmaAddMaskBytes (mask, dest);

  if (ioctl(fd, DMA_Set_MaskBytes, mask)<0) {
    perror("DMA_Set_MaskBytes");
    return -1;
  }

  time(&t);
  srandom(t);

  pthread_attr_t tattr;
  pthread_attr_init(&tattr);
  pthread_t thr;
  if (lRead) {
    if (pthread_create(&thr, &tattr, &readThread, &fd))
      perror("Error creating read thread");

    if (lquiet) {
      pthread_attr_t qattr;
      pthread_attr_init(&qattr);
      pthread_t qthr;
      if (pthread_create(&qthr, &qattr, &countThread, &fd))
        perror("Error creating count thread");
    }
  }

  data = (uint *)malloc(sizeof(uint)*size);
  for (x=0; x<size; x++) {
    data[x] = random();
    if (!lquiet) {
      cout << " 0x" << setw(8) << setfill('0') << hex << data[x];
      if ( ((x+1)%10) == 0 ) cout << endl << "   ";
    }
  }
  if (!lquiet)
    cout << endl;
  
  for(uint count=0; count<nevents; count++) {
    // DMA Write
    if (!lquiet) {
      cout << "Sending:";
      cout << " Lane=" << dec << lane;
      cout << ", Vc=" << dec << vc << endl;
    }

    pollfd pfd;
    pfd.fd      = fd;
    pfd.events  = POLLOUT;
    pfd.revents = 0;

    int result = poll(&pfd, 1, -1);
    if (!lquiet)
      printf("pollout result = %d\n",result);
    if (result < 0) {
      perror("poll");
      return -1;
    }

    //  Convert Size argument to bytes
    ret = dmaWrite(fd, data, size*4, axisSetFlags(2, 0, 0), dest);
    if (!lquiet)
      cout << "Returned " << dec << ret << endl;

  }
  free(data);

  if (lRead)
    pthread_join(thr,NULL);

  close(fd);

  return(0);
}

void* readThread(void* args)
{
  int fd = *(int*)args;

  // Allocate a buffer
  const uint maxSize = 1024*256;
  const uint maxPrint = 32;

  uint* data = (uint *)malloc(sizeof(uint)*maxSize);

  for(uint count=0; count<nevents; count++) {
    pollfd pfd;
    pfd.fd      = fd;
    pfd.events  = POLLIN;
    pfd.revents = 0;

    int result = poll(&pfd, 1, -1);
    if (!lquiet)
      printf("pollin result = %d\n",result);
    if (result < 0) {
      perror("poll");
      return 0;
    }

    uint32_t flags=0, err=0, dest=0;
    int ret = dmaRead(fd,data,maxSize,&flags,&err,&dest);

    gcount++;
    gbytes += ret;

    //  Return value is in bytes (convert to words)
    ret = (ret+3)>>2;

    if (lquiet) continue;

    if ( ret == 0 ) {
      cout << "Ret=" << dec << ret << endl;
    }
    else {
      cout << "Ret=" << dec << ret;
      cout << ", dest=" << hex << dest;
      cout << ", EOFE=" << hex << err;
      cout << ", size=" << dec << ret;
      cout << endl << "   ";
      
      for (uint x=0; x<ret && x<maxPrint; x++) {
        cout << " 0x" << setw(8) << setfill('0') << hex << data[x];
        if ( ((x+1)%10) == 0 ) cout << endl << "   ";
      }
      cout << endl;
    }
  }

  return 0;
}

void* countThread(void* args)
{
  timespec tv;
  clock_gettime(CLOCK_REALTIME,&tv);
  unsigned ocount = gcount;
  int64_t  obytes = gbytes;
  while(1) {
    usleep(1000000);
    timespec otv = tv;
    clock_gettime(CLOCK_REALTIME,&tv);
    unsigned ncount = gcount;
    int64_t  nbytes = gbytes;

    double dt     = double( tv.tv_sec - otv.tv_sec) + 1.e-9*(double(tv.tv_nsec)-double(otv.tv_nsec));
    double rate   = double(ncount-ocount)/dt;
    double dbytes = double(nbytes-obytes)/dt;
    double tbytes = dbytes/rate;
    unsigned dbsc = 0, rsc=0, tbsc=0;
    
    if (gcount < 0) break;

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
    
    if (tbytes > 1.e6) {
      tbsc    = 2;
      tbytes *= 1.e-6;
    }
    else if (tbytes > 1.e3) {
      tbsc    = 1;
      tbytes *= 1.e-3;
    }
    
    printf("Rate %7.2f %cHz [%u]:  Size %7.2f %cBps [%lld B] (%7.2f %cB/evt)\n", 
           rate  , scchar[rsc ], ncount, 
           dbytes, scchar[dbsc], (long long)nbytes, 
           tbytes, scchar[tbsc]);

    ocount = ncount;
    obytes = nbytes;
  }
  return 0;
}
