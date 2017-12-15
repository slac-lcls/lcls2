
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

int main (int argc, char **argv) {
  int           fd;
  uint          x;
  int           ret;
  time_t        t;
  uint          lane;
  uint          vc;
  uint          size;
  uint          *data;
  unsigned      count = 0;
  unsigned      number = 0;

  if (argc < 5) {
    printf("Usage: %s device lane vc size [number]\n", argv[0]);
    return(1);
  }

  // Get args
  lane  = atoi(argv[2]);
  vc    = atoi(argv[3]);
  size  = atoi(argv[4]);

  if (argc == 6) {
    number = atoi(argv[5]);
  }

  // Check ranges
  if ( size == 0 || lane > 7 || vc > 3 ) {
    printf("Invalid size, lane or vc value : %u, %u or %u\n", size, lane, vc);
    return(1);
  }

  if ( (fd = open(argv[1], O_RDWR)) <= 0 ) {
    cout << "Error opening " << argv[1] << endl;
    perror(argv[1]);
    return(1);
  }

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

  data = (uint *)malloc(sizeof(uint)*size);
  
  do {
    // DMA Write
    cout << "Sending:";
    cout << " Lane=" << dec << lane;
    cout << ", Vc=" << dec << vc << endl;
    for (x=0; x<size; x++) {
      data[x] = random();
      cout << " 0x" << setw(8) << setfill('0') << hex << data[x];
      if ( ((x+1)%10) == 0 ) cout << endl << "   ";
    }
    cout << endl;

    pollfd pfd;
    pfd.fd      = fd;
    pfd.events  = POLLOUT;
    pfd.revents = 0;

    int result = poll(&pfd, 1, -1);
    printf("poll result = %d\n",result);
    if (result < 0) {
      perror("poll");
      return -1;
    }

    ret = dmaWrite(fd, data, size, axisSetFlags(1, 1, 0), dest);
    cout << "Returned " << dec << ret << endl;

  } while ( count++ < number );
  free(data);

  close(fd);
  return(0);
}
