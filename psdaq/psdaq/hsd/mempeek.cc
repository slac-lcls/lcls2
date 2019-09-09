#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <fcntl.h>
#include <stdint.h>
#include <sys/mman.h>

void usage(const char* p) {
  printf("Usage: %s [options]\n",p);
  printf("Options: -a <address> \n"
         "         -n <bytes>   \n");
}

int main(int argc, char** argv)
{
  extern char* optarg;

  int c;
  off_t    adx    = 0;
  unsigned nbytes = 1;

  while ( (c=getopt( argc, argv, "a:n:")) != EOF ) {
    switch(c) {
    case 'a':
      adx    = strtoull(optarg, NULL, 0);
      break;
    case 'n':
      nbytes = strtoul(optarg, NULL, 0);
      break;
    default:
      usage(argv[0]);
      exit(1);
    }
  }
  
  int fd = open("/dev/mem", O_RDWR | O_SYNC);
  if (fd<0) {
    perror("Open device failed");
    return -1;
  }

  uint8_t* ptr = (uint8_t*)mmap(0, nbytes, PROT_READ|PROT_WRITE, MAP_SHARED, fd, adx);
  if (ptr == MAP_FAILED) {
    perror("Failed to map");
    return 0;
  }

  for(unsigned i=0; i<nbytes; i++)
    printf("%02x%c", ptr[i], (i%16)==15 ? '\n':'.');

  close(fd);

  return 0;
}
