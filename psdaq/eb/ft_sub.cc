#include "Endpoint.hh"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FI_ERR(cmd, ret) fprintf(stderr, "%s: %s(%d)\n", cmd, fi_strerror(-ret), ret)

using namespace Pds::Fabrics;

int connect(Endpoint* endp, char* buff, size_t buff_size, int max_count)
{
  // get a pointer to the fabric
  Fabric* fab = endp->fabric();
  // register the memory buffer
  MemoryRegion* mr = fab->register_memory(buff, buff_size);
  if (!mr) {
    fprintf(stderr, "Failed to register memory region: %s\n", fab->error());
    return fab->error_num();
  }

  if(!endp->connect()) {
    fprintf(stderr, "Failed to connect endpoint: %s\n", endp->error());
    return endp->error_num();
  }

  int count = 0;
  bool cm_entry;
  struct fi_eq_cm_entry entry;
  uint32_t event;

  int comp_num;
  struct fi_cq_data_entry comp;

  while(count < max_count) {
    if (endp->event(&event, &entry, &cm_entry)) {
      if (cm_entry && event == FI_SHUTDOWN) {
        printf("Remote endpoint has shut down!\n");
        break;
      }
    }

    if (!endp->recv(buff, buff_size, mr)) {
      fprintf(stderr, "Failed receiving data from remote server: %s\n", endp->error());
      return endp->error_num();
    }

    if (endp->comp_wait(&comp, &comp_num, 1, 1000)) {
      if (comp_num == 1 && comp.flags & FI_RECV) {
        printf("data");
        uint64_t* data_buff = (uint64_t*) buff;
        for(unsigned j = 0; j < (buff_size/sizeof(uint64_t)); j++) {
          printf(" %lx", data_buff[j]);
        }
        printf("\n");

        count++;
      }
    }
  }

  printf("Received %d out of %d recvs!\n", count, max_count);

  return 0;
}

int main(int argc, char *argv[])
{
  if (argc != 3) {
    fprintf(stderr, "usage: %s addr count\n", argv[0]);
    return -1;
  }

  int ret = 0;
  char *addr = argv[1];
  int count = atoi(argv[2]);
  size_t buff_size = sizeof(uint64_t)*10;
  char* buff = new char[buff_size];

  Endpoint* endp = new Endpoint(addr, "1234");
  if (endp->state() != EP_UP) {
    fprintf(stderr, "Failed to initialize fabrics endpoint: %s\n", endp->error());
    ret = endp->error_num();
  } else {
    ret = connect(endp, buff, buff_size, count);
  }

  if (endp) delete endp;
  delete[] buff;

  return ret;
}
