#include "Endpoint.hh"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FI_ERR(cmd, ret) fprintf(stderr, "%s: %s(%d)\n", cmd, fi_strerror(-ret), ret)

using namespace Pds::Fabrics;

int connect(Endpoint* endp, char* buff, size_t buff_size, int size, int count)
{
  int* comp_num;
  int num_comp;
  struct fi_cq_data_entry comp;
  RemoteAddress keys;

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

  if (!endp->recv_sync(buff, sizeof (keys), mr)) {
    fprintf(stderr, "Failed receiving memory keys from remote server: %s\n", endp->error());
    return endp->error_num();
  }

  memcpy(&keys, buff, sizeof (keys));
  if (size*sizeof(uint64_t) > keys.extent) {
    fprintf(stderr, "Size of remote memory region (%lu) is less than size to read: %lu\n", keys.extent, size*sizeof(uint64_t));
    return -1;
  }

  for (int i = 0; i < count; i++) {
    if (!endp->read(buff, size*sizeof(uint64_t), &keys, &i, mr)) {
      fprintf(stderr, "Failed posting read to endpoint: %s\n", endp->error());
      return endp->error_num();;
    }

    if (!endp->comp_wait(&comp, &num_comp, 1)) {
      fprintf(stderr, "Failed waiting for read completion: %s\n", endp->error());
      return endp->error_num();
    }

    if (comp.flags & FI_READ) {
      comp_num = (int*) comp.op_context;
      printf("finished read %d\n", *comp_num);
      uint64_t* data_buff = (uint64_t*) buff;
      printf("data (key, addr, values): %lu %lu", keys.rkey, keys.addr);
      for(int j = 0; j < size; j++) {
        printf(" %lx", data_buff[j]);
      }
      printf("\n");
    }
  }

  return 0;
}

int main(int argc, char *argv[])
{
  if (argc != 4) {
    fprintf(stderr, "usage: %s addr size count\n", argv[0]);
    return -1;
  }

  int ret = 0;
  char *addr = argv[1];
  int size = atoi(argv[2]);
  int count = atoi(argv[3]);
  size_t buff_size = 32*1024*1024;
  char* buff = new char[buff_size];

  Endpoint* endp = new Endpoint(addr, "1234");
  if (endp->state() != EP_UP) {
    fprintf(stderr, "Failed to initialize fabrics endpoint: %s\n", endp->error());
    ret = endp->error_num();
  } else {
    ret = connect(endp, buff, buff_size, size, count);
  }

  if (endp) delete endp;
  delete[] buff;

  return ret;
}
