#include "Endpoint.hh"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FI_ERR(cmd, ret) fprintf(stderr, "%s: %s(%d)\n", cmd, fi_strerror(-ret), ret)

using namespace Pds::Fabrics;

int connect(Endpoint* endp, char* buff, size_t buff_size, int size)
{
  int num_comp;
  struct fi_cq_data_entry comp;

  // get a pointer to the fabric
  Fabric* fab = endp->fabric();
  // register the memory buffer
  MemoryRegion* mr = fab->register_memory(buff, buff_size);
  if (!mr) {
    fprintf(stderr, "Failed to register memory region: %s\n", fab->error());
    return fab->error_num();
  }

  RemoteAddress keys(mr->rkey(), (uint64_t) buff, buff_size);
  memcpy(buff, &keys, sizeof(keys));

  if(!endp->connect()) {
    fprintf(stderr, "Failed to connect endpoint: %s\n", endp->error());
    return endp->error_num();
  }

  // post a recv buffer for the completion data
  if (!endp->recv_comp_data(buff)) {
    fprintf(stderr, "Failed posting completion recv from remote server: %s\n", endp->error());
    return endp->error_num();
  }

  // Send to the server where we want it to put the data
  if (!endp->send_sync(buff, sizeof (keys), mr)) {
    fprintf(stderr, "Failed receiving memory keys from remote server: %s\n", endp->error());
    return endp->error_num();
  }

  // wait for the completion request from the remote write
  if (!endp->comp_wait(&comp, &num_comp, 1)) {
    fprintf(stderr, "Failed waiting for write completion data: %s\n", endp->error());
    return endp->error_num();
  }

  if ((comp.flags & FI_REMOTE_WRITE) && (comp.op_context == buff)) {
    printf("write completion received! - server data value: %lu\n", comp.data);
    uint64_t* data_buff = (uint64_t*) buff;
    printf("data (key, addr, values): %lu %lu", keys.rkey, keys.addr);
    for(int j = 0; j < size; j++) {
      printf(" %lx", data_buff[j]);
    }
    printf("\n");
  }

  return 0;
}

int main(int argc, char *argv[])
{
  if (argc != 3) {
    fprintf(stderr, "usage: %s addr size\n", argv[0]);
    return -1;
  }

  int ret = 0;
  char *addr = argv[1];
  int size = atoi(argv[2]);
  size_t buff_size = (size_t) size*sizeof(uint64_t);
  char* buff = new char[buff_size];

  Endpoint* endp = new Endpoint(addr, "1234");
  if (endp->state() != EP_UP) {
    fprintf(stderr, "Failed to initialize fabrics endpoint: %s\n", endp->error());
    ret = endp->error_num();
  } else {
    ret = connect(endp, buff, buff_size, size);
  }

  if (endp) delete endp;
  delete[] buff;

  return ret;
}
