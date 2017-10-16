#include "Endpoint.hh"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

using namespace Pds::Fabrics;

int listen(PassiveEndpoint* pendp, MemoryRegion* mr, char* key_buff, size_t key_size, char* data_buff)
{
  int ret = FI_SUCCESS;
  uint64_t value = 0;

  if(!pendp->listen()) {
    fprintf(stderr, "Failed to set passive fabrics endpoint to listening state: %s\n", pendp->error());
    return pendp->error_num();
  }
  printf("listening\n");

  Endpoint *endp = NULL;
  while(1) {
    endp = pendp->accept();
    if (!endp) {
      fprintf(stderr, "Endpoint accept failed: %s\n", pendp->error());
      return pendp->error_num();
    }

    printf("client connected!\n");
    memcpy(data_buff, &value, sizeof(value));
    value++;

    if (!endp->send_sync(key_buff, key_size, mr)) {
      fprintf(stderr, "Sending of local memory keys failed: %s\n", endp->error());
      ret = endp->error_num();
      pendp->close(endp);
      break;
    }

    bool cm_entry;
    struct fi_eq_cm_entry entry;
    uint32_t event;
    if(endp->event_wait(&event, &entry, &cm_entry)) {
      if (!cm_entry || event != FI_SHUTDOWN) {
        fprintf(stderr, "unexpected event %u - expected FI_SHUTDOWN (%u)", event, FI_SHUTDOWN);
        ret = endp->error_num();
        pendp->close(endp);
        break;
      }
    } else {
      fprintf(stderr, "Wating for event failed: %s\n", endp->error());
      ret = endp->error_num();
      pendp->close(endp);
      break;
    }

    pendp->close(endp);
  }

  return ret;
}

int main(int argc, char *argv[])
{
  int ret = FI_SUCCESS;
  size_t buff_size = 32*1024*1024;
  char* buff = new char[buff_size];
  char* data_buff = buff + sizeof(RemoteAddress);
  
  
  ((uint64_t*) data_buff)[1] = 0xadd;
  ((uint64_t*) data_buff)[2] = 0xdeadbeef;
  for (unsigned i=3; i< ((buff_size-sizeof(RemoteAddress)) / sizeof(uint64_t)); i++)
    ((uint64_t*) data_buff)[i] = i;

  PassiveEndpoint* pendp = new PassiveEndpoint(NULL, "1234");
  if (pendp->state() != EP_UP) {
    fprintf(stderr, "Failed to initialize fabrics endpoint: %s\n", pendp->error());
    return pendp->error_num();
  }

  // get a pointer to the fabric
  Fabric* fab = pendp->fabric();
  // register the memory buffer
  MemoryRegion* mr = fab->register_memory(buff, buff_size);
  if (!mr) {
    fprintf(stderr, "Failed to register memory region: %s\n", fab->error());
    return fab->error_num();
  }
  RemoteAddress keys(mr->rkey(), (uint64_t) data_buff, buff_size-sizeof(RemoteAddress));
  memcpy(buff, &keys, sizeof(keys));

  ret = listen(pendp, mr, buff, sizeof(keys), data_buff);

  delete pendp;
  delete[] buff;

  return ret;
}
