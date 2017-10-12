#include "Endpoint.hh"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

using namespace Pds::Fabrics;

int listen(PassiveEndpoint* pendp, MemoryRegion* mr, char* key_buff, size_t key_size, char* data_buff, size_t max_data_size)
{
  int ret = FI_SUCCESS;
  uint64_t value = 0;
  RemoteAddress* keys = (RemoteAddress*) key_buff;

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

    if (!endp->recv_sync(key_buff, key_size, mr)) {
      fprintf(stderr, "Reciveing of remote memory keys failed: %s\n", endp->error());
      ret = endp->error_num();
      pendp->close(endp);
      break;
    }

    printf("received memory address from client!\n");

    if (keys->extent > max_data_size) {
      fprintf(stderr, "Client has requested more data (%lu) than size of local buffer: %lu\n", keys->extent, max_data_size);
      ret = -FI_ENOMEM;
      pendp->close(endp);
      break;
    }

    if(!endp->write_data_sync(data_buff, keys->extent, keys, value++, mr)) {
      fprintf(stderr, "Writeing of data to remote client failed: %s\n", endp->error());
      ret = endp->error_num();
      pendp->close(endp);
      break;
    }

    printf("posted requested write of size %lu to client\n", keys->extent);

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

    printf("client has disconnected!\n");

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
  
  
  ((uint64_t*) data_buff)[0] = 0xadd;
  ((uint64_t*) data_buff)[1] = 0xdeadbeef;
  for (unsigned i=2; i< ((buff_size-sizeof(RemoteAddress)) / sizeof(uint64_t)); i++)
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

  ret = listen(pendp, mr, buff, sizeof(RemoteAddress), data_buff, buff_size-sizeof(RemoteAddress));

  delete pendp;
  delete[] buff;

  return ret;
}
