#include "psdaq/eb/Endpoint.hh"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>


const static int MSG_LEN = 40;


using namespace Pds::Fabrics;

int main(int argc, char *argv[])
{
  if (argc != 2) {
    fprintf(stderr, "usage: %s nbuffers\n", argv[0]);
    return -1;
  }

  int ret = 0;
  int num = atoi(argv[1]);
  PassiveEndpoint* pendp = new PassiveEndpoint(NULL, "1234");
  char* data_buffer = new char[MSG_LEN * num];
  RemoteAddress* raddr = new RemoteAddress();
  MemoryRegion* mr = pendp->fabric()->register_memory(data_buffer, MSG_LEN*num);
  MemoryRegion* mr_keys = pendp->fabric()->register_memory(raddr, sizeof(RemoteAddress));
  *raddr = RemoteAddress(mr->rkey(), (uint64_t) data_buffer, MSG_LEN*num);
  Endpoint* endp = NULL;
  
  pendp->listen();

  while(1) {
    memset(data_buffer, 0, MSG_LEN*num);
    endp = pendp->accept();
    if (!endp) {
      fprintf(stderr, "Endpoint accept failed: %s\n", pendp->error());
      ret = pendp->error_num();
    }

    printf("client connected!\n");

    memcpy(data_buffer, &num, sizeof(num));

    endp->recv_comp_data(data_buffer);
    if (!endp->send_sync(data_buffer, sizeof(num))) {
      fprintf(stderr, "Endpoint send error %s\n", endp->error());
    }
    if (!endp->send_sync(raddr, sizeof(RemoteAddress), mr_keys)) {
      fprintf(stderr, "Endpoint send error %s\n", endp->error());
    }

    int num_comp;
    struct fi_cq_data_entry comp;

    printf("waiting!\n");
    endp->comp_wait(&comp, &num_comp, 1);

    if ((comp.flags & FI_REMOTE_WRITE) && (comp.op_context == data_buffer)) {
      for (int i=0; i<num; i++) {
        printf("%s\n", &data_buffer[i*MSG_LEN]);
        if (i%3 == 0) {
          strcpy(&data_buffer[i*MSG_LEN], "cat");
        }
      }
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
      } else {
        pendp->close(endp);
        printf("client disconnected!\n");
      }
    } else {
      fprintf(stderr, "Wating for event failed: %s\n", endp->error());
      ret = endp->error_num();
      pendp->close(endp);
      break;
    }
  }

  if (endp) delete endp;
  delete pendp;
  delete raddr;
  delete data_buffer;

  return ret;
}
