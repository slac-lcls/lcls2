#include "psdaq/eb/Endpoint.hh"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>

using namespace Pds::Fabrics;

const static int MSG_LEN = 40;
static const char* PORT_DEF = "12345";
static const unsigned NUM_DEF = 3;

static void showUsage(const char* p)
{
  printf("\nSimple libfabrics writemsg server example (its pair is the ft_writemsg_client example).\n"
         "\n"
         "This example uses the writemsg/readmsg api in libfabrics to pingpong the specified number of\n"
         "buffers back and forth making a few changes\n"
         "\n"
         "Usage: %s [-h|--help]\n"
         "    -a|--addr     the local address to which the server binds (default: libfabrics 'best' choice)\n"
         "    -p|--port     the port or libfaric 'service' the server will use (default: %s)\n"
         "    -n|--nbuffer  the number of the local memory buffers to allocate (default: %u)\n"
         "    -h|--help     print this message and exit\n", p, PORT_DEF, NUM_DEF);
}

int main(int argc, char *argv[])
{
  int ret = FI_SUCCESS;
  bool show_usage = false;
  unsigned num = NUM_DEF;
  const char* addr = NULL;
  const char* port = PORT_DEF;

  const char* str_opts = ":ha:p:n:";
  const struct option lo_opts[] =
  {
    {"help",    0, 0, 'h'},
    {"addr",    1, 0, 'a'},
    {"nbuffer", 1, 0, 'n'},
    {"port",    1, 0, 'p'},
    {0,         0, 0,  0 }
  };

  int option_idx = 0;
  while (int opt = getopt_long(argc, argv, str_opts, lo_opts, &option_idx )) {
    if ( opt == -1 ) break;

    switch(opt) {
      case 'h':
        showUsage(argv[0]);
        return 0;
      case 'a':
        addr = optarg;
        break;
      case 'p':
        port = optarg;
        break;
      case 'n':
        num = strtoul(optarg, NULL, 0);
        break;
      default:
        show_usage = true;
        break;
    }
  }

  if (optind < argc) {
    printf("%s: invalid argument -- %s\n", argv[0], argv[optind]);
    show_usage = true;
  }

  if (!num) {
    printf("%s: invalid number of buffers requested -- %u\n", argv[0], num);
    return 1;
  }

  if (show_usage) {
    showUsage(argv[0]);
    return 1;
  }

  PassiveEndpoint* pendp = new PassiveEndpoint(addr, port);
  char* data_buffer = new char[MSG_LEN * num];
  RemoteAddress* raddr = new RemoteAddress();
  MemoryRegion* mr = pendp->fabric()->register_memory(data_buffer, MSG_LEN*num);
  MemoryRegion* mr_keys = pendp->fabric()->register_memory(raddr, sizeof(RemoteAddress));
  *raddr = RemoteAddress(mr->rkey(), (uint64_t) data_buffer, MSG_LEN*num);
  Endpoint* endp = NULL;

  printf("using %s provider\n", pendp->fabric()->provider());

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

    endp->recv_comp_data();
    if (endp->send_sync(data_buffer, sizeof(num))) {
      fprintf(stderr, "Endpoint send error %s\n", endp->error());
    }
    if (endp->send_sync(raddr, sizeof(RemoteAddress), mr_keys)) {
      fprintf(stderr, "Endpoint send error %s\n", endp->error());
    }

    struct fi_cq_data_entry comp;

    printf("waiting!\n");
    endp->txcq()->comp_wait(&comp, 1);

    if ((comp.flags & FI_REMOTE_WRITE) && (comp.flags & FI_REMOTE_CQ_DATA)) {
      for (unsigned i=0; i<num; i++) {
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
