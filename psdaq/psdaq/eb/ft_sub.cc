#include "psdaq/eb/Endpoint.hh"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>

using namespace Pds::Fabrics;

static const char* PORT_DEF = "12345";
static const uint64_t COUNT_DEF = 10;

static void showUsage(const char* p)
{
  printf("\nSimple libfabrics subscriber example (its pair is the ft_publish example).\n"
         "\n"
         "This example is simulating something similar to zeromq's publish/subscribe model - a.k.a\n"
         "a simulated multicast using an underlying connected interface. The server periodically sends a\n"
         "data buffer to all of its subscribers. A client connects to server to 'subscribe' and disconnects\n"
         "to unsubscribe.\n"
         "\n"
         "Usage: %s [-h|--help]\n"
         "    -a|--addr     the address of the publish server\n"
         "    -p|--port     the port or libfaric 'service' the server uses (default: %s)\n"
         "    -c|--count    the max number of times to the data to subscribers before exiting (default: %lu)\n"
         "    -h|--help     print this message and exit\n", p, PORT_DEF, COUNT_DEF);
}

int connect(Endpoint* endp, char* buff, size_t buff_size, uint64_t max_count)
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

  uint64_t count = 0;
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

    if (endp->recv(buff, buff_size, mr)) {
      fprintf(stderr, "Failed receiving data from remote server: %s\n", endp->error());
      return endp->error_num();
    }

    if ( (comp_num = endp->txcq()->comp_wait(&comp, 1, 1000)) > 0) {
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

  printf("Received %lu out of %lu recvs!\n", count, max_count);

  return 0;
}

int main(int argc, char *argv[])
{
  int ret = FI_SUCCESS;
  bool show_usage = false;
  char *addr = NULL;
  const char *port = PORT_DEF;
  uint64_t count = COUNT_DEF;
  size_t buff_size = sizeof(uint64_t)*10;
  char* buff = new char[buff_size];

  const char* str_opts = ":ha:p:c:";
  const struct option lo_opts[] =
  {
      {"help",  0, 0, 'h'},
      {"addr",  1, 0, 'a'},
      {"port",  1, 0, 'p'},
      {"count", 1, 0, 'c'},
      {0,       0, 0,  0 }
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
      case 'c':
        count = strtoul(optarg, NULL, 0);
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

  if (!addr) {
    printf("%s: server address is required\n", argv[0]);
    show_usage = true;
  }

  if (show_usage) {
    showUsage(argv[0]);
    return 1;
  }

  Endpoint* endp = new Endpoint(addr, port);
  if (endp->state() != EP_UP) {
    fprintf(stderr, "Failed to initialize fabrics endpoint: %s\n", endp->error());
    ret = endp->error_num();
  } else {
    printf("using %s provider\n", endp->fabric()->provider());
    ret = connect(endp, buff, buff_size, count);
  }

  if (endp) delete endp;
  delete[] buff;

  return ret;
}
