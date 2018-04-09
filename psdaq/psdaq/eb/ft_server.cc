#include "psdaq/eb/Endpoint.hh"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>

using namespace Pds::Fabrics;

static const char* PORT_DEF = "12345";
static const size_t BUFFER_DEF = 800;

static void showUsage(const char* p)
{
  printf("\nSimple libfabrics server example (its pair is the ft_client example).\n"
         "\n"
         "In this simple example the server waits for a connection, and when a client connects the server\n"
         "sends a message containing info on a readable memory region. The client then initiates a specified\n"
         "number of remote reads on that memory and then disconnects. After disconnect the server can accept\n"
         "new incoming connections.\n"
         "\n"
         "Usage: %s [-h|--help]\n"
         "    -a|--addr     the local address to which the server binds (default: libfabrics 'best' choice)\n"
         "    -b|--buffer   the size of the local memory buffer to allocate (default: %lu bytes)\n"
         "    -p|--port     the port or libfaric 'service' the server will use (default: %s)\n"
         "    -h|--help     print this message and exit\n", p, BUFFER_DEF, PORT_DEF);
}

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

    ssize_t rc;
    if ((rc = endp->send_sync(key_buff, key_size, mr))) {
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
  bool show_usage = false;
  const char* addr = NULL;
  const char* port = PORT_DEF;
  size_t buff_size = sizeof(RemoteAddress);
  size_t data_size = BUFFER_DEF;
  char* buff = NULL;
  char* data_buff = NULL;

  const char* str_opts = ":ha:b:p:";
  const struct option lo_opts[] =
  {
    {"help",    0, 0, 'h'},
    {"addr",    1, 0, 'a'},
    {"buffer",  1, 0, 'b'},
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
      case 'b':
        data_size = strtoul(optarg, NULL, 0);
        break;
      case 'p':
        port = optarg;
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

  if (!data_size) {
    printf("%s: invalid buffer size requested -- %lu\n", argv[0], data_size);
    return 1;
  }

  if (show_usage) {
    showUsage(argv[0]);
    return 1;
  }

  buff_size += data_size;
  buff = new char[buff_size];
  data_buff = buff + sizeof(RemoteAddress);

  ((uint64_t*) data_buff)[1] = 0xadd;
  ((uint64_t*) data_buff)[2] = 0xdeadbeef;
  for (unsigned i=3; i< (data_size / sizeof(uint64_t)); i++)
    ((uint64_t*) data_buff)[i] = i;

  PassiveEndpoint* pendp = new PassiveEndpoint(addr, port);
  if (pendp->state() != EP_UP) {
    fprintf(stderr, "Failed to initialize fabrics endpoint: %s\n", pendp->error());
    return pendp->error_num();
  }

  // get a pointer to the fabric
  Fabric* fab = pendp->fabric();
  printf("using %s provider\n", fab->provider());
  // register the memory buffer
  MemoryRegion* mr = fab->register_memory(buff, buff_size);
  if (!mr) {
    fprintf(stderr, "Failed to register memory region: %s\n", fab->error());
    return fab->error_num();
  }
  RemoteAddress keys(mr->rkey(), (uint64_t) data_buff, data_size);
  memcpy(buff, &keys, sizeof(keys));

  ret = listen(pendp, mr, buff, sizeof(keys), data_buff);

  delete pendp;
  delete[] buff;

  return ret;
}
