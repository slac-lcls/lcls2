#include "psdaq/eb/Endpoint.hh"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>

using namespace Pds::Fabrics;

static const char* PORT_DEF = "12345";
static const unsigned SIZE_DEF = 100;
static const unsigned COUNT_DEF = 1;

static void showUsage(const char* p)
{
  printf("\nSimple libfabrics client example (its pair is the ft_server example).\n"
         "\n"
         "In this simple example the server waits for a connection, and when a client connects the server\n"
         "sends a message containing info on a readable memory region. The client then initiates a specified\n"
         "number of remote reads on that memory and then disconnects. After disconnect the server can accept\n"
         "new incoming connections.\n"
         "\n"
         "Usage: %s [-h|--help]\n"
         "    -a|--addr     the address of the server\n"
         "    -p|--port     the port or libfaric 'service' the server uses (default: %s)\n"
         "    -s|--size     the number of 64bit ints to read from the server (default: %u)\n"
         "    -c|--count    the number of times to read the data from the server (default: %u)\n"
         "    -h|--help     print this message and exit\n", p, PORT_DEF, SIZE_DEF, COUNT_DEF);
}

int connect(Endpoint* endp, char* buff, size_t buff_size, unsigned size, unsigned count)
{
  int* comp_num;
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

  if (endp->recv_sync(buff, sizeof (keys), mr)) {
    fprintf(stderr, "Failed receiving memory keys from remote server: %s\n", endp->error());
    return endp->error_num();
  }

  memcpy(&keys, buff, sizeof (keys));
  if (size*sizeof(uint64_t) > keys.extent) {
    fprintf(stderr, "Size of remote memory region (%lu) is less than size to read: %lu\n", keys.extent, size*sizeof(uint64_t));
    return -1;
  }

  for (unsigned i = 0; i < count; i++) {
    if (endp->read(buff, size*sizeof(uint64_t), &keys, &i, mr)) {
      fprintf(stderr, "Failed posting read to endpoint: %s\n", endp->error());
      return endp->error_num();;
    }

    if (endp->rxcq()->comp_wait(&comp, 1) < 0) {
      fprintf(stderr, "Failed waiting for read completion: %s\n", endp->error());
      return endp->error_num();
    }

    if (comp.flags & FI_READ) {
      comp_num = (int*) comp.op_context;
      printf("finished read %d\n", *comp_num);
      uint64_t* data_buff = (uint64_t*) buff;
      printf("data (key, addr, values): %lu %lu", keys.rkey, keys.addr);
      for(unsigned j = 0; j < size; j++) {
        printf(" %lx", data_buff[j]);
      }
      printf("\n");
    }
  }

  return 0;
}

int main(int argc, char *argv[])
{
  int ret = FI_SUCCESS;
  bool show_usage = false;
  const char *addr = NULL;
  const char *port = PORT_DEF;
  unsigned size = SIZE_DEF;
  unsigned count = COUNT_DEF;
  size_t buff_size = 0;
  char* buff = NULL;

  const char* str_opts = ":ha:p:s:c:";
  const struct option lo_opts[] =
  {
      {"help",  0, 0, 'h'},
      {"addr",  1, 0, 'a'},
      {"port",  1, 0, 'p'},
      {"size",  1, 0, 's'},
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
      case 's':
        size = strtoul(optarg, NULL, 0);
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

  if (!size) {
    printf("%s: invalid size requested -- %u\n", argv[0], size);
    return 1;
  }

  if (show_usage) {
    showUsage(argv[0]);
    return 1;
  }

  buff_size = size * sizeof(uint64_t);
  buff = new char[buff_size];

  Endpoint* endp = new Endpoint(addr, port);
  if (endp->state() != EP_UP) {
    fprintf(stderr, "Failed to initialize fabrics endpoint: %s\n", endp->error());
    ret = endp->error_num();
  } else {
    printf("using %s provider\n", endp->fabric()->provider());
    ret = connect(endp, buff, buff_size, size, count);
  }

  if (endp) delete endp;
  delete[] buff;

  return ret;
}
