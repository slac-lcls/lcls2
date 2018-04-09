#include "psdaq/eb/Endpoint.hh"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <getopt.h>

using namespace Pds::Fabrics;

const static int MSG_LEN = 40;
static const char* PORT_DEF = "12345";
static const unsigned NUM_DEF = 3;

static void showUsage(const char* p)
{
  printf("\nSimple libfabrics writemsg client example (its pair is the ft_writemsg_server example).\n"
         "\n"
         "This example uses the writemsg/readmsg api in libfabrics to pingpong the specified number of\n"
         "buffers back and forth making a few changes\n"
         "\n"
         "Usage: %s [-h|--help]\n"
         "    -a|--addr     the address of the server\n"
         "    -p|--port     the port or libfaric 'service' the server uses (default: %s)\n"
         "    -n|--nbuffer  the number of the local memory buffers to allocate (default: %u)\n"
         "    -h|--help     print this message and exit\n", p, PORT_DEF, NUM_DEF);
}

int connect(Endpoint* endp, LocalIOVec *iov)
{
  int ret = 0;
  int num_remote = 0;
  size_t key_buff_size = sizeof(RemoteAddress);
  char* key_buff = new char[key_buff_size];
  RemoteAddress* keys = (RemoteAddress*) key_buff;

  MemoryRegion* mr = endp->fabric()->register_memory(key_buff, key_buff_size);
  if (mr) {
    if (endp->connect()) {
      if (!endp->recv_sync(key_buff, sizeof (int), mr)) {
        num_remote = *((int*) key_buff);
        if ((unsigned) num_remote < iov->count()) {
          fprintf(stderr, "Requested munber of remote buffers (%d) is less than local iov size: %lu\n", num_remote, iov->count());
          ret = -1;
        } else {
          if (!endp->recv_sync(key_buff, sizeof(RemoteAddress), mr)) {
            RemoteIOVec rem_iov(keys, 1);
            RmaMessage msg(iov, &rem_iov, NULL, num_remote);
            if (endp->writemsg_sync(&msg, FI_REMOTE_CQ_DATA)) {
              fprintf(stderr, "Failed writing data to remote addresses: %s\n", endp->error());
              ret = endp->error_num();
            } else {
              sleep(2);
              if (endp->readmsg_sync(&msg, 0)) {
                fprintf(stderr, "Failed reading back data from remote addresses: %s\n", endp->error());
                ret = endp->error_num();
              }
            }
          } else {
            fprintf(stderr, "Failed reading requested remote address keys: %s\n", endp->error());
            ret = endp->error_num();
          }
        }
      } else {
        fprintf(stderr, "Failed reading number of remote buffers requested: %s\n", endp->error());
        ret = endp->error_num();
      }
    } else {
      fprintf(stderr, "Failed to connect to remote: %s\n", endp->error());
      ret = endp->error_num();
    }
  } else {
    fprintf(stderr, "Failed to register memory region: %s\n", endp->fabric()->error());
    ret = endp->fabric()->error_num();
  }

  endp->shutdown();

  delete key_buff;

  return ret;
}

int main(int argc, char *argv[])
{
  int ret = FI_SUCCESS;
  bool show_usage = false;
  char *addr = NULL;
  const char* port = PORT_DEF;
  unsigned nbuffers = NUM_DEF;

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
        nbuffers = strtoul(optarg, NULL, 0);
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

  if (!nbuffers) {
    printf("%s: invalid number of buffers requested -- %u\n", argv[0], nbuffers);
    return 1;
  }

  if (show_usage) {
    showUsage(argv[0]);
    return 1;
  }

  char* buffers[nbuffers];
  LocalAddress laddrs[nbuffers];

  Endpoint* endp = new Endpoint(addr, port);
  if (endp->state() != EP_UP) {
    fprintf(stderr, "Failed to initialize fabrics endpoint: %s\n", endp->error());
    delete endp;
    return endp->error_num();
  }

  printf("using %s provider\n", endp->fabric()->provider());

  for (unsigned i=0; i<nbuffers; i++) {
    buffers[i] = new char[MSG_LEN];
    snprintf(buffers[i], MSG_LEN, "my favorite number is %d", i);
    laddrs[i] = LocalAddress(buffers[i], MSG_LEN);
    endp->fabric()->register_memory(&laddrs[i]);
  }

  LocalIOVec iov(laddrs, nbuffers);

  ret = connect(endp, &iov);
  if (ret == FI_SUCCESS) {
    for (unsigned i=0; i<nbuffers; i++) {
      printf("%s\n", buffers[i]);
    }
  }

  // cleanup
  if (endp) delete endp;
  for (unsigned i=0; i<nbuffers; i++) {
    if (buffers[i])
      delete[] buffers[i];
  }

  return ret;
}
