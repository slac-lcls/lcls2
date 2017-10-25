#include "psdaq/eb/Endpoint.hh"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <unistd.h>
const static int MSG_LEN = 40;


using namespace Pds::Fabrics;

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
      if (endp->recv_sync(key_buff, sizeof (int), mr)) {
        num_remote = *((int*) key_buff);
        if ((unsigned) num_remote < iov->count()) {
          fprintf(stderr, "Requested munber of remote buffers (%d) is less than local iov size: %lu\n", num_remote, iov->count());
          ret = -1;
        } else {
          if (endp->recv_sync(key_buff, sizeof(RemoteAddress), mr)) {
            RemoteIOVec rem_iov(keys, 1);
            RmaMessage msg(iov, &rem_iov, NULL, num_remote);
            if (!endp->writemsg_sync(&msg, FI_REMOTE_CQ_DATA)) {
              fprintf(stderr, "Failed writing data to remote addresses: %s\n", endp->error());
              ret = endp->error_num();
            } else {
              sleep(2);
              if (!endp->readmsg_sync(&msg, FI_REMOTE_CQ_DATA)) {
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
  if (argc != 3) {
    fprintf(stderr, "usage: %s addr nbuffers\n", argv[0]);
    return -1;
  }

  int ret = 0;
  char *addr = argv[1];
  int nbuffers = atoi(argv[2]);
  char* buffers[nbuffers];
  LocalAddress laddrs[nbuffers];

  Endpoint* endp = new Endpoint(addr, "1234");
  if (endp->state() != EP_UP) {
    fprintf(stderr, "Failed to initialize fabrics endpoint: %s\n", endp->error());
    delete endp;
    return endp->error_num();
  }

  for (int i=0; i<nbuffers; i++) {
    buffers[i] = new char[MSG_LEN];
    snprintf(buffers[i], MSG_LEN, "my favorite number is %d", i);
    laddrs[i] = LocalAddress(buffers[i], MSG_LEN);
    endp->fabric()->register_memory(&laddrs[i]);
  }

  LocalIOVec iov(laddrs, nbuffers);

  ret = connect(endp, &iov);
  if (ret == FI_SUCCESS) {
    for (int i=0; i<nbuffers; i++) {
      printf("%s\n", buffers[i]);
    }
  }

  // cleanup
  if (endp) delete endp;
  for (int i=0; i<nbuffers; i++) {
    if (buffers[i])
      delete[] buffers[i];
  }

  return ret;
}
