#include "pdsdata/xtc/Descriptor.hh"
#include "pdsdata/xtc/Dgram.hh"
#include "xtchdf5/hdf5/Hdf5Writer.hh"

#include <stdio.h>
#include <fcntl.h>
#include <errno.h>
#include <unistd.h>

using namespace Pds;
#define BUFSIZE 0x4000000

int main()
{
  Dgram *dgram = (Dgram*)malloc(BUFSIZE);

  int fd = open("data.xtc", O_RDONLY | O_LARGEFILE);

  if(::read(fd,dgram,sizeof(*dgram))<=0){ 
    printf("read was unsuccessful: %s\n",strerror(errno));
  }

  size_t payloadSize = dgram->xtc.sizeofPayload();
  size_t sz = ::read(fd,dgram->xtc.payload(),payloadSize);

  HDF5File file("data.h5");
  HDF5LevelIter iter(&dgram->xtc, file);
  // iterate through the datagram twice to simulate two events
  iter.iterate();
  iter.iterate();

  return 0;
}
