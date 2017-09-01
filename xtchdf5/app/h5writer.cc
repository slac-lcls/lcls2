#include "pdsdata/xtc/Descriptor.hh"
#include "pdsdata/xtc/Dgram.hh"
#include "pdsdata/xtc/TypeId.hh"
#include "pdsdata/xtc/XtcIterator.hh"
#include "xtchdf5/hdf5/Hdf5Writer.hh"

#include <stdio.h>
#include <fcntl.h>
#include <errno.h>
#include <unistd.h>

using namespace Pds;
#define BUFSIZE 0x4000000

class myLevelIter : public XtcIterator {
public:
  enum { Stop, Continue };
  myLevelIter(Xtc* xtc, HDF5File &file) : XtcIterator(xtc), _file(file) {}

  int process(Xtc* xtc) {
    switch (xtc->contains.id()) {
    case (TypeId::Parent): {
      iterate(xtc);
      break;
    }
    case (TypeId::DescData): {
      DescData& descdata = *(DescData*)xtc->payload();
      Descriptor& desc = descdata.desc();

      _file.addDatasets(desc);
      _file.appendData(descdata);

      break;
    }
    default:
      printf("TypeId %s (value = %d)\n", TypeId::name(xtc->contains.id()), (int)xtc->contains.id());
      break;
    }
    return Continue;
  }

private:
  HDF5File &_file;
};

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
  myLevelIter iter(&dgram->xtc, file);
  // iterate through the datagram twice to simulate two events
  iter.iterate();
  iter.iterate();

  return 0;
}
