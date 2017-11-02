#include "psdaq/hdf5/Hdf5Writer.hh"
#include "xtcdata/xtc/DescData.hh"
#include "xtcdata/xtc/XtcFileIterator.hh"
#include "xtcdata/xtc/Dgram.hh"
#include "xtcdata/xtc/NamesIter.hh"

#include <fcntl.h>
#include <stdio.h>
#include <string.h>

using namespace XtcData;
#define BUFSIZE 0x4000000

int main()
{
    Dgram* config = (Dgram*)malloc(BUFSIZE);

    int fd = open("data.xtc", O_RDONLY | O_LARGEFILE);
    XtcFileIterator iter(fd,BUFSIZE);
    Dgram* dgram = iter.next();
    memcpy(config,dgram,sizeof(Dgram)+dgram->xtc.sizeofPayload());
    NamesIter namesiter(&config->xtc);
    namesiter.iterate();

    HDF5File file("data.h5");
    while ((dgram = iter.next())) {
        HDF5Iter iter(&dgram->xtc, file, namesiter.namesVec());
        iter.iterate();
    }

    return 0;
}
