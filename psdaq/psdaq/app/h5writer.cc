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
    int fd = open("data.xtc", O_RDONLY);
    XtcFileIterator xtciter(fd,BUFSIZE);

    NamesIter namesiter;
    HDF5File h5file("data.h5", namesiter.namesVec());
    bool firstTime=true;
    Dgram* dgram;
    while ((dgram = xtciter.next())) {
        if (firstTime) {
            namesiter.iterate(&dgram->xtc);
            firstTime=false;
        } else {
            h5file.save(*dgram);
        }
    }

    return 0;
}
