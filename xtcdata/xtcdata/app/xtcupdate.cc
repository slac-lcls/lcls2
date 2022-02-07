#include <iostream>
#include <unistd.h>
#include <fcntl.h>

#include "xtcdata/xtc/XtcFileIterator.hh"
#include "xtcdata/xtc/Dgram.hh"
#include "xtcdata/xtc/XtcUpdateIter.hh"
#include "xtcdata/xtc/TransitionId.hh"
#include "xtcdata/xtc/ShapesData.hh"

using namespace XtcData;

void usage(char* progname)
{
    fprintf(stderr, "Usage: %s -f <filename> [-n <nEvents>] [-h]\n", progname);
}

int main(int argc, char* argv[])
{
    int c;
    char* xtcname = 0;
    int parseErr = 0;
    unsigned neventreq = 0xffffffff;
    unsigned numWords = 3;

    while ((c=getopt(argc, argv, "hf:n:")) != -1) {
        switch (c) {
            case 'h':
                usage(argv[0]);
                exit(0);
            case 'f':
                xtcname = optarg;
                break;
            case 'n':
                neventreq = atoi(optarg);
                break;
            default:
                parseErr++;

        }
    }

    if (!xtcname) {
        usage(argv[0]);
        exit(2);
    }

    // open xtc file 
    int fd = open(xtcname, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "Unable to open file '%s'\n", xtcname);
        exit(2);
    }

    // iterate through the xtc file
    XtcFileIterator iter(fd, 0x4000000);
    Dgram* dg;
    unsigned nevent=0;

    // create update iterator
    XtcUpdateIter uiter(numWords);
    
    char detName[] = "xpphsd";
    char detType[] = "hsd";
    char detId[] = "detnum1234";
    unsigned nodeId = 1;
    unsigned namesId = 1;
    unsigned segment = 0;
    char algName[] = "fex";
    uint8_t major = 4;
    uint8_t minor = 5;
    uint8_t micro = 6;

    DataDef datadef;
    char defName[] = "arrayFex";
    datadef.add(defName, Name::UINT8, 2);

    while(dg = iter.next()) {
        if (nevent >= neventreq) break;

        nevent++;
        printf("event %d ", nevent);
        
        // add Names to configure
        if (dg->service() == TransitionId::Configure) {
            printf("[Configure] ");
            uiter.addNames(dg->xtc, detName, detType, detId,
                    nodeId, namesId, segment,
                    algName, major, minor, micro, datadef);
            uiter.iterate(&(dg->xtc));
        }

        // add data to L1Accept
        if (dg->service() == TransitionId::L1Accept) {
            printf("[L1Accept] ");
            unsigned shape[MaxRank] = {2,3};
            uiter.createData(dg->xtc, nodeId, namesId);
            uint8_t data[6] = {141,142,143,144,145,146};
            uiter.addData(nodeId, namesId, shape, (char *)data, datadef, defName);
            uiter.iterate(&(dg->xtc));
        }
        printf("\n");
    }

    close(fd);
    return 0;
}
