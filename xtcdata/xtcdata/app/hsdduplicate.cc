#include <fcntl.h>
#include <map>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <assert.h>

#include "xtcdata/xtc/XtcFileIterator.hh"

using namespace XtcData;

void usage(char* progname)
{
    fprintf(stderr, "Usage: %s -i <xtc filename> -o <xtc filename> [-h]\n", progname);
}

void writedgram(Dgram* dg, unsigned& count, FILE* outFile) {
    TimeStamp& stamp = const_cast<TimeStamp&>(dg->seq.stamp());
    stamp = TimeStamp(count,0);
    PulseId& pulseid = const_cast<PulseId&>(dg->seq.pulseId());
    unsigned control = pulseid.control();
    pulseid = PulseId(count,control);
    count++;

    if (fwrite(dg, sizeof(*dg) + dg->xtc.sizeofPayload(), 1, outFile) != 1) {
        printf("Error writing to output xtc file.\n");
    }
    
    return;
}

int main(int argc, char* argv[])
{
    int c;
    char* outname = 0;
    char* inname = 0;
    unsigned nevents = 0;
    int parseErr = 0;

    while ((c = getopt(argc, argv, "hi:o:")) != -1) {
        switch (c) {
        case 'h':
            usage(argv[0]);
            exit(0);
        case 'i':
            inname = optarg;
            break;
        case 'o':
            outname = optarg;
            break;
        default:
            parseErr++;
        }
    }

    if (!outname || !inname) {
        usage(argv[0]);
        exit(2);
    }

    Dgram* cfg = (Dgram*)malloc(0x100000);
    Dgram* big = (Dgram*)malloc(0x100000);
    Dgram* small = (Dgram*)malloc(0x100000);

    int fd = open(inname, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "Unable to open file '%s'\n", inname);
        exit(2);
    }

    XtcFileIterator iter(fd, 0x4000000);
    Dgram* dg;
    bool configDone = false;
    bool bigDone = false;
    bool smallDone = false;
    for (unsigned ievt = 0; ievt < 1000000; ievt++) {
        dg = iter.next();
        if (dg==0) {
            printf("*** Error reading dg\n");
            break;
        }
        if (ievt==0) {
            memcpy(cfg,dg,sizeof(*dg)+dg->xtc.sizeofPayload());
            printf("config evt %d size %d\n",ievt,dg->xtc.sizeofPayload());
            configDone=true;
        } else if (dg->xtc.sizeofPayload()>1000) {
            memcpy(big,dg,sizeof(*dg)+dg->xtc.sizeofPayload());
            printf("big evt %d size %d\n",ievt,dg->xtc.sizeofPayload());
            bigDone=true;
        } else if (dg->xtc.sizeofPayload()<=1000 && !smallDone) {
            memcpy(small,dg,sizeof(*dg)+dg->xtc.sizeofPayload());
            printf("small evt %d size %d\n",ievt,dg->xtc.sizeofPayload());
            smallDone=true;
        }
        if (configDone && bigDone && smallDone) break;
    }
    ::close(fd);
    assert (configDone && bigDone && smallDone);

    unsigned nbig = 10000;
    unsigned nsmall = 10000;

    FILE* outFile = fopen(outname, "w");

    unsigned count=0;
    writedgram(cfg,count,outFile);
    for (unsigned ibig = 0; ibig<nbig; ibig++) {
        printf("nbig %d\n",ibig);
        writedgram(big,count,outFile);
        for (unsigned ismall = 0; ismall<nsmall; ismall++)
            writedgram(small,count,outFile);
    }

    fclose(outFile);
    return 0;
}
