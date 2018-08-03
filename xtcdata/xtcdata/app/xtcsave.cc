#include <fcntl.h>
#include <map>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <unistd.h>

#include "xtcdata/xtc/XtcFileIterator.hh"

using namespace XtcData;

void usage(char* progname)
{
    fprintf(stderr, "Usage: %s -i <xtc filename> -o <xtc filename> -n nevents [-h]\n", progname);
}

int main(int argc, char* argv[])
{
    int c;
    char* outname = 0;
    char* inname = 0;
    unsigned nevents = 0;
    int parseErr = 0;

    while ((c = getopt(argc, argv, "hi:o:n:")) != -1) {
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
        case 'n':
            nevents = atoi(optarg);
            break;
        default:
            parseErr++;
        }
    }

    if (!outname || !inname || !nevents) { // Check for inname and outname and nevents
        usage(argv[0]);
        exit(2);
    }

    FILE* outFile = fopen(outname, "w");

    int fd = open(inname, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "Unable to open file '%s'\n", inname);
        exit(2);
    }
    printf("fd: %d\n", fd);

    XtcFileIterator iter(fd, 0x4000000);
    Dgram* dg;
    for (unsigned i = 0; i < nevents; i++) {
        dg = iter.next();
        printf("Event: %d\n", i);
        printf("%s transition: time %d.%09d, pulseId 0x%lux, env 0x%lux, "
               "payloadSize %d\n",
               TransitionId::name(dg->seq.service()), dg->seq.stamp().seconds(),
               dg->seq.stamp().nanoseconds(), dg->seq.pulseId().value(),
               dg->env, dg->xtc.sizeofPayload());
        printf("*** dg xtc extent %d\n",dg->xtc.extent);
        printf("$$$$ %u %u %u\n", sizeof(*dg), sizeof(dg), sizeof(&dg));

        if (fwrite(dg, sizeof(*dg) + dg->xtc.sizeofPayload(), 1, outFile) != 1) {
            printf("Error writing to output xtc file.\n");
            return -1;
        }
    }

    ::close(fd);
    return 0;
}
