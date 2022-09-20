/*
 * This emulates how the DRP would behave
 */

#include <stdio.h>
#include <iostream>
#include <vector>
#include <stdlib.h>
#include <fstream>
#include <assert.h>

#include <fcntl.h>
#include <map>
#include <string>
#include <unistd.h>

#include "psalg/alloc/AllocArray.hh"
#include "psalg/alloc/Allocator.hh"
#include "psalg/digitizer/Hsd.hh"

#include "xtcdata/xtc/XtcFileIterator.hh"
#include "xtcdata/xtc/XtcIterator.hh"
#include "xtcdata/xtc/ShapesData.hh"
#include "xtcdata/xtc/NamesLookup.hh"

#include "xtcdata/xtc/NamesIter.hh"

//using namespace psalgos;
using namespace psalg;

using namespace XtcData;
using std::string;

void usage(char* progname)
{
    fprintf(stderr, "Usage: %s -f <xtc> [-h]\n", progname);
}

class HsdIter : public XtcIterator
{
public:
    enum { Stop, Continue };
    HsdIter(Xtc* xtc, const void* bufEnd, NamesLookup& namesLookup) : XtcIterator(xtc, bufEnd), _namesLookup(namesLookup), eventHeader(0)
    {
    }

    int process(Xtc* xtc, const void* bufEnd)
    {
        switch (xtc->contains.id()) {

        case (TypeId::Parent): {
            iterate(xtc, bufEnd);
            break;
        }
        case (TypeId::ShapesData): {
            ShapesData& shapesdata = *(ShapesData*)xtc;
            // lookup the index of the names we are supposed to use
            NamesId namesId = shapesdata.namesId();
            DescData descdata(shapesdata, _namesLookup[namesId]);
            Names& names = descdata.nameindex().names();

            for (unsigned i = 0; i < names.num(); i++) {
                Name& name = names.get(i);
                if (strcmp(name.name(), "eventHeader") == 0) {
                    auto eventHeader_array = descdata.get_array<uint32_t>(i);
                    eventHeader = eventHeader_array.data();
                }
                // Here we check algorithm and version can be analyzed
                if (strcmp(name.alg().name(), "fpga") == 0) {
                    if (name.alg().version() == 0x010203) {
                        auto array = descdata.get_array<uint8_t>(i);
                        chans[std::string(name.name())] = array.data();
                    }
                }
            }
            break;
        }
        default:
            break;
        }
        return Continue;
    }
    NamesLookup& _namesLookup;


public:
    std::map <std::string, uint8_t*> chans;
    uint32_t* eventHeader; // the "event header" words for the HSD
};

int main (int argc, char* argv[]) {

    int c;
    char* xtcname = 0;
    int parseErr = 0;

    while ((c = getopt(argc, argv, "hf:")) != -1) {
        switch (c) {
        case 'h':
            usage(argv[0]);
            exit(0);
        case 'f':
            xtcname = optarg;
            break;
        default:
            parseErr++;
        }
    }

    if (!xtcname) {
        usage(argv[0]);
        exit(2);
    }

    int fd = open(xtcname, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "Unable to open file '%s'\n", xtcname);
        exit(2);
    }

    XtcFileIterator iter(fd, 0x4000000);
    Dgram* dg = iter.next();
    const void* bufEnd = ((char*)dg) + iter.size();
    //Stack stack;
    Heap stack;

    NamesIter namesIter(&dg->xtc, bufEnd);
    namesIter.iterate();



    unsigned counter = 0;
    while ((dg = iter.next())) {
        printf("####################################################\n");
        printf("Event: %u\n", counter++);

        printf("%s transition: time %d.%09d, env 0x%x"
               "payloadSize %d\n",
               TransitionId::name(dg->service()), dg->time.seconds(),
               dg->time.nanoseconds(),
               dg->env, dg->xtc.sizeofPayload());
        printf("*** dg xtc extent %d\n",dg->xtc.extent);

        if (dg->isEvent()) { // FIXME: this is always false
            printf("Found event\n");
        }

        HsdIter hsdIter(&dg->xtc, bufEnd, namesIter.namesLookup());
        hsdIter.iterate();

        // Test DRP situation
        unsigned nChan = hsdIter.chans.size();
        printf("nChan: %d\n", nChan);

        assert(hsdIter.eventHeader);
        // iterate over all available channels
        for (std::map<std::string, uint8_t*>::iterator it=hsdIter.chans.begin(); it!=hsdIter.chans.end(); ++it){
            std::cout << "Channel name: " << it->first << std::endl;
            //Pds::HSD::Channel mychan = Pds::HSD::Channel(&stack, vHsd, (const uint8_t*)hsdIter.chans[it->first]);
            //printf("npeaks: %d\n", mychan.npeaks());
        }
    }

    ::close(fd);

    return 0;
}
