/*
 * This emulates how the DRP would behave
 */

#include <stdio.h>
#include <iostream>
#include <vector>
#include <stdlib.h>
#include <fstream>

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
#include "xtcdata/xtc/NamesVec.hh"

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
    HsdIter(Xtc* xtc, NamesVec& namesVec) : XtcIterator(xtc), _namesVec(namesVec)
    {
    }

    int process(Xtc* xtc)
    {
        switch (xtc->contains.id()) {

        case (TypeId::Parent): {
            iterate(xtc);
            break;
        }
        case (TypeId::ShapesData): {
            ShapesData& shapesdata = *(ShapesData*)xtc;
            // lookup the index of the names we are supposed to use
            NamesId namesId = shapesdata.namesId();
            DescData descdata(shapesdata, _namesVec[namesId]);
            Names& names = descdata.nameindex().names();

            for (unsigned i = 0; i < names.num(); i++) {
                Name& name = names.get(i);
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
    NamesVec& _namesVec;


public:
    std::map <std::string, uint8_t*> chans;
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
    Dgram* dg;
    //Stack stack;
    Heap stack;

    dg = iter.next();
    NamesIter namesIter(&dg->xtc);
    namesIter.iterate();



    unsigned counter = 0;
    while ((dg = iter.next())) {
        printf("####################################################\n");
        printf("Event: %u\n", counter++);

        printf("%s transition: time %d.%09d, pulseId 0x%x, env 0x%x, 0x%x, "
               "payloadSize %d\n",
               TransitionId::name(dg->seq.service()), dg->seq.stamp().seconds(),
               dg->seq.stamp().nanoseconds(), dg->seq.pulseId().value(),
               dg->env[1], dg->env[2], dg->xtc.sizeofPayload());
        printf("*** dg xtc extent %d\n",dg->xtc.extent);

        if (dg->seq.isEvent()) { // FIXME: this is always false
            printf("Found event\n");
        }

        HsdIter hsdIter(&dg->xtc, namesIter.namesVec());
        hsdIter.iterate();

        // Test DRP situation
        unsigned nChan = hsdIter.chans.size();
        printf("nChan: %d\n", nChan);
        /*
        // Create Hsd using the factory
        Pds::HSD::Factory *pFactory = new Pds::HSD::Factory(&heap, "1.2.3", nChan);
        Pds::HSD::HsdEventHeaderV1 *pHsd = pFactory->getHsd();
        Pds::HSD::Hsd_v1_2_3 *vHsd = (Pds::HSD::Hsd_v1_2_3*) pHsd;
        */
        Pds::HSD::Hsd_v1_2_3 *vHsd = new Pds::HSD::Hsd_v1_2_3(&stack); //, dg, nChan);
        vHsd->init(dg->env);
        printf("####### hsd: %u %u %u %u\n", vHsd->samples(), vHsd->streams(), vHsd->channels(), vHsd->sync());
        // iterate over all available channels
        for (std::map<std::string, uint8_t*>::iterator it=hsdIter.chans.begin(); it!=hsdIter.chans.end(); ++it){
            std::cout << "Channel name: " << it->first << std::endl;
            Pds::HSD::Channel mychan = Pds::HSD::Channel(&stack, vHsd, (const uint8_t*)hsdIter.chans[it->first]);
            printf("npeaks: %d\n", mychan.npeaks());
        }
        //delete pFactory;
        delete vHsd;
    }

    ::close(fd);

    return 0;
}

