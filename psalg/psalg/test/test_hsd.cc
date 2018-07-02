//g++ -g -Wall -std=c++11 -I /reg/neh/home/yoon82/Software/lcls2/install/include test_hsd.cpp -o test_hsd
//valgrind ./test_hsd

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

#include "xtcdata/xtc/DetInfo.hh"
#include "xtcdata/xtc/ProcInfo.hh"
#include "xtcdata/xtc/XtcFileIterator.hh"
#include "xtcdata/xtc/XtcIterator.hh"
#include "xtcdata/xtc/ShapesData.hh"

#include "xtcdata/xtc/NamesIter.hh"

//using namespace psalgos;
using namespace psalg;

using namespace XtcData;
using std::string;

void usage(char* progname)
{
    fprintf(stderr, "Usage: %s -f <filename> [-h]\n", progname);
}

class HsdIter : public XtcIterator
{
public:
    enum { Stop, Continue };
    HsdIter(Xtc* xtc, std::vector<NameIndex>& namesVec) : XtcIterator(xtc), _namesVec(namesVec)
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
            unsigned namesId = shapesdata.shapes().namesId();
            DescData descdata(shapesdata, _namesVec[namesId]);
            Names& names = descdata.nameindex().names();

            for (unsigned i = 0; i < names.num(); i++) {
                Name& name = names.get(i);
                // Here we check algorithm and version can be analyzed
                if (strcmp(name.alg().name(), "fpga") == 0) {
                    if (name.alg().version() == 0x010203) {
                        auto array = descdata.get_array<uint8_t>(i);
                        chanPtr[std::string(name.name())] = array.data();
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
    std::vector<NameIndex>& _namesVec;

public:
    std::map <std::string, uint8_t*> chanPtr;
};

int main (int argc, char* argv[]) {
/*
    //open file
    std::ifstream infile("/reg/neh/home/yoon82/temp1/lcls2/chan0_e0.bin");
    //get length of file
    infile.seekg(0, infile.end);
    size_t length = infile.tellg();
    infile.seekg(0, infile.beg);
    char buffer[length];
    //read file
    infile.read(buffer, length);
*/

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

    int fd = open(xtcname, O_RDONLY | O_LARGEFILE);
    if (fd < 0) {
        fprintf(stderr, "Unable to open file '%s'\n", xtcname);
        exit(2);
    }

    XtcFileIterator iter(fd, 0x4000000);
    Dgram* dg;
    unsigned counter = 0;

    dg = iter.next();
    NamesIter namesIter(&dg->xtc);
    namesIter.iterate();

    while ((dg = iter.next())) {


        printf("%s transition: time %d.%09d, pulseId 0x%lux, env 0x%lux, "
               "payloadSize %d\n",
               TransitionId::name(dg->seq.service()), dg->seq.stamp().seconds(),
               dg->seq.stamp().nanoseconds(), dg->seq.pulseId().value(),
               dg->env, dg->xtc.sizeofPayload());
        printf("*** dg xtc extent %d\n",dg->xtc.extent);

        if (dg->seq.isEvent()) {
            printf("Found event\n");
        }

        if (counter == 1) {


            Heap heap;
            //unsigned nChan = 4;

            //Pds::HSD::Client *pClient = new Pds::HSD::Client(&heap, "1.2.3", nChan);
            //Pds::HSD::HsdEventHeaderV1 *pHsd = pClient->getHsd();
            //Pds::HSD::Hsd_v1_2_3 *vHsd = (Pds::HSD::Hsd_v1_2_3*) pHsd;

            const Pds::HSD::EventHeader& eh = *reinterpret_cast<const Pds::HSD::EventHeader*>(dg);
            printf("Address of eh: %u %u\n", &eh, &eh+1);
            printf("Dumping eventheader\n");
            eh.dump();

            HsdIter hsdIter(&dg->xtc, namesIter.namesVec());
            hsdIter.iterate();
            const Pds::HSD::StreamHeader* sh_rawx = 0;
            sh_rawx = reinterpret_cast<const Pds::HSD::StreamHeader*>(hsdIter.chanPtr["chan3"]);
            sh_rawx->dump();

/*
            const Pds::HSD::EventHeader* ehx = reinterpret_cast<const Pds::HSD::EventHeader*>(dg);
            ehx->dump();

            printf("sizeof: %u %u\n", sizeof(Pds::HSD::EventHeader), sizeof(Pds::HSD::StreamHeader));

            //const char* nextx = reinterpret_cast<const char*>(ehx+1);
            const Pds::HSD::StreamHeader* sh_rawx = 0;
            sh_rawx = reinterpret_cast<const Pds::HSD::StreamHeader*>(ehx+1);
            const uint16_t* rawx = reinterpret_cast<const uint16_t*>(sh_rawx+1);
            sh_rawx->dump();
*/
            //numPixels.push_back((unsigned) (sh_rawx->samples()-sh_rawx->eoffs()-sh_rawx->boffs()));
            //rawPtr.push_back((uint16_t *) (rawx+sh_rawx->boffs()));
            //numFexPeaksx.push_back(0);

            // ------------ FEX --------------
            //nextx = reinterpret_cast<const char*>(&rawx[sh_rawx->samples()]);
            //const Pds::HSD::StreamHeader& sh_fexx = *reinterpret_cast<const Pds::HSD::StreamHeader*>(nextx);
            //sh_fexx.dump();
            //const unsigned end = sh_fexx.samples() - sh_fexx.eoffs() - sh_fexx.boffs();
            //const uint16_t* p_thr = &reinterpret_cast<const uint16_t*>(&sh_fexx+1)[sh_fexx.boffs()];


/*
            for (unsigned chan = 0; chan < nChan; chan++) {
                vHsd->parseChan((const uint8_t*)dg+sizeof(Pds::HSD::EventHeader)+chan*sizeof(Pds::HSD::StreamHeader), chan);
                printf("\n\n");
            }
*/
            break;
        }
        counter++;

    }

    ::close(fd);

/*
    // Test with heap
    Heap heap;
    unsigned nChan = 4;

    Pds::HSD::Client *pClient = new Pds::HSD::Client(&heap, "1.0.0", nChan);
    Pds::HSD::HsdEventHeaderV1 *pHsd = pClient->getHsd();
    Pds::HSD::Hsd_v1_0_0 *vHsd = (Pds::HSD::Hsd_v1_0_0*) pHsd;
    for (unsigned i = 0; i < nChan; i++) {
        vHsd->parseChan((const uint8_t*)buffer, i);
    }

    delete pClient;
*/
    return 0;
}

