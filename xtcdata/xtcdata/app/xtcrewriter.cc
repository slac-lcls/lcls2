//
//  Copy the L1Accept ShapesData from a set of runs over the ShapesData of a run with steps
//

#include "xtcdata/xtc/ShapesData.hh"
#include "xtcdata/xtc/Dgram.hh"
#include "xtcdata/xtc/TypeId.hh"
#include "xtcdata/xtc/XtcIterator.hh"
#include "xtcdata/xtc/XtcFileIterator.hh"

#include <vector>
#include <string>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <unistd.h>
#include <type_traits>
#include <cstring>
#include <sys/time.h>
#include <stdint.h>

using namespace XtcData;

void usage(char* p) {
    printf("Usage: %s [arguments]\n",p);
    printf("  A program to pull events from separate runs and copy them into a template run: \n");
    printf("     one step in the template run for each input xtc\n");
    printf("  -i <filename> : input xtc used as template for steps, events\n");
    printf("  -o <filename> : output xtc to write\n");
    printf("  -d <filename> : event data for step <i> : use as many -d arguments as steps\n",p);
}

class MyIter : public XtcIterator
{
public:
    enum { Stop, Continue };
    MyIter() : shapesdata(0)
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
            shapesdata = xtc;
            return Stop;
        }
        default:
            break;
        }
        return Continue;
    }

public:
    Xtc* shapesdata;
};

int main(int argc, char* argv[])
{
    int c;
    char* ixtcname = 0;
    char* oxtcname = 0;
    std::vector<std::string> dxtcnames;
    int parseErr = 0;

    while ((c = getopt(argc, argv, "hi:o:d:")) != -1) {
        switch (c) {
        case 'h':
            usage(argv[0]);
            exit(0);
        case 'i':
            ixtcname = optarg;
            break;
        case 'o':
            oxtcname = optarg;
            break;
        case 'd':
            dxtcnames.push_back(std::string(optarg));
            break;
        default:
            printf("unexpected argument %c\n",c);
            parseErr++;
        }
    }

    if (!ixtcname || dxtcnames.empty()) {
        usage(argv[0]);
        exit(2);
    }

#define OPEN(fd, xtcname, mode) {                                       \
        fd = open(xtcname, mode);                                      \
        if (fd < 0) {                                                   \
            fprintf(stderr, "Unable to open file '%s'\n", xtcname);     \
            exit(2);                                                    \
        }                                                               \
    }

    int ifd, dfd, ofd;
    OPEN(ifd, ixtcname, O_RDONLY);
    OPEN(ofd, oxtcname, O_WRONLY|O_CREAT);
        
    XtcFileIterator iiter(ifd, 0x4000000);
    XtcFileIterator* diter = 0;
    Dgram* dg;
    Dgram* ddg;
    unsigned nstep=0;
    unsigned nevt =0;
    while ((dg = iiter.next())) {
        if (dg->service()==TransitionId::BeginStep) {
            printf("BeginStep: opening %s\n",dxtcnames[nstep].c_str());
            OPEN(dfd, dxtcnames[nstep].c_str(), O_RDONLY);
            diter = new XtcFileIterator(dfd, 0x400000);
            nevt = 0;
        }
        if (dg->service()==TransitionId::EndStep) {
            printf("EndStep after %d events\n",nevt);
            delete diter;
            close(dfd);
            nstep++;
        }
        if (dg->service()==TransitionId::L1Accept) {
            while ((ddg = diter->next())) {
                if (ddg->service() == TransitionId::L1Accept)
                    break;
                if (ddg->service() == TransitionId::Disable) {
                    close(dfd);
                    delete diter;
                    OPEN(dfd, dxtcnames[nstep].c_str(), O_RDONLY);
                    diter = new XtcFileIterator(dfd, 0x400000);
                }
            }
            MyIter idgiter; idgiter.iterate(&(dg->xtc));
            MyIter ddgiter; ddgiter.iterate(&(ddg->xtc));
            if (idgiter.shapesdata->extent != ddgiter.shapesdata->extent) {
                printf("event %d  inputlen %d  datalen %d\n", 
                       nevt, idgiter.shapesdata->extent, ddgiter.shapesdata->extent);
                exit(2);
            }
            ddgiter.shapesdata->src = idgiter.shapesdata->src;            
            memcpy(idgiter.shapesdata, ddgiter.shapesdata, ddgiter.shapesdata->extent);
            nevt++;
        }
        //  Write the output
        if (write(ofd, dg, sizeof(*dg) + dg->xtc.sizeofPayload()) < 0) {
            printf("Error writing to output xtc file.\n");
        }
    }
    ::close(ofd);
    ::close(ifd);
    ::close(dfd);
    return 0;
}
