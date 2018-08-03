#include <fcntl.h>
#include <map>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <unistd.h>

#include "xtcdata/xtc/DetInfo.hh"
#include "xtcdata/xtc/ProcInfo.hh"
#include "xtcdata/xtc/XtcFileIterator.hh"
#include "xtcdata/xtc/XtcIterator.hh"
#include "xtcdata/xtc/ShapesData.hh"
//#include "xtcdata/xtc/DescData.hh"

using namespace XtcData;
using std::string;

class DebugIter : public XtcIterator
{
public:
    enum { Stop, Continue };
    DebugIter(Xtc* xtc) : XtcIterator(xtc)
    {
    }

    int process(Xtc* xtc)
    {
        printf("found typeid %s extent %d\n",XtcData::TypeId::name(xtc->contains.id()),xtc->extent);
        switch (xtc->contains.id()) {
        case (TypeId::Parent): {
            iterate(xtc);
            break;
        }
        case (TypeId::Names): {
            Names& names = *(Names*)xtc;
	    printf("Found %d names\n",names.num());
            for (unsigned i = 0; i < names.num(); i++) {
                Name& name = names.get(i);
                printf("name %s\n",name.name());
            }
            break;
        }
        case (TypeId::ShapesData): {
            ShapesData& shapesdata = *(ShapesData*)xtc;
            printf("found shapesdata\n");
            iterate(xtc);
            // lookup the index of the names we are supposed to use
            // unsigned namesId = shapesdata.shapes().namesId();
            // DescData descdata(shapesdata, _namesVec[namesId]);
            // Names& names = descdata.nameindex().names();
            break;
        }
        default:
            break;
        }
        return Continue;
    }
    // std::vector<NameIndex>& _namesVec;
};


void usage(char* progname)
{
    fprintf(stderr, "Usage: %s -f <filename> [-h]\n", progname);
}

int main(int argc, char* argv[])
{
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
    while ((dg = iter.next())) {
        printf("%s transition: time %d.%09d, pulseId 0x%lux, env 0x%lux, "
               "payloadSize %d\n",
               TransitionId::name(dg->seq.service()), dg->seq.stamp().seconds(),
               dg->seq.stamp().nanoseconds(), dg->seq.pulseId().value(),
               dg->env, dg->xtc.sizeofPayload());
        printf("*** dg xtc extent %d\n",dg->xtc.extent);
        DebugIter iter(&(dg->xtc));
        iter.iterate();
    }

    ::close(fd);
    return 0;
}
