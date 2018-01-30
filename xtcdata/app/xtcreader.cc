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

using namespace XtcData;
using std::string;

class myLevelIter : public XtcIterator
{
public:
    enum { Stop, Continue };
    myLevelIter(Xtc* xtc, unsigned depth)
    : XtcIterator(xtc), _depth(depth)
    {
    }

    int process(Xtc* xtc)
    {
        unsigned i = _depth;
        while (i--)
            printf("  ");
        Level::Type level = xtc->src.level();
        printf("%s level payload size %d contains %s damage "
               "0x%x\n",
               Level::name(level), xtc->sizeofPayload(),
               TypeId::name(xtc->contains.id()), xtc->damage.value());

        switch (xtc->contains.id()) {
        case (TypeId::Parent): {
            myLevelIter iter(xtc, _depth + 1);
            iter.iterate();
            break;
        }
        default:
            break;
        }
        return Continue;
    }

private:
    unsigned _depth;
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

    int fd = open(xtcname, O_RDONLY | O_LARGEFILE);
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
        myLevelIter iter(&(dg->xtc), 0);
        iter.iterate();
    }

    ::close(fd);
    return 0;
}
