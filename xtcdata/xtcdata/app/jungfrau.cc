#include <fcntl.h>
#include <map>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <unistd.h>

#include "xtcdata/xtc/XtcFileIterator.hh"
#include "xtcdata/xtc/XtcIterator.hh"
#include "xtcdata/xtc/ShapesData.hh"
#include "xtcdata/xtc/DescData.hh"
#include "xtcdata/xtc/NamesIter.hh"

using namespace XtcData;
using std::string;

class MyXtcIter : public XtcIterator
{
public:
    enum { Stop, Continue };
    MyXtcIter(Xtc* xtc) :
        XtcIterator(xtc)
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
            ShapesData* tmp = (ShapesData*)xtc;
            _shapesData[tmp->namesId().namesId()] = tmp;
            iterate(xtc);
            break;
        }
        default:
            break;
        }
        return Continue;
    }

    ShapesData& config() {return *_shapesData[0];}
    ShapesData& event()  {return *_shapesData[1];}

private:
    ShapesData* _shapesData[2];
};


void usage(char* progname)
{
    fprintf(stderr, "Usage: %s -f <filename> [-h]\n", progname);
}

void dump(const char* transition, Names& names, DescData& descdata) {
    printf("------ Names for %s transition ---------\n",transition);
    for (unsigned i = 0; i < names.num(); i++) {
        Name& name = names.get(i);
        printf("rank %d type %d name %s\n",name.rank(),name.type(),name.name());
    }
    printf("------ Values for %s transition ---------\n",transition);
    for (unsigned i = 0; i < names.num(); i++) {
        Name& name = names.get(i);
        if (name.type()==Name::INT64 and name.rank()==0) {
            printf("Name %s has value %d\n",name.name(),descdata.get_value<int64_t>(name.name()));
        }
    }
    printf("\n\n");
}

int main(int argc, char* argv[])
{
    int c;
    char* xtcname = 0;
    int parseErr = 0;
    unsigned neventreq = 0xffffffff;

    while ((c = getopt(argc, argv, "hf:n:")) != -1) {
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

    int fd = open(xtcname, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "Unable to open file '%s'\n", xtcname);
        exit(2);
    }

    XtcFileIterator iter(fd, 0x4000000);
    Dgram* cfg = iter.next();
    NamesIter& namesIter = *new NamesIter(&(cfg->xtc));
    namesIter.iterate();
    NamesLookup& namesLookup = namesIter.namesLookup();

    // get data out of the configure transition
    MyXtcIter cfgiter(&(cfg->xtc));
    cfgiter.iterate();
    NamesId& namesId = cfgiter.config().namesId();
    DescData descdata(cfgiter.config(), namesLookup[namesId]);
    Names& names = descdata.nameindex().names();
    dump("Configure",names,descdata);

    Dgram* dg;
    unsigned nevent=0;
    while ((dg = iter.next())) {
        if (nevent>=neventreq) break;
        nevent++;
        MyXtcIter iter(&(dg->xtc));
        iter.iterate();
        NamesId& namesId = iter.event().namesId();
        DescData descdata(iter.event(), namesLookup[namesId]);
        Names& names = descdata.nameindex().names();
        dump("Event",names,descdata);
    }

    ::close(fd);
    return 0;
}
