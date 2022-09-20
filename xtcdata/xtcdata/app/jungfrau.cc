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
    MyXtcIter(Xtc* xtc, const void* bufEnd) :
        XtcIterator(xtc, bufEnd)
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
            ShapesData* tmp = (ShapesData*)xtc;
            _shapesData[tmp->namesId().namesId()] = tmp;
            break;
        }
        default:
            break;
        }
        return Continue;
    }

    ShapesData& shape() {return *_shapesData[0];}
    ShapesData& value() {return *_shapesData[1];}

private:
    ShapesData* _shapesData[2];
};


void usage(char* progname)
{
    fprintf(stderr, "Usage: %s -f <filename> [-h]\n", progname);
}

void dump(const char* comment, DescData& descdata) {

    Names& names = descdata.nameindex().names();
    ShapesData& shapesData = descdata.shapesdata();
    NamesId& namesId       = shapesData.namesId();
    printf("========= transition: %d  0/1 = config/data\n", namesId.namesId());
    printf("Names:: detName: %s  detType: %s  detId: %s  segment: %d alg.name: %s\n",
	   names.detName(), names.detType(), names.detId(), names.segment(), names.alg().name());

    printf("------ Names for %s ---------\n",comment);
    for (unsigned i = 0; i < names.num(); i++) {
        Name& name = names.get(i);
        printf("rank %d type %d name %s\n",name.rank(),name.type(),name.name());
    }
    printf("------ Values for %s ---------\n",comment);
    for (unsigned i = 0; i < names.num(); i++) {
        Name& name = names.get(i);
        if (name.type()==Name::INT64 and name.rank()==0) {
            printf("Name %s has value %lld\n",name.name(),descdata.get_value<int64_t>(name.name()));
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

    XtcFileIterator iter_fdg(fd, 0x4000000);
    Dgram* cfg = iter_fdg.next();
    const char* bufEnd = ((char*) cfg) + iter_fdg.size();

    NamesIter& namesIter = *new NamesIter(&(cfg->xtc), bufEnd);
    namesIter.iterate();
    NamesLookup& namesLookup = namesIter.namesLookup();

    // get data out of the configure transition
    MyXtcIter cfgiter(&(cfg->xtc), bufEnd);
    cfgiter.iterate();
    NamesId& namesId = cfgiter.shape().namesId();
    printf("*** namesid %d\n",namesId.namesId());
    DescData descdata(cfgiter.shape(), namesLookup[namesId]);
    dump("Configure",descdata);

    Dgram* dg;
    unsigned nevent=0;
    while ((dg = iter_fdg.next())) {
        if (nevent>=neventreq) break;
        nevent++;
        MyXtcIter iter(&(dg->xtc), bufEnd);
        iter.iterate();
        NamesId& namesId = iter.value().namesId();
        printf("*** namesid %d\n",namesId.namesId());
        DescData descdata(iter.value(), namesLookup[namesId]);
        dump("Event",descdata);
    }

    ::close(fd);
    return 0;
}
