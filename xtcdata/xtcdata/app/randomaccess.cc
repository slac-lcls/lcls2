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

using namespace XtcData;
using std::string;

class SmdIter : public XtcIterator
{
public:
    enum { Stop, Continue };
    SmdIter() : XtcIterator()
    {
    }

    int process(Xtc* xtc)
    {
        switch (xtc->contains.id()) {
        case (TypeId::Parent): {
            iterate(xtc);
            break;
        }
        case (TypeId::Names): {
            Names& names = *(Names*)xtc;
            _namesLookup[names.namesId()] = NameIndex(names);
            break;
        }
        case (TypeId::ShapesData): {
            ShapesData& shapesdata = *(ShapesData*)xtc;
            // lookup the index of the names we are supposed to use
            NamesId namesId = shapesdata.namesId();
            // protect against the fact that this namesid
            // may not have a NamesLookup.  cpo thinks this
            // should be fatal, since it is a sign the xtc is "corrupted",
            // in some sense.
            if (_namesLookup.count(namesId)<=0) {
                printf("*** Corrupt xtc: namesid 0x%x not found in NamesLookup\n",(int)namesId);
                throw "invalid namesid";
                break;
            }
            DescData descdata(shapesdata, _namesLookup[namesId]);
            // see if the offset is in this ShapesData xtc
            Names& names = descdata.nameindex().names();
            for (unsigned i = 0; i < names.num(); i++) {
                Name& name = names.get(i);
                if (strcmp(name.name(),"intOffset")==0) {
                    offset = descdata.get_value<uint64_t>(i);
                }
            }
            break;
        }
        default:
            break;
        }
        return Continue;
    }
    void reset() {offset=-1;}
    int64_t offset;
    
private:
    NamesLookup _namesLookup;
};


void usage(char* progname)
{
    fprintf(stderr, "Usage: %s -f <filename> [-h]\n", progname);
}

int main(int argc, char* argv[])
{
    int c;
    string xtcname;
    int parseErr = 0;
    unsigned neventreq = 0xffffffff;

    while ((c = getopt(argc, argv, "hf:n")) != -1) {
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

    if (xtcname.empty()) {
        usage(argv[0]);
        exit(-1);
    }

    int smallfd = open((xtcname+".smd.xtc2").c_str(), O_RDONLY);
    if (smallfd < 0) {
        fprintf(stderr, "Unable to open smd file '%s'\n", xtcname.c_str());
        exit(-1);
    }
    int bigfd = open((xtcname+".xtc2").c_str(), O_RDONLY);
    if (bigfd < 0) {
        fprintf(stderr, "Unable to open big file '%s'\n", xtcname.c_str());
        exit(-1);
    }

    static const unsigned bigdgBufferSize = 0x4000000;
    Dgram* bigdg = (Dgram*)malloc(bigdgBufferSize);

    XtcFileIterator iter(smallfd, 0x4000000);
    Dgram* smalldg;
    unsigned nevent=0;
    SmdIter smditer;
    while ((smalldg = iter.next())) {
        if (nevent>=neventreq) break;
        nevent++;
        smditer.reset();
        smditer.iterate(&(smalldg->xtc));
        printf("Small event %d, %s transition: time %d.%09d, "
               "extent %d offset 0x%llx\n",
               nevent,
               TransitionId::name(smalldg->service()),
               smalldg->time.seconds(),
               smalldg->time.nanoseconds(), smalldg->xtc.extent,
               smditer.offset);
        if (smditer.offset<=0) continue; // non-L1 transitions are stored in smd
        if (lseek(bigfd, (off_t)smditer.offset, SEEK_SET) < 0) {
            printf("lseek error\n");
            exit(-1);
        }
        if (::read(bigfd, bigdg, sizeof(Dgram)) == 0) {
            printf("Big dgram header read error\n");
            exit(-1);
        }
        unsigned bigdgsize = sizeof(Dgram)+bigdg->xtc.extent;
        if (bigdgsize>bigdgBufferSize) {
            printf("Big dgram too large %s\n",bigdgsize);
            exit(-1);
        }
        if (::read(bigfd, bigdg->xtc.payload(), bigdg->xtc.extent) == 0) {
            printf("Big dgram payload read error\n");
            exit(-1);
        }
        printf("Big   event %d, %s transition: time %d.%09d, "
               "extent %d\n",
               nevent,
               TransitionId::name(bigdg->service()), bigdg->time.seconds(),
               bigdg->time.nanoseconds(), bigdg->xtc.extent);
    }

    ::close(smallfd);
    ::close(bigfd);
    free(bigdg);
    return 0;
}
