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

class DebugIter : public XtcIterator
{
public:
    enum { Stop, Continue };
    DebugIter() : XtcIterator()
    {
    }

    void get_value(int i, Name& name, DescData& descdata){
        int data_rank = name.rank();
        int data_type = name.type();
        printf("%d: '%s' rank %d, type %d\n", i, name.name(), data_rank, data_type);

        switch(name.type()){
        case(Name::UINT8):{
            if(data_rank > 0){
                Array<uint8_t> arrT = descdata.get_array<uint8_t>(i);
                printf("'%s': %d, %d, %d\n",name.name(),arrT.data()[0],arrT.data()[1], arrT.data()[2]);
                    }
            else{
                printf("'%s': %d\n",name.name(),descdata.get_value<uint8_t>(i));
            }
            break;
        }

        case(Name::UINT16):{
            if(data_rank > 0){
                Array<uint16_t> arrT = descdata.get_array<uint16_t>(i);
                printf("'%s': %d, %d, %d\n",name.name(),arrT.data()[0],arrT.data()[1], arrT.data()[2]);
                    }
            else{
                printf("'%s': %d\n",name.name(),descdata.get_value<uint16_t>(i));
            }
            break;
        }

        case(Name::UINT32):{
            if(data_rank > 0){
                Array<uint32_t> arrT = descdata.get_array<uint32_t>(i);
                printf("'%s': %d, %d, %d\n",name.name(),arrT.data()[0],arrT.data()[1], arrT.data()[2]);
                    }
            else{
                printf("'%s': %d\n",name.name(),descdata.get_value<uint32_t>(i));
            }
            break;
        }

        case(Name::UINT64):{
            if(data_rank > 0){
                Array<uint64_t> arrT = descdata.get_array<uint64_t>(i);
                printf("'%s': %d, %d, %d\n",name.name(),arrT.data()[0],arrT.data()[1], arrT.data()[2]);
                    }
            else{
                printf("'%s': %d\n",name.name(),descdata.get_value<uint64_t>(i));
            }
            break;
        }

        case(Name::INT8):{
            if(data_rank > 0){
                Array<int8_t> arrT = descdata.get_array<int8_t>(i);
                printf("'%s': %d, %d, %d\n",name.name(),arrT.data()[0],arrT.data()[1], arrT.data()[2]);
                    }
            else{
                printf("'%s': %d\n",name.name(),descdata.get_value<int8_t>(i));
            }
            break;
        }

        case(Name::INT16):{
            if(data_rank > 0){
                Array<int16_t> arrT = descdata.get_array<int16_t>(i);
                printf("'%s': %d, %d, %d\n",name.name(),arrT.data()[0],arrT.data()[1], arrT.data()[2]);
                    }
            else{
                printf("'%s': %d\n",name.name(),descdata.get_value<int16_t>(i));
            }
            break;
        }

        case(Name::INT32):{
            if(data_rank > 0){
                Array<int32_t> arrT = descdata.get_array<int32_t>(i);
                printf("'%s': %d, %d, %d\n",name.name(),arrT.data()[0],arrT.data()[1], arrT.data()[2]);
                    }
            else{
                printf("'%s': %d\n",name.name(),descdata.get_value<int32_t>(i));
            }
            break;
        }

        case(Name::INT64):{
            if(data_rank > 0){
                Array<int64_t> arrT = descdata.get_array<int64_t>(i);
                printf("'%s': %d, %d, %d\n",name.name(),arrT.data()[0],arrT.data()[1], arrT.data()[2]);
                    }
            else{
                printf("'%s': %d\n",name.name(),descdata.get_value<int64_t>(i));
            }
            break;
        }

        case(Name::FLOAT):{
            if(data_rank > 0){
                Array<float> arrT = descdata.get_array<float>(i);
                printf("'%s': %f, %f\n",name.name(),arrT.data()[0],arrT.data()[1]);
                    }
            else{
                printf("'%s': %f\n",name.name(),descdata.get_value<float>(i));
            }
            break;
        }

        case(Name::DOUBLE):{
            if(data_rank > 0){
                Array<double> arrT = descdata.get_array<double>(i);
                printf("'%s': %f, %f, %f\n",name.name(),arrT.data()[0],arrT.data()[1], arrT.data()[2]);
                    }
            else{
                printf("'%s': %f\n",name.name(),descdata.get_value<double>(i));
            }
            break;
        }

        case(Name::CHARSTR):{
            if(data_rank > 0){
                Array<char> arrT = descdata.get_array<char>(i);
                printf("'%s': \"%s\"\n",name.name(),arrT.data());
                    }
            else{
                printf("'%s': string with no rank?!?\n",name.name());
            }
            break;
        }

        case(Name::ENUMVAL):{
            if(data_rank > 0){
                Array<int32_t> arrT = descdata.get_array<int32_t>(i);
                printf("'%s': %d, %d, %d\n",name.name(),arrT.data()[0],arrT.data()[1], arrT.data()[2]);
                    }
            else{
                printf("'%s': %d\n",name.name(),descdata.get_value<int32_t>(i));
            }
            break;
        }

        case(Name::ENUMDICT):{
            if(data_rank > 0){
                printf("'%s': enumdict with rank?!?\n", name.name());
            } else{
                printf("'%s': %d\n",name.name(),descdata.get_value<int32_t>(i));
            }
            break;
        }
        }
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
            Alg& alg = names.alg();
	    printf("*** DetName: %s, Segment %d, DetType: %s, Alg: %s, Version: 0x%6.6x, namesid: 0x%x, Names:\n",
                   names.detName(), names.segment(), names.detType(),
                   alg.name(), alg.version(), (int)names.namesId());

            for (unsigned i = 0; i < names.num(); i++) {
                Name& name = names.get(i);
                printf("Name: '%s' Type: %d Rank: %d\n",name.name(),name.type(), name.rank());
            }

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
            Names& names = descdata.nameindex().names();
            Data& data = shapesdata.data();
	    printf("Found %d names\n",names.num());
            for (unsigned i = 0; i < names.num(); i++) {
                Name& name = names.get(i);
                get_value(i, name, descdata);
            }
            break;
        }
        default:
            break;
        }
        return Continue;
    }

private:
    NamesLookup _namesLookup;
};


void usage(char* progname)
{
    fprintf(stderr, "Usage: %s -f <filename> [-d] [-n <nEvents>] [-h]\n", progname);
}

int main(int argc, char* argv[])
{
    int c;
    char* xtcname = 0;
    int parseErr = 0;
    unsigned neventreq = 0xffffffff;
    bool debugprint = false;

    while ((c = getopt(argc, argv, "hf:n:d")) != -1) {
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
        case 'd':
            debugprint = true;
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
    unsigned nevent=0;
    DebugIter dbgiter;
    while ((dg = iter.next())) {
        if (nevent>=neventreq) break;
        nevent++;
        printf("event %d, %11s transition: time %d.%09d, pulseId 0x%014lx, env 0x%08x, "
               "payloadSize %d extent %d\n",
               nevent,
               TransitionId::name(dg->seq.service()), dg->seq.stamp().seconds(),
               dg->seq.stamp().nanoseconds(), dg->seq.pulseId().value(),
               dg->env, dg->xtc.sizeofPayload(),dg->xtc.extent);
        if (debugprint) dbgiter.iterate(&(dg->xtc));
    }

    ::close(fd);
    return 0;
}
