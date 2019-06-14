#include "xtcdata/xtc/ShapesData.hh"
#include "xtcdata/xtc/DescData.hh"
#include "xtcdata/xtc/Dgram.hh"
#include "xtcdata/xtc/TypeId.hh"
#include "xtcdata/xtc/XtcIterator.hh"
#include "xtcdata/xtc/VarDef.hh"
#include "xtcdata/xtc/NamesLookup.hh"
#include "rapidjson/document.h"
#include "xtcdata/xtc/XtcFileIterator.hh"

#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <unistd.h>
#include <type_traits>
#include <cstring>
#include <sys/time.h>
#include <stdint.h>
#include <fstream>
#include <iostream>
#include <array>
#include <fcntl.h>

using namespace XtcData;
using namespace rapidjson;
using namespace std;
using std::string;

#define BUFSIZE 0x4000000

class EpicsDefFast:public VarDef
{
public:
  enum index
    {
        HX2_DVD_GCC_01_PMON,
        HX2_DVD_GPI_01_PMON,
    };

  EpicsDefFast()
   {
        NameVec.push_back({"HX2:DVD:GCC:01:PMON",Name::DOUBLE});
        NameVec.push_back({"HX2:DVD:GPI:01:PMON",Name::DOUBLE});
   }
} EpicsDefFast;

class EpicsDefSlow:public VarDef
{
public:
  enum index
    {
        XPP_GON_MMS_01_RBV,
        XPP_GON_MMS_02_RBV,
    };

  EpicsDefSlow()
   {
        NameVec.push_back({"XPP:GON:MMS:01:RBV",Name::DOUBLE});
        NameVec.push_back({"XPP:GON:MMS:02:RBV",Name::DOUBLE});
   }
} EpicsDefSlow;

class EpicsDefS2Fast:public VarDef
{
public:
  enum index
    {
        XPP_VARS_FLOAT_02,
        XPP_VARS_FLOAT_03,
    };

  EpicsDefS2Fast()
   {
        NameVec.push_back({"XPP:VARS:FLOAT:02",Name::DOUBLE});
        NameVec.push_back({"XPP:VARS:FLOAT:03",Name::DOUBLE});
   }
} EpicsDefS2Fast;

class EpicsDefS2Slow:public VarDef
{
public:
  enum index
    {
        XPP_VARS_STRING_01,
        XPP_VARS_STRING_02,
    };

  EpicsDefS2Slow()
   {
        NameVec.push_back({"XPP:VARS:STRING:01",Name::CHARSTR,1});
        NameVec.push_back({"XPP:VARS:STRING:02",Name::CHARSTR,1});
   }
} EpicsDefS2Slow;


class DebugIter : public XtcIterator
{
public:
    enum { Stop, Continue };
    DebugIter(Xtc* xtc, NamesLookup& namesLookup) : XtcIterator(xtc), _namesLookup(namesLookup)
    {
    }

    void get_value(int i, Name& name, DescData& descdata){
        int data_rank = name.rank();
        int data_type = name.type();

        switch(name.type()){
        case(0):{
            if(data_rank > 0){
                Array<uint8_t> arrT = descdata.get_array<uint8_t>(i);
                // printf("%s: %d, %d, %d\n",name.name(),arrT(0),arrT(1), arrT(2));
                    }
            else{
                // printf("%s: %d\n",name.name(),descdata.get_value<uint8_t>(i));
            }
            break;
        }

        case(1):{
            if(data_rank > 0){
                Array<uint16_t> arrT = descdata.get_array<uint16_t>(i);
                // printf("%s: %d, %d, %d\n",name.name(),arrT(0),arrT(1), arrT(2));
                    }
            else{
                // printf("%s: %d\n",name.name(),descdata.get_value<uint16_t>(i));
            }
            break;
        }

        case(2):{
            if(data_rank > 0){
                Array<uint32_t> arrT = descdata.get_array<uint32_t>(i);
                // printf("%s: %d, %d, %d\n",name.name(),arrT(0),arrT(1), arrT(2));
                    }
            else{
                // printf("%s: %d\n",name.name(),descdata.get_value<uint32_t>(i));
            }
            break;
        }

        case(3):{
            if(data_rank > 0){
                Array<uint64_t> arrT = descdata.get_array<uint64_t>(i);
                // printf("%s: %d, %d, %d\n",name.name(),arrT(0),arrT(1), arrT(2));
                    }
            else{
                // printf("%s: %d\n",name.name(),descdata.get_value<uint64_t>(i));
            }
            break;
        }

        case(4):{
            if(data_rank > 0){
                Array<int8_t> arrT = descdata.get_array<int8_t>(i);
                // printf("%s: %d, %d, %d\n",name.name(),arrT(0),arrT(1), arrT(2));
                    }
            else{
                // printf("%s: %d\n",name.name(),descdata.get_value<int8_t>(i));
            }
            break;
        }

        case(5):{
            if(data_rank > 0){
                Array<int16_t> arrT = descdata.get_array<int16_t>(i);
                // printf("%s: %d, %d, %d\n",name.name(),arrT(0),arrT(1), arrT(2));
                    }
            else{
                // printf("%s: %d\n",name.name(),descdata.get_value<int16_t>(i));
            }
            break;
        }

        case(6):{
            if(data_rank > 0){
                Array<int32_t> arrT = descdata.get_array<int32_t>(i);
                // printf("%s: %d, %d, %d\n",name.name(),arrT(0),arrT(1), arrT(2));
                    }
            else{
                // printf("%s: %d\n",name.name(),descdata.get_value<int32_t>(i));
            }
            break;
        }

        case(7):{
            if(data_rank > 0){
                Array<int64_t> arrT = descdata.get_array<int64_t>(i);
                // printf("%s: %d, %d, %d\n",name.name(),arrT(0),arrT(1), arrT(2));
                    }
            else{
                // printf("%s: %d\n",name.name(),descdata.get_value<int64_t>(i));
            }
            break;
        }

        case(8):{
            if(data_rank > 0){
                Array<float> arrT = descdata.get_array<float>(i);
                // printf("%s: %f, %f\n",name.name(),arrT(0),arrT(1));
                    }
            else{
                // printf("%s: %f\n",name.name(),descdata.get_value<float>(i));
            }
            break;
        }

        case(9):{
            if(data_rank > 0){
                Array<double> arrT = descdata.get_array<double>(i);
                // printf("%s: %f, %f, %f\n",name.name(),arrT(0),arrT(1), arrT(2));
                    }
            else{
                // printf("%s: %f\n",name.name(),descdata.get_value<double>(i));
            }
            break;
        }

        }

    }

    int process(Xtc* xtc)
    {
        // printf("found typeid %s\n",XtcData::TypeId::name(xtc->contains.id()));
        switch (xtc->contains.id()) {

        case (TypeId::Names): {
            // printf("Names pointer is %p\n", xtc);
            break;
        }
        case (TypeId::Parent): {
            iterate(xtc);
            break;
        }
        case (TypeId::ShapesData): {
            ShapesData& shapesdata = *(ShapesData*)xtc;
            // lookup the index of the names we are supposed to use
            NamesId& namesId = shapesdata.namesId();
            DescData descdata(shapesdata, _namesLookup[namesId]);
            Names& names = descdata.nameindex().names();
	    //   printf("Found %d names\n",names.num());
            // printf("data shapes extents %d %d %d\n", shapesdata.data().extent,
                   // shapesdata.shapes().extent, sizeof(double));
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
    NamesLookup& _namesLookup;
    void printOffset(const char* str, void* base, void* ptr) {
        printf("***%s at offset %li addr %p\n",str,(char*)ptr-(char*)base,ptr);
    }
};

void epicsExample(Xtc& parent, NamesLookup& namesLookup, NamesId& namesId, int streamId, int nameId)
{ 
    CreateData epics(parent, namesLookup, namesId);
    if (streamId == 1) {

        if (nameId == 0) {
            epics.set_value(EpicsDefFast::HX2_DVD_GCC_01_PMON, (double)41.0);
            epics.set_value(EpicsDefFast::HX2_DVD_GPI_01_PMON, (double)41.0);
        } else if (nameId == 1) {
            epics.set_value(EpicsDefSlow::XPP_GON_MMS_01_RBV, (double)41.0);
            epics.set_value(EpicsDefSlow::XPP_GON_MMS_02_RBV, (double)41.0);
        }

    } else if (streamId == 2) {

        if (nameId == 0) {
            epics.set_value(EpicsDefS2Fast::XPP_VARS_FLOAT_02, (double)41.0);
            epics.set_value(EpicsDefS2Fast::XPP_VARS_FLOAT_03, (double)41.0);
        } else if (nameId == 1) {
            epics.set_string(EpicsDefS2Slow::XPP_VARS_STRING_01, "Test String");
            epics.set_string(EpicsDefS2Slow::XPP_VARS_STRING_02, "Test String");
        }
    } 
}
   

void addNames(Xtc& xtc, NamesLookup& namesLookup, unsigned& nodeId, unsigned segment, int streamId) {
    Alg xppEpicsFastAlg("fast",0,0,0);
    NamesId namesId0(nodeId,0+10*segment);
    Names& epicsFastNames = *new(xtc) Names("xppepics", xppEpicsFastAlg, "epics","detnum1234", namesId0, segment);
    if (streamId == 1) {
        epicsFastNames.add(xtc, EpicsDefFast);
    } else if (streamId == 2) {
        epicsFastNames.add(xtc, EpicsDefS2Fast);
    }
    namesLookup[namesId0] = NameIndex(epicsFastNames);

    Alg xppEpicsSlowAlg("slow",0,0,0);
    NamesId namesId1(nodeId,1+10*segment);
    Names& epicsSlowNames = *new(xtc) Names("xppepics", xppEpicsSlowAlg, "epics","detnum1234", namesId1, segment);
    if (streamId == 1) {
        epicsSlowNames.add(xtc, EpicsDefSlow);
    } else if (streamId == 2) {
        epicsSlowNames.add(xtc, EpicsDefS2Slow);
    }
    namesLookup[namesId1] = NameIndex(epicsSlowNames);
}

void addData(Xtc& xtc, NamesLookup& namesLookup, unsigned nodeId, unsigned segment, int streamId, int nameId) {
    // depending on namesId - add fast-slow (-1), fast (0), or slow (1) data
    if (nameId == -1) {
        NamesId namesId0(nodeId,0+10*segment);
        epicsExample(xtc, namesLookup, namesId0, streamId, 0);
        NamesId namesId1(nodeId,1+10*segment);
        epicsExample(xtc, namesLookup, namesId1, streamId, 1);
    } else {
        NamesId namesId(nodeId,nameId+10*segment);
        epicsExample(xtc, namesLookup, namesId, streamId, nameId);
    }
}

void usage(char* progname)
{
    fprintf(stderr, "Usage: %s [-f <filename> -n <numEvents> -t -h]\n", progname);
}

Dgram& createTransition(TransitionId::Value transId) {
    TypeId tid(TypeId::Parent, 0);
    uint64_t pulseId = 0xffff;
    uint32_t env = 0;
    struct timeval tv;
    void* buf = malloc(BUFSIZE);
    gettimeofday(&tv, NULL);
    Sequence seq(Sequence::Event, transId, TimeStamp(tv.tv_sec, tv.tv_usec), PulseId(pulseId,0));
    return *new(buf) Dgram(Transition(seq, env), Xtc(tid));
}

void save(Dgram& dg, FILE* xtcFile) {
    if (fwrite(&dg, sizeof(dg) + dg.xtc.sizeofPayload(), 1, xtcFile) != 1) {
        printf("Error writing to output xtc file.\n");
    }
}

#define MAX_FNAME_LEN 256

int main(int argc, char* argv[])
{
    int c;
    int parseErr = 0;
    unsigned nevents = 2;
    char outname[MAX_FNAME_LEN];
    char* tsname = 0;
    char* bdXtcname = 0; // Bigdata xtc - if specified, timestamps are from this file
    strncpy(outname, "epics.xtc2", MAX_FNAME_LEN);
    int streamId = 1;

    while ((c = getopt(argc, argv, "ht:f:n:s:o:")) != -1) {
        switch (c) {
            case 'h':
                usage(argv[0]);
                exit(0);
            case 't':
                tsname = optarg;
                break;
            case 'n':
                nevents = atoi(optarg);
                break;
            case 'f':
                bdXtcname = optarg;
                break;
            case 'o':
                strncpy(outname, optarg, MAX_FNAME_LEN);
                break;
            case 's':
                streamId = atoi(optarg);
                break;
            default:
                parseErr++;
        }
    }

    // Output xtc file
    FILE* xtcFile = fopen(outname, "w");
    if (!xtcFile) {
        printf("Error opening output xtc file.\n");
        return -1;
    }

    // Bigdata xtc file
    int fd = -1;
    if (bdXtcname != 0) {
        fd = open(bdXtcname, O_RDONLY);
        if (fd < 0) {
            fprintf(stderr, "Unable to open bigdata xtc file %s\n", bdXtcname);
            exit(2);
        }
    }

    XtcFileIterator dgInIter(fd, BUFSIZE);
    Dgram* dgIn;

    // Read timestamp 
    array<time_t, 500> sec_arr = {};
    array<long, 500> nsec_arr = {};
    sec_arr[0] = 0;
    nsec_arr[0] = 0;
    if (tsname != 0) {
        ifstream tsfile(tsname);
        time_t sec = 0;
        long nsec = 0;
        int i = 0;
        while (tsfile >> sec >> nsec) {
            sec_arr[i] = sec;
            nsec_arr[i] = nsec;
            cout << i << ":" << sec_arr[i] << " " << nsec_arr[i] << endl;
            i++;
        }  
        printf("found %d timestamps\n", i); 
    }

    struct timeval tv;
    TypeId tid(TypeId::Parent, 0);
    uint32_t env = 0;
    uint64_t pulseId = 0;

    Dgram& config = createTransition(TransitionId::Configure);

    unsigned nodeid1 = 1;
    NamesLookup namesLookup1;
    unsigned nSegments=1;
    for (unsigned iseg=0; iseg<nSegments; iseg++) {
        addNames(config.xtc, namesLookup1, nodeid1, iseg, streamId);
        addData(config.xtc, namesLookup1, nodeid1, iseg, streamId, -1); // nameId = -1 for adding both fast and slow
    }

    save(config,xtcFile);

    DebugIter iter(&config.xtc, namesLookup1);
    iter.iterate();

    void* buf = malloc(BUFSIZE);
    for (int i = 0; i < nevents; i++) {
        // Determine how which timestamps to be used: from bigdata xtc,
        // from text file (space separated), or current time.
        Sequence seq;
        if (bdXtcname != 0) {
            dgIn = dgInIter.next();
            seq = dgIn->seq;
        } else if (tsname != 0) {
            seq = Sequence(Sequence::Event, TransitionId::L1Accept, TimeStamp(sec_arr[i], nsec_arr[i]), PulseId(pulseId, 0));
            cout << "Timestamp from file: " << sec_arr[i] << " " << nsec_arr[i] << endl;
        } else {
            gettimeofday(&tv, NULL);
            seq = Sequence(Sequence::Event, TransitionId::L1Accept, TimeStamp(tv.tv_sec, tv.tv_usec), PulseId(pulseId,0));
        }

        Dgram& dgram = *new(buf) Dgram(Transition(seq, env), Xtc(tid));
        
        // Create a sequence for fast-slow, fast-only epics events
        int nameId = 0;
        if (i % 2 == 0) {
            nameId = -1;
        } else {
            nameId = 0;
        }
        
        // Add event
        for (unsigned iseg=0; iseg<nSegments; iseg++) {
            addData(dgram.xtc, namesLookup1, nodeid1, iseg, streamId, nameId);
        }

        DebugIter iter(&dgram.xtc, namesLookup1);
        iter.iterate();

        save(dgram,xtcFile);
     }

    fclose(xtcFile);

    return 0;
}
