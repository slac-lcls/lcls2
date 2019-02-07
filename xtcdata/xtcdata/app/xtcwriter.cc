#include "xtcdata/xtc/ShapesData.hh"
#include "xtcdata/xtc/DescData.hh"
#include "xtcdata/xtc/Dgram.hh"
#include "xtcdata/xtc/TypeId.hh"
#include "xtcdata/xtc/XtcIterator.hh"
#include "xtcdata/xtc/VarDef.hh"
#include "xtcdata/xtc/NamesLookup.hh"
#include "rapidjson/document.h"

#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <unistd.h>
#include <type_traits>
#include <cstring>
#include <sys/time.h>
// #include <Python.h>
#include <stdint.h>

using namespace XtcData;
using namespace rapidjson;

#define BUFSIZE 0x4000000

class FexDef:public VarDef
{
public:
  enum index
    {
      floatFex,
      arrayFex,
      intFex
    };

  FexDef()
   {
       NameVec.push_back({"floatFex",Name::DOUBLE});
       NameVec.push_back({"arrayFex",Name::FLOAT,2});
       NameVec.push_back({"intFex",Name::INT64});
   }
} FexDef;

class PgpDef:public VarDef
{
public:
  enum index
    {
      floatPgp,
      array0Pgp,
      intPgp,
      array1Pgp
    };

  
   PgpDef()
   {
     NameVec.push_back({"floatPgp",Name::DOUBLE,0});
     NameVec.push_back({"array0Pgp",Name::FLOAT,2});
     NameVec.push_back({"intPgp",Name::INT64,0});
     NameVec.push_back({"array1Pgp",Name::FLOAT,2});     
   }
} PgpDef;


class PadDef:public VarDef
{
public:
  enum index
    {
      arrayRaw
    };
  
  
  PadDef()
   {
     Alg segmentAlg("cspadseg",2,3,42);
     NameVec.push_back({"arrayRaw", segmentAlg});
   }
} PadDef;



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
                printf("%s: %d, %d, %d\n",name.name(),arrT(0),arrT(1), arrT(2));
                    }
            else{
                printf("%s: %d\n",name.name(),descdata.get_value<uint8_t>(i));
            }
            break;
        }

        case(1):{
            if(data_rank > 0){
                Array<uint16_t> arrT = descdata.get_array<uint16_t>(i);
                printf("%s: %d, %d, %d\n",name.name(),arrT(0),arrT(1), arrT(2));
                    }
            else{
                printf("%s: %d\n",name.name(),descdata.get_value<uint16_t>(i));
            }
            break;
        }

        case(2):{
            if(data_rank > 0){
                Array<uint32_t> arrT = descdata.get_array<uint32_t>(i);
                printf("%s: %d, %d, %d\n",name.name(),arrT(0),arrT(1), arrT(2));
                    }
            else{
                printf("%s: %d\n",name.name(),descdata.get_value<uint32_t>(i));
            }
            break;
        }

        case(3):{
            if(data_rank > 0){
                Array<uint64_t> arrT = descdata.get_array<uint64_t>(i);
                printf("%s: %d, %d, %d\n",name.name(),arrT(0),arrT(1), arrT(2));
                    }
            else{
                printf("%s: %d\n",name.name(),descdata.get_value<uint64_t>(i));
            }
            break;
        }

        case(4):{
            if(data_rank > 0){
                Array<int8_t> arrT = descdata.get_array<int8_t>(i);
                printf("%s: %d, %d, %d\n",name.name(),arrT(0),arrT(1), arrT(2));
                    }
            else{
                printf("%s: %d\n",name.name(),descdata.get_value<int8_t>(i));
            }
            break;
        }

        case(5):{
            if(data_rank > 0){
                Array<int16_t> arrT = descdata.get_array<int16_t>(i);
                printf("%s: %d, %d, %d\n",name.name(),arrT(0),arrT(1), arrT(2));
                    }
            else{
                printf("%s: %d\n",name.name(),descdata.get_value<int16_t>(i));
            }
            break;
        }

        case(6):{
            if(data_rank > 0){
                Array<int32_t> arrT = descdata.get_array<int32_t>(i);
                printf("%s: %d, %d, %d\n",name.name(),arrT(0),arrT(1), arrT(2));
                    }
            else{
                printf("%s: %d\n",name.name(),descdata.get_value<int32_t>(i));
            }
            break;
        }

        case(7):{
            if(data_rank > 0){
                Array<int64_t> arrT = descdata.get_array<int64_t>(i);
                printf("%s: %d, %d, %d\n",name.name(),arrT(0),arrT(1), arrT(2));
                    }
            else{
                printf("%s: %d\n",name.name(),descdata.get_value<int64_t>(i));
            }
            break;
        }

        case(8):{
            if(data_rank > 0){
                Array<float> arrT = descdata.get_array<float>(i);
                printf("%s: %f, %f\n",name.name(),arrT(0),arrT(1));
                    }
            else{
                printf("%s: %f\n",name.name(),descdata.get_value<float>(i));
            }
            break;
        }

        case(9):{
            if(data_rank > 0){
                Array<double> arrT = descdata.get_array<double>(i);
                printf("%s: %f, %f, %f\n",name.name(),arrT(0),arrT(1), arrT(2));
                    }
            else{
                printf("%s: %f\n",name.name(),descdata.get_value<double>(i));
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

#pragma pack(push,1)
class PgpData
{
public:
    void* operator new(size_t size, void* p)
    {
        return p;
    }
    PgpData(float f, int i1)
    {
        _fdata = f;
        _idata = i1;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                array[i][j] = i * j;
                array2[i][j] = i * j + 2;
            }
        }
    }

    double _fdata;
    float array[3][3];
    int64_t _idata;
    float array2[3][3];
};
#pragma pack(pop)

class PadData
{
public:
  
    void* operator new(size_t size, void* p)
    {
        return p;
    }
    PadData()
    {
        uint8_t counter = 0;
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 3; k++) {
                    array[counter] = counter;
                    counter++;
                }
            }
        }
    }
    uint8_t array[18];
};

void pgpExample(Xtc& parent, NamesLookup& namesLookup, NamesId& namesId)
{
    DescribedData frontEnd(parent, namesLookup, namesId);

    // simulates PGP data arriving, and shows the address that should be given to PGP driver
    // we should perhaps worry about DMA alignment issues if in the future
    // we avoid the pgp driver copy into user-space.
    new (frontEnd.data()) PgpData(1, 2);

    // now that data has arrived update with the number of bytes received
    // it is required to call this before set_array_shape
    // printf("sizeof pgpdata %d\n",sizeof(PgpData));
    frontEnd.set_data_length(sizeof(PgpData));

    unsigned shape[] = { 3, 3 };
    frontEnd.set_array_shape(PgpDef::array0Pgp, shape);
    frontEnd.set_array_shape(PgpDef::array1Pgp, shape);
}

void fexExample(Xtc& parent, NamesLookup& namesLookup, NamesId& namesId)
{ 
    CreateData fex(parent, namesLookup, namesId);
    fex.set_value(FexDef::floatFex, (double)41.0);

    unsigned shape[MaxRank] = {2,3};
    Array<float> arrayT = fex.allocate<float>(FexDef::arrayFex,shape);
    for(unsigned i=0; i<shape[0]; i++){
        for (unsigned j=0; j<shape[1]; j++) {
            arrayT(i,j) = 142.0+i*shape[1]+j;
        }
    };
    
    fex.set_value(FexDef::intFex, (int64_t) 42);
}
   

void padExample(Xtc& parent, NamesLookup& namesLookup, NamesId& namesId)
{ 
    DescribedData pad(parent, namesLookup, namesId);

    // simulates PGP data arriving, and shows the address that should be given to PGP driver
    // we should perhaps worry about DMA alignment issues if in the future
    // we avoid the pgp driver copy into user-space.
    new (pad.data()) PadData();

    // now that data has arrived update with the number of bytes received
    // it is required to call this before set_array_shape
    pad.set_data_length(sizeof(PadData));

    unsigned shape[] = { 18 };
    pad.set_array_shape(PadDef::arrayRaw, shape);
}

void addNames(Xtc& xtc, NamesLookup& namesLookup, unsigned& nodeId) {
    Alg hsdRawAlg("raw",0,0,0);
    NamesId namesId0(nodeId,0);
    Names& frontEndNames = *new(xtc) Names("xpphsd", hsdRawAlg, "hsd", "detnum1234", namesId0);
    frontEndNames.add(xtc,PgpDef);
    namesLookup[namesId0] = NameIndex(frontEndNames);

    Alg hsdFexAlg("fex",4,5,6);
    NamesId namesId1(nodeId,1);
    Names& fexNames = *new(xtc) Names("xpphsd", hsdFexAlg, "hsd","detnum1234", namesId1);
    fexNames.add(xtc, FexDef);
    namesLookup[namesId1] = NameIndex(fexNames);

    unsigned segment = 0;
    Alg cspadRawAlg("raw",2,3,42);
    NamesId namesId2(nodeId,2);
    Names& padNames = *new(xtc) Names("xppcspad", cspadRawAlg, "cspad", "detnum1234", namesId2, segment);
    Alg segmentAlg("cspadseg",2,3,42);
    padNames.add(xtc, PadDef);
    namesLookup[namesId2] = NameIndex(padNames);
}

void addData(Xtc& xtc, NamesLookup& namesLookup, unsigned nodeId) {
    NamesId namesId0(nodeId,0);
    pgpExample(xtc, namesLookup, namesId0);
    NamesId namesId1(nodeId,1);
    fexExample(xtc, namesLookup, namesId1);
    NamesId namesId2(nodeId,2);
    padExample(xtc, namesLookup, namesId2);
}

class HsdConfigDef:public VarDef
{
public:
  enum index
    {
      enable,
      raw_prescale
    };

  HsdConfigDef()
   {
       NameVec.push_back({"enable",Name::UINT64,1});
       NameVec.push_back({"raw_prescale",Name::UINT64,1});
   }
} HsdConfigDef;

// void addJson(Xtc& xtc, NamesLookup& namesLookup) {

//     FILE* file;
//     Py_Initialize();
//     PyObject* main_module = PyImport_AddModule("__main__");
//     PyObject* main_dict = PyModule_GetDict(main_module);
//     file = fopen("hsdConfig.py","r");
//     PyObject* a = PyRun_File(file, "hsdConfig.py",Py_file_input,main_dict,main_dict);
//     printf("PyRun: %p \n", a);
//     printf("size %d\n",PyDict_Size(main_dict));
//     PyObject* mybytes = PyDict_GetItemString(main_dict,"dgram");
//     printf("%p\n",mybytes);
//     char* json = (char*)PyBytes_AsString(mybytes); // TODO: Add PyUnicode_AsEncodedString()?
//     printf("%p\n",mybytes);
//     printf("%s\n",json);
//     Py_Finalize();
//     printf("Done\n");

//     Document d;
//     d.Parse(json);

//     Value& software = d["alg"]["software"];
//     Value& version = d["alg"]["version"];
//     for (SizeType i = 0; i < version.Size(); i++) // Uses SizeType instead of size_t
//         printf("version[%d] = %d\n", i, version[i].GetInt());

//     Alg hsdConfigAlg(software.GetString(),version[0].GetInt(),version[1].GetInt(),version[2].GetInt());
//     Names& configNames = *new(xtc) Names("xpphsd", hsdConfigAlg, "hsd", "detnum1235");
//     configNames.add(xtc, HsdConfigDef);
//     namesLookup.push_back(NameIndex(configNames));

//     CreateData fex(xtc, namesLookup, 3); //FIXME: avoid hardwiring namesId

//     // TODO: dynamically discover

//     Value& enable = d["xtc"]["ENABLE"];
//     unsigned shape[MaxRank] = {enable.Size()};
//     Array<uint64_t> arrayT = fex.allocate<uint64_t>(HsdConfigDef::enable,shape); //FIXME: figure out avoiding hardwired zero
//     for(unsigned i=0; i<shape[0]; i++){
//         arrayT(i) = (uint64_t) enable[i].GetInt();
//     };

//     Value& raw_prescale = d["xtc"]["RAW_PS"];
//     shape[MaxRank] = {raw_prescale.Size()};
//     Array<uint64_t> arrayT1 = fex.allocate<uint64_t>(HsdConfigDef::raw_prescale,shape); //FIXME: figure out avoiding hardwired zero
//     for(unsigned i=0; i<shape[0]; i++){
//         arrayT1(i) = (uint64_t) raw_prescale[i].GetInt();
//     };
// }

void usage(char* progname)
{
    fprintf(stderr, "Usage: %s [-f <filename> -n <numEvents> -t -h]\n", progname);
}

#define MAX_FNAME_LEN 256

int main(int argc, char* argv[])
{
    int c;
    int writeTs = 0;
    int parseErr = 0;
    unsigned nevents = 2;
    char xtcname[MAX_FNAME_LEN];
    strncpy(xtcname, "data.xtc2", MAX_FNAME_LEN);

    while ((c = getopt(argc, argv, "htf:n:")) != -1) {
        switch (c) {
            case 'h':
                usage(argv[0]);
                exit(0);
            case 't':
                writeTs = 1;
                break;
            case 'n':
                nevents = atoi(optarg);
                break;
            case 'f':
                strncpy(xtcname, optarg, MAX_FNAME_LEN);
                break;
            default:
                parseErr++;
        }
    }

    FILE* xtcFile = fopen(xtcname, "w");
    if (!xtcFile) {
        printf("Error opening output xtc file.\n");
        return -1;
    }

    void* configbuf = malloc(BUFSIZE);
    Dgram& config = *(Dgram*)configbuf;
    TypeId tid(TypeId::Parent, 0);
    config.xtc.contains = tid;
    config.xtc.damage = 0;
    config.xtc.extent = sizeof(Xtc);

    unsigned nodeid1 = 1;
    unsigned nodeid2 = 2;
    NamesLookup namesLookup1;
    // NamesLookup namesLookup2;
    addNames(config.xtc, namesLookup1, nodeid1);
    // addNames(config.xtc, namesLookup2, nodeid2);
    // addData(config.xtc, namesLookup2, nodeid2);
    addData(config.xtc, namesLookup1, nodeid1);

    //addJson(config.xtc,namesLookup);
    //std::cout << "Done addJson" << std::endl;

    DebugIter iter(&config.xtc, namesLookup1);
    iter.iterate();
    std::cout << "Done iter" << std::endl;

    if (fwrite(&config, sizeof(config) + config.xtc.sizeofPayload(), 1, xtcFile) != 1) {
        printf("Error writing configure to output xtc file.\n");
        return -1;
    }

     // for(auto const& value: namesLookup[0].nameMap()){
     //     std::cout<<value.first<<std::endl;
     // }


    void* buf = malloc(BUFSIZE);
    struct timeval tv;
    uint64_t pulseId = 0;
 
    for (int i = 0; i < nevents; i++) {
        Dgram& dgram = *(Dgram*)buf;
        TypeId tid(TypeId::Parent, 0);
        dgram.xtc.contains = tid;
        dgram.xtc.damage = 0;
        dgram.xtc.extent = sizeof(Xtc);
        
        if (writeTs != 0) {
            gettimeofday(&tv, NULL);
            dgram.seq = Sequence(TimeStamp(tv.tv_sec, tv.tv_usec), PulseId(pulseId,0));
        }

        addData(dgram.xtc, namesLookup1, nodeid1);
        // addData(dgram.xtc, namesLookup2, nodeid2);

        printf("*** event %d ***\n",i);

        if (writeTs != 0) {
            std::cout << "timestamp: " << dgram.seq.stamp().value() << std::endl;
        }

        DebugIter iter(&dgram.xtc, namesLookup1);
        iter.iterate();

        if (fwrite(&dgram, sizeof(dgram) + dgram.xtc.sizeofPayload(), 1, xtcFile) != 1) {
            printf("Error writing to output xtc file.\n");
            return -1;
        }
     }
    fclose(xtcFile);

    return 0;
}
