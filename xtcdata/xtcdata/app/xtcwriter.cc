#include "xtcdata/xtc/ShapesData.hh"
#include "xtcdata/xtc/DescData.hh"
#include "xtcdata/xtc/Dgram.hh"
#include "xtcdata/xtc/TypeId.hh"
#include "xtcdata/xtc/XtcIterator.hh"
#include "xtcdata/xtc/VarDef.hh"
#include "xtcdata/xtc/NamesLookup.hh"

#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <unistd.h>
#include <type_traits>
#include <cstring>
#include <sys/time.h>
#include <stdint.h>

using namespace XtcData;

#define BUFSIZE 0x4000000
static const unsigned nSegments=2;

enum MyNamesId {HsdRaw,HsdFex,Cspad,Epics,EpicsInfo,Scan,RunInfo,HsdCfg,HsdRun,ChunkInfo,NumberOf};

class RunInfoDef:public VarDef
{
public:
  enum index
    {
        EXPT,
        RUNNUM
    };

  RunInfoDef()
   {
       NameVec.push_back({"expt",Name::CHARSTR,1});
       NameVec.push_back({"runnum",Name::UINT32});
   }
} RunInfoDef;

class ChunkInfoDef:public VarDef
{
public:
  enum index
    {
        CHUNKID,
        FILENAME
    };

  ChunkInfoDef()
   {
       NameVec.push_back({"chunkid",Name::UINT32});
       NameVec.push_back({"filename",Name::CHARSTR,1});
   }
} ChunkInfoDef;

class EpicsDef:public VarDef
{
public:
  enum index
    {
        HX2_DVD_GCC_01_PMON,
        HX2_DVD_GPI_01_PMON,
    };

  EpicsDef()
   {
        NameVec.push_back({"HX2:DVD:GCC:01:PMON",Name::DOUBLE});
        NameVec.push_back({"HX2:DVD:GPI:01:PMON",Name::CHARSTR,1});
   }
} EpicsDef;

void epicsExample(Xtc& parent, const void* bufEnd, NamesLookup& namesLookup, NamesId& namesId)
{
    CreateData epics(parent, bufEnd, namesLookup, namesId);
    epics.set_value(EpicsDef::HX2_DVD_GCC_01_PMON, (double)41.0);
    epics.set_string(EpicsDef::HX2_DVD_GPI_01_PMON, "Test String");
}

class FexDef:public VarDef
{
public:
  enum index
    {
      floatFex,
      arrayFex,
      intFex,
      charStrFex,
      enumFex1,
      enumFex1_HighGain,
      enumFex1_LowGain,
      enumFex2,
      enumFex3,
      enumFex3_On,
      enumFex3_Off,
    };

  FexDef()
   {
       NameVec.push_back({"floatFex",Name::DOUBLE});
       NameVec.push_back({"arrayFex",Name::FLOAT,2});
       NameVec.push_back({"intFex",Name::INT64});
       NameVec.push_back({"charStrFex",Name::CHARSTR,1});
       // the enum dict requires ":EnumString" at the end of the name
       NameVec.push_back({"enumFex1:MyEnumName",Name::ENUMVAL});
       NameVec.push_back({"HighGain:MyEnumName",Name::ENUMDICT});
       NameVec.push_back({"LowGain:MyEnumName",Name::ENUMDICT});
       // enumFex2 reuses the same dictionary as enumFex1
       NameVec.push_back({"enumFex2:MyEnumName",Name::ENUMVAL});
       NameVec.push_back({"enumFex3:MyEnumName2",Name::ENUMVAL});
       NameVec.push_back({"On:MyEnumName2",Name::ENUMDICT});
       NameVec.push_back({"Off:MyEnumName2",Name::ENUMDICT});
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


class PgpScanDef:public VarDef
{
public:
  enum index
    {
      intPgp
    };


   PgpScanDef()
   {
     NameVec.push_back({"intPgp",Name::INT64,0});
   }
} PgpScanDef;


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
     NameVec.push_back({"arrayRaw", Name::UINT16, 2, segmentAlg});
   }
} PadDef;


class ScanDef:public VarDef
{
public:
  enum index
    {
        Motor1,
        Motor2
    };


  ScanDef()
   {
     Alg scanAlg("scanalg",2,3,42);
     NameVec.push_back({"motor1", Name::FLOAT, 0, scanAlg});
     NameVec.push_back({"motor2", Name::DOUBLE, 0, scanAlg});
   }
} ScanDef;

void scanExample(Xtc& parent, const void* bufEnd, NamesLookup& namesLookup, NamesId& namesId)
{
    CreateData scan(parent, bufEnd, namesLookup, namesId);
    scan.set_value(ScanDef::Motor1, (float)41.0);
    scan.set_value(ScanDef::Motor2, (double)42.0);
}

class HsdCfgDef:public VarDef
{
public:
  enum index
    {
      enable,
      raw_prescale
    };

  HsdCfgDef()
   {
       NameVec.push_back({"enable",Name::UINT64,1});
       NameVec.push_back({"raw_prescale",Name::UINT64,1});
   }
} HsdCfgDef;

class HsdRunDef:public VarDef
{
public:
  enum index
    {
        raw_prescale
    };

  HsdRunDef()
   {
       NameVec.push_back({"raw_prescale",Name::UINT64,1});
   }
} HsdRunDef;

class DebugIter : public XtcIterator
{
public:
    enum { Stop, Continue };
    DebugIter(Xtc* xtc, const void* bufEnd, NamesLookup& namesLookup) : XtcIterator(xtc, bufEnd), _namesLookup(namesLookup)
    {
    }

    void get_value(int i, Name& name, DescData& descdata){
        int data_rank = name.rank();

        switch(name.type()){
        case(Name::UINT8):{
            if(data_rank > 0){
                Array<uint8_t> arrT = descdata.get_array<uint8_t>(i);
                // printf("%s: %d, %d, %d\n",name.name(),arrT(0),arrT(1), arrT(2));
                    }
            else{
                // printf("%s: %d\n",name.name(),descdata.get_value<uint8_t>(i));
            }
            break;
        }

        case(Name::UINT16):{
            if(data_rank > 0){
                Array<uint16_t> arrT = descdata.get_array<uint16_t>(i);
                // printf("%s: %d, %d, %d\n",name.name(),arrT(0),arrT(1), arrT(2));
                    }
            else{
                // printf("%s: %d\n",name.name(),descdata.get_value<uint16_t>(i));
            }
            break;
        }

        case(Name::UINT32):{
            if(data_rank > 0){
                Array<uint32_t> arrT = descdata.get_array<uint32_t>(i);
                // printf("%s: %d, %d, %d\n",name.name(),arrT(0),arrT(1), arrT(2));
                    }
            else{
                // printf("%s: %d\n",name.name(),descdata.get_value<uint32_t>(i));
            }
            break;
        }

        case(Name::UINT64):{
            if(data_rank > 0){
                Array<uint64_t> arrT = descdata.get_array<uint64_t>(i);
                // printf("%s: %d, %d, %d\n",name.name(),arrT(0),arrT(1), arrT(2));
                    }
            else{
                // printf("%s: %d\n",name.name(),descdata.get_value<uint64_t>(i));
            }
            break;
        }

        case(Name::INT8):{
            if(data_rank > 0){
                Array<int8_t> arrT = descdata.get_array<int8_t>(i);
                // printf("%s: %d, %d, %d\n",name.name(),arrT(0),arrT(1), arrT(2));
                    }
            else{
                // printf("%s: %d\n",name.name(),descdata.get_value<int8_t>(i));
            }
            break;
        }

        case(Name::INT16):{
            if(data_rank > 0){
                Array<int16_t> arrT = descdata.get_array<int16_t>(i);
                // printf("%s: %d, %d, %d\n",name.name(),arrT(0),arrT(1), arrT(2));
                    }
            else{
                // printf("%s: %d\n",name.name(),descdata.get_value<int16_t>(i));
            }
            break;
        }

        case(Name::INT32):{
            if(data_rank > 0){
                Array<int32_t> arrT = descdata.get_array<int32_t>(i);
                // printf("%s: %d, %d, %d\n",name.name(),arrT(0),arrT(1), arrT(2));
                    }
            else{
                // printf("%s: %d\n",name.name(),descdata.get_value<int32_t>(i));
            }
            break;
        }

        case(Name::INT64):{
            if(data_rank > 0){
                Array<int64_t> arrT = descdata.get_array<int64_t>(i);
                // printf("%s: %d, %d, %d\n",name.name(),arrT(0),arrT(1), arrT(2));
                    }
            else{
                // printf("%s: %d\n",name.name(),descdata.get_value<int64_t>(i));
            }
            break;
        }

        case(Name::FLOAT):{
            if(data_rank > 0){
                Array<float> arrT = descdata.get_array<float>(i);
                // printf("%s: %f, %f\n",name.name(),arrT(0),arrT(1));
                    }
            else{
                // printf("%s: %f\n",name.name(),descdata.get_value<float>(i));
            }
            break;
        }

        case(Name::DOUBLE):{
            if(data_rank > 0){
                Array<double> arrT = descdata.get_array<double>(i);
                // printf("%s: %f, %f, %f\n",name.name(),arrT(0),arrT(1), arrT(2));
                    }
            else{
                // printf("%s: %f\n",name.name(),descdata.get_value<double>(i));
            }
            break;
        }

        case(Name::CHARSTR):{
            if(data_rank > 0){
                Array<char> arrT = descdata.get_array<char>(i);
                //printf("%s: \"%s\"\n",name.name(),arrT.data());
                    }
            else{
                printf("%s: string with no rank?!?\n",name.name());
            }
            break;
        }

        case(Name::ENUMVAL):{
            if(data_rank > 0){
                Array<int32_t> arrT = descdata.get_array<int32_t>(i);
                //printf("%s: %d, %d, %d\n",name.name(),arrT.data()[0],arrT.data()[1], arrT.data()[2]);
                    }
            else{
                //printf("%s: %d\n",name.name(),descdata.get_value<int32_t>(i));
            }
            break;
        }

        case(Name::ENUMDICT):{
            if(data_rank > 0){
                printf("%s: enumdict with rank?!?\n", name.name());
            } else{
                //printf("%s: %d\n",name.name(),descdata.get_value<int32_t>(i));
            }
            break;
        }

        }

    }

    int process(Xtc* xtc, const void* bufEnd)
    {
        // printf("found typeid %s\n",XtcData::TypeId::name(xtc->contains.id()));
        switch (xtc->contains.id()) {

        case (TypeId::Names): {
            // printf("Names pointer is %p\n", xtc);
            break;
        }
        case (TypeId::Parent): {
            iterate(xtc, bufEnd);
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
    uint16_t array[18];
};

void pgpExample(Xtc& parent, const void* bufEnd, NamesLookup& namesLookup, NamesId& namesId)
{
    DescribedData frontEnd(parent, bufEnd, namesLookup, namesId);

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

void fexExample(Xtc& parent, const void* bufEnd, NamesLookup& namesLookup, NamesId& namesId)
{
    CreateData fex(parent, bufEnd, namesLookup, namesId);
    fex.set_value(FexDef::floatFex, (double)41.0);

    unsigned shape[MaxRank] = {2,3};
    Array<float> arrayT = fex.allocate<float>(FexDef::arrayFex,shape);
    for(unsigned i=0; i<shape[0]; i++){
        for (unsigned j=0; j<shape[1]; j++) {
            arrayT(i,j) = 142.0+i*shape[1]+j;
        }
    };

    fex.set_value(FexDef::intFex, (int64_t) 42);

    fex.set_string(FexDef::charStrFex, "Test String");

    fex.set_value(FexDef::enumFex1, (int32_t) -2);
    fex.set_value(FexDef::enumFex1_HighGain, (int32_t) -2);
    fex.set_value(FexDef::enumFex1_LowGain, (int32_t) 5);
    fex.set_value(FexDef::enumFex2, (int32_t) 5);
    fex.set_value(FexDef::enumFex3, (int32_t) 12);
    fex.set_value(FexDef::enumFex3_On, (int32_t) -7);
    fex.set_value(FexDef::enumFex3_Off, (int32_t) 12);
}


void padExample(Xtc& parent, const void* bufEnd, NamesLookup& namesLookup, NamesId& namesId)
{
    DescribedData pad(parent, bufEnd, namesLookup, namesId);

    // simulates PGP data arriving, and shows the address that should be given to PGP driver
    // we should perhaps worry about DMA alignment issues if in the future
    // we avoid the pgp driver copy into user-space.
    new (pad.data()) PadData();

    // now that data has arrived update with the number of bytes received
    // it is required to call this before set_array_shape
    pad.set_data_length(sizeof(PadData));

    unsigned shape[MaxRank] = {3,6};
    pad.set_array_shape(PadDef::arrayRaw, shape);
}

void addNames(Xtc& xtc, const void* bufEnd, NamesLookup& namesLookup, unsigned& nodeId, unsigned segment) {
    Alg hsdRawAlg("raw",0,0,0);
    NamesId namesId0(nodeId,MyNamesId::HsdRaw+MyNamesId::NumberOf*(segment%nSegments));
    Names& frontEndNames = *new(xtc, bufEnd) Names(bufEnd, "xpphsd", hsdRawAlg, "hsd", "detnum1234", namesId0, segment);
    frontEndNames.add(xtc, bufEnd, PgpDef);
    namesLookup[namesId0] = NameIndex(frontEndNames);

    Alg hsdFexAlg("fex",4,5,6);
    NamesId namesId1(nodeId,MyNamesId::HsdFex+MyNamesId::NumberOf*(segment%nSegments));
    Names& fexNames = *new(xtc, bufEnd) Names(bufEnd, "xpphsd", hsdFexAlg, "hsd","detnum1234", namesId1, segment);
    fexNames.add(xtc, bufEnd, FexDef);
    namesLookup[namesId1] = NameIndex(fexNames);

    Alg cspadRawAlg("raw",2,3,42);
    NamesId namesId2(nodeId,MyNamesId::Cspad+MyNamesId::NumberOf*(segment%nSegments));
    Names& padNames = *new(xtc, bufEnd) Names(bufEnd, "xppcspad", cspadRawAlg, "cspad", "detnum1234", namesId2, segment);
    padNames.add(xtc, bufEnd, PadDef);
    namesLookup[namesId2] = NameIndex(padNames);
}

void addCfgNames(Xtc& xtc, const void* bufEnd, NamesLookup& namesLookup, unsigned& nodeId, unsigned segment) {
    // Configuration data
    Alg hsdCfgAlg("config",2,0,0);
    NamesId namesId0(nodeId,MyNamesId::HsdCfg+MyNamesId::NumberOf*(segment%nSegments));
    Names& cfgNames = *new(xtc, bufEnd) Names(bufEnd, "xpphsd", hsdCfgAlg, "hsd", "detnum1234", namesId0, segment);
    cfgNames.add(xtc, bufEnd, HsdCfgDef);
    namesLookup[namesId0] = NameIndex(cfgNames);

    // Configuration update on BeginRun data
    Alg hsdRunAlg("config",2,0,0);
    NamesId namesId1(nodeId,MyNamesId::HsdRun+MyNamesId::NumberOf*(segment%nSegments));
    Names& runNames = *new(xtc, bufEnd) Names(bufEnd, "xpphsd", hsdRunAlg, "hsd", "detnum1234", namesId1, segment);
    runNames.add(xtc, bufEnd, HsdRunDef);
    namesLookup[namesId1] = NameIndex(runNames);
}

void addCfgData(Xtc& xtc, const void* bufEnd, NamesLookup& namesLookup, unsigned nodeId, unsigned segment) {
    NamesId namesId(nodeId,MyNamesId::HsdCfg+MyNamesId::NumberOf*(segment%nSegments));
    CreateData cd(xtc, bufEnd, namesLookup, namesId);
    unsigned shape[MaxRank];
    shape[0] = 4;
    { Array<uint64_t> arrayT = cd.allocate<uint64_t>(HsdCfgDef::enable,shape);
        for(unsigned i=0; i<shape[0]; i++)
            arrayT(i) = i; }
    { Array<uint64_t> arrayT = cd.allocate<uint64_t>(HsdCfgDef::raw_prescale,shape);
        for(unsigned i=0; i<shape[0]; i++)
            arrayT(i) = i+100*segment; }
}

void addCfgRunData(Xtc& xtc, const void* bufEnd, NamesLookup& namesLookup, unsigned nodeId, unsigned segment) {
    NamesId namesId(nodeId,MyNamesId::HsdRun+MyNamesId::NumberOf*(segment%nSegments));
    CreateData cd(xtc, bufEnd, namesLookup, namesId);
    unsigned shape[MaxRank];
    shape[0] = 4;
    { Array<uint64_t> arrayT = cd.allocate<uint64_t>(HsdRunDef::raw_prescale,shape);
        for(unsigned i=0; i<shape[0]; i++)
            arrayT(i) = i+100*segment+1000; }
}

void addData(Xtc& xtc, const void* bufEnd, NamesLookup& namesLookup, unsigned nodeId, unsigned segment) {
    NamesId namesId0(nodeId,MyNamesId::HsdRaw+MyNamesId::NumberOf*(segment%nSegments));
    pgpExample(xtc, bufEnd, namesLookup, namesId0);
    NamesId namesId1(nodeId,MyNamesId::HsdFex+MyNamesId::NumberOf*(segment%nSegments));
    fexExample(xtc, bufEnd, namesLookup, namesId1);
    NamesId namesId2(nodeId,MyNamesId::Cspad+MyNamesId::NumberOf*(segment%nSegments));
    padExample(xtc, bufEnd, namesLookup, namesId2);
}

Dgram& createTransition(TransitionId::Value transId, bool counting_timestamps,
                        unsigned& timestamp_val, void** bufEnd) {
    TypeId tid(TypeId::Parent, 0);
    uint64_t pulseId = 0;
    uint32_t env = 0;
    struct timeval tv;
    void* buf = malloc(BUFSIZE);
    *bufEnd = ((char*)buf) + BUFSIZE;

    if (counting_timestamps) {
        tv.tv_sec = 1;
        tv.tv_usec = timestamp_val;
        timestamp_val++;
    } else {
        gettimeofday(&tv, NULL);
    }
    Transition tr(Dgram::Event, transId, TimeStamp(tv.tv_sec, tv.tv_usec), env);
    return *new(buf) Dgram(tr, Xtc(tid));
}

void save(Dgram& dg, FILE* xtcFile) {
    if (fwrite(&dg, sizeof(dg) + dg.xtc.sizeofPayload(), 1, xtcFile) != 1) {
        printf("Error writing to output xtc file.\n");
    }
}

void addEpicsNames(Xtc& xtc, const void* bufEnd, NamesLookup& namesLookup, unsigned& nodeId, unsigned segment) {
    Alg xppEpicsAlg("raw",2,0,0);
    NamesId namesId(nodeId,MyNamesId::Epics+MyNamesId::NumberOf*(segment%nSegments));
    Names& epicsNames = *new(xtc, bufEnd) Names(bufEnd, "epics", xppEpicsAlg, "epics","detnum1234", namesId, segment);
    epicsNames.add(xtc, bufEnd, EpicsDef);
    namesLookup[namesId] = NameIndex(epicsNames);
}

void addEpicsData(Xtc& xtc, const void* bufEnd, NamesLookup& namesLookup, unsigned nodeId, unsigned segment) {
    NamesId namesId(nodeId,MyNamesId::Epics+MyNamesId::NumberOf*(segment%nSegments));
    epicsExample(xtc, bufEnd, namesLookup, namesId);
}

void addEpicsInfo(Xtc& xtc, const void* bufEnd, NamesLookup& namesLookup, unsigned& nodeId, unsigned segment) {
    Alg xppEpicsInfoAlg("epicsinfo",1,0,0);
    NamesId namesId(nodeId,MyNamesId::EpicsInfo+MyNamesId::NumberOf*(segment%nSegments));
    Names& epicsInfoNames = *new(xtc, bufEnd) Names(bufEnd, "epicsinfo", xppEpicsInfoAlg, "epicsinfo","detnum1234", namesId, segment);
    VarDef epicsInfo;
    epicsInfo.NameVec.push_back({"keys",Name::CHARSTR,1});
    epicsInfo.NameVec.push_back({"HX2:DVD:GCC:01:PMON",Name::CHARSTR,1});
    epicsInfo.NameVec.push_back({"HX2:DVD:GPI:01:PMON",Name::CHARSTR,1});
    epicsInfoNames.add(xtc, bufEnd, epicsInfo);
    namesLookup[namesId] = NameIndex(epicsInfoNames);
    // add dictionary of information for each epics detname above.
    // first name is required to be "keys".  keys and values
    // are delimited by "\n".
    CreateData epicsinfo(xtc, bufEnd, namesLookup, namesId);
    epicsinfo.set_string(0, "epicsname"",""otherfieldname");
    epicsinfo.set_string(1, "HX2:DVD:GCC:01:PMON"",""hello1");
    epicsinfo.set_string(2, "HX2:DVD:GPI:01:PMON"",""hello2");
}

void addScanNames(Xtc& xtc, const void* bufEnd, NamesLookup& namesLookup, unsigned& nodeId, unsigned segment) {
    Alg scanAlg("raw",2,0,0);
    NamesId namesId(nodeId,MyNamesId::Scan+MyNamesId::NumberOf*(segment%nSegments));
    Names& scanNames = *new(xtc, bufEnd) Names(bufEnd, "scan", scanAlg, "scan", "detnum1234", namesId, segment);
    scanNames.add(xtc, bufEnd, ScanDef);
    namesLookup[namesId] = NameIndex(scanNames);
}

void addScanData(Xtc& xtc, const void* bufEnd, NamesLookup& namesLookup, unsigned nodeId, unsigned segment) {
    NamesId namesId(nodeId,MyNamesId::Scan+MyNamesId::NumberOf*(segment%nSegments));
    scanExample(xtc, bufEnd, namesLookup, namesId);
}

void addRunInfoNames(Xtc& xtc, const void* bufEnd, NamesLookup& namesLookup, unsigned& nodeId) {
    Alg runInfoAlg("runinfo",0,0,1);
    NamesId namesId(nodeId,MyNamesId::RunInfo);
    unsigned segment = 0;
    Names& runInfoNames = *new(xtc, bufEnd) Names(bufEnd, "runinfo", runInfoAlg, "runinfo", "", namesId, segment);
    runInfoNames.add(xtc, bufEnd, RunInfoDef);
    namesLookup[namesId] = NameIndex(runInfoNames);
}

void addRunInfoData(Xtc& xtc, const void* bufEnd, NamesLookup& namesLookup, unsigned nodeId) {
    NamesId namesId(nodeId,MyNamesId::RunInfo);
    CreateData runinfo(xtc, bufEnd, namesLookup, namesId);
    runinfo.set_string(RunInfoDef::EXPT, "xpptut15");
    runinfo.set_value(RunInfoDef::RUNNUM, (uint32_t)14);
}

void addChunkInfoNames(Xtc& xtc, const void* bufEnd, NamesLookup& namesLookup, unsigned& nodeId, unsigned segment) {
    Alg chunkInfoAlg("chunkinfo",0,0,1);
    NamesId namesId(nodeId,MyNamesId::ChunkInfo+MyNamesId::NumberOf*(segment%nSegments));
    Names& chunkInfoNames = *new(xtc, bufEnd) Names(bufEnd, "chunkinfo", chunkInfoAlg, "chunkinfo", "", namesId, segment);
    chunkInfoNames.add(xtc, bufEnd, ChunkInfoDef);
    namesLookup[namesId] = NameIndex(chunkInfoNames);
}

void addChunkInfoData(Xtc& xtc, const void* bufEnd, NamesLookup& namesLookup, unsigned nodeId, unsigned segment) {
    NamesId namesId(nodeId,MyNamesId::ChunkInfo+MyNamesId::NumberOf*(segment%nSegments));
    CreateData chunkinfo(xtc, bufEnd, namesLookup, namesId);
    chunkinfo.set_value(ChunkInfoDef::CHUNKID, (uint32_t)1);
    chunkinfo.set_string(ChunkInfoDef::FILENAME, "data-r0001-s000-c001.xtc2");
}


#define MAX_FNAME_LEN 256

int main(int argc, char* argv[])
{
    int c;
    int parseErr = 0;
    unsigned nevents = 2;
    unsigned nmotorsteps = 1;
    unsigned epicsPeriod = 4;
    char xtcname[MAX_FNAME_LEN];
    strncpy(xtcname, "data.xtc2", MAX_FNAME_LEN);
    unsigned starting_segment = 0;
    // this is used to create uniform timestamps across files
    // so we can do offline event-building.
    bool counting_timestamps = false;
    unsigned starting_nodeid = 1;
    bool adding_chunkinfo = false;
    auto usage = [](const char* progname) {
        fprintf(stderr, "Usage: %s [-f <filename> -n <numEvents> -t -h]\n", progname);
    };

    while ((c = getopt(argc, argv, "hf:n:s:e:m:ti:c")) != -1) {
        switch (c) {
            case 'h':
                usage(argv[0]);
                exit(0);
            case 'n':
                nevents = atoi(optarg);
                break;
            case 'i':
                starting_nodeid = atoi(optarg);
                break;
            case 'm':
                nmotorsteps = atoi(optarg);
                break;
            case 'e':
                epicsPeriod = atoi(optarg);
                break;
            case 's':
                starting_segment = atoi(optarg);
                break;
            case 'f':
                strncpy(xtcname, optarg, MAX_FNAME_LEN);
                break;
            case 't':
                counting_timestamps = true;
                break;
            case 'c':
                adding_chunkinfo = true;
                break;
            default:
                parseErr++;
        }
    }

    if (parseErr) {
        usage(argv[0]);
        return -1;
    }

    FILE* xtcFile = fopen(xtcname, "w");
    if (!xtcFile) {
        printf("Error opening output xtc file.\n");
        return -1;
    }

    struct timeval tv;
    TypeId tid(TypeId::Parent, 0);
    uint32_t env = 0;
    uint64_t pulseId = 0;
    unsigned timestamp_val = 0;
    void* bufEnd;

    Dgram& config = createTransition(TransitionId::Configure,
                                     counting_timestamps,
                                     timestamp_val,
                                     &bufEnd);

    unsigned nodeid1 = starting_nodeid;
    unsigned nodeid2 = starting_nodeid+1;
    NamesLookup namesLookup;
    unsigned iseg = 1; // add to 2nd segment for testing
    addRunInfoNames(config.xtc, bufEnd, namesLookup, nodeid1);
    // only add epics and scan info to the first stream
    if (starting_segment==0) {
        addEpicsNames(config.xtc, bufEnd, namesLookup, nodeid1, iseg);
        addEpicsInfo(config.xtc, bufEnd, namesLookup, nodeid1, iseg);
        addScanNames(config.xtc, bufEnd, namesLookup, nodeid1, iseg);
    }

    if (adding_chunkinfo) {
        addChunkInfoNames(config.xtc, bufEnd, namesLookup, nodeid1, starting_segment);
    }

    for (unsigned iseg=0; iseg<nSegments; iseg++) {
        addNames   (config.xtc, bufEnd, namesLookup, nodeid1, iseg+starting_segment);
        addData    (config.xtc, bufEnd, namesLookup, nodeid1, iseg+starting_segment);
        addCfgNames(config.xtc, bufEnd, namesLookup, nodeid1, iseg+starting_segment);
        addCfgData (config.xtc, bufEnd, namesLookup, nodeid1, iseg+starting_segment);
    }

    save(config,xtcFile);

    DebugIter iter(&config.xtc, bufEnd, namesLookup);
    iter.iterate();

    Dgram& beginRunTr = createTransition(TransitionId::BeginRun,
                                         counting_timestamps,
                                         timestamp_val,
                                         &bufEnd);
    addRunInfoData(beginRunTr.xtc, bufEnd, namesLookup, nodeid1);
    for (unsigned iseg=0; iseg<nSegments; iseg++) {
        addCfgRunData (beginRunTr.xtc, bufEnd, namesLookup, nodeid1, iseg+starting_segment);
    }
    save(beginRunTr, xtcFile);

    for (unsigned istep=0; istep<nmotorsteps; istep++) {

        Dgram& beginStepTr = createTransition(TransitionId::BeginStep,
                                              counting_timestamps,
                                              timestamp_val,
                                              &bufEnd);
        // cpo comments this out because it somehow causes
        // the BeginStep transition to end up in the epics store
        if (starting_segment==0) addScanData(beginStepTr.xtc, bufEnd, namesLookup, nodeid1, iseg);
        save(beginStepTr, xtcFile);

        Dgram& enableTr = createTransition(TransitionId::Enable,
                                           counting_timestamps,
                                           timestamp_val,
                                           &bufEnd);
        save(enableTr, xtcFile);

        void* buf = malloc(BUFSIZE);
        bufEnd = ((char*)buf) + BUFSIZE;
        for (unsigned ievt=0; ievt<nevents; ievt++) {
            if (epicsPeriod>0) {
                if (ievt>0 and ievt%epicsPeriod==0) {

                    if (adding_chunkinfo) {
                        // make an Enable prior to SlowUpdate if chunkinfo is needed
                        if (counting_timestamps) {
                            tv.tv_sec = 1;
                            tv.tv_usec = timestamp_val;
                            timestamp_val++;
                        } else {
                            gettimeofday(&tv, NULL);
                        }
                        Transition chunkEnableTr(Dgram::Event, TransitionId::Enable, TimeStamp(tv.tv_sec, tv.tv_usec), env);
                        Dgram& chunkEnableDgram = *new(buf) Dgram(chunkEnableTr, Xtc(tid));

                        addChunkInfoData(chunkEnableDgram.xtc, bufEnd, namesLookup, nodeid1, starting_segment);

                        save(chunkEnableDgram,xtcFile);
                    }

                    // make a SlowUpdate with epics data
                    if (counting_timestamps) {
                        tv.tv_sec = 1;
                        tv.tv_usec = timestamp_val;
                        timestamp_val++;
                    } else {
                        gettimeofday(&tv, NULL);
                    }
                    Transition tr(Dgram::Event, TransitionId::SlowUpdate, TimeStamp(tv.tv_sec, tv.tv_usec), env);
                    Dgram& dgram = *new(buf) Dgram(tr, Xtc(tid));

                    unsigned iseg = 1; // add epics to second segment
                    // only add epics to the first stream
                    if (starting_segment==0) {
                        addEpicsData(dgram.xtc, bufEnd, namesLookup, nodeid1, iseg);
                    }

                    save(dgram,xtcFile);
                }
            }

            // generate a normal L1
            if (counting_timestamps) {
                tv.tv_sec = 1;
                tv.tv_usec = timestamp_val;
                timestamp_val++;
            } else {
                gettimeofday(&tv, NULL);
            }
            Transition tr(Dgram::Event, TransitionId::L1Accept, TimeStamp(tv.tv_sec, tv.tv_usec), env);
            Dgram& dgram = *new(buf) Dgram(tr, Xtc(tid));

            for (unsigned iseg=0; iseg<nSegments; iseg++) {
                addData(dgram.xtc, bufEnd, namesLookup, nodeid1, iseg+starting_segment);
            }
            DebugIter iter(&dgram.xtc, bufEnd, namesLookup);
            iter.iterate();
            save(dgram,xtcFile);

        } // events

        Dgram& disableTr = createTransition(TransitionId::Disable,
                                            counting_timestamps,
                                            timestamp_val,
                                            &bufEnd);
        save(disableTr, xtcFile);

        Dgram& endStepTr = createTransition(TransitionId::EndStep,
                                            counting_timestamps,
                                            timestamp_val,
                                            &bufEnd);
        save(endStepTr, xtcFile);
    } // steps

    // make an EndRun
    Dgram& endRunTr = createTransition(TransitionId::EndRun,
                                       counting_timestamps,
                                       timestamp_val,
                                       &bufEnd);
    save(endRunTr, xtcFile);

    fclose(xtcFile);

    return 0;
}
