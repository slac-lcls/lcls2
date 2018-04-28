// to do:
// - * add set_array()
// - * figure out how to associate nameindex with correct xtc's
// - X put names in real configure transition
// - create new autoalloc that also allocs xtc header size
// - X faster version of routines that takes index vs. string
// - X* better namespacing (ShapesData, DescData)
// - X minimize number of name lookups
// - X provide both "safe" name-lookup xface and "performant" index xface
// - X automated testing in cmake (and jenkins/continuous integration?)
// - Array should support slicing
// - maybe put non-vlen array shapes in configure (use this to distinguish
//   vlen/non-vlen?).  but this may make things too complex. could use name instead.
// - * fix wasted space in CreateData ctor. CPO to think about the DescribedData ctor
// - protection:
//   o pass in full list of names (ensures we get early error if the
//     nameindex number is incorrect, but breaks object-oriented encapsulation)
//   o X error when name not in map
//   o X make sure things go in name order (not a problem if we do string lookup)
//     (are offsets messed up if user leaves a "gap" with set_array/set_array_shape?)
//   o X* check maxrank limit
//   o X* check maxnamesize limit

#include "xtcdata/xtc/ShapesData.hh"
#include "xtcdata/xtc/DescData.hh"
#include "xtcdata/xtc/Dgram.hh"
#include "xtcdata/xtc/TypeId.hh"
#include "xtcdata/xtc/XtcIterator.hh"
#include "xtcdata/xtc/VarDef.hh"


#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <unistd.h>
#include <type_traits>
#include <cstring>

using namespace XtcData;
#define BUFSIZE 0x4000000
#define NEVENT 2


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
    DebugIter(Xtc* xtc, std::vector<NameIndex>& namesVec) : XtcIterator(xtc), _namesVec(namesVec)
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
            unsigned namesId = shapesdata.shapes().namesId();
            DescData descdata(shapesdata, _namesVec[namesId]);
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
    std::vector<NameIndex>& _namesVec;
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

void pgpExample(Xtc& parent, std::vector<NameIndex>& NamesVec, unsigned nameId)
{
    DescribedData frontEnd(parent, NamesVec, nameId);

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

void fexExample(Xtc& parent, std::vector<NameIndex>& NamesVec, unsigned nameId)
{ 
    CreateData fex(parent, NamesVec, nameId);
    fex.set_value(FexDef::floatFex, (double)41.0);

    unsigned shape[Name::MaxRank] = {2,3};
    Array<float> arrayT = fex.allocate<float>(FexDef::arrayFex,shape);
    for(unsigned i=0; i<shape[0]; i++){
        for (unsigned j=0; j<shape[1]; j++) {
            arrayT(i,j) = 142.0+i*shape[1]+j;
        }
    };
    
    fex.set_value(FexDef::intFex, (int64_t) 42);
}
   

void padExample(Xtc& parent, std::vector<NameIndex>& NamesVec, unsigned nameId)
{ 
    DescribedData pad(parent, NamesVec, nameId);

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

void add_names(Xtc& parent, std::vector<NameIndex>& namesVec) {
    Alg hsdRawAlg("raw",0,0,0);
    Names& frontEndNames = *new(parent) Names("xpphsd", hsdRawAlg, "hsd", "detnum1234");
    frontEndNames.add(parent,PgpDef);
    namesVec.push_back(NameIndex(frontEndNames));

    Alg hsdFexAlg("fex",4,5,6);
    Names& fexNames = *new(parent) Names("xpphsd", hsdFexAlg, "hsd","detnum1234");
    fexNames.add(parent, FexDef);
    namesVec.push_back(NameIndex(fexNames));

    unsigned segment = 0;
    Alg cspadRawAlg("raw",2,3,42);
    Names& padNames = *new(parent) Names("xppcspad", cspadRawAlg, "cspad", "detnum1234", segment);
    Alg segmentAlg("cspadseg",2,3,42);
    padNames.add(parent, PadDef);
    namesVec.push_back(NameIndex(padNames));
}

void addData(Xtc& xtc, std::vector<NameIndex>& namesVec) {
    // need to protect against putting in the wrong nameId here
    unsigned nameId = 0;
    pgpExample(xtc, namesVec, nameId);
    nameId++;
    fexExample(xtc, namesVec, nameId);
    nameId++;
    padExample(xtc, namesVec, nameId);
}

int main()
{
    
    FILE* xtcFile = fopen("data.xtc", "w");
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
    std::vector<NameIndex> namesVec;
    add_names(config.xtc, namesVec);
    addData(config.xtc,namesVec);

    DebugIter iter(&config.xtc, namesVec);
    iter.iterate();

    if (fwrite(&config, sizeof(config) + config.xtc.sizeofPayload(), 1, xtcFile) != 1) {
        printf("Error writing configure to output xtc file.\n");
        return -1;
    }

     // for(auto const& value: namesVec[0].nameMap()){
     //     std::cout<<value.first<<std::endl;
     // }


    void* buf = malloc(BUFSIZE);
 
    for (int i = 0; i < NEVENT; i++) {
        Dgram& dgram = *(Dgram*)buf;
        TypeId tid(TypeId::Parent, 0);
        dgram.xtc.contains = tid;
        dgram.xtc.damage = 0;
        dgram.xtc.extent = sizeof(Xtc);

        addData(dgram.xtc,namesVec);

        printf("*** event %d ***\n",i);
        DebugIter iter(&dgram.xtc, namesVec);
        iter.iterate();

        if (fwrite(&dgram, sizeof(dgram) + dgram.xtc.sizeofPayload(), 1, xtcFile) != 1) {
            printf("Error writing to output xtc file.\n");
            return -1;
        }
     }
    fclose(xtcFile);

    return 0;
}
