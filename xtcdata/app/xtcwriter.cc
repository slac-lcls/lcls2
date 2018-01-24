
// to do:
// - add set_array()
// - figure out how to associate nameindex with correct xtc's
// - put names in real configure transition
// - create new autoalloc that also allocs xtc header size
// - faster version of routines that takes index vs. string
// - better namespacing
// - minimize number of name lookups
// - provide both "safe" name-lookup xface and "performant" index xface
// - automated testing in cmake (and jenkins/continuous integration?)
// - use Array for both getting/setting data
// - Array should support slicing
// - maybe put non-vlen array shapes in configure (use this to distinguish
//   vlen/non-vlen?).  but this may make things too complex. could use name instead.
// - fix wasted space in CreateData ctor
// - protection:
//   o pass in full list of names (ensures we get early error if the
//     nameindex number is incorrect, but breaks object-oriented encapsulation)
//   o error when name not in map
//   o make sure things go in name order (not a problem if we do string lookup)
//     (are offsets messed up if user leaves a "gap" with set_array/set_array_shape?)
//   o check maxrank limit
//   o check maxnames limit
//   o check total offsets equal to xtc sizeofPayload (not necessary with autoalloc?)

#include "xtcdata/xtc/ShapesData.hh"
#include "xtcdata/xtc/DescData.hh"
#include "xtcdata/xtc/Dgram.hh"
#include "xtcdata/xtc/TypeId.hh"
#include "xtcdata/xtc/XtcIterator.hh"

#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <unistd.h>
#include <type_traits>
#include <cstring>

using namespace XtcData;
#define BUFSIZE 0x4000000
#define NEVENT 1

class DebugIter : public XtcIterator
{
public:
    enum { Stop, Continue };
    DebugIter(Xtc* xtc, std::vector<NameIndex>& namesVec) : XtcIterator(xtc), _namesVec(namesVec)
    {
    }

    int process(Xtc* xtc)
    {
        // printf("found typeid %s\n",XtcData::TypeId::name(xtc->contains.id()));
        switch (xtc->contains.id()) {
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
            printf("Found %d names\n",names.num());
            for (unsigned i = 0; i < names.num(); i++) {
                Name& name = names.get(i);
                if (strncmp(name.name(),"int",3)==0) {
                    printf("integ value %s: %d\n",name.name(),descdata.get_value<int32_t>(name.name()));
                }
                if (strncmp(name.name(),"float",5)==0) {
                    printf("float value %s: %f\n",name.name(),descdata.get_value<float>(name.name()));
                }
                if (strncmp(name.name(),"array",5)==0) {
                    float* arr = (float *)descdata.address(i);
                    printf("array value %s: %f %f\n",name.name(),arr[0],arr[1]);
                }
                //unsigned index = descdata.nameindex()[name.name()];
                //void* addr = descdata.address(index);
                //printOffset(name.name(),xtc,addr);
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

    float _fdata;
    float array[3][3];
    int _idata;
    float array2[3][3];
};

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

void pgpExample(Xtc& parent, NameIndex& nameindex, unsigned nameId)
{
    DescribedData frontEnd(parent, nameindex, nameId);

    // simulates PGP data arriving, and shows the address that should be given to PGP driver
    // we should perhaps worry about DMA alignment issues if in the future
    // we avoid the pgp driver copy into user-space.
    new (frontEnd.data()) PgpData(1, 2);

    // now that data has arrived update with the number of bytes received
    // it is required to call this before set_array_shape
    frontEnd.set_data_length(sizeof(PgpData));

    unsigned shape[] = { 3, 3 };
    frontEnd.set_array_shape("array0Pgp", shape);
    frontEnd.set_array_shape("array1Pgp", shape);
}

void fexExample(Xtc& parent, NameIndex& nameindex, unsigned nameId)
{
    CreateData fex(parent, nameindex, nameId);

    // have to be careful to set the correct type here, unfortunately.
    // because of variable length issues, user is currently required to
    // fill the fields in the order that they exist in the Names.
    // if that is not done an error will be asserted.  I think
    // we could remove that requirement if we indicate which
    // array is being written in the data as they are written.

    // Weirdness. When the map lookup fails, it returns 0. Thus, it is protected from
    // failing on the first element because this would return 0 anyway.

    fex.set_value("floatFex000", (float)99.0);
    float* ptr = (float*)fex.get_ptr();
    unsigned shape[Name::MaxRank];
    shape[0]=2;
    shape[1]=3;
    for (unsigned i=0; i<shape[0]*shape[1]; i++) ptr[i] = 142.0+i;
    fex.set_array_shape("arrayFex",shape);
    fex.set_value("intFex", 42);
    //    fex.set_value("floatFex", (float)99.0);
}

void padExample(Xtc& parent, NameIndex& nameindex, unsigned nameId)
{
    DescribedData pad(parent, nameindex, nameId);

    // simulates PGP data arriving, and shows the address that should be given to PGP driver
    // we should perhaps worry about DMA alignment issues if in the future
    // we avoid the pgp driver copy into user-space.
    new (pad.data()) PadData();

    // now that data has arrived update with the number of bytes received
    // it is required to call this before set_array_shape
    pad.set_data_length(sizeof(PadData));

    unsigned shape[] = { 18 };
    pad.set_array_shape("arrayRaw", shape);
}

void add_names(Xtc& parent, std::vector<NameIndex>& namesVec) {
    Alg alg0("hsd",1,2,3);
    Names& frontEndNames = *new(parent) Names("hsd1", "raw");
    frontEndNames.add(parent, "floatPgp",  Name::FLOAT);
    frontEndNames.add(parent, "array0Pgp", Name::FLOAT, 2);
    frontEndNames.add(parent, "intPgp",    Name::INT32);
    frontEndNames.add(parent, "array1Pgp", Name::FLOAT, 2);
    namesVec.push_back(NameIndex(frontEndNames));

    Alg alg1("abc",4,5,6);
    Names& fexNames = *new(parent) Names("hsd1", "fex");
    fexNames.add(parent, "floatFex", Name::FLOAT);
    fexNames.add(parent, "arrayFex", Name::FLOAT, 2);
    fexNames.add(parent, "intFex",   Name::INT32);
    namesVec.push_back(NameIndex(fexNames));

    Alg alg2("cspad",2,3,42);
    Names& padNames = *new(parent) Names("cspad0", "raw");
    padNames.add(parent, "arrayRaw", alg2); //, Name::FLOAT, parent, 3);
    namesVec.push_back(NameIndex(padNames));
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
    if (fwrite(&config, sizeof(config) + config.xtc.sizeofPayload(), 1, xtcFile) != 1) {
        printf("Error writing configure to output xtc file.\n");
        return -1;
    }

    void* buf = malloc(BUFSIZE);

    for (int i = 0; i < NEVENT; i++) {
        Dgram& dgram = *(Dgram*)buf;
        TypeId tid(TypeId::Parent, 0);
        dgram.xtc.contains = tid;
        dgram.xtc.damage = 0;
        dgram.xtc.extent = sizeof(Xtc);

        // need to protect against putting in the wrong nameId here
        unsigned nameId = 0;
	//	pgpExample(dgram.xtc, namesVec[nameId], nameId);
	nameId++;
        fexExample(dgram.xtc, namesVec[nameId], nameId);
	//   nameId++;
	//	padExample(dgram.xtc, namesVec[nameId], nameId);

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
