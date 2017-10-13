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
#define NDGRAM 1

class DebugIter : public XtcIterator
{
public:
    enum { Stop, Continue };
    DebugIter(Xtc* xtc) : XtcIterator(xtc)
    {
    }

    int process(Xtc* xtc)
    {
        switch (xtc->contains.id()) {
        case (TypeId::Parent):
        case (TypeId::ShapesData): {
            ShapesData& sd = *(ShapesData*)xtc;
            NameIndex nameindex(*_names);
            DescData descdata(sd, nameindex);
            Names& names = descdata.nameindex().names();
            for (unsigned i = 0; i < names.num(); i++) {
                Name& name = names.get(i);
                if (strncmp(name.name(),"int",3)==0) {
                    descdata.get_value<int32_t>(name.name());
                }
                if (strncmp(name.name(),"float",5)==0) {
                    descdata.get_value<float>(name.name());
                }
                if (strncmp(name.name(),"array",5)==0) {
                    (float *)descdata.address(i);
                }
                //unsigned index = descdata.nameindex()[name.name()];
                //void* addr = descdata.address(index);
                //printOffset(name.name(),xtc,addr);
            }
            iterate(xtc);
            break;
        }
        case (TypeId::Names): {
            _names = (Names*)xtc;
            break;
        }
        default:
            break;
        }
        return Continue;
    }
    Names* _names;
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
    PgpData(float f, int i1, int i2)
    {
        _fdata = f;
        _idata = i1;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                array[i][j] = i * j;
            }
        }
    }

    float _fdata;
    float array[3][3];
    int _idata;
};

void pgpExample(Xtc& parent, char* intName, char* floatName, char* arrayName, int vals[3],
                NameIndex& nameindex)
{
    FrontEndData frontEnd(parent, nameindex);

    // simulates PGP data arriving, and shows the address that should be given to PGP driver
    // we should perhaps worry about DMA alignment issues if in the future
    // we avoid the pgp driver copy into user-space.
    new (frontEnd.data()) PgpData(vals[0], vals[1], vals[2]);

    // now that data has arrived update with the number of bytes received
    // it is required to call this before set_array_shape
    frontEnd.set_data_length(sizeof(PgpData));

    unsigned shape[] = { 3, 3 };
    frontEnd.set_array_shape(arrayName, shape);
}

void fexExample(Xtc& parent)
{
    // would normally get Names from configure transition
    Names& names = *new(parent) Names();
    names.add("fexfloat1", Name::FLOAT, parent);
    names.add("fexint1", Name::INT32, parent);
    NameIndex nameindex(names);

    FexData fex(parent, nameindex);

    // have to be careful to set the correct type here, unfortunately.
    // because of variable length issues, user is currently required to
    // fill the fields in the order that they exist in the Names.
    // if that is not done an error will be asserted.  I think
    // we could remove that requirement if we indicate which
    // array is being written in the data as they are written.
    fex.set_value("fexfloat1", (float)41.0);
    fex.set_value("fexint1", 42);
}

NameIndex addNameIndex(Xtc& parent, const char* intName, const char* floatName,
                   const char* arrayName)
{
    // would normally get Names from configure transition
    Names& names = *new(parent) Names();
    // needs to match the order of the data 
    names.add(floatName, Name::FLOAT, parent);
    names.add(arrayName, Name::FLOAT, parent, 2);
    names.add(intName, Name::INT32, parent);

    return NameIndex(names);
}

int main()
{
    void* buf = malloc(BUFSIZE);
    int vals1[3];
    int vals2[3];
    FILE* xtcFile = fopen("data.xtc", "w");
    if (!xtcFile) {
        printf("Error opening output xtc file.\n");
        return -1;
    }

    for (int i = 0; i < NDGRAM; i++) {
        Dgram& dgram = *(Dgram*)buf;
        TypeId tid(TypeId::Parent, 0);
        dgram.xtc.contains = tid;
        dgram.xtc.damage = 0;
        dgram.xtc.extent = sizeof(Xtc);

        for (int j = 0; j < 3; j++) {
            vals1[j] = i + j;
            vals2[j] = 1000 + i + j;
        }
        char intname[10], floatname[10], arrayname[10];
        sprintf(intname, "%s%d", "int", i);
        sprintf(floatname, "%s%d", "float", i);
        sprintf(arrayname, "%s%d", "array", i);
        NameIndex nameindex1 = addNameIndex(dgram.xtc, intname, floatname, arrayname);
        pgpExample(dgram.xtc, intname, floatname, arrayname, vals1, nameindex1);
        sprintf(intname, "%s%d", "int", i + 1);
        sprintf(floatname, "%s%d", "float", i + 1);
        sprintf(arrayname, "%s%d", "array", i + 1);
        NameIndex nameindex2 = addNameIndex(dgram.xtc, intname, floatname, arrayname);
        pgpExample(dgram.xtc, intname, floatname, arrayname, vals2, nameindex2);
        fexExample(dgram.xtc);

        DebugIter iter(&dgram.xtc);
        iter.iterate();

        if (fwrite(&dgram, sizeof(dgram) + dgram.xtc.sizeofPayload(), 1, xtcFile) != 1) {
            printf("Error writing to output xtc file.\n");
            return -1;
        }
    }
    fclose(xtcFile);

    return 0;
}
