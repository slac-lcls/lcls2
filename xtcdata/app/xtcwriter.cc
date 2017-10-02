#include "xtcdata/xtc/Descriptor.hh"
#include "xtcdata/xtc/Dgram.hh"
#include "xtcdata/xtc/TypeId.hh"
#include "xtcdata/xtc/XtcIterator.hh"

#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <unistd.h>

using namespace XtcData;
#define BUFSIZE 0x4000000
#define NDGRAM 1

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

void pgpExample(Xtc* parent, char* intName, char* floatName, char* arrayName, int vals[3])
{
    DescData& descdata = *new (parent)    DescData();
    Data& data         = *new (&descdata) Data();

    // simulates PGP data arriving, and shows the address that should be given to PGP driver
    new (data.payload()) PgpData(vals[0], vals[1], vals[2]);

    // now that data has arrived manually update with the number of bytes received
    data.alloc(sizeof(PgpData), descdata);

    // now add the descriptor
    Desc& desc = *new (&descdata) Desc();

    uint32_t offset;
    desc.add(floatName, FLOAT, offset, descdata);

    int shape[] = { 3, 3 };
    desc.add(arrayName, FLOAT, 2, shape, offset, descdata);
    desc.add(intName, INT32, offset, descdata);

    // update parent xtc with our new size.
    parent->alloc(descdata.sizeofPayload());
}

void fexExample(Xtc* parent)
{
    DescData& descdata = *new (parent) DescData();
    Desc& desc = *new (&descdata) Desc();

    uint32_t offset;
    desc.add("fexfloat1", FLOAT, offset, descdata);
    desc.add("fexint1", INT32, offset, descdata);

    Data& data = *new (&descdata) Data();
    // have to be careful to set the correct type here, unfortunately
    descdata.set_value("fexfloat1", (float)41.0);
    descdata.set_value("fexint1", 42);

    // update parent xtc with our new size.
    parent->alloc(descdata.sizeofPayload());
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
        // this Parent xtc allows for attaching multiple DescData xtc's.
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
        pgpExample(&dgram.xtc, intname, floatname, arrayname, vals1);
        sprintf(intname, "%s%d", "int", i + 1);
        sprintf(floatname, "%s%d", "float", i + 1);
        sprintf(arrayname, "%s%d", "array", i + 1);
        pgpExample(&dgram.xtc, intname, floatname, arrayname, vals2);
        fexExample(&dgram.xtc);

        if (fwrite(&dgram, sizeof(dgram) + dgram.xtc.sizeofPayload(), 1, xtcFile) != 1) {
            printf("Error writing to output xtc file.\n");
            return -1;
        }
    }
    fclose(xtcFile);

    return 0;
}
