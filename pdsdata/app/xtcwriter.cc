#include "pdsdata/xtc/Descriptor.hh"
#include "pdsdata/xtc/Dgram.hh"
#include "pdsdata/xtc/TypeId.hh"
#include "pdsdata/xtc/XtcIterator.hh"

#include "pdsdata/xtc/Hdf5Writer.hh"
#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <unistd.h>

using namespace Pds;
#define BUFSIZE 0x4000000

// this represent the "analysis" code
class myLevelIter : public XtcIterator
{
    public:
    enum { Stop, Continue };
    myLevelIter(Xtc* xtc, unsigned depth) : XtcIterator(xtc), _depth(depth)
    {

    }

    int process(Xtc* xtc)
    {
        unsigned i = _depth;
        switch (xtc->contains.id()) {
        case (TypeId::Parent): {
            myLevelIter iter(xtc, _depth + 1);
            iter.iterate();
            break;
        }
        case (TypeId::Data): {
            Data& d = *(Data*)xtc->payload();
            Descriptor& desc = d.desc();
            printf("Found fields named:\n");
            for (int i = 0; i < desc.num_fields; i++) {
                printf("%s  offset: %d\n", desc.get(i).name, desc.get(i).offset);
            }

            //std::cout << d.get_value<float>("myfloat") << std::endl;

            //auto array = d.get_value<Array<float>>("array");
            //for (int i = 0; i < 3; i++) {
            //    for (int j = 0; j < 3; j++) {
            //        std::cout << array(i, j) << "  ";
            //    }
            //    std::cout << std::endl;
            //}

            break;
        }
        default:
            printf("TypeId %s (value = %d)\n", TypeId::name(xtc->contains.id()), (int)xtc->contains.id());
            break;
        }
        return Continue;
    }

    private:
    unsigned _depth;
};

// everything below here is inletWire code

class MyData : public Data
{
    public:
    void* operator new(size_t size, void* p)
    {
        return p;
    }
    MyData(float f, int i1, int i2) : Data(sizeof(*this))
    {
        _fdata = f;
        _idata = i1;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                array[i][j] = i * j;
            }
        }
    }

    private:
    float _fdata;
    float array[3][3];
    int _idata;
};

void createXtcChild(Xtc* parent, char* intName, char* floatName, char* arrayName, int vals[3])
{
   // make a child xtc with detector data and descriptor
    TypeId tid_child(TypeId::Data, 0);
    Xtc& xtcChild = *new (parent->next()) Xtc(tid_child);

    // creation of fixed-length data in xtc
    MyData& d = *new (xtcChild.alloc(sizeof(MyData))) MyData(vals[0],vals[1],vals[2]);

    // creation of variable-length data in xtc
    DescriptorManager descMgr(xtcChild.next());
    descMgr.add(floatName, FLOAT);

    int shape[] = {3, 3};
    descMgr.add(arrayName, FLOAT, 2, shape);

    descMgr.add(intName, INT32);

    xtcChild.alloc(descMgr.size());

    // update parent xtc with our new size.
    parent->alloc(xtcChild.extent);

}
 

int main()
{
    // this is the datagram, which gives you an "xtc" for free
    Dgram& dgram1 = *(Dgram*)malloc(BUFSIZE);
    TypeId tid(TypeId::Parent, 0);
    dgram1.xtc.contains = tid;
    dgram1.xtc.damage = 0;
    dgram1.xtc.extent = sizeof(Xtc);

    int vals1[] = {4,5,6};
    int vals2[] = {7,8,9};
    int vals3[] = {10,11,12};
    int vals4[] = {13,14,15};

    createXtcChild(&dgram1.xtc, "int1", "float1", "array1", vals1);
    createXtcChild(&dgram1.xtc, "int2", "float2", "array2", vals2); 

    Dgram& dgram2 = *(Dgram*)malloc(BUFSIZE);
    dgram2.xtc.contains = tid;
    dgram2.xtc.damage = 0;
    dgram2.xtc.extent = sizeof(Xtc);

    createXtcChild(&dgram2.xtc, "int3", "float3", "array3", vals3);
    createXtcChild(&dgram2.xtc, "int4", "float4", "array4", vals4); 



    FILE* xtcFile = fopen("data.xtc","w");

    if(!xtcFile){
      printf("Error opening output xtc file.\n");
      return -1;
    }

    if(fwrite(&dgram1, sizeof(dgram1)+dgram1.xtc.sizeofPayload(), 1, xtcFile)!=1){
        printf("Error writing to output xtc file.\n");
        return -1;
    }
    if(fwrite(&dgram2, sizeof(dgram2)+dgram2.xtc.sizeofPayload(), 1, xtcFile)!=1){
        printf("Error writing to output xtc file.\n");
        return -1;
    }
    fclose(xtcFile);

    //HDF5File file("test.h5");
    //file.addDatasets(descMgr._desc);
    //file.appendData(d);
    
    myLevelIter iter1(&dgram1.xtc, 0);
    iter1.iterate();

    printf("dgram1 pointer: %p\n\ndgram1.xtc: %p\n\n",&dgram1,&dgram1.xtc);

    printf("Did first iteration. Here is second: \n\n");

    myLevelIter iter2(&dgram2.xtc, 0);
    iter2.iterate();

    printf("dgram2 pointer: %p\n\ndgram2.xtc: %p\n\n",&dgram2,&dgram2.xtc);

    free((void*)&dgram1);
    free((void*)&dgram2);

    return 0;
}
