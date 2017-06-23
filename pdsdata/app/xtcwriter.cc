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

            std::cout << d.get_value<float>("myfloat") << std::endl;

            auto array = d.get_value<Array<float>>("array");
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    std::cout << array(i, j) << "  ";
                }
                std::cout << std::endl;
            }

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

int main()
{
    // this is the datagram, which gives you an "xtc" for free
    Dgram& dgram = *(Dgram*)malloc(BUFSIZE);
    TypeId tid(TypeId::Parent, 0);
    dgram.xtc.contains = tid;
    dgram.xtc.damage = 0;
    dgram.xtc.extent = sizeof(Xtc);

    // make a child xtc with detector data and descriptor
    TypeId tid_child(TypeId::Data, 0);
    Xtc& xtcChild = *new (&dgram.xtc) Xtc(tid_child);

    // creation of fixed-length data in xtc
    MyData& d = *new (xtcChild.alloc(sizeof(MyData))) MyData(1, 2, 3);

    // creation of variable-length data in xtc
    DescriptorManager descMgr(xtcChild.next());
    descMgr.add("myfloat", FLOAT);

    int shape[] = {3, 3};
    descMgr.add("array", FLOAT, 2, shape);

    descMgr.add("myint", INT32);

    xtcChild.alloc(descMgr.size());

    // update parent xtc with our new size.
    dgram.xtc.alloc(xtcChild.sizeofPayload());

    HDF5File file("test.h5");
    file.addDatasets(descMgr._desc);
    file.appendData(d);
    
    myLevelIter iter(&dgram.xtc, 0);
    iter.iterate();

    free((void*)&dgram);

    return 0;
}
