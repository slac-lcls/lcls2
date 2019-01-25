#ifndef BLOCKDGRAM__H
#define BLOCKDGRAM__H

// class used by cython to create dgrams with "blocks" of memory,
// the three block types being Shapes, Data, and ShapesData

#include "xtcdata/xtc/ShapesData.hh"
#include "xtcdata/xtc/NamesId.hh"

namespace XtcData {

class BlockDgram : public Xtc
{
public:
    BlockDgram(uint8_t* buffdgram, uint64_t tstamp, uint64_t pulseId,
               unsigned control):
        _dgram(*new(buffdgram) Dgram())
    {
        _dgram.seq = Sequence(TimeStamp(tstamp),PulseId(pulseId,control));
        TypeId tid(TypeId::Parent, 0);
        _dgram.xtc.contains = tid;
        _dgram.xtc.damage = 0;
        _dgram.xtc.extent = sizeof(Xtc);

        _sizeDgram =sizeof(Dgram)+_dgram.xtc.sizeofPayload();
    };

    void addNamesBlock(uint8_t* name_block, size_t block_elems,
                       unsigned nodeId, unsigned namesId){
        Xtc& namesxtc = *new((char*)_dgram.xtc.alloc(sizeof(Xtc))) Xtc(TypeId(TypeId::Names, 0), NamesId(nodeId,namesId));
        size_t nameblock_size = sizeof(NameInfo) + block_elems*sizeof(Name);
        memcpy(namesxtc.payload(), name_block, nameblock_size);
        namesxtc.alloc(nameblock_size);
        _dgram.xtc.alloc(nameblock_size);

        _sizeDgram =sizeof(Dgram)+_dgram.xtc.sizeofPayload();
    }

    void addShapesDataBlock(uint8_t* shape_block, uint8_t* data_block, size_t sizeofdata, size_t block_elems, unsigned nodeId, unsigned namesId){
        Xtc& shapesdata = *new((char*)_dgram.xtc.alloc(sizeof(Xtc))) Xtc(TypeId(TypeId::ShapesData, 0), NamesId(nodeId,namesId));

        Xtc& shapes = *new((char*)shapesdata.alloc(sizeof(Xtc))) Xtc(TypeId(TypeId::Shapes, 0));
        // cpo: need to take away this uint32_t when we eliminate
        // it from the Shapes class
        size_t shapeblock_size = sizeof(uint32_t) + block_elems*sizeof(Shape);
        memcpy(shapes.payload(), shape_block, shapeblock_size);
        shapes.alloc(shapeblock_size);
        shapesdata.alloc(shapeblock_size);
        _dgram.xtc.alloc(shapeblock_size+sizeof(Xtc));


        Xtc& data = *new((char*)shapesdata.alloc(sizeof(Xtc))) Xtc(TypeId(TypeId::Data, 0));
        memcpy(data.payload(), data_block, sizeofdata);

        data.alloc(sizeofdata);
        shapesdata.alloc(sizeofdata);
        _dgram.xtc.alloc(sizeofdata+sizeof(Xtc));
        _sizeDgram =sizeof(Dgram)+_dgram.xtc.sizeofPayload();
    }

    void addDataBlock(uint8_t* data_block, size_t sizeofdata){
        Xtc& shapesdata = *new((char*)_dgram.xtc.alloc(sizeof(Xtc))) Xtc(TypeId(TypeId::ShapesData, 0));

        Xtc& data = *new((char*)shapesdata.alloc(sizeof(Xtc))) Xtc(TypeId(TypeId::Data, 0));
        memcpy(data.payload(), data_block, sizeofdata);

        data.alloc(sizeofdata);
        shapesdata.alloc(sizeofdata);
        _dgram.xtc.alloc(sizeofdata+sizeof(Xtc));
        _sizeDgram =sizeof(Dgram)+_dgram.xtc.sizeofPayload();
    }

    uint32_t dgramSize(){
        return _sizeDgram;
    };

private:
    size_t _sizeDgram = 0;
    Dgram& _dgram;

};

}; // namespace XtcData

#endif // BLOCKDGRAM__H
