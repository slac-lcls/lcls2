/**
 * @file
 * @brief Declares XtcData::BlockDgram, which builds a Dgram from pre-built memory blocks (Names, Shapes, Data).
 */
#ifndef BLOCKDGRAM__H
#define BLOCKDGRAM__H

// class used by cython to create dgrams with "blocks" of memory,
// the three block types being Shapes, Data, and ShapesData

#include "xtcdata/xtc/ShapesData.hh"
#include "xtcdata/xtc/NamesId.hh"

namespace XtcData {

/**
 * Builds a Dgram in a caller-supplied buffer by appending copies of pre-built Names, Shapes and Data blocks. The comment above says it is used by cython.
 * It derives from Xtc, but these methods do not use that base.
 */
class BlockDgram : public Xtc
{
public:
    /**
     * Placement-construct a Dgram at buffdgram with time TimeStamp(tstamp), env control << 24, and an empty TypeId::Parent (version 0) xtc.
     * buffdgram + buffSize is the end bound used by later alloc() calls, which abort if it would be passed.
     */
    BlockDgram(uint8_t* buffdgram, size_t buffSize, uint64_t tstamp,
               unsigned control):
        _dgram(*new(buffdgram) Dgram()),
        _buffEnd(buffdgram + buffSize)
    {
        _dgram.time = TimeStamp(tstamp);
        _dgram.env = control<<24;
        _dgram.xtc = Xtc(TypeId(TypeId::Parent, 0));

        _sizeDgram =sizeof(Dgram)+_dgram.xtc.sizeofPayload();
    };

    /** Append to the top-level xtc a TypeId::Names xtc (src NamesId(nodeId, namesId)) whose payload is a copy of sizeof(NameInfo) + block_elems*sizeof(Name) bytes from name_block. */
    void addNamesBlock(uint8_t* name_block, size_t block_elems,
                       unsigned nodeId, unsigned namesId){
        size_t nameblock_size = sizeof(NameInfo) + block_elems*sizeof(Name);
        Xtc& namesxtc = *::new(_dgram.xtc.alloc(sizeof(Xtc) + nameblock_size, _buffEnd)) Xtc(TypeId(TypeId::Names, 0), NamesId(nodeId,namesId));
        auto payload = namesxtc.alloc(nameblock_size, _buffEnd);
        memcpy(payload, name_block, nameblock_size);

        _sizeDgram =sizeof(Dgram)+_dgram.xtc.sizeofPayload();
    }

    /**
     * Append a TypeId::ShapesData xtc (src NamesId(nodeId, namesId)) holding a Shapes xtc and then a Data xtc.
     * The Shapes payload is sizeof(uint32_t) + block_elems*sizeof(Shape) bytes copied from shape_block; the Data payload is sizeofdata bytes copied from data_block.
     */
    void addShapesDataBlock(uint8_t* shape_block, uint8_t* data_block, size_t sizeofdata, size_t block_elems, unsigned nodeId, unsigned namesId){
        // cpo: need to take away this uint32_t when we eliminate
        // it from the Shapes class
        size_t shapeblock_size = sizeof(uint32_t) + block_elems*sizeof(Shape);
        size_t shapes_size = sizeof(Xtc) + shapeblock_size;
        size_t data_size = sizeof(Xtc) + sizeofdata;
        size_t shapesdata_size = sizeof(Xtc) + shapes_size + data_size;
        Xtc& shapesdata = *::new(_dgram.xtc.alloc(shapesdata_size, _buffEnd)) Xtc(TypeId(TypeId::ShapesData, 0), NamesId(nodeId,namesId));

        Xtc& shapes = *::new(shapesdata.alloc(shapes_size, _buffEnd)) Xtc(TypeId(TypeId::Shapes, 0));
        auto payload = shapes.alloc(shapeblock_size, _buffEnd);
        memcpy(payload, shape_block, shapeblock_size);

        Xtc& data = *::new(shapesdata.alloc(data_size, _buffEnd)) Xtc(TypeId(TypeId::Data, 0));
        payload = data.alloc(sizeofdata, _buffEnd);
        memcpy(payload, data_block, sizeofdata);

        _sizeDgram =sizeof(Dgram)+_dgram.xtc.sizeofPayload();
    }

    /** Append a TypeId::ShapesData xtc with default src (0) holding only a Data xtc whose payload is sizeofdata bytes copied from data_block. */
    void addDataBlock(uint8_t* data_block, size_t sizeofdata){
        size_t data_size = sizeof(Xtc) + sizeofdata;
        size_t shapesdata_size = sizeof(Xtc) + data_size;
        Xtc& shapesdata = *::new((char*)_dgram.xtc.alloc(shapesdata_size, _buffEnd)) Xtc(TypeId(TypeId::ShapesData, 0));

        Xtc& data = *::new(shapesdata.alloc(data_size, _buffEnd)) Xtc(TypeId(TypeId::Data, 0));
        auto payload = data.alloc(sizeofdata, _buffEnd);
        memcpy(payload, data_block, sizeofdata);

        _sizeDgram =sizeof(Dgram)+_dgram.xtc.sizeofPayload();
    }

    /** Return sizeof(Dgram) plus the top-level xtc's payload size, as updated by the constructor and each add*Block() call. */
    uint32_t dgramSize(){
        return _sizeDgram;
    };

private:
    size_t _sizeDgram = 0;
    Dgram& _dgram;
    const void* _buffEnd;
};

}; // namespace XtcData

#endif // BLOCKDGRAM__H
