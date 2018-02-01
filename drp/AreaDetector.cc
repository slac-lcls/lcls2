#include "AreaDetector.hh"

using namespace XtcData;

void roiExample(Xtc& parent, NameIndex& nameindex, unsigned nameId, Pebble* pebble_data, uint32_t** dma_buffers)
{
    CreateData fex(parent, nameindex, nameId);

    uint16_t* ptr = (uint16_t*)fex.get_ptr();
    unsigned shape[Name::MaxRank];
    shape[0] = 30;
    shape[1] = 30;
    uint32_t dma_index = pebble_data->pgp_data->buffers[0].dma_index;
    uint16_t* img = reinterpret_cast<uint16_t*>(dma_buffers[dma_index]);
    for (unsigned i=0; i<shape[0]*shape[1]; i++) {
        ptr[i] = img[i];
    }
    fex.set_array_shape("array_fex",shape);
}

void add_roi_names(Xtc& parent, std::vector<NameIndex>& namesVec) {
    Alg detAlg("cspadTop",1,2,3);
    Names& fexNames = *new(parent) Names("cspad", "fex", detAlg);
    Alg alg("roi", 1, 0, 0);
    fexNames.add(parent, "array_fex", alg); //Name::UINT16, parent, 2);
    namesVec.push_back(NameIndex(fexNames));
}
