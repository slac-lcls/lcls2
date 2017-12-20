#include "Digitizer.hh"

using namespace XtcData;

void hsdExample(Xtc& parent, NameIndex& nameindex, unsigned nameId, Pebble* pebble_data, uint32_t** dma_buffers, std::vector<unsigned>& lanes)
{
    char chan_name[8];
    CreateData hsd(parent, nameindex, nameId);
    PGPBuffer* buffers = pebble_data->pgp_data->buffers;
    uint32_t shape[1];
    for (unsigned i=0; i<lanes.size(); i++) {
        sprintf(chan_name,"chan%d",i);
        shape[0] = buffers[lanes[i]].length*sizeof(uint32_t);
        hsd.set_array_shape(chan_name, shape);
    }
}

void add_hsd_names(Xtc& parent, std::vector<NameIndex>& namesVec) {
    Alg alg("hsd",1,2,3);
    Names& fexNames = *new(parent) Names("hsd1", "raw");
    
    fexNames.add(parent, "chan0", alg);
    fexNames.add(parent, "chan1", alg);
    fexNames.add(parent, "chan2", alg);
    fexNames.add(parent, "chan3", alg);
    namesVec.push_back(NameIndex(fexNames));
}
