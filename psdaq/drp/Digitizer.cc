
#include "Digitizer.hh"

#include "xtcdata/xtc/VarDef.hh"


using namespace XtcData;

class HsdDef : public VarDef
{
public:
    enum index
    {
        chan0,
        chan1,
        chan2,
        chan3,
    };

    HsdDef()
    {
        Alg alg("hsdchan", 1, 2, 3);
        NameVec.push_back({"chan0", alg});
        NameVec.push_back({"chan1", alg});
        NameVec.push_back({"chan2", alg});
        NameVec.push_back({"chan3", alg});
    }
};

void hsdExample(Xtc& parent, std::vector<NameIndex>& NamesVec, unsigned nameId, Pebble* pebble_data, uint32_t** dma_buffers, std::vector<unsigned>& lanes)
{
    CreateData hsd(parent, NamesVec, nameId);
    PGPBuffer* buffers = pebble_data->pgp_data->buffers;
    uint32_t shape[1];
    for (unsigned i=0; i<lanes.size(); i++) {
        shape[0] = buffers[lanes[i]].length*sizeof(uint32_t);
        hsd.set_array_shape(i, shape);
    }
}

void add_hsd_names(Xtc& parent, std::vector<NameIndex>& namesVec)
{
    Alg hsdAlg("hsdalg", 1, 2, 3);
    Names& fexNames = *new(parent) Names("xpphsd", hsdAlg, "hsd", "detnum1234");
    fexNames.add(parent, HsdDef);
    namesVec.push_back(NameIndex(fexNames));
}
