#include "Digitizer.hh"
#include "xtcdata/xtc/VarDef.hh"


using namespace XtcData;

class HsdDef:public VarDef
{
public:
  enum index
    {
      chan0,
      chan1,
      chan2,
      chan3,
      maxNum
    };

   HsdDef()
   {
     detVec.push_back({"chan0"});
     detVec.push_back({"chan1"});
     detVec.push_back({"chan2"});
     detVec.push_back({"chan3"});
   }
};

void hsdExample(Xtc& parent, NameIndex& nameindex, unsigned nameId, Pebble* pebble_data, uint32_t** dma_buffers, std::vector<unsigned>& lanes)
{
    CreateData hsd(parent, nameindex, nameId);
    PGPBuffer* buffers = pebble_data->pgp_data->buffers;
    uint32_t shape[1];
    for (unsigned i=0; i<lanes.size(); i++) {
        shape[0] = buffers[lanes[i]].length*sizeof(uint32_t);
        hsd.set_array_shape(i, shape);
    }
}

void add_hsd_names(Xtc& parent, std::vector<NameIndex>& namesVec) {
    Alg hsdAlg("hsdalg",1,2,3);
    Names& fexNames = *new(parent) Names("xpphsd", hsdAlg, "hsd");
    
    Alg alg("hsdchan",1,2,3);
    fexNames.add_vec<HsdDef>(parent, alg);
    namesVec.push_back(NameIndex(fexNames));
}
