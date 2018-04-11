
#include "AreaDetector.hh"
#include "xtcdata/xtc/VarDef.hh"

using namespace XtcData;

class RoiDef : public VarDef
{
public:
    enum index
    {
      array_fex
    };

    RoiDef()
    {
        Alg roi("roi", 1, 0, 0);
        NameVec.push_back({"array_fex", Name::FLOAT, 2, roi});
    }
};

void AreaDetector::configure(Xtc& parent)
{
    Alg cspadRawAlg("cspadRawAlg", 1, 2, 3);
    unsigned segment = 0;

    Names& fexNames = *new(parent) Names("cspad", cspadRawAlg, "cspad", "detnum1234", segment);
    Alg roi("roi", 1, 0, 0);
    // fexNames.add(parent, RoiDef); FIXME
    m_namesVec.push_back(NameIndex(fexNames));
}

void AreaDetector::event(Xtc& parent)
{
    /*
    CreateData fex(parent, m_namesVec, nameId);

    uint16_t* ptr = (uint16_t*)fex.get_ptr();
    unsigned shape[Name::MaxRank];
    shape[0] = 30;
    shape[1] = 30;
    uint32_t dma_index = pebble_data->pgp_data->buffers[0].dma_index;
    uint16_t* img = reinterpret_cast<uint16_t*>(dma_buffers[dma_index]);
    for (unsigned i=0; i<shape[0]*shape[1]; i++) {
        ptr[i] = img[i];
    }
    fex.set_array_shape(RoiDef::array_fex,shape);
    */
}

