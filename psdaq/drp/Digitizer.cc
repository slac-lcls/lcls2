
#include "Digitizer.hh"
#include "xtcdata/xtc/VarDef.hh"
#include "xtcdata/xtc/DescData.hh"

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
        Alg alg("fpga", 1, 2, 3);
        NameVec.push_back({"chan0", alg});
        NameVec.push_back({"chan1", alg});
        NameVec.push_back({"chan2", alg});
        NameVec.push_back({"chan3", alg});
    }
};
static HsdDef myHsdDef;

Digitizer::Digitizer() : m_evtcount(0) {}

void Digitizer::configure(Dgram& dgram, PGPData* pgp_data)
{
    printf("Digitizer configure\n");

    // copy Event header into beginning of Datagram
    int index = __builtin_ffs(pgp_data->buffer_mask) - 1;
    Transition* event_header = reinterpret_cast<Transition*>(pgp_data->buffers[index]->virt);
    memcpy(&dgram, event_header, sizeof(Transition));

    Alg hsdAlg("hsd", 1, 2, 3);
    unsigned segment = 0;
    Names& hsdNames = *new(dgram.xtc) Names("xpphsd", hsdAlg, "hsd", "detnum0", segment);
    hsdNames.add(dgram.xtc, myHsdDef);
    m_namesVec.push_back(NameIndex(hsdNames));
}

void Digitizer::event(Dgram& dgram, PGPData* pgp_data)
{
    m_evtcount+=1;
    int index = __builtin_ffs(pgp_data->buffer_mask) - 1;
    unsigned nameId=0;
    Transition* event_header = reinterpret_cast<Transition*>(pgp_data->buffers[index]->virt);
    memcpy(&dgram, event_header, sizeof(Transition));
    CreateData hsd(dgram.xtc, m_namesVec, nameId);
    
    unsigned data_size;
    unsigned shape[MaxRank];
    for (int l=0; l<8; l++) {
        if (pgp_data->buffer_mask & (1 << l)) {
            // size without Event header
            data_size = pgp_data->buffers[l]->size - sizeof(Transition);
            shape[0] = data_size;
            Array<uint8_t> arrayT = hsd.allocate<uint8_t>(l, shape);
            memcpy(arrayT.data(), (uint8_t*)pgp_data->buffers[l]->virt + sizeof(Transition), data_size);
         }
    }
}
