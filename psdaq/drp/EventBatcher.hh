// see https://confluence.slac.stanford.edu/display/ppareg/AxiStream+Batcher+Protocol+Version+1

namespace Drp {
#pragma pack(push,1)
    class EvtBatcherHeader {
    public:
        unsigned version:4;
        unsigned width:4;
        uint8_t  sequence_count;
        uint8_t  _unused[14]; // set for width==3
    };
    class EvtBatcherSubFrameTail {
    public:
        uint32_t size;
        uint8_t  tdest;
        uint8_t  tuser_first;
        uint8_t  tuser_last;
        uint8_t  width;
        uint8_t  _unused[8]; // set for width==3
    };
#pragma pack(pop)
};
