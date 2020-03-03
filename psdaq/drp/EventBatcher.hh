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
        void* data() {return (void*)((char*)(this)-size);}
        uint32_t size;
        uint8_t  tdest;
        uint8_t  tuser_first;
        uint8_t  tuser_last;
        uint8_t  width;
        uint8_t  _unused[8]; // set for width==3
    };
#pragma pack(pop)
    class EvtBatcherIterator {
    public:
        EvtBatcherIterator(EvtBatcherHeader* ebh, size_t bytes) :
            // compute the first subframe ptr
            _next(((char*)(ebh))+bytes-sizeof(EvtBatcherSubFrameTail)),
            _end((char*)(ebh+1)) {}

        // iterate backwards over the subframes
        EvtBatcherSubFrameTail* next() {
            EvtBatcherSubFrameTail* save = (EvtBatcherSubFrameTail*)_next;
            if (!save) return save; // no more subframes
            if (_next-save->size < _end) {
                // we've jumped backwards too far
                printf("*** corrupt EvtBatcherOutput: %p %p\n",_next,_end);
                throw "*** corrupt EvtBatcherOutput";
            }
            // compute the next subframe ptr
            if (_next-save->size == _end) {
                // indicates this is the last one
                _next = 0;
            } else {
                _next -= (save->size+sizeof(EvtBatcherSubFrameTail));
            }
            return save;
        }
    private:
        char* _next;
        char* _end;
    };
};
