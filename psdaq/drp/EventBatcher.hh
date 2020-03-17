// see https://confluence.slac.stanford.edu/display/ppareg/AxiStream+Batcher+Protocol+Version+1

#include <stdint.h>
#include "psalg/utils/SysLog.hh"

namespace Drp {
#pragma pack(push,1)
    class EvtBatcherHeader {
    public:
        unsigned version:4;
        unsigned width:4;
        uint8_t  sequence_count;
        uint8_t  _unused[14]; // for width==3 only
    };
    class EvtBatcherSubFrameTail {
    public:
        void* data() {return (void*)((char*)(this)-_totSize());}
        unsigned width() {return _width;}
        unsigned tdest() {return _tdest;}
        unsigned size() {return _size;}
    private:
        friend class EvtBatcherIterator;
        unsigned _totSize() {
            // round up the size to the nearest "line boundary",
            // which depends on the "width" parameter.
            if (_size==0) psalg::SysLog::critical("*** Error: EventBatcher found corrupt size=0");
            return ((_size-1)/16)*16+16; // for width==3 only
        }
        uint32_t _size;
        uint8_t  _tdest;
        uint8_t  _tuser_first;
        uint8_t  _tuser_last;
        uint8_t  _width;
        uint8_t  _unused[8]; // for width==3 only
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
            // see if we've jumped backwards too far
            if (_next-(save->_totSize()) < _end) psalg::SysLog::critical("*** corrupt EvtBatcherOutput: %li %d\n",_next-_end,save->_totSize());
            // compute the next subframe ptr
            if (_next-(save->_totSize()) == _end) {
                // indicates this is the last one
                _next = 0;
            } else {
                _next -= (save->_totSize()+sizeof(EvtBatcherSubFrameTail));
            }
            return save;
        }
    private:
        char* _next;
        char* _end;
    };
};
