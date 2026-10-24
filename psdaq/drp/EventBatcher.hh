/**
 * @file
 * @brief Layout of AxiStream batcher events (header, subframe tails) and an iterator over their subframes.
 */
// see https://confluence.slac.stanford.edu/display/ppareg/AxiStream+Batcher+Protocol+Version+1
#pragma once
#include <stdint.h>
#include "psalg/utils/SysLog.hh"

namespace Drp {
#pragma pack(push,1)
    /** Packed header at the start of an AxiStream batcher event: two 4-bit fields (version, width code) and an 8-bit sequence count. It occupies one line of lineWidth(width) bytes. */
    class EvtBatcherHeader {
    public:
      /** Return the address one line (lineWidth(width) bytes) after this header, where the first subframe starts. */
      void* next() const { return (void*)((char*)this + lineWidth(width)); }
      /** Return the line width in bytes for width code w: 2 shifted left by w. */
      static unsigned lineWidth(unsigned w) { return 2<<w; }
    public:
        unsigned version:4;  ///< 4-bit version field; not read by the code in this header.
        unsigned width:4;  ///< 4-bit width code; lineWidth(width) is the line size used by next() and EvtBatcherIterator.
        uint8_t  sequence_count;  ///< 8-bit sequence count; not read by the code in this header.
    };
    /** Packed tail that follows each subframe: payload size, tdest, first and last tuser bytes and a width code. The subframe data precedes the tail, padded to a whole line. */
    class EvtBatcherSubFrameTail {
    public:
        /** Return the start of the subframe data: this tail address minus the size rounded up to a whole line (a size of 0 is logged as corrupt). */
        void*     data () {return (void*)((char*)(this)-_totSize());}
        /** Return the width code of the subframe. */
        unsigned  width() {return _width;}
        /** Return the tdest of the subframe; BEBDetector uses it as the subframe index. */
        unsigned  tdest() {return _tdest;}
        /** Return a reference to the payload size in bytes. */
        unsigned& size () {return _size;}
    private:
        friend class EvtBatcherIterator;
        unsigned _totSize() {
            // round up the size to the nearest "line boundary",
            // which depends on the "width" parameter.
            if (_size==0) psalg::SysLog::critical("*** Error: EventBatcher found corrupt size=0");
            unsigned lw = EvtBatcherHeader::lineWidth(_width);
            return ((_size+lw-1)&~(lw-1));
        }
        uint32_t _size;
        uint8_t  _tdest;
        uint8_t  _tuser_first;
        uint8_t  _tuser_last;
        uint8_t  _width;
      };
#pragma pack(pop)
    /** Iterates backwards over the subframe tails of a batcher event, from the end of the buffer to the first line after the header. */
    class EvtBatcherIterator {
    public:
        /** Start at the last line of the bytes-long buffer at ebh; iteration stops at the line after the header. */
        EvtBatcherIterator(EvtBatcherHeader* ebh, size_t bytes) :
            // compute the first subframe ptr
            _lw(ebh->lineWidth(ebh->width)),
            _next((char*)ebh+bytes-_lw),
            _end((char*)(ebh->next())) {}

        // iterate backwards over the subframes
        /** Return the next tail going backwards, or nullptr when all have been returned. Logs and throws std::runtime_error if a subframe would extend before the first line after the header. */
        EvtBatcherSubFrameTail* next() {
            EvtBatcherSubFrameTail* save = (EvtBatcherSubFrameTail*)_next;
            if (!save) return save; // no more subframes
            // see if we've jumped backwards too far
            if (_next-(save->_totSize()) < _end) {
                psalg::SysLog::critical("*** corrupt EvtBatcherOutput: %li %d\n",_next-_end,save->_totSize());
                throw std::runtime_error("corrupt AxiStreamEventBuilder");
            }
            // compute the next subframe ptr
            if (_next-(save->_totSize()) == _end) {
                // indicates this is the last one
                _next = 0;
            } else {
                _next -= (save->_totSize()+_lw);
            }
            return save;
        }
    private:
        unsigned _lw;
        char* _next;
        char* _end;
    };
};
