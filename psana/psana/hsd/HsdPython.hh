#ifndef HSD_EVENTHEADER_HH
#define HSD_EVENTHEADER_HH

// See Hsd.hh (the DRP version of this) for a description of the
// hsd data structure from the front-end FPGA.

#include <stdint.h>
#include <stdio.h>
#include <cinttypes>

#include "psalg/digitizer/Stream.hh"

namespace Pds {
  namespace HSD {

    class ChannelPython {
    public:
        ChannelPython() {}
        ChannelPython(uint32_t *evtheader, uint8_t* data) :
            _evtheader(evtheader), _data(data)
        {
            memset(_sh, sizeof(_sh), 0);
            unsigned streams((evtheader[0]>>20)&0x3);
            const uint8_t* p = data;
            while(streams) {
                const StreamHeader* sh = reinterpret_cast<const StreamHeader*>(p);
                _sh[sh->stream_id()] = sh;
                p += sizeof(StreamHeader)+sh->num_samples()*2;
                streams &= ~(1<<sh->stream_id());
            }
        }

        ~ChannelPython(){}

        //  Raw stream : "NTR"
        uint16_t* waveform(unsigned& numsamples, unsigned istr) {
            if (!_sh[istr]) return 0;
            numsamples = _sh[istr]->num_samples();
            uint16_t* wf = (uint16_t*)(_sh[0]+1);
            return wf;
        }

        //  For debugging the processing in this class
        // uint16_t* sparse(unsigned& numsamples) {
        //     if (!_sh_fex) return 0;
        //     numsamples = _sh_fex->num_samples();
        //     uint16_t* wf = (uint16_t*)(_sh_fex+1);
        //     return wf;
        // }

        //  Fex stream ("NAF")
        unsigned next_peak(unsigned& startPos, uint16_t** peakPtr, unsigned istr) {
            unsigned peakLen = 0; // indicate that, by default, we haven't found a peak
            if (!_sh[istr]) return peakLen; // no more peaks to look for
            const uint16_t* q = reinterpret_cast<const uint16_t*>(_sh[istr]+1);

            unsigned i;
            for(i=_startSample; i<_sh[istr]->num_samples();) {
                if (q[i]&0x8000) { // are we a "skip" sample?
                    for (unsigned j=0; j<4; j++, i++) {
                        _ns += (q[i]&0x7fff); // increment the number-of-skips
                    }
                    if (_in) { // if we were previously in a peak, this completes that peak
                        _totWidth += _width; // add the width of the last peak to the cumulative sum
                        peakLen = _width;
                        _startSample = i; // remember where to start for next call
                        _in = false; // we're not in a peak anymore
                        return peakLen;
                    }
                } else {
                    if (!_in) { // we weren't previously in a peak, so start a new one
                        _width = 0;
                        startPos = _ns+_totWidth; // the index into the raw waveform: number-of-skips plus width of all previous peaks
                        *peakPtr = (uint16_t *) (q+i); // pointer to the start of the peak array
                    }
                    i += 4; // move to the next (interleaved) sample
                    _width += 4; // increment the width of this peak
                    _in = true; // we are in a peak
                }
            }
            if (_in) {
                // I think this case happens when the last peak includes
                // the very last uint16_t in sh_fex payload.
                peakLen = _width;
            }
            // these two lines will cause the iterator to return 0
            // on the next call, ending the iteration.
            _startSample = i;
            _in = false;
            return peakLen;
        }

        //  Window stream("WFX"), cfdWindow("CFP"), timestampCFD("CFD")
        unsigned next_window(unsigned& startPos, uint16_t** peakPtr, unsigned istr) {
            unsigned peakLen = 0; // indicate that, by default, we haven't found a peak
            if (!_sh[istr]) return peakLen; // no more peaks to look for
            const uint16_t* q = reinterpret_cast<const uint16_t*>(_sh[istr]+1);

            unsigned i;
            for(i=_startSample; i<_sh[istr]->num_samples();) {
                // The modified skip sample is interpreted differently by its position
                // in the super sample.  Pos 0 is the index of the gate opening.  Pos 3 is
                // the index of the gate closing.  Pos 1 is the nominal skip since last
                // recording.  We only care about Pos 1... I think.
                if (q[i]&0x8000) { // are we a "skip" sample?
                    //  Test if we ever have any other entries in the supersample
                    //  If not, then next_peak can replace this.
                    // unsigned qt = q[i] | q[i+2] | q[i+3];
                    // if (qt & 0x7fff)
                    //     printf("Found extra skip entries [0x%04x,0x%04x,0x%04x,0x%04x]\n",
                    //            q[i], q[i+1], q[i+2], q[i+3]);

                    _ns += (q[i+1]&0x7fff);
                    i += 4;
                    if (_in) { // if we were previously in a peak, this completes that peak
                        _totWidth += _width; // add the width of the last peak to the cumulative sum
                        peakLen = _width;
                        _startSample = i; // remember where to start for next call
                        _in = false; // we're not in a peak anymore
                        return peakLen;
                    }
                } else {
                    if (!_in) { // we weren't previously in a peak, so start a new one
                        _width = 0;
                        startPos = _ns+_totWidth; // the index into the raw waveform: number-of-skips plus width of all previous peaks
                        *peakPtr = (uint16_t *) (q+i); // pointer to the start of the peak array
                    }
                    i += 4; // move to the next (interleaved) sample
                    _width += 4; // increment the width of this peak
                    _in = true; // we are in a peak
                }
            }
            if (_in) {
                // I think this case happens when the last peak includes
                // the very last uint16_t in sh_fex payload.
                peakLen = _width;
            }
            // these two lines will cause the iterator to return 0
            // on the next call, ending the iteration.
            _startSample = i;
            _in = false;
            return peakLen;
        }

        unsigned char fex_out_of_range() { return _sh[1] ? _sh[1]->out_of_range() : 0; }

        void reset_peak_iter() {
            _ns=0;
            _in = false;
            _width = 0;
            _totWidth = 0;
            _startSample = 0;
        }

    private:
        uint32_t* _evtheader;
        uint8_t*  _data;
        const StreamHeader* _sh[4];

        unsigned _ns;
        bool     _in;
        unsigned _width;
        unsigned _totWidth;
        unsigned _startSample;

    };
  } // HSD
} // Pds

#endif
