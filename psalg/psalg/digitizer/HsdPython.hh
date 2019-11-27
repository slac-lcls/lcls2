#ifndef HSD_EVENTHEADER_HH
#define HSD_EVENTHEADER_HH

// See Hsd.hh (the DRP version of this) for a description of the
// hsd data structure from the front-end FPGA.

#include <stdint.h>
#include <stdio.h>
#include <cinttypes>

#include "xtcdata/xtc/Dgram.hh"
#include "Stream.hh"

namespace Pds {
  namespace HSD {

    class ChannelPython {
    public:
        ChannelPython() {}
        ChannelPython(uint32_t *evtheader, uint8_t* data) :
            _evtheader(evtheader), _data(data) {}

        ~ChannelPython(){}

    public:
        uint32_t* _evtheader;
        uint8_t*  _data;

        uint16_t* waveform(unsigned& numsamples) {
            // find out if we have raw/fex or both
            unsigned streams((_evtheader[0]>>20)&0x3);
            uint8_t* p = _data;
            while(streams) {
                StreamHeader& s = *reinterpret_cast<StreamHeader*>(p);
                if (s.stream_id() == 0) {
                    numsamples = s.num_samples();
                    uint16_t* wf = (uint16_t*)(&s+1);
                    return wf;
                }
                streams &= ~(1<<s.stream_id());
            }
            return 0;
        }

        // void _parse_peaks(const StreamHeader& s, uint16_t* sPos,
        //                   uint16_t* len, uint16_t** fexPtr) {
        //     const Pds::HSD::StreamHeader& sh_fex = *reinterpret_cast<const Pds::HSD::StreamHeader*>(&s);
        //     const uint16_t* q = reinterpret_cast<const uint16_t*>(&sh_fex+1);

        //     unsigned ns=0;
        //     bool in = false;
        //     unsigned width = 0;
        //     unsigned totWidth = 0;
        //     for(unsigned i=0; i<s.num_samples();) {
        //         if (q[i]&0x8000) {
        //             for (unsigned j=0; j<4; j++, i++) {
        //                 ns += (q[i]&0x7fff);
        //             }
        //             totWidth += width;
        //             if (in) {
        //                 *len++ = width;
        //                 numFexPeaks++;
        //             }
        //             width = 0;
        //             in = false;
        //         } else {
        //             if (!in) {
        //                 *sPos++ = ns+totWidth;
        //                 *fexPtr++ = (uint16_t *) (q+i);
        //             }
        //             for (unsigned j=0; j<4; j++, i++) {
        //                 width++;
        //             }
        //             in = true;
        //         }
        //     }
        //     if (in) {
        //         *len++ = width;
        //         numFexPeaks++;
        //     }
        // }
    };
  } // HSD
} // Pds

#endif
