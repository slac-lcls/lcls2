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
        // find out how long the arrays need to be so we can malloc
        // the right amount of space
        ChannelPython(const StreamHeader& s, unsigned& waveform_len,
                      unsigned& len_len, unsigned& sPos_len,
                      unsigned& fexPtr_len) : numPixels(0), numFexPeaks(0)
        {
            waveform_len = s.num_samples();

            const Pds::HSD::StreamHeader& sh_fex = *reinterpret_cast<const Pds::HSD::StreamHeader*>(&s);
            const uint16_t* q = reinterpret_cast<const uint16_t*>(&sh_fex+1);

            bool in = false;
            for(unsigned i=0; i<s.num_samples();) {
                if (q[i]&0x8000) {
                    if (in) {
                        len_len++;
                    }
                    in = false;
                } else {
                    if (!in) {
                        sPos_len++;
                        fexPtr_len++;
                    }
                    in = true;
                }
            }
            if (in) {
                len_len++;
            }
        }

        void fill_arrays(const uint32_t *evtheader, const uint8_t *data,
                         uint16_t* waveform, uint16_t* sPos, uint16_t* len,
                         uint16_t** fexPtr);

        ~ChannelPython(){}

        unsigned npeaks(){
            return numFexPeaks;
        }

    public:
        unsigned numPixels;
        unsigned numFexPeaks;
        unsigned content;

    private:

        void _parse_waveform(const StreamHeader& s, uint16_t* waveform) {
              const uint16_t* q = reinterpret_cast<const uint16_t*>(&s+1);
              numPixels = s.num_samples();
              for(unsigned i=0; i<numPixels; i++) {
                  waveform[i] = q[i];
              }
        }

        void _parse_peaks(const StreamHeader& s, uint16_t* sPos,
                          uint16_t* len, uint16_t** fexPtr) {
            const Pds::HSD::StreamHeader& sh_fex = *reinterpret_cast<const Pds::HSD::StreamHeader*>(&s);
            const uint16_t* q = reinterpret_cast<const uint16_t*>(&sh_fex+1);

            unsigned ns=0;
            bool in = false;
            unsigned width = 0;
            unsigned totWidth = 0;
            for(unsigned i=0; i<s.num_samples();) {
                if (q[i]&0x8000) {
                    for (unsigned j=0; j<4; j++, i++) {
                        ns += (q[i]&0x7fff);
                    }
                    totWidth += width;
                    if (in) {
                        *len++ = width;
                        numFexPeaks++;
                    }
                    width = 0;
                    in = false;
                } else {
                    if (!in) {
                        *sPos++ = ns+totWidth;
                        *fexPtr++ = (uint16_t *) (q+i);
                    }
                    for (unsigned j=0; j<4; j++, i++) {
                        width++;
                    }
                    in = true;
                }
            }
            if (in) {
                *len++ = width;
                numFexPeaks++;
            }
        }
    };
  } // HSD
} // Pds

#endif
