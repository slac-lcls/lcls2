#ifndef HSD_EVENTHEADER_HH
#define HSD_EVENTHEADER_HH

/*
 * Summary of design ideas from cpo/yoon82 (9/11/19)
 *
 * The Channel class here are intended to be used both in the drp (C++) and
 * psana (via cython wrapper psana/psana/hsd/hsd.pyx).  The code
 * unpacks the firmware-compressed data into more usable in-memory
 * arrays (which can be exported to python, for example).
 * The guts of the code are Channel::_parse_waveforms and
 * Channel::_parse_peaks.  There is (at least) one complex idea:
 * The unpacked arrays are variable-length and we wanted to avoid calling
 * malloc on every event in the drp (OK to do that for psana) so
 * we should try to use the simple Stack obj in psalg/alloc/Allocator.hh
 * in the drp.  the python uses a similar Heap obj which calls malloc.
 * In the drp each core should have its own Stack to avoid reentrancy
 * problems.
 *
 * The "event header" is formed from the two uint32_t _opaque fields of the
 * TimingHeader and contains information about whether the event contains
 * raw, fex, or both.  Originally we had a class for this, but this
 * was switched to an array uint32_t[2] for two reasons:
 *
 * - easier inter-operability with python
 * - we only ever accessed the "streams" field
 * 
 * Structure of the event header:
 * 20b  unused
 * 4b   stream mask (raw is bit 0, fex is bit 1)
 * 8b   x"01"
 * 16b  trigger phase on the sample clock
 * 16b  trigger waveform on the sample clock
 *
 * Immediately following the EventHeader in the dma buffer are the stream
 * headers (raw and fex) and their associated data:
 * Both raw/fex streamheaders are present on every event, even if their
 * associated data is missing (e.g. if prescale is not set to 1).
 * Per-channel structure:
 *
 * streamheader raw
 * raw data
 * streamheader fex
 * fex data
 *
 */

#include <stdint.h>
#include <stdio.h>
#include <cinttypes>

#include "xtcdata/xtc/Dgram.hh"
#include "Stream.hh"
#include "psalg/alloc/Allocator.hh"
#include "psalg/alloc/AllocArray.hh"

using namespace psalg;

namespace Pds {
  namespace HSD {

    class Channel {
    public:
        Channel(Allocator *allocator, const uint32_t *evtheader, const uint8_t *data);

        ~Channel(){}

        unsigned npeaks(){
            return numFexPeaks;
        }

    public:
        unsigned maxSize = 10000;
        Allocator *m_allocator;
        unsigned numPixels;
        unsigned numFexPeaks;
        unsigned content;
        uint16_t* rawPtr; // pointer to raw data

        AllocArray1D<uint16_t> waveform;
        AllocArray1D<uint16_t> sPos; // maxLength
        AllocArray1D<uint16_t> len; // maxLength
        AllocArray1D<uint16_t*> fexPtr; // maxLength

    private:
        void _parse_waveform(const StreamHeader& s) {
              const uint16_t* q = reinterpret_cast<const uint16_t*>(&s+1);
              numPixels = s.num_samples();
              for(unsigned i=0; i<numPixels; i++) {
                  waveform.push_back(q[i]);
              }
        }

        void _parse_peaks(const StreamHeader& s) {
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
                        len.push_back(width);
                        numFexPeaks++;
                    }
                    width = 0;
                    in = false;
                } else {
                    if (!in) {
                        sPos.push_back(ns+totWidth);
                        fexPtr.push_back((uint16_t *) (q+i));
                    }
                    for (unsigned j=0; j<4; j++, i++) {
                        width++;
                    }
                    in = true;
                }
            }
            if (in) {
                len.push_back(width);
                numFexPeaks++;
            }
        }
    };
  } // HSD
} // Pds

#endif
