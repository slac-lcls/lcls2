/**
 * @file
 * @brief Declares Pds::HSD::Channel, which unpacks the raw and fex streams of one HSD channel into arrays.
 */
#ifndef HSD_EVENTHEADER_HH
#define HSD_EVENTHEADER_HH

/*
 * This file was an early attempt to have C code that could use both
 * malloc (for psana) and a non-malloc "stack" (for the DAQ).  in the
 * end this code felt too complex, so it has been removed from the DAQ.
 */

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
 * Note: if raw or fex data is missing, then the associated header is also
 * missing (e.g. if prescale is not set to 1).
 * Per-channel structure, if both raw/fex data are present:
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

namespace Pds {
  namespace HSD {

    /**
     * Unpacks one HSD channel: the raw stream (stream id 0) is copied into waveform, and the fex stream (stream id 1) is parsed into peak start positions (sPos), widths (len) and data pointers (fexPtr).
     * The arrays are allocated from the given Allocator with capacity maxSize each.
     */
    class Channel {
    public:
        /**
         * Allocate the four arrays from allocator, then walk the streams flagged in bits 21:20 of evtheader[0]. Each StreamHeader at data is followed by num_samples() 16-bit words; stream id 0 is copied into waveform and stream id 1 is parsed into peaks.
         * In the fex stream, groups of four words whose first word has bit 15 set are skips (their low 15 bits add to the sample position); other groups of four belong to a peak.
         */
        Channel(Allocator *allocator, const uint32_t *evtheader, const uint8_t *data);

        /** Destructor; does nothing itself (the arrays release their memory in their own destructors). */
        ~Channel(){}

        /** Return numFexPeaks. */
        unsigned npeaks(){
            return numFexPeaks;
        }

    public:
        unsigned maxSize = 1000000;  ///< Capacity (1000000 elements) of each array allocated by the constructor.
        Allocator *m_allocator;  ///< Allocator passed to the constructor.
        unsigned numPixels;  ///< Number of raw samples copied into waveform (0 if there is no raw stream).
        unsigned numFexPeaks;  ///< Number of peaks found in the fex stream.
        unsigned content;  ///< Not set by the constructor or by anything in Hsd.hh or Hsd.cc.
        /** Not set by the constructor or by anything in Hsd.hh or Hsd.cc; the trailing comment calls it a pointer to raw data. */
        uint16_t* rawPtr; // pointer to raw data

        psalg::AllocArray1D<uint16_t> waveform;  ///< Raw samples of stream id 0.
        /** Start sample of each fex peak: the skipped samples plus the widths of earlier peaks. */
        psalg::AllocArray1D<uint16_t> sPos; // maxLength
        /** Width in samples of each fex peak. */
        psalg::AllocArray1D<uint16_t> len; // maxLength
        /** Pointer to the first data word of each fex peak inside the input buffer. */
        psalg::AllocArray1D<uint16_t*> fexPtr; // maxLength

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
