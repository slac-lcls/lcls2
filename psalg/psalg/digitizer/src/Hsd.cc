#include "psalg/digitizer/Hsd.hh"

#include <stdio.h>
#include <ctype.h>

using namespace Pds::HSD;
// TODO: test with fex only data
Channel::Channel(Allocator *allocator, Pds::HSD::Hsd_v1_2_3 *vHsd, const uint8_t* data)
: m_allocator(allocator)
, numPixels(0)
, numFexPeaks(0)
, waveform(allocator, maxSize)
, sPos(allocator, maxSize)
, len(allocator, maxSize)
, fexPos(allocator, maxSize)
, fexPtr(allocator, maxSize)
{
    const char* nextx = reinterpret_cast<const char*>(data);

    if (vHsd->raw()) {
        const Pds::HSD::StreamHeader* sh_rawx = 0;
        sh_rawx = reinterpret_cast<const Pds::HSD::StreamHeader*>(nextx);
        printf("raw samples: %u %u %u %u\n", sh_rawx->samples(), sh_rawx->eoffs(), sh_rawx->boffs(), sh_rawx->samples()-sh_rawx->eoffs()-sh_rawx->boffs());
        const uint16_t* rawx = reinterpret_cast<const uint16_t*>(sh_rawx+1);
        if (sh_rawx->samples() > 0 && sh_rawx->samples() <= 6456) {

            numPixels = (unsigned) (sh_rawx->samples()-sh_rawx->eoffs()-sh_rawx->boffs());
            rawPtr = (uint16_t *) (rawx+sh_rawx->boffs());

            for (unsigned i =0; i < numPixels; i++) {
                waveform.push_back(*(rawPtr+i));
            }

            nextx = reinterpret_cast<const char*>(&rawx[sh_rawx->samples()]);
        } else {
            nextx = reinterpret_cast<const char*>(rawx+24);
        }
    }

    if (vHsd->fex()) {
        // ------------ FEX --------------
        const Pds::HSD::StreamHeader& sh_fex = *reinterpret_cast<const Pds::HSD::StreamHeader*>(nextx);
        const unsigned end = sh_fex.samples() - sh_fex.eoffs() - sh_fex.boffs();
        printf("fex samples: %u %u %u\n", sh_fex.samples(), sh_fex.eoffs(), sh_fex.boffs());
        if (sh_fex.samples() > 0 && sh_fex.samples() <= 6456) {
            const uint16_t* p_thr = &reinterpret_cast<const uint16_t*>(&sh_fex+1)[sh_fex.boffs()];

            unsigned i=0, j=0;
            bool skipped = true;
            bool in = false;
            if (p_thr[i] & 0x8000) { // skip to the sample with the trigger
              i++;
              j++;
            }
            while(i<end) {
                if (p_thr[i] & 0x8000) {
                    j += p_thr[i] & 0x7fff;
                    if (skipped) {
                        //printf(" consecutive skip\n"); // TODO: remove
                    } else {
                        //printf(" SKIP\n");
                        if (in) {
                            len.push_back(i-fexPos(numFexPeaks));
                            numFexPeaks++;
                        }
                    }
                    in = false;
                    skipped = true;
                } else {
                    if (skipped) {
                        sPos.push_back(j);
                        fexPos.push_back(i);
                        fexPtr.push_back((uint16_t *) (p_thr+i));

                        in = true;
                    }
                    j++;
                    skipped = false;
                }
                i++;
            }
            if (in) {
                len.push_back(i-fexPos(numFexPeaks));
                numFexPeaks++;
            }
        }
    }
}

Hsd_v1_2_3::Hsd_v1_2_3(Allocator *allocator)
: m_allocator(allocator)
{
    version = "1.2.3";
}
