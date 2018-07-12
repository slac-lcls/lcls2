#ifndef HSD_EVENTHEADER_HH
#define HSD_EVENTHEADER_HH

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
    class HsdEventHeaderV1 : public XtcData::L1Transition {
    public:
        static HsdEventHeaderV1* Create(Allocator *allocator, const char* version, const unsigned nChan);
        virtual ~HsdEventHeaderV1(){}
        virtual void printVersion() = 0;
        virtual unsigned samples   () = 0;
        virtual unsigned streams   () = 0;
        virtual unsigned channels  () = 0;
        virtual unsigned sync      () = 0;
        virtual void dump          () = 0;
        virtual int parseChan(const uint8_t* data, const unsigned chanNum) = 0;
    protected:
        uint32_t _syncword;
        std::string version;
    };

    class Hsd_v1_2_3 : public HsdEventHeaderV1 {
    public:
        Hsd_v1_2_3(Allocator *allocator, const unsigned nChan);

        ~Hsd_v1_2_3(){
        }

        void printVersion() {
            std::cout << "hsd version " << version << std::endl;
        }

        unsigned samples   ()  { return env[1]&0xfffff; }    // NOTE: These 3 functions assume
        unsigned streams   ()  { return (env[1]>>20)&0xf; }  // all event headers in each
        unsigned channels  ()  { return (env[1]>>24)&0xff; } // channel are identical
        unsigned sync      ()  { return env[2]&0x7; }

        void dump()
        {
            uint32_t* word = (uint32_t*) this;
            for(unsigned i=0; i<8; i++)
                printf("%08x%c", word[i], i<7 ? '.' : '\n');
        }

        // FIXME: find out what to parse Raw, Fex, or Raw+Fex
        // TODO: find out how to convert bin number to time
        int parseChan(const uint8_t* data, const unsigned chanNum) // byte offset to get the next channel
        {
            const char* nextx = reinterpret_cast<const char*>(data);
            const Pds::HSD::StreamHeader* sh_rawx = 0;
            sh_rawx = reinterpret_cast<const Pds::HSD::StreamHeader*>(nextx);
            const uint16_t* rawx = reinterpret_cast<const uint16_t*>(sh_rawx+1);
            //sh_rawx->dump();

            numPixels.push_back((unsigned) (sh_rawx->samples()-sh_rawx->eoffs()-sh_rawx->boffs()));
            rawPtr.push_back((uint16_t *) (rawx+sh_rawx->boffs()));
            numFexPeaks.push_back(0);

            // ------------ FEX --------------
            nextx = reinterpret_cast<const char*>(&rawx[sh_rawx->samples()]);
            const Pds::HSD::StreamHeader& sh_fexx = *reinterpret_cast<const Pds::HSD::StreamHeader*>(nextx);
            //sh_fexx.dump();
            const unsigned end = sh_fexx.samples() - sh_fexx.eoffs() - sh_fexx.boffs();
            const uint16_t* p_thr = &reinterpret_cast<const uint16_t*>(&sh_fexx+1)[sh_fexx.boffs()];

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
                            len(chanNum).push_back(i-fexPos(chanNum)(numFexPeaks(chanNum)));
                            numFexPeaks(chanNum) = numFexPeaks(chanNum)+1;
                        }
                    }
                    in = false;
                    skipped = true;
                } else {
                    if (skipped) {
                        //printf(" New beginning\n");
                        sPos(chanNum).push_back(j);
                        fexPos(chanNum).push_back(i);
                        fexPtr(chanNum).push_back((uint16_t *) (p_thr+i));

                        in = true;
                    }
                    j++;
                    skipped = false;
                }
                i++;
            }
            if (in) {
                len(chanNum).push_back(i-fexPos(chanNum)(numFexPeaks(chanNum)));
                numFexPeaks(chanNum) = numFexPeaks(chanNum)+1;
            }
            return 0;
        }
    public:
        Allocator *m_allocator;
        AllocArray1D<unsigned> numPixels;
        AllocArray1D<AllocArray1D<uint16_t> > sPos; // nChan x maxLength
        AllocArray1D<AllocArray1D<uint16_t> > len; // nChan x maxLength
        AllocArray1D<AllocArray1D<uint16_t> > fexPos; // nChan x maxLength
        AllocArray1D<AllocArray1D<uint16_t*> > fexPtr; // nChan x maxLength
        AllocArray1D<unsigned> numFexPeaks; // nChan x 1
        AllocArray1D<uint16_t*> rawPtr; // nChan x 1
    };

    HsdEventHeaderV1* HsdEventHeaderV1::Create(Allocator *allocator, const char* version, const unsigned nChan) {
        if ( strcmp(version, "1.2.3") == 0 )
            return new Hsd_v1_2_3(allocator, nChan);
        else
            return NULL;
    }

    // Client class
    class Client {
    public:
        // Client doesn't explicitly create objects
        // but passes type to factory method "Create()"
        Client(Allocator* allocator, const char* version, const unsigned nChan)
        {
            // assert version = 2
            pHsd = HsdEventHeaderV1::Create(allocator, version, nChan);
        }
        ~Client() {
            if (pHsd) {
                delete pHsd;
                pHsd = NULL;
            }
        }
        HsdEventHeaderV1* getHsd()  {
            return pHsd;
        }

    private:
        HsdEventHeaderV1 *pHsd = NULL;
    };
  } // HSD
} // Pds

#endif
