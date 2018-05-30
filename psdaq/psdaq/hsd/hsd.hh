#ifndef HSD_EVENTHEADER_HH
#define HSD_EVENTHEADER_HH

#include <stdint.h>
#include <stdio.h>
#include <cinttypes>

#include "xtcdata/xtc/Dgram.hh"
#include "stream.hh"
#include "../../../psalg/psalg/include/Allocator.hh"
#include "../../../psalg/psalg/include/AllocArray.hh"
using namespace psalg;

namespace Pds {
  namespace HSD {
    class HsdEventHeaderV1 : public XtcData::L1Transition {  // TODO: check whether we need to inherit L1Transition
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

    class Hsd_v1_0_0 : public HsdEventHeaderV1 {
    public:
        Hsd_v1_0_0(Allocator *allocator, const unsigned nChan): m_allocator(allocator)
        , numPixels(allocator, 0)
        , rawPtr(allocator, 0)
        , sPosx(allocator, 0)
        , numFexPeaksx(allocator, 0)
        , lenx(allocator, 0)
        , fexPtr(allocator, 0)
        , fexPos(allocator, 0)
        {
            version = "1.0.0";
            sPosx = AllocArray1D<AllocArray1D<uint16_t> >(m_allocator, nChan);
            lenx = AllocArray1D<AllocArray1D<uint16_t> >(m_allocator, nChan);
            fexPos = AllocArray1D<AllocArray1D<uint16_t> >(m_allocator, nChan);
            fexPtr = AllocArray1D<AllocArray1D<uint16_t*> >(m_allocator, nChan);
            for (unsigned i=0; i<nChan; i++) {
                auto _t = AllocArray1D<uint16_t>(m_allocator, 1600); // FIXME: better way to set array length?
                sPosx.push_back(_t);
                auto _p = AllocArray1D<uint16_t>(m_allocator, 1600);
                lenx.push_back(_p);
                auto _r = AllocArray1D<uint16_t>(m_allocator, 1600);
                fexPos.push_back(_r);
                auto _q = AllocArray1D<uint16_t*>(m_allocator, 1600);
                fexPtr.push_back(_q);
            }
            numFexPeaksx = AllocArray1D<unsigned>(m_allocator, nChan);
            rawPtr = AllocArray1D<uint16_t*>(m_allocator, nChan);
            numPixels = AllocArray1D<unsigned>(m_allocator, nChan);
        }

        ~Hsd_v1_0_0(){
        }

        void printVersion() {
            std::cout << "I am v" << version << std::endl;
        }

        // FIXME: channels have different lengths
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
        int parseChan(const uint8_t* data, const unsigned chanNum) // byte offset to get the next channel
        {
            printf("----------- parseChan %d\n", chanNum);
            printf("Address data: %p\n", (void *) data);
            const Pds::HSD::EventHeader& ehx = *reinterpret_cast<const Pds::HSD::EventHeader*>(data);
            //printf("Address ehx: %p\n", (void *) &ehx);
            //ehx.dump();
            const char* nextx = reinterpret_cast<const char*>(&ehx);
            const Pds::HSD::StreamHeader* sh_rawx = 0;
            sh_rawx = reinterpret_cast<const Pds::HSD::StreamHeader*>(nextx);
            const uint16_t* rawx = reinterpret_cast<const uint16_t*>(sh_rawx+1);
            printf("Address rawx: %p\n", (void*)rawx);
            sh_rawx->dump();

            printf("\t"); for(unsigned i=0; i<8; i++) printf(" %04x", rawx[i]); printf("\n");
            printf("boffs %d, samples %d, eoffs %d\n", sh_rawx->boffs(), sh_rawx->samples(), sh_rawx->eoffs());

            numPixels.push_back((unsigned) (sh_rawx->samples()-sh_rawx->eoffs()-sh_rawx->boffs()));
            rawPtr.push_back((uint16_t *) (rawx+sh_rawx->boffs()));
            numFexPeaksx.push_back(0);

            // ------------ FEX --------------
            nextx = reinterpret_cast<const char*>(&rawx[sh_rawx->samples()]);
            const Pds::HSD::StreamHeader& sh_fexx = *reinterpret_cast<const Pds::HSD::StreamHeader*>(nextx);
            const uint16_t* fexx = reinterpret_cast<const uint16_t*>(&sh_fexx+1);
            //printf("Address fexx: %p\n", (void*)fexx);

            const unsigned end = sh_fexx.samples() - sh_fexx.eoffs() - sh_fexx.boffs();

            //unsigned counter = 0;
            //for(unsigned i = sh_fexx.boffs(); i < sh_fexx.samples() - sh_fexx.eoffs(); i++){
            //    printf("%d %d\n",counter,fexx[i]);
            //    counter++;
            //}

            const uint16_t* p_thr = &reinterpret_cast<const uint16_t*>(&sh_fexx+1)[sh_fexx.boffs()];
            //printf("Address p_thr: %p %p\n", (void*)p_thr, p_thr);

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
                        printf(" consecutive skip\n"); // TODO: remove
                    } else {
                        printf(" SKIP\n");
                        if (in) {
                            lenx(chanNum).push_back(i-fexPos(chanNum)(numFexPeaksx(chanNum)));
                            numFexPeaksx(chanNum) = numFexPeaksx(chanNum)+1;
                        }
                    }
                    in = false;
                    skipped = true;
                } else {
                    if (skipped) {
                        printf(" New beginning\n");
                        sPosx(chanNum).push_back(j);
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
                lenx(chanNum).push_back(i-fexPos(chanNum)(numFexPeaksx(chanNum)));
                numFexPeaksx(chanNum) = numFexPeaksx(chanNum)+1;
            }
            printf("numPeaks: %d\n", numFexPeaksx(chanNum));

            return 0;
        }
    public:
        Allocator *m_allocator;
        AllocArray1D<unsigned> numPixels; // FIXME: channels have different lengths
        AllocArray1D<AllocArray1D<uint16_t> > sPosx; // nChan x maxLength
        AllocArray1D<AllocArray1D<uint16_t> > lenx; // nChan x maxLength
        AllocArray1D<AllocArray1D<uint16_t> > fexPos; // nChan x maxLength
        AllocArray1D<AllocArray1D<uint16_t*> > fexPtr; // nChan x maxLength
        AllocArray1D<unsigned> numFexPeaksx; // nChan x 1
        AllocArray1D<uint16_t*> rawPtr; // nChan x 1
    };

    HsdEventHeaderV1* HsdEventHeaderV1::Create(Allocator *allocator, const char* version, const unsigned nChan) {
        if ( strcmp(version, "1.0.0") == 0 )
            return new Hsd_v1_0_0(allocator, nChan);
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
