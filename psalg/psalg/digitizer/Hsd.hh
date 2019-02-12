#ifndef HSD_EVENTHEADER_HH
#define HSD_EVENTHEADER_HH

/*
 * How to handle changing version number:
 * The factory method can return a completely different object if there is a big change
 * The algorithm name is some ways corresponds to the major version number
 * For some small backwardly compatible changes (minor version inc.), consider adding methods using inheritance
 * We have tried to demonstrate this with HsdEventHeaderV1, where additional virtual methods can be specified
 * as the objects evolve in time.
 *
 * The factory method is for use in python only
 * In the DRP, hard code version number and create object directly
 *
 *
 *
 * Hsd object is created per event
 *
 * EventHeader is in the env of the dgram and contains information about it containing raw, fex, or both.
 * (1/Nov/18: Currently, we can only determine whether the dgram contains raw, fex, or both by checking the size
 * of the payload. This makes the fex results ambiguous when empty. Matt will look into fixing this.)
 * Both streamheaders for raw and fex exist, even if prescale is not set to 1.
 * Per-channel structure:
 * streamheader raw
 * raw data
 * streamheader fex
 * fex data
 *
 * (7/Feb/19): HSD Header
 * 128b sequence
 * 32b  env
 * 24b  event counter
 * 8b   version
 * --
 * 20b  unused
 * 4b   stream mask
 * 8b   x"01"
 * 16b  trigger phase on the sample clock
 * 16b  trigger waveform on the sample clock
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
    // V1 corresponds to the major version of the high level alg/version number
    class HsdEventHeaderV1 {
    public:
        static HsdEventHeaderV1* Create(Allocator *allocator, Dgram* dg, const char* version, const unsigned nChan);
        virtual ~HsdEventHeaderV1(){}
        virtual void printVersion  () = 0;
        virtual unsigned samples   () = 0;
        virtual unsigned streams   () = 0;
        virtual unsigned channels  () = 0;
        virtual unsigned sync      () = 0;
        virtual bool raw           () = 0;
        virtual bool fex           () = 0;
    protected:
        std::string version;
    };

    class Hsd_v1_2_3 : public HsdEventHeaderV1 {
    // This uses the high level alg/version number for the detector to reflect changes in the datagram env.
    public:
        Hsd_v1_2_3(Allocator *allocator);

        ~Hsd_v1_2_3(){}

        void init(uint32_t *e) {
            env = e;
        }

        void printVersion() {
            std::cout << "hsd version " << version << std::endl;
        }

        // TODO: find out how to convert bin number to time (we need the configure transition)
        unsigned samples   ()  { return env[1]&0xfffff; }    // NOTE: These 3 functions assume
        unsigned streams   ()  { return (env[1]>>20)&0xf; }  // all event headers in each
        unsigned channels  ()  { return (env[1]>>24)&0xff; } // channel are identical
        unsigned sync      ()  { return env[2]&0x7; }

        bool raw() {
            if (streams() & 1) return true;
            return false;
        }

        bool fex() {
            if (streams() & 2) return true;
            return false;
        }

    public:
        Allocator *m_allocator; // python uses this interface to use the heap
        // TODO: get rid of m_allocator if not needed
    private:
        uint32_t *env;
    };

    HsdEventHeaderV1* HsdEventHeaderV1::Create(Allocator *allocator, Dgram* dg, const char* version, const unsigned nChan) {
        if ( strcmp(version, "1.2.3") == 0 )
            return new Hsd_v1_2_3(allocator); //, dg, nChan);
        else
            return NULL;
    }

    class Channel { // TODO: ideally get a version number from low level alg/version like we do for HsdEventHeaderV1
    public:
        Channel(Allocator *allocator, Hsd_v1_2_3 *vHsd, const uint32_t *evtheader, const uint8_t *data);

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
        AllocArray1D<uint16_t> fexPos; // maxLength
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
            const uint16_t* p_thr = reinterpret_cast<const uint16_t*>(&sh_fex+1);
            const unsigned end = s.num_samples();

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
                            len.push_back(i-fexPos(numFexPeaks));
                            numFexPeaks++;
                        }
                    }
                    in = false;
                    skipped = true;
                } else {
                    sPos.push_back(j);
                    fexPos.push_back(i);
                    fexPtr.push_back((uint16_t *) (p_thr+i));
                    in = true;
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
    };

    class Factory {
    public:
        // Factory doesn't explicitly create objects
        // but passes type to factory method "Create()"
        Factory(Allocator* allocator, const char* version, const unsigned nChan, Dgram* dg)
        {
            // assert version = v1
            printf("@@@@ Hsd Factory Create\n");
            pHsd = HsdEventHeaderV1::Create(allocator, dg, version, nChan);
        }
        ~Factory() {
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
