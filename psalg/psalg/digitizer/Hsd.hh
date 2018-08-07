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
 * Per-channel structure:
 * streamheader raw
 * raw data
 * streamheader fex
 * fex data
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
        static HsdEventHeaderV1* Create(Allocator *allocator, const char* version, const unsigned nChan, Dgram* dg);
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
        Hsd_v1_2_3(Allocator *allocator, const unsigned nChan, Dgram *dg);

        ~Hsd_v1_2_3(){}

        void printVersion() {
            std::cout << "hsd version " << version << std::endl;
        }

        // TODO: find out how to convert bin number to time (we need the configure transition)
        unsigned samples   ()  { return m_dg->env[1]&0xfffff; }    // NOTE: These 3 functions assume
        unsigned streams   ()  { return (m_dg->env[1]>>20)&0xf; }  // all event headers in each
        unsigned channels  ()  { return (m_dg->env[1]>>24)&0xff; } // channel are identical
        unsigned sync      ()  { return m_dg->env[2]&0x7; }

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
        Dgram *m_dg;
    };

    HsdEventHeaderV1* HsdEventHeaderV1::Create(Allocator *allocator, const char* version, const unsigned nChan, Dgram* dg) {
        if ( strcmp(version, "1.2.3") == 0 )
            return new Hsd_v1_2_3(allocator, nChan, dg);
        else
            return NULL;
    }

    class Channel { // TODO: ideally get a version number from low level alg/version like we do for HsdEventHeaderV1
    public:
        Channel(Allocator *allocator, const uint8_t *data, Hsd_v1_2_3 *vHsd);

        ~Channel(){}

        unsigned npeaks(){
            return numFexPeaks;
        }

    public:
        unsigned maxSize = 1600;
        Allocator *m_allocator;
        unsigned numPixels;
        unsigned numFexPeaks;
        unsigned content;
        uint16_t* rawPtr; // pointer to raw data
        AllocArray1D<uint16_t> sPos; // maxLength
        AllocArray1D<uint16_t> len; // maxLength
        AllocArray1D<uint16_t> fexPos; // maxLength
        AllocArray1D<uint16_t*> fexPtr; // maxLength
    };

    class Factory {
    public:
        // Factory doesn't explicitly create objects
        // but passes type to factory method "Create()"
        Factory(Allocator* allocator, const char* version, const unsigned nChan, Dgram* dg)
        {
            // assert version = v1
            pHsd = HsdEventHeaderV1::Create(allocator, version, nChan, dg);
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
