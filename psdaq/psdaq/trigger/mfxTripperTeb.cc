#include "Trigger.hh"
#include "TripperTebData.hh"

#include "utilities.hh"

#include <cadef.h>

#include <cstdint>
#include <stdio.h>
#include <iostream>
#include <cstring>

using namespace rapidjson;
using json = nlohmann::json;

/**
 * TEB Trigger data for detector protection.
 * "Trips" shutter when number of hot pixels (ADU above a threshold) exceeds a count.
 */

namespace Pds {
    namespace Trg {

    class MfxTripperTrigger : public Trigger
    {
    public:
        int  configure(const json&              connectMsg,
                       const Document&          top,
                       const Pds::Eb::EbParams& prms) override;
        void event(const Pds::EbDgram* const* start,
                   const Pds::EbDgram**       end,
                   Pds::Eb::ResultDgram&      result) override;
        ~MfxTripperTrigger() {
            SEVCHK(ca_clear_channel(m_blockId), "Clear channel :BLOCK failed..");
            SEVCHK(ca_clear_channel(m_aduId), "Clear channel :ADU failed..");
            SEVCHK(ca_clear_channel(m_npixId), "Clear channel :NPIX failed..");
            SEVCHK(ca_clear_channel(m_npix_overId), "Clear channel :NPIX_OT failed..");
        }
    private:
        chid m_blockId; ///< Channel to trigger blocking of beam
        chid m_aduId;  ///< ADU threshold to consider pixel "hot"
        chid m_npixId; ///< Maximum number of hot pixels
        chid m_npix_overId; ///< Number of hot pixels found
        std::string m_tripperPV;
        bool m_tripped {false};
        ca_client_context* m_context;
    };
    };
};


int Pds::Trg::MfxTripperTrigger::configure(const json&              connectMsg,
                                           const Document&          top,
                                           const Pds::Eb::EbParams& prms)
{
    int rc = 0;
    if (top.HasMember("tripBasePV")) {
        m_tripperPV = top["tripBasePV"].GetString();
        std::cout << "Using tripper base PV: " << m_tripperPV << std::endl;
    } else {
        std::cout << "Configuration requires tripBasePV to work! Check configdb!" << std::endl;
        return 1;
    }
    SEVCHK(ca_context_create(ca_enable_preemptive_callback), "ca_context_create failed!");
    m_context = ca_current_context();
    int status;
    chid block;
    std::string fullBlockPV = m_tripperPV + ":BLOCK";
    ca_create_channel(fullBlockPV.c_str(), NULL, NULL, 0, &block);
    status = ca_pend_io(1);
    SEVCHK(status, "Connection to tripper :BLOCK PV failed!");
    m_blockId = block;

    chid adu;
    std::string fullAduPV = m_tripperPV + ":ADU";
    ca_create_channel(fullAduPV.c_str(), NULL, NULL, 0, &adu);
    status = ca_pend_io(1);
    SEVCHK(status, "Connection to tripper ADU PV failed!");
    m_aduId = adu;

    chid npix;
    std::string fullNpixPV = m_tripperPV + ":NPIX";
    ca_create_channel(fullNpixPV.c_str(), NULL, NULL, 0, &npix);
    status = ca_pend_io(1);
    SEVCHK(status, "Connection to tripper NPIX PV failed!");
    m_npixId = npix;

    chid npix_over;
    std::string fullNpix_overPV = m_tripperPV + ":NPIX_OT";
    ca_create_channel(fullNpix_overPV.c_str(), NULL, NULL, 0, &npix_over);
    status = ca_pend_io(1);
    SEVCHK(status, "Connection to tripper NPIX_OT PV failed!");
    m_npix_overId = npix_over;
    return rc;
}

void Pds::Trg::MfxTripperTrigger::event(const Pds::EbDgram* const* start,
                                        const Pds::EbDgram**       end,
                                        Pds::Eb::ResultDgram&      result)
{
    const Pds::EbDgram* const* ctrb = start;

    // Always write and monitor for now
    bool wrt {true};
    bool mon {true};

    uint32_t hotPixelThresh {0};
    uint32_t hotPixelCnt {0};
    uint32_t maxHotPixels {0};
    bool foundJungfrau {false};

    if (ca_current_context() == NULL) {
        ca_attach_context(m_context);
    }

    // Accumulate each contribution's input into some sort of overall summary
    do {
        auto data = reinterpret_cast<TripperTebData*>((*ctrb)->xtc.payload());
        if (strcmp(data->detType, "jungfrau") == 0) {
            hotPixelCnt += data->numHotPixels;
            maxHotPixels = data->maxHotPixels;
            hotPixelThresh = data->hotPixelThresh;
            foundJungfrau = true;
        }
    } while (++ctrb != end);

    if (foundJungfrau) {
        if (hotPixelCnt > maxHotPixels) {
            m_tripped = true;
            std::cout << "Tripped: " << hotPixelCnt << ", " << maxHotPixels << ", "
                      << hotPixelThresh << std::endl;
        }
        SEVCHK(ca_put(DBR_LONG, m_aduId, &hotPixelThresh), "Put to tripper :ADU PV failed!");
        SEVCHK(ca_put(DBR_LONG, m_npixId, &maxHotPixels), "Put to tripper :NPIX PV failed!");
        SEVCHK(ca_put(DBR_LONG, m_npix_overId, &hotPixelCnt), "Put to tripper :NPIX_OT PV failed!");
        ca_flush_io();
    }

    if (m_tripped) {
        long enable = 1;
        SEVCHK(ca_put(DBR_LONG, m_blockId, &enable), "Put to tripper :BLOCK PV failed!");
        ca_flush_io();
        wrt = false;
        mon = false;
    }
    // Insert the trigger values into a Result EbDgram
    result.persist(wrt);
    result.monitor(mon ? -1 : 0); // All MEBs
}


// The class factory

extern "C" Pds::Trg::Trigger* create_consumer()
{
  return new Pds::Trg::MfxTripperTrigger;
}
