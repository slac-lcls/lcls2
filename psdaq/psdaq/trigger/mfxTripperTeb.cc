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
            SEVCHK(ca_clear_channel(m_block), "Clear channel failed..");
        }
    public:
        chid m_block;
        std::string m_tripperPV;
        bool m_tripped;
    };
    };
};


int Pds::Trg::MfxTripperTrigger::configure(const json&              connectMsg,
                                           const Document&          top,
                                           const Pds::Eb::EbParams& prms)
{
    int rc = 0;

    chid block;

    if (top.HasMember("tripBasePV")) {
        m_tripperPV = top["tripBasePV"].GetString();
        std::cout << "Using tripper base PV: " << m_tripperPV << std::endl;
    } else {
        std::cout << "Configuration requires tripBasePV to work! Check configdb!" << std::endl;
        rc = 1;
    }
    SEVCHK(ca_context_create(ca_enable_preemptive_callback), "ca_context_create failed!");
    std::string fullBlockPV = m_tripperPV + ":BLOCK";
    ca_create_channel(fullBlockPV.c_str(), NULL, NULL, 0, &block);
    int status = ca_pend_io(1);
    SEVCHK(status, "Connection to tripper PV failed!");
    m_block = block;
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

    uint32_t hotPixelCnt {0};

    uint32_t maxHotPixels {0};

    // Accumulate each contribution's input into some sort of overall summary
    do {
        auto data = reinterpret_cast<TripperTebData*>((*ctrb)->xtc.payload());
        if (strcmp(data->detType, "jungfrau") == 0) {
            hotPixelCnt += data->numHotPixels;
            maxHotPixels = data->maxHotPixels;
        }
    } while (++ctrb != end);

    if (hotPixelCnt > maxHotPixels) {
        m_tripped = true;
    }

    if (m_tripped) {
        long enable = 1;
        SEVCHK(ca_put(DBR_LONG, m_block, &enable), "Put to tripper PV failed!");
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
