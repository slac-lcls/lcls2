#pragma once

#include <string>
#include <vector>
#include "drp.hh"
#include "Detector.hh"
#include "EventBatcher.hh"
#include "xtcdata/xtc/Xtc.hh"
#include "xtcdata/xtc/NamesId.hh"
#include "psalg/alloc/Allocator.hh"

namespace Drp {

class Wave8 : public Detector
{
public:
    Wave8(Parameters* para, MemPool* pool);
    nlohmann::json connectionInfo() override;
    void connect(const nlohmann::json&, const std::string& collectionId) override;
    unsigned configure(const std::string& config_alias, XtcData::Xtc& xtc) override;
    void event(XtcData::Dgram& dgram, PGPEvent* event) override;
    void shutdown() override;
    Pds::TimingHeader* getTimingHeader(uint32_t index) const override
    {
        EvtBatcherHeader& ebh = *static_cast<EvtBatcherHeader*>(m_pool->dmaBuffers[index]);
        // skip past the event-batcher header
        return static_cast<Pds::TimingHeader*>(ebh.next());
    }
private:
    unsigned _addJson(XtcData::Xtc& xtc, XtcData::NamesId& configNamesId, const std::string&);
    unsigned _getPaddr();
private:
    enum {ConfigNamesIndex = NamesIndex::BASE, EventNamesIndex};
    unsigned          m_readoutGroup;
    XtcData::NamesId  m_evtNamesRaw;
    XtcData::NamesId  m_evtNamesFex;
    std::string       m_connect_json;
    std::string       m_epics_name;
    Heap              m_allocator;
    unsigned          m_paddr;
};

}
