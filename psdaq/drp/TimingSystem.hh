#pragma once

#include "drp.hh"
#include "XpmDetector.hh"
#include "xtcdata/xtc/Xtc.hh"
#include "xtcdata/xtc/NamesId.hh"

namespace Drp {

class TimingSystem : public XpmDetector
{
public:
    TimingSystem(Parameters* para, MemPool* pool);
    void connect(const nlohmann::json& msg, const std::string& collectionId) override;
    unsigned configure(const std::string& config_alias, XtcData::Xtc& xtc) override;
    void beginstep(XtcData::Xtc& xtc, const nlohmann::json& stepInfo) override;
    void event(XtcData::Dgram& dgram, PGPEvent* event) override;
private:
    void _addJson(XtcData::Xtc& xtc, XtcData::NamesId& configNamesId, const std::string& config_alias);
    enum {ConfigNamesIndex, EventNamesIndex};
    unsigned          m_evtcount;
    XtcData::NamesId  m_evtNamesId;
    std::string       m_connect_json;
};

}
