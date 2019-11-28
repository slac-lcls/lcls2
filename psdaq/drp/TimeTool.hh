#pragma once

#include "drp.hh"
#include "XpmDetector.hh"
#include "xtcdata/xtc/Xtc.hh"
#include "xtcdata/xtc/NamesId.hh"

namespace Drp {

class TimeTool : public XpmDetector
{
public:
    TimeTool(Parameters* para, MemPool* pool);
    unsigned configure(const std::string& config_alias, XtcData::Xtc& xtc) override;
    void event(XtcData::Dgram& dgram, PGPEvent* event) override;
private:
    void _addJson(XtcData::Xtc& xtc, XtcData::NamesId& configNamesId);
    enum {ConfigNamesIndex = NAMES_INDEX_BASE, EventNamesIndex};
    unsigned          m_evtcount;
    XtcData::NamesId  m_evtNamesId;
    std::string       m_connect_json;
};

}
