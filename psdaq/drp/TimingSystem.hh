#pragma once

#include "drp.hh"
#include "Detector.hh"
#include "xtcdata/xtc/Xtc.hh"
#include "xtcdata/xtc/NamesId.hh"

namespace Drp {

class TimingSystem : public Detector
{
public:
    TimingSystem(Parameters* para, MemPool* pool, unsigned nodeId);
    unsigned configure(XtcData::Xtc& xtc) override;
    void event(XtcData::Dgram& dgram, PGPEvent* event) override;
private:
    static unsigned _addJson(XtcData::Xtc& xtc, XtcData::NamesId& configNamesId);
    enum {ConfigNamesIndex, EventNamesIndex};
    unsigned          m_evtcount;
    XtcData::NamesId  m_evtNamesId;
};

}
