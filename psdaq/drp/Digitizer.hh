#pragma once

#include <vector>
#include "drp.hh"
#include "Detector.hh"
#include "xtcdata/xtc/Xtc.hh"
#include "xtcdata/xtc/NamesId.hh"

namespace Drp {

class Digitizer : public Detector
{
public:
    Digitizer(Parameters* para, MemPool* pool, unsigned nodeId);
    unsigned configure(XtcData::Dgram& dgram) override;
    void event(XtcData::Dgram& dgram, PGPEvent* event) override;
private:
    enum {ConfigNamesIndex, EventNamesIndex};
    unsigned          m_evtcount;
    XtcData::NamesId  m_evtNamesId;
};

}
