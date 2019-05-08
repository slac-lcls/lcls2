#pragma once

#include "drp.hh"
#include "Detector.hh"
#include "xtcdata/xtc/Xtc.hh"
#include "xtcdata/xtc/NamesLookup.hh"

namespace Drp {

class AreaDetector : public Detector
{
public:
    AreaDetector(Parameters* para, MemPool* pool, unsigned nodeId);
    void connect() override;
    unsigned configure(XtcData::Dgram& dgram) override;
    void event(XtcData::Dgram& dgram, PGPEvent* event) override;
private:
    enum {RawNamesIndex, FexNamesIndex};
    XtcData::NamesLookup m_namesLookup;
    unsigned m_evtcount;
};

}
