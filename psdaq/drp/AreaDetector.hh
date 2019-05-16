#pragma once

#include "drp.hh"
#include "Detector.hh"
#include "xtcdata/xtc/Xtc.hh"
#include "xtcdata/xtc/NamesLookup.hh"

namespace Drp {

class AreaDetector : public Detector
{
public:
    AreaDetector(Parameters* para, MemPool* pool);
    void connect() override;
    unsigned configure(XtcData::Xtc& xtc) override;
    void event(XtcData::Dgram& dgram, PGPEvent* event) override;
private:
    enum {RawNamesIndex, FexNamesIndex};
    XtcData::NamesLookup m_namesLookup;
    unsigned m_evtcount;
};

}
