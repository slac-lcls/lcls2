#ifndef AREA_DETECTOR_H
#define AREA_DETECTOR_H

#include "drp.hh"
#include "Detector.hh"
#include "xtcdata/xtc/Xtc.hh"
#include "xtcdata/xtc/NamesLookup.hh"

class AreaDetector : public Detector
{
public:
    AreaDetector(unsigned nodeId);
    void connect() override;
    void configure(XtcData::Dgram& dgram, PGPData* pgp_data) override;
    void event(XtcData::Dgram& dgram, PGPData* pgp_data) override;
private:
    enum {RawNamesIndex, FexNamesIndex};
    XtcData::NamesLookup m_namesLookup;
    unsigned m_evtcount;
};

#endif // AREA_DETECTOR_H
