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
    virtual void configure(XtcData::Dgram& dgram, PGPData* pgp_data);
    virtual void event(XtcData::Dgram& dgram, PGPData* pgp_data);
private:
    enum {RawNamesIndex, FexNamesIndex};
    XtcData::NamesLookup m_namesLookup;
    unsigned m_evtcount;
};

#endif // AREA_DETECTOR_H
