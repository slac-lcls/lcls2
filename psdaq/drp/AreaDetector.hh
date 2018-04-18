#ifndef AREA_DETECTOR_H
#define AREA_DETECTOR_H

#include <vector>
#include "drp.hh"
#include "Detector.hh"
#include "xtcdata/xtc/Xtc.hh"
#include "xtcdata/xtc/ShapesData.hh"
#include "xtcdata/xtc/DescData.hh"

class AreaDetector : public Detector
{
public:
    AreaDetector();
    virtual void configure(XtcData::Xtc& xtc);
    virtual void event(XtcData::Xtc& xtc, PGPData* pgp_data);
private:
    std::vector<XtcData::NameIndex> m_namesVec;
    unsigned m_evtcount;
};

#endif // AREA_DETECTOR_H
