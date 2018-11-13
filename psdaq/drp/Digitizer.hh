#ifndef DIGITIZER_H
#define DIGITIZER_H

#include <vector>
#include "drp.hh"
#include "Detector.hh"
#include "xtcdata/xtc/Xtc.hh"
#include "xtcdata/xtc/ShapesData.hh"
#include "xtcdata/xtc/DescData.hh"

class Digitizer : public Detector
{
public:
    Digitizer(unsigned src);
    virtual void configure(XtcData::Dgram& dgram, PGPData* pgp_data);
    virtual void event(XtcData::Dgram& dgram, PGPData* pgp_data);
private:
    std::vector<XtcData::NameIndex> m_namesVec;
    unsigned m_evtcount;
};

#endif // DIGITIZER_H
