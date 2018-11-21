#ifndef DIGITIZER_H
#define DIGITIZER_H

#include <vector>
#include "drp.hh"
#include "Detector.hh"
#include "xtcdata/xtc/Xtc.hh"
#include "xtcdata/xtc/NamesId.hh"

class Digitizer : public Detector
{
public:
    Digitizer(unsigned nodeId);
    virtual void configure(XtcData::Dgram& dgram, PGPData* pgp_data);
    virtual void event(XtcData::Dgram& dgram, PGPData* pgp_data);
private:
    enum {ConfigNamesIndex, EventNamesIndex};
    unsigned          m_evtcount;
    XtcData::NamesId  m_evtNamesId;
};

#endif // DIGITIZER_H
