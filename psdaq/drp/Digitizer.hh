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
    Digitizer(Parameters* para);
    unsigned configure(XtcData::Dgram& dgram) override;
    void event(XtcData::Dgram& dgram, PGPData* pgp_data) override;
private:
    enum {ConfigNamesIndex, EventNamesIndex};
    unsigned          m_evtcount;
    XtcData::NamesId  m_evtNamesId;
};

#endif // DIGITIZER_H
