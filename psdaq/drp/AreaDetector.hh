#ifndef AREA_DETECTOR_H
#define AREA_DETECTOR_H

#include <vector>

#include "drp.hh"
#include "Detector.hh"
#include "xtcdata/xtc/Xtc.hh"
#include "xtcdata/xtc/ShapesData.hh"
#include "xtcdata/xtc/DescData.hh"

//void roiExample(XtcData::Xtc& parent, std::vector<XtcData::NameIndex>& nameindex, unsigned nameId, Pebble* pebble_data, uint32_t** dma_buffers);
//void add_roi_names(XtcData::Xtc& parent, std::vector<XtcData::NameIndex>& namesVec);

class AreaDetector : public Detector
{
public:
    virtual void configure(XtcData::Xtc& parent);
    virtual void event(XtcData::Xtc& parent);
private:
    std::vector<XtcData::NameIndex> m_namesVec;
};

#endif // AREA_DETECTOR_H
