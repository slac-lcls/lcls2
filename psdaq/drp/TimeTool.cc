#include "TimeTool.hh"

#include "xtcdata/xtc/VarDef.hh"
#include "xtcdata/xtc/DescData.hh"
#include "xtcdata/xtc/NamesLookup.hh"
#include "psalg/utils/SysLog.hh"

#include <fcntl.h>
#include <Python.h>
#include <stdint.h>
#include <stdio.h>
#include <assert.h>

using namespace XtcData;
using logging = psalg::SysLog;

namespace Drp {

class TTDef : public VarDef
{
public:
    enum index {
        image
    };
    TTDef()
    {
        Alg alg("tt", 2, 0, 0);
        NameVec.push_back({"image", Name::UINT8, 1});
    }
} TTDef;

TimeTool::TimeTool(Parameters* para, MemPool* pool) :
    BEBDetector(para, pool),
    m_evtNamesId(-1, -1) // placeholder
{
    _init(para->detName.c_str());
    _init_feb();
}

unsigned TimeTool::_configure(XtcData::Xtc& xtc, XtcData::ConfigIter& configo)
{
    // set up the names for L1Accept data
    m_evtNamesId = NamesId(nodeId, EventNamesIndex);
    Alg alg("raw", 2, 0, 0);
    Names& eventNames = *new(xtc) Names(m_para->detName.c_str(), alg,
                                        m_para->detType.c_str(), m_para->serNo.c_str(), m_evtNamesId, m_para->detSegment);

    eventNames.add(xtc, TTDef);
    m_namesLookup[m_evtNamesId] = NameIndex(eventNames);

    XtcData::DescData& descdata = configo.desc_shape();
    std::string uarts("expert.ClinkFeb[0].ClinkTop.Ch[0].UartPiranha4.");
    m_roiLen = descdata.get_value<uint32_t>((uarts+"ROI[1]").c_str()) -
      descdata.get_value<uint32_t>((uarts+"ROI[0]").c_str()) + 1;

    return 0;
}

void TimeTool::_event(XtcData::Xtc& xtc, std::vector< XtcData::Array<uint8_t> >& subframes)
{
    CreateData cd(xtc, m_namesLookup, m_evtNamesId);

    if (subframes[2].shape()[0] != m_roiLen)
      xtc.damage.increase(XtcData::Damage::UserDefined);
    else {
      unsigned shape[MaxRank];
      shape[0] = subframes[2].shape()[0];
      Array<uint8_t> arrayT = cd.allocate<uint8_t>(TTDef::image, shape);
      memcpy(arrayT.data(), subframes[2].data(), subframes[2].shape()[0]);
    }
}

}

