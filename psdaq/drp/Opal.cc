#include "Opal.hh"
#include "xtcdata/xtc/VarDef.hh"
#include "xtcdata/xtc/DescData.hh"
#include "xtcdata/xtc/NamesLookup.hh"
#include "psalg/utils/SysLog.hh"

#include <Python.h>
#include <stdint.h>
#include <stdio.h>
#include <assert.h>

using namespace XtcData;
using logging = psalg::SysLog;
using json = nlohmann::json;

namespace Drp {

class RawDef : public VarDef
{
public:
    enum index {
        image
    };
    RawDef()
    {
        Alg alg("raw", 2, 0, 0);
        NameVec.push_back({"image", Name::UINT16, 2});  // Does the data need to be reformatted?
    }
} rawDef;

Opal::Opal(Parameters* para, MemPool* pool) :
    BEBDetector   (para, pool),
    m_evtNamesId  (-1, -1) // placeholder
{
  _init(para->detName.c_str());  // an argument is required here
}

void Opal::_connect(PyObject* mbytes)
{
    unsigned modelnum = strtoul( _string_from_PyDict(mbytes,"model").c_str(), NULL, 10);
#define MODEL(num,rows,cols) case num: m_rows = rows; m_columns = cols; break
    switch(modelnum) {
        MODEL(1000,1024,1024);
        MODEL(1600,1200,1600);
        MODEL(2000,1080,1920);
        MODEL(4000,1752,2336);
        MODEL(8000,2472,3296);
#undef MODEL
    default:
        throw std::string("Opal camera model not recognized");
        break;
    }

    m_para->serNo = _string_from_PyDict(mbytes,"serno");
}

json Opal::connectionInfo()
{
    // Exclude connection info until cameralink-gateway timingTxLink is fixed
    logging::error("Returning NO XPM link; implementation incomplete");
    return json({});
}

unsigned Opal::_configure(XtcData::Xtc& xtc)
{
    // set up the names for L1Accept data
    m_evtNamesId = NamesId(nodeId, EventNamesIndex);
    Alg alg("raw", 2, 0, 0);
    Names& eventNames = *new(xtc) Names(m_para->detName.c_str(), alg,
                                        m_para->detType.c_str(), m_para->serNo.c_str(), m_evtNamesId, m_para->detSegment);

    eventNames.add(xtc, rawDef);
    m_namesLookup[m_evtNamesId] = NameIndex(eventNames);
    return 0;
}

void Opal::_event(XtcData::Xtc& xtc, std::vector< XtcData::Array<uint8_t> >& subframes)
{
    CreateData cd(xtc, m_namesLookup, m_evtNamesId);

    unsigned shape[MaxRank];
    shape[0] = m_rows;
    shape[1] = m_columns;
    Array<uint8_t> arrayT = cd.allocate<uint8_t>(RawDef::image, shape);
    memcpy(arrayT.data(), subframes[2].data(), subframes[2].shape()[0]);
}

}
