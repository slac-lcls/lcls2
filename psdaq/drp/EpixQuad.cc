#include "EpixQuad.hh"
#include "psdaq/service/Semaphore.hh"
#include "xtcdata/xtc/VarDef.hh"
#include "xtcdata/xtc/DescData.hh"
#include "xtcdata/xtc/NamesLookup.hh"
#include "psalg/utils/SysLog.hh"
#include "psalg/calib/NDArray.hh"
#include "psalg/detector/UtilsConfig.hh"

#include <Python.h>
#include <stdint.h>
#include <stdio.h>
#include <assert.h>
#include <fcntl.h>
#include <stdint.h>

using namespace XtcData;
using logging = psalg::SysLog;
using json = nlohmann::json;

namespace Drp {

  class RawDef : public VarDef
  {
  public:
    enum index { image };
    RawDef() { NameVec.push_back({"image", Name::UINT16, 3}); }
  } rawDef;
};

using Drp::EpixQuad;

EpixQuad::EpixQuad(Parameters* para, MemPool* pool) :
    BEBDetector   (para, pool),
    m_evtNamesId  (-1, -1) // placeholder
{
  _init(para->detName.c_str());  // an argument is required here
}

EpixQuad::~EpixQuad()
{
}

void EpixQuad::_connect(PyObject* mbytes)
{
    m_para->serNo = _string_from_PyDict(mbytes,"serno");
}

json EpixQuad::connectionInfo()
{
  return BEBDetector::connectionInfo();
}

unsigned EpixQuad::_configure(XtcData::Xtc& xtc,XtcData::ConfigIter& configo)
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

void EpixQuad::_event(XtcData::Xtc& xtc, std::vector< XtcData::Array<uint8_t> >& subframes)
{
  CreateData cd(xtc, m_namesLookup, m_evtNamesId);

  unsigned shape[MaxRank];
  shape[0] = m_rows;
  shape[1] = m_columns;
  shape[2] = 4;         // 4 ASICs in a quad
  Array<uint16_t> arrayT = cd.allocate<uint16_t>(RawDef::image, shape);
  memcpy(arrayT.data(), subframes[2].data(), subframes[2].shape()[0]);
}

void     EpixQuad::slowupdate(XtcData::Xtc& xtc)
{
}

void     EpixQuad::shutdown()
{
}

