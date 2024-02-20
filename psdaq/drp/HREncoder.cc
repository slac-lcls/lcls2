#include "HREncoder.hh"

#include "xtcdata/xtc/Array.hh"
#include "xtcdata/xtc/VarDef.hh"
#include "xtcdata/xtc/DescData.hh"
#include "xtcdata/xtc/NamesLookup.hh"
#include "DataDriver.h"
#include "psalg/utils/SysLog.hh"

#include <Python.h>
#include <stdint.h>
#include <stdio.h>
#include <assert.h>
#include <fstream>
#include <string>
#include <iostream>

using namespace XtcData;
using logging = psalg::SysLog;
using json = nlohmann::json;

namespace Drp {
  namespace Enc {
  struct Stream {
  public:
      Stream() {}
      uint32_t getPosition() { return m_position; }
      uint8_t getEncErrCnt() { return m_encErrCnt; }
      uint8_t getMissedTrigCnt() { return m_missedTrigCnt; }
      uint8_t getLatches() { return m_latches; }

      void printValues() const
      {
          std::cout << "Position: " << m_position << std::endl;
          std::cout << "Error Count: " << m_encErrCnt << std::endl;
          std::cout << "Missed Triggers: " << m_missedTrigCnt << std::endl;
          std::cout << "Latch bits: " << m_latches << std::endl;
          std::cout << "Reserved: " << m_reserved << std::endl;
      }
  private:
      uint32_t m_position{};
      uint8_t m_encErrCnt{};
      uint8_t m_missedTrigCnt{};
      uint8_t m_latches{}; // upper 3 bits define 3 latch bits
      uint8_t m_reserved{}; // actually 13 bits
    };

  } // Enc

HREncoder::HREncoder(Parameters* para, MemPool* pool) :
    BEBDetector(para, pool)
{
    _init(para->kwargs["epics_prefix"].c_str());

    if (para->kwargs.find("timebase")!=para->kwargs.end() &&
        para->kwargs["timebase"]==std::string("119M"))
        m_debatch = true;

}

unsigned HREncoder::_configure(Xtc& xtc, const void* bufEnd, ConfigIter&)
{
    // set up the names for the event data
    m_evtNamesRaw = NamesId(nodeId, EventNamesIndex+0);

    Alg alg("raw", 0, 1, 0);
    Names& eventNames = *new (xtc, bufEnd) Names(bufEnd, m_para->detName.c_str(),
                                                 alg, m_para->detType.c_str(),
                                                 m_para->serNo.c_str(), m_evtNamesRaw);
    VarDef v;
    v.NameVec.push_back(XtcData::Name("position", XtcData::Name::UINT32));
    v.NameVec.push_back(XtcData::Name("error_cnt", XtcData::Name::UINT8));
    v.NameVec.push_back(XtcData::Name("missedTrig_cnt", XtcData::Name::UINT8));
    v.NameVec.push_back(XtcData::Name("latches", XtcData::Name::UINT8)); // Only 3 bits
    eventNames.add(xtc, bufEnd, v);
    m_namesLookup[m_evtNamesRaw] = NameIndex(eventNames);

    return 0;
}

void HREncoder::_event(XtcData::Xtc& xtc,
                   const void* bufEnd,
                   std::vector< XtcData::Array<uint8_t> >& subframes)
{
    unsigned index = 0;
    CreateData cd(xtc, bufEnd, m_namesLookup, m_evtNamesRaw);

    Enc::Stream& p = *new (subframes[2].data()) Enc::Stream;
    cd.set_value(index++, p.getPosition());
    cd.set_value(index++, p.getEncErrCnt());
    cd.set_value(index++, p.getMissedTrigCnt());
    cd.set_value(index++, p.getLatches());
}
} // Drp
