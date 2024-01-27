#include "Wave8.hh"

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

using namespace XtcData;
using logging = psalg::SysLog;
using json = nlohmann::json;

namespace Drp {

  namespace W8 {
    class RawStream {
    public:
        static void varDef(VarDef& v,unsigned ch) {
            char name[32];
            // raw streams
            sprintf(name,"raw_%d",ch);
            v.NameVec.push_back(XtcData::Name(name, XtcData::Name::UINT16,1));
        }
        static void createData(CreateData& cd, unsigned& index, unsigned ch, Array<uint8_t>& seg) {
            unsigned shape[MaxRank];
            shape[0] = seg.shape()[0]>>1;
            Array<uint16_t> arrayT = cd.allocate<uint16_t>(index++, shape);
            memcpy(arrayT.data(), seg.data(), seg.shape()[0]);
        }
    };
    class IntegralStream {
    public:
        static void varDef(XtcData::VarDef& v) {
            char name[32];
            // v.NameVec.push_back(XtcData::Name("integralSize", XtcData::Name::UINT8));
            // v.NameVec.push_back(XtcData::Name("trigDelay"   , XtcData::Name::UINT8));
            v.NameVec.push_back(XtcData::Name("itrigCount"   , XtcData::Name::UINT16));
            // v.NameVec.push_back(XtcData::Name("baselineSize", XtcData::Name::UINT16));
            for(unsigned i=0; i<8; i++) {
                sprintf(name,"integral_%d",i);
                v.NameVec.push_back(XtcData::Name(name, XtcData::Name::INT32));  // actually 24-bit signed, need to sign-extend
            }
            for(unsigned i=0; i<8; i++) {
                sprintf(name,"base_%d",i);
                v.NameVec.push_back(XtcData::Name(name, XtcData::Name::UINT16));
            }
        }

        static void createData(CreateData& cd, unsigned& index, Array<uint8_t>& seg) {
            IntegralStream& p = *new(seg.data()) IntegralStream;
            cd.set_value(index++, p._itrigCount);
            for(unsigned i=0; i<8; i++) {
                uint32_t v = p._integral[i];
                if (v & (1<<23))   // sign-extend
                    v |= 0xff000000;
                cd.set_value(index++, (int32_t)v);
            }
            for(unsigned i=0; i<8; i++)
                cd.set_value(index++, p._base[i]);
            //            p._dump();
        }
        IntegralStream() {}
    private:
        void _dump() const {
          printf("integralSize %02x  trigDelay %02x  itrigCount %04x  baseSize %04x\n",
                 _integralSize, _trigDelay, _itrigCount, _baselineSize);
          for(unsigned i=0; i<8; i++)
              printf(" %d/%u", _integral[i], _base[i]);
          printf("\n");
        }
        uint8_t  _integralSize;
        uint8_t  _trigDelay;
        uint16_t _reserved;
        uint16_t _itrigCount;
        uint16_t _baselineSize;
        int32_t  _integral[8];
        uint16_t _base[8];
    };
    class ProcStream {
    public:
        static void varDef(VarDef& v) {
            // v.NameVec.push_back(XtcData::Name("quadrantSel", XtcData::Name::UINT32));
            v.NameVec.push_back(XtcData::Name("ptrigCount" , XtcData::Name::UINT32));
            v.NameVec.push_back(XtcData::Name("intensity"  , XtcData::Name::DOUBLE));
            v.NameVec.push_back(XtcData::Name("posX"       , XtcData::Name::DOUBLE));
            v.NameVec.push_back(XtcData::Name("posY"       , XtcData::Name::DOUBLE));
        }
        static void createData(CreateData& cd, unsigned& index, Array<uint8_t>& seg) {
            ProcStream& p = *new(seg.data()) ProcStream;
            cd.set_value(index++, p._ptrigCount);
            cd.set_value(index++, p._intensity);
            cd.set_value(index++, p._posX);
            cd.set_value(index++, p._posY);
            //            p._dump();
        }
        ProcStream() {}
    private:
        void _dump() const {
            printf("quadrantSel %08x  ptrigCount %08x  intensity %f  posX %f  posY %f\n",
                   _quadrantSel, _ptrigCount, _intensity, _posX, _posY);
        }
        uint32_t _quadrantSel;
        uint32_t _ptrigCount;
        double   _intensity;
        double   _posX;
        double   _posY;
    };
    class Streams {
    public:
        static void defineData(Xtc& xtc, const void* bufEnd, const char* detName,
                               const char* detType, const char* detNum,
                               NamesLookup& lookup, NamesId& raw, NamesId& fex) {
          // set up the names for L1Accept data
          { Alg alg("raw", 0, 0, 1);
            Names& eventNames = *new(xtc, bufEnd) Names(bufEnd,
                                                        detName, alg,
                                                        detType, detNum, raw);
            VarDef v;
            for(unsigned i=0; i<8; i++)
                RawStream::varDef(v,i);
            eventNames.add(xtc, bufEnd, v);
            lookup[raw] = NameIndex(eventNames); }
          { Alg alg("fex", 0, 0, 1);
            Names& eventNames = *new(xtc, bufEnd) Names(bufEnd,
                                                        detName, alg,
                                                        detType, detNum, fex);
            VarDef v;
            IntegralStream::varDef(v);
            ProcStream    ::varDef(v);
            eventNames.add(xtc, bufEnd, v);
            lookup[fex] = NameIndex(eventNames); }
        }
        static void createData(XtcData::Xtc&         xtc,
                               const void*           bufEnd,
                               XtcData::NamesLookup& lookup,
                               XtcData::NamesId&     rawId,
                               XtcData::NamesId&     fexId,
                               XtcData::Array<uint8_t>* streams) {
            CreateData raw(xtc, bufEnd, lookup, rawId);

            unsigned index=0;
            for(unsigned i=0; i<8; i++)
                RawStream::createData(raw,index,i,streams[i]);

            index=0;
            CreateData fex(xtc, bufEnd, lookup, fexId);

            if (streams[8].data())
                IntegralStream::createData(fex,index,streams[8]);

            if (streams[9].data())
                ProcStream::createData(fex,index,streams[9]);
       }
    };

  };

Wave8::Wave8(Parameters* para, MemPool* pool) :
    BEBDetector(para, pool)
{
    _init(para->kwargs["epics_prefix"].c_str());

    if (para->kwargs.find("timebase")!=para->kwargs.end() &&
        para->kwargs["timebase"]==std::string("119M"))
        m_debatch = true;
}

unsigned Wave8::_configure(Xtc& xtc, const void* bufEnd, ConfigIter&)
{
    // set up the names for the event data
    m_evtNamesRaw = NamesId(nodeId, EventNamesIndex+0);
    m_evtNamesFex = NamesId(nodeId, EventNamesIndex+1);
    W8::Streams::defineData(xtc,bufEnd,m_para->detName.c_str(),
                            m_para->detType.c_str(),m_para->serNo.c_str(),
                            m_namesLookup,m_evtNamesRaw,m_evtNamesFex);
    return 0;
}

void Wave8::_event(XtcData::Xtc& xtc,
                   const void* bufEnd,
                   std::vector< XtcData::Array<uint8_t> >& subframes)
{
    W8::Streams::createData(xtc, bufEnd, m_namesLookup, m_evtNamesRaw, m_evtNamesFex, &subframes[2]);
}
}
