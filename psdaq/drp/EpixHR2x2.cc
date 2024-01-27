#include "EpixHR2x2.hh"
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
#include <signal.h>

using namespace XtcData;
using logging = psalg::SysLog;
using json = nlohmann::json;

namespace Drp {

#define ADD_FIELD(name,ntype,ndim)  NameVec.push_back({#name, Name::ntype, ndim})

    class EpixHRPanelDef : public VarDef
    {
    public:
        //        enum index { raw, aux, numfields };
        enum index { raw, numfields };

        EpixHRPanelDef() {
            ADD_FIELD(raw              ,UINT16,2);
            //            ADD_FIELD(aux              ,UINT16,2);
        }
    } epixHRPanelDef;

    class EpixHRHwDef : public VarDef
    {
    public:
        EpixHRHwDef() {
            ADD_FIELD(var, UINT32, 0);
        }
    } epixHRHwDef;

    class EpixHRDef : public VarDef
    {
    public:
        enum index { sht31Hum, sht31TempC,
                     nctLocTempC, nctFpgaTempC,
                     asicA0_2V5_CurrmA,
                     asicA1_2V5_CurrmA,
                     asicA2_2V5_CurrmA,
                     asicA3_2V5_CurrmA,
                     asicD0_2V5_CurrmA,
                     asicD1_2V5_CurrmA,
                     therm0TempC,
                     therm1TempC,
                     pwrDigCurr,
                     pwrDigVin,
                     pwrDigTempC,
                     pwrAnaCurr,
                     pwrAnaVin,
                     pwrAnaTempC,
                     asic_temp,
                     num_fields };

        EpixHRDef() {
            ADD_FIELD(sht31Hum         ,FLOAT,0);
            ADD_FIELD(sht31TempC       ,FLOAT,0);
            ADD_FIELD(nctLocTempC      ,UINT16 ,0);
            ADD_FIELD(nctFpgaTempC     ,FLOAT,0);
            ADD_FIELD(asicA0_2V5_CurrmA,FLOAT,0);
            ADD_FIELD(asicA1_2V5_CurrmA,FLOAT,0);
            ADD_FIELD(asicA2_2V5_CurrmA,FLOAT,0);
            ADD_FIELD(asicA3_2V5_CurrmA,FLOAT,0);
            ADD_FIELD(asicD0_2V5_CurrmA,FLOAT,0);
            ADD_FIELD(asicD1_2V5_CurrmA,FLOAT,0);
            ADD_FIELD(therm0TempC      ,FLOAT,0);
            ADD_FIELD(therm1TempC      ,FLOAT,0);
            ADD_FIELD(pwrDigCurr       ,FLOAT,0);
            ADD_FIELD(pwrDigVin        ,FLOAT,0);
            ADD_FIELD(pwrDigTempC      ,FLOAT,0);
            ADD_FIELD(pwrAnaCurr       ,FLOAT,0);
            ADD_FIELD(pwrAnaVin        ,FLOAT,0);
            ADD_FIELD(pwrAnaTempC      ,FLOAT,0);
            ADD_FIELD(asic_temp        ,UINT16,1);
        }
    } epixHRDef;
};

#undef ADD_FIELD

using Drp::EpixHR2x2;

static EpixHR2x2* epix = 0;

static void sigHandler(int signal)
{
  psignal(signal, "epixhr2x2 received signal");
  epix->monStreamEnable();
  ::exit(signal);
}

EpixHR2x2::EpixHR2x2(Parameters* para, MemPool* pool) :
    BEBDetector   (para, pool),
    m_env_sem     (Pds::Semaphore::FULL),
    m_env_empty   (true)
{
    _init(para->detName.c_str());  // an argument is required here

    m_descramble = false;

    epix = this;

    struct sigaction sa;
    sa.sa_handler = sigHandler;
    sa.sa_flags = SA_RESETHAND;

    sigaction(SIGINT ,&sa,NULL);
    sigaction(SIGABRT,&sa,NULL);
    sigaction(SIGKILL,&sa,NULL);
    sigaction(SIGSEGV,&sa,NULL);
}

EpixHR2x2::~EpixHR2x2()
{
}

void EpixHR2x2::_connectionInfo(PyObject* mbytes)
{
    m_para->serNo = _string_from_PyDict(mbytes,"serno");
}

unsigned EpixHR2x2::enable(XtcData::Xtc& xtc, const void* bufEnd, const nlohmann::json& info)
{
    logging::debug("EpixHR2x2 enable");
    monStreamDisable();
    return 0;
}

unsigned EpixHR2x2::disable(XtcData::Xtc& xtc, const void* bufEnd, const nlohmann::json& info)
{
    logging::debug("EpixHR2x2 disable");
    monStreamEnable();
    return 0;
}

unsigned EpixHR2x2::_configure(XtcData::Xtc& xtc, const void* bufEnd, XtcData::ConfigIter& configo)
{
    // set up the names for L1Accept data
    // Generic panel data
    {
        Alg alg("raw", 2, 0, 1);
        // copy the detName, detType, detId from the Config Names
        Names& configNames = configo.namesLookup()[NamesId(nodeId, ConfigNamesIndex+1)].names();
        NamesId nid = m_evtNamesId[0] = NamesId(nodeId, EventNamesIndex);
        logging::debug("Constructing panel eventNames src 0x%x",
                       unsigned(nid));
        Names& eventNames = *new(xtc, bufEnd) Names(bufEnd,
                                                    configNames.detName(), alg,
                                                    configNames.detType(),
                                                    configNames.detId(),
                                                    nid,
                                                    m_para->detSegment);

        eventNames.add(xtc, bufEnd, epixHRPanelDef);
        m_namesLookup[nid] = NameIndex(eventNames);
    }

    {
        XtcData::Names&    names    = detector::configNames(configo);
        XtcData::DescData& descdata = configo.desc_shape();
        for(unsigned i=0; i< names.num(); i++) {
            XtcData::Name& name = names.get(i);
            if (strcmp(name.name(),"user.asic_enable")==0)
                m_asics = descdata.get_value<uint32_t>(name.name());
        }
    }

    return 0;
}

//
//  The timing header is in each ASIC pair batch
//

Pds::TimingHeader* EpixHR2x2::getTimingHeader(uint32_t index) const
{
    EvtBatcherHeader* ebh = static_cast<EvtBatcherHeader*>(m_pool->dmaBuffers[index]);
    ebh = reinterpret_cast<EvtBatcherHeader*>(ebh->next());
    //  This may get called multiple times, so we can't overwrite input we need
    uint32_t* p = reinterpret_cast<uint32_t*>(ebh->next());

    if (m_descramble) {
        //  The nested AxiStreamBatcherEventBuilder seems to have padded every 8B with 8B
        if (p[2]==0 && p[3]==0) {
            // A zero timestamp means the data has not been rearranged.
            for(unsigned i=1; i<5; i++) {
                p[2*i+0] = p[4*i+0];
                p[2*i+1] = p[4*i+1];
            }
        }
    }
    else {
    }
    return reinterpret_cast<Pds::TimingHeader*>(p);
}

//
//  Subframes:  3:   ASIC0/1
//                   0:  Timing
//                   2:  ASIC0/1 2B interleaved
//              4:   ASIC2/3
//                   0:  Timing
//                   2:  ASIC2/3 2B interleaved
//
void EpixHR2x2::_event(XtcData::Xtc& xtc, const void* bufEnd, std::vector< XtcData::Array<uint8_t> >& subframes)
{
    unsigned shape[MaxRank] = {0,0,0,0,0};

    //  A super row crosses 2 elements; each element contains 2x2 ASICs
    const unsigned elemRows     = 144;
    const unsigned elemRowSize  = 192;
    const unsigned timHdrSize   =  60; // timing header prepended to every ASIC segment

    //  The epix10kT unit cell is 2x2 ASICs
    CreateData cd(xtc, bufEnd, m_namesLookup, m_evtNamesId[0]);
    logging::debug("Writing panel event src 0x%x",unsigned(m_evtNamesId[0]));
    shape[0] = elemRows*2; shape[1] = elemRowSize*2;
    Array<uint16_t> aframe = cd.allocate<uint16_t>(EpixHRPanelDef::raw, shape);

    if (!m_descramble) {
        if (subframes.size()<5) {
            logging::error("Missing data: subframe size %d [5]\n",
                           subframes.size());
            xtc.damage.increase(XtcData::Damage::MissingData);
            return;
        }

        if (0) {  // debug
            uint8_t* p = reinterpret_cast<uint8_t*>(aframe.data());
            for(unsigned i=3; i<5; i++) {
                std::vector< XtcData::Array<uint8_t> > ssf =
                    _subframes(subframes[i].data(),subframes[i].shape()[0]);
                for(unsigned j=0; j<ssf.size(); j++)
                    printf("Subframe %u/%u  Num_Elem %lu\n",
                           i, j, ssf[j].num_elem());
            }
        }

        uint8_t* p = reinterpret_cast<uint8_t*>(aframe.data());
        for(unsigned i=3; i<5; i++) {
            std::vector< XtcData::Array<uint8_t> > ssf =
                _subframes(subframes[i].data(),subframes[i].shape()[0]);
            if (ssf.size()<3) {
                logging::error("Missing data: subframe[%d] size %d [3]\n",
                               i,ssf.size());
                xtc.damage.increase(XtcData::Damage::MissingData);
                return;
            }

            unsigned sz = elemRows*elemRowSize*4;

            if (ssf[2].num_elem() < sz+2*elemRowSize*4) {
                logging::error("Missing data: subframe[%d] num_elems %u [%u]\n",
                               i,ssf[2].num_elem(),sz + 2*elemRowSize*4);
                xtc.damage.increase(XtcData::Damage::MissingData);
                return;
            }

            // skip the first row, drop the last row
            memcpy(p,ssf[2].data()+elemRowSize*4,sz);
            p += sz;
        }
        return;
    }
    //  Missing ASICS are padded with zeroes
    m_asics = 0xf;

    memset(aframe.data(),0,4*elemRows*elemRowSize*2);

    logging::debug("m_asics[%d] subframes.num_elem[%d]",m_asics,subframes.size());

    //  Validate timing headers

    //
    //    A1   |   A3       (A1,A3) rotated 180deg
    // --------+--------
    //    A0   |   A2
    //

    //  Check which ASICs are in the streams
    unsigned q_asics = m_asics;
    for(unsigned q=0; q<4; q++) {
        if (q_asics & (1<<q)) {
            if (subframes.size()<(q/2+4)) {
                logging::error("Missing data from asic %d\n",q);
                xtc.damage.increase(XtcData::Damage::MissingData);
                q_asics ^= (1<<q);
            }
            else if (subframes[q/2+3].num_elem()!=2*(56076+timHdrSize)) {
                logging::error("Wrong size frame %d [%d] from asic %d\n",
                               subframes[q/2+3].num_elem()/2,56076+timHdrSize,q);
                xtc.damage.increase(XtcData::Damage::MissingData);
                q_asics ^= (1<<q);
            }
        }
    }

    char dline[5*128+32];
    //  Copy A0,A1 into the 2x2 buffer
    for(unsigned iq=0; iq<2; iq++) {
        unsigned q = iq;
        if ((q_asics & (1<<q))==0)
            continue;
        logging::debug("asic[%d] nelem[%d]",q,subframes[q/2+3].num_elem()/2);
        {
            const uint16_t* u = reinterpret_cast<const uint16_t*>(subframes[q/2+3].data());
            { for(unsigned i=0; i<128; i++)
                    sprintf(&dline[i*5]," %04x", u[i]);
                logging::debug("asic[%d] sz[%d] [0:127] %s",q,subframes[q/2+3].num_elem()/2,dline); }
            u += 192*144+timHdrSize;
            { for(unsigned i=0; i<128; i++)
                    sprintf(&dline[i*5]," %04x", u[i]);
                logging::debug("asic[%d] sz[%d] [next:127] %s",q,subframes[q/2+3].num_elem()/2,dline); }
        }
        const uint16_t* u = reinterpret_cast<const uint16_t*>(subframes[q/2+3].data());
        u += timHdrSize;
        u += 6*(iq&1);
        for(unsigned row=0, e=0; row<elemRows; row++, e+=2*elemRowSize) {
            uint16_t* dst = &aframe(row+elemRows,elemRowSize*(q>>1));
            for(unsigned m=0; m<elemRowSize; m++) {
                //  special fixup for the last two columns
                if (row > 1 && (m&0x1f) > 0x1d)
                    dst[m] = u[e+12*(m&0x1f)+(m>>5)-2*elemRowSize];
                else
                    dst[m] = u[e+12*(m&0x1f)+(m>>5)];
            }
        }
    }
    //  Copy A2,A3 into the 2x2 buffer (rotated 180)
    for(unsigned iq=0; iq<2; iq++) {
        unsigned q = iq+2;
        if ((q_asics & (1<<q))==0)
            continue;
        logging::debug("asic[%d] nelem[%d]",q,subframes[q/2+3].num_elem()/2);
        {
            const uint16_t* u = reinterpret_cast<const uint16_t*>(subframes[q/2+3].data());
            { for(unsigned i=0; i<128; i++)
                    sprintf(&dline[i*5]," %04x", u[i]);
                logging::debug("asic[%d] sz[%d] [0:127] %s",q,subframes[q/2+3].num_elem()/2,dline); }
            u += 192*144+timHdrSize;
            { for(unsigned i=0; i<128; i++)
                    sprintf(&dline[i*5]," %04x", u[i]);
                logging::debug("asic[%d] sz[%d] [next:127] %s",q,subframes[q/2+3].num_elem()/2,dline); }
        }
        const uint16_t* u = reinterpret_cast<const uint16_t*>(subframes[q/2+3].data());
        u += timHdrSize;
        u += 12;
        u += 6*(iq&1);
        for(unsigned row=0, e=0; row<elemRows; row++, e+=2*elemRowSize) {
            uint16_t* dst = &aframe(elemRows-1-row,elemRowSize*(1-(q>>1)));
            for(unsigned m=0; m<elemRowSize; m++) {
                //  special fixup for the last two columns
                if (row > 1 && (m&0x1f) > 0x1d)
                    dst[elemRowSize-1-m] = u[e+12*(m&0x1f)+(m>>5)-2*elemRowSize];
                else
                    dst[elemRowSize-1-m] = u[e+12*(m&0x1f)+(m>>5)];
            }
        }
    }
}

void     EpixHR2x2::slowupdate(XtcData::Xtc& xtc, const void* bufEnd)
{
    this->Detector::slowupdate(xtc, bufEnd);
}

bool     EpixHR2x2::scanEnabled()
{
    //  Only needed this when the TimingSystem could not be used
    //    return true;
    return false;
}

void     EpixHR2x2::shutdown()
{
}

void     EpixHR2x2::monStreamEnable()
{
    PyObject* pDict = _check(PyModule_GetDict(m_module));
    char func_name[64];
    sprintf(func_name,"%s_disable",m_para->detType.c_str());
    PyObject* pFunc = _check(PyDict_GetItemString(pDict, (char*)func_name));

    // returns new reference
    PyObject* mybytes = _check(PyObject_CallFunction(pFunc, "O", m_root));
    Py_DECREF(mybytes);
}

void     EpixHR2x2::monStreamDisable()
{
    PyObject* pDict = _check(PyModule_GetDict(m_module));
    char func_name[64];
    sprintf(func_name,"%s_enable",m_para->detType.c_str());
    PyObject* pFunc = _check(PyDict_GetItemString(pDict, (char*)func_name));

    // returns new reference
    PyObject* mybytes = _check(PyObject_CallFunction(pFunc, "O", m_root));
    Py_DECREF(mybytes);
}
