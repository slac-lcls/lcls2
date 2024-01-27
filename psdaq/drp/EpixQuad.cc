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

//#define INCLUDE_ENV
//#define SLOW_UPDATE_ENV

namespace Drp {

#define ADD_FIELD(name,ntype,ndim)  NameVec.push_back({#name, Name::ntype, ndim})

    class EpixPanelDef : public VarDef
    {
    public:
        enum index { raw, aux, numfields };

        EpixPanelDef() {
            ADD_FIELD(raw              ,UINT16,2);
            ADD_FIELD(aux              ,UINT16,2);
        }
    } epixPanelDef;

    class EpixQuadHwDef : public VarDef
    {
    public:
        EpixQuadHwDef() {
            ADD_FIELD(var, UINT32, 0);
        }
    } epixQuadHwDef;

    class EpixQuadDef : public VarDef
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

        EpixQuadDef() {
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
    } epixQuadDef;
};

#undef ADD_FIELD

using Drp::EpixQuad;

EpixQuad::EpixQuad(Parameters* para, MemPool* pool) :
    BEBDetector   (para, pool),
    m_env_sem     (Pds::Semaphore::FULL),
    m_env_empty   (true)
{
    virtChan = 0;

    _init(para->detName.c_str());  // an argument is required here
}

EpixQuad::~EpixQuad()
{
}

void EpixQuad::_connectionInfo(PyObject* mbytes)
{
    m_para->serNo = _string_from_PyDict(mbytes,"serno");
}

unsigned EpixQuad::enable(XtcData::Xtc& xtc, const void* bufEnd, const nlohmann::json& info)
{
    _monStreamDisable();
    return 0;
}

unsigned EpixQuad::disable(XtcData::Xtc& xtc, const void* bufEnd, const nlohmann::json& info)
{
    _monStreamEnable();
    return 0;
}

unsigned EpixQuad::_configure(XtcData::Xtc& xtc, const void* bufEnd, XtcData::ConfigIter& configo)
{
    // set up the names for L1Accept data
    unsigned seg=0;
    for (unsigned q=0; q < 4; q++) {
        // Generic panel data
        {
            Alg alg("raw", 2, 0, 1);
            // copy the detName, detType, detId from the Config Names
            Names& configNames = configo.namesLookup()[NamesId(nodeId, ConfigNamesIndex+seg+1)].names();
            NamesId nid = m_evtNamesId[seg] = NamesId(nodeId, EventNamesIndex+seg);
            logging::debug("Constructing panel eventNames src 0x%x",
                           unsigned(nid));
            Names& eventNames = *new(xtc, bufEnd) Names(bufEnd,
                                                        configNames.detName(), alg,
                                                        configNames.detType(),
                                                        configNames.detId(),
                                                        nid,
                                                        q+4*m_para->detSegment);

            eventNames.add(xtc, bufEnd, epixPanelDef);
            m_namesLookup[nid] = NameIndex(eventNames);
        }
#ifdef INCLUDE_ENV
        // Quad-specific data
        {
            Alg alg("raw", 2, 0, 0);
            // copy the detName, detType, detId from the Config Names
            Names& configNames = configo.namesLookup()[NamesId(nodeId, ConfigNamesIndex)].names();
            NamesId nid = m_evtNamesId[seg+4] = NamesId(nodeId, EventNamesIndex+seg+4);
            logging::debug("Constructing panel eventNames src 0x%x",
                           unsigned(nid));
            Names& eventNames = *new(xtc, bufEnd) Names(bufEnd,
                                                        configNames.detName(), alg,
                                                        configNames.detType(),
                                                        configNames.detId(),
                                                        nid,
                                                        q+4*m_para->detSegment);

            eventNames.add(xtc, bufEnd, epixQuadDef);
            m_namesLookup[nid] = NameIndex(eventNames);
        }
#endif
        seg++;
    }

    return 0;
}

#ifdef INCLUDE_ENV
static float _getThermistorTemp(uint16_t x)
{
    float tthermk = 0.;
    if (x) {
        float umeas = float(x)*2.5/16383.;
        float itherm = umeas/100000.;
        float rtherm = (2.5-umeas)/itherm;
        if (rtherm>0) {
            float lnrtr25 = log(rtherm/10000.);
            tthermk = 1.0 / (3.3538646E-03 + 2.5654090E-04 * lnrtr25 + 1.9243889E-06 * (lnrtr25*lnrtr25) + 1.0969244E-07 * (lnrtr25*lnrtr25*lnrtr25));
            tthermk -= 273.15;
        }
    }
    return 0.;
}
#endif

void EpixQuad::_event(XtcData::Xtc& xtc, const void* bufEnd, std::vector< XtcData::Array<uint8_t> >& subframes)
{
    unsigned shape[MaxRank] = {0,0,0,0,0};

    //  A super row crosses 2 elements; each element contains 2x2 ASICs
    const unsigned asicRows     = 176;
    const unsigned elemRowSize  = 2*192;

#ifdef SLOW_UPDATE_ENV
    bool env_empty = m_env_empty;
    if (env_empty) {
        m_env_sem.take();
        transitionXtc().extent = sizeof(Xtc);
    }
#endif

    unsigned seg=0;
    for(unsigned q=0; q<4; q++) {
        CreateData cd(xtc, bufEnd, m_namesLookup, m_evtNamesId[seg]);
        logging::debug("Writing panel event src 0x%x",unsigned(m_evtNamesId[seg]));
        shape[0] = asicRows*2; shape[1] = elemRowSize;
        Array<uint16_t> aframe = cd.allocate<uint16_t>(EpixPanelDef::raw, shape);

        const uint16_t* u = reinterpret_cast<const uint16_t*>(subframes[2].data()) + 16;

#define MMCPY(a,el,row,src,sz) {                \
            if (q==el) {                        \
                uint16_t* dst = &a(row,0);      \
                for(unsigned k=0; k<sz; k++) {  \
                    dst[sz-1-k] = src[k];       \
                }                               \
            }                                   \
            src += sz;                          \
        }

        // Frame data
        for(unsigned i=0; i<asicRows; i++) { // 4 super rows at a time
            unsigned dnRow = asicRows+i;
            unsigned upRow = asicRows-i-1;

            MMCPY(aframe, 2, upRow, u, elemRowSize);
            MMCPY(aframe, 3, upRow, u, elemRowSize);
            MMCPY(aframe, 2, dnRow, u, elemRowSize);
            MMCPY(aframe, 3, dnRow, u, elemRowSize);
            MMCPY(aframe, 0, upRow, u, elemRowSize);
            MMCPY(aframe, 1, upRow, u, elemRowSize);
            MMCPY(aframe, 0, dnRow, u, elemRowSize);
            MMCPY(aframe, 1, dnRow, u, elemRowSize);
        }

        // Calibration rows
        const unsigned calibRows = 4;
        shape[0] = calibRows; shape[1] = elemRowSize;
        Array<uint16_t> acalib = cd.allocate<uint16_t>(EpixPanelDef::aux, shape);

        for(unsigned i=0; i<calibRows; i++) {
            MMCPY(acalib, 2, i, u, elemRowSize);
            MMCPY(acalib, 3, i, u, elemRowSize);
            MMCPY(acalib, 0, i, u, elemRowSize);
            MMCPY(acalib, 1, i, u, elemRowSize);
        }

#ifdef INCLUDE_ENV
#define ADD_FIELD(name,ntype,val) qcd.set_value<ntype>(EpixQuadDef::name, val)
#ifdef SLOW_UPDATE_ENV
        if (env_empty) {
            CreateData qcd(transitionXtc(), trXtcBufEnd(), m_namesLookup, m_evtNamesId[seg+4]);
#else
        if (1) {
            CreateData qcd(xtc, bufEnd, m_namesLookup, m_evtNamesId[seg+4]);
#endif
            logging::debug("Writing quad event src 0x%x",unsigned(m_evtNamesId[seg+4]));
            const uint8_t* u8 = reinterpret_cast<const uint8_t*>(u);
            ADD_FIELD(sht31Hum         ,float_t, (float(u[0])*100./65535.));
            ADD_FIELD(sht31TempC       ,float_t, (float(u[1])*175./65535.-45.));
            ADD_FIELD(nctLocTempC      ,uint16_t, u[2]);
            ADD_FIELD(nctFpgaTempC     ,float_t, (float(u8[7])+float(u8[6]>>6)*0.25));
            ADD_FIELD(asicA0_2V5_CurrmA,float_t, (float(u[4])*2.5e6/(16383.*330.)));
            ADD_FIELD(asicA1_2V5_CurrmA,float_t, (float(u[5])*2.5e6/(16383.*330.)));
            ADD_FIELD(asicA2_2V5_CurrmA,float_t, (float(u[6])*2.5e6/(16383.*330.)));
            ADD_FIELD(asicA3_2V5_CurrmA,float_t, (float(u[7])*2.5e6/(16383.*330.)));
            ADD_FIELD(asicD0_2V5_CurrmA,float_t, (float(u[8])*2.5e6/(16383.*330.*2.)));
            ADD_FIELD(asicD1_2V5_CurrmA,float_t, (float(u[9])*2.5e6/(16383.*330.*2.)));
            ADD_FIELD(therm0TempC      ,float_t, _getThermistorTemp(u[10]));
            ADD_FIELD(therm1TempC      ,float_t, _getThermistorTemp(u[11]));
            ADD_FIELD(pwrDigCurr       ,float_t, (float(u[12])*0.1024/(4095.*0.02)));
            ADD_FIELD(pwrDigVin        ,float_t, (float(u[13])*102.4/4095.));
            ADD_FIELD(pwrDigTempC      ,float_t, (float(u[14])*2.048/4095.*(130./(0.882-1.951)) + 0.882/0.0082 + 100));
            ADD_FIELD(pwrAnaCurr       ,float_t, (float(u[15])*0.1024/(4095.*0.02)));
            ADD_FIELD(pwrAnaVin        ,float_t, (float(u[16])*102.4/4095.));
            ADD_FIELD(pwrAnaTempC      ,float_t, (float(u[17])*2.048/4095.*(130./(0.882-1.951)) + 0.882/0.0082 + 100));

            u += 2*elemRowSize;

            // Temperatures
            const unsigned tempSize = 4;
            u += tempSize*q;
            shape[0] = tempSize; shape[1] = 0;
            Array<uint16_t> atemp = qcd.allocate<uint16_t>(EpixQuadDef::asic_temp, shape);
            for(unsigned i=0; i<tempSize; i++)
                atemp(tempSize-i-1) = *u++;
        }
#endif
        seg++;
    }

#ifdef SLOW_UPDATE_ENV
    if (env_empty) {
        m_env_empty = false;
        m_env_sem.give();
    }
#endif

}

void     EpixQuad::slowupdate(XtcData::Xtc& xtc, const void* bufEnd)
{
#ifdef SLOW_UPDATE_ENV
    m_env_sem.take();
    XtcData::Xtc& trXtc = transitionXtc();
    xtc = trXtc; // Preserve header info, but allocate to check fit
    auto payload = xtc.alloc(trXtc.sizeofPayload(), bufEnd);
    memcpy(payload, (const void*)trXtc.payload(), trXtc.sizeofPayload());
    m_env_empty = true;
    m_env_sem.give();
#else
    this->Detector::slowupdate(xtc, bufEnd);
#endif
}

bool     EpixQuad::scanEnabled()
{
    //  Only needed this when the TimingSystem could not be used
    //    return true;
    return false;
}

void     EpixQuad::shutdown()
{
}

void     EpixQuad::_monStreamEnable()
{
    PyObject* pDict = _check(PyModule_GetDict(m_module));
    char func_name[64];
    sprintf(func_name,"%s_disable",m_para->detType.c_str());
    PyObject* pFunc = _check(PyDict_GetItemString(pDict, (char*)func_name));

    // returns new reference
    PyObject* mybytes = _check(PyObject_CallFunction(pFunc, "O", m_root));
    Py_DECREF(mybytes);
}

void     EpixQuad::_monStreamDisable()
{
    PyObject* pDict = _check(PyModule_GetDict(m_module));
    char func_name[64];
    sprintf(func_name,"%s_enable",m_para->detType.c_str());
    PyObject* pFunc = _check(PyDict_GetItemString(pDict, (char*)func_name));

    // returns new reference
    PyObject* mybytes = _check(PyObject_CallFunction(pFunc, "O", m_root));
    Py_DECREF(mybytes);
}
