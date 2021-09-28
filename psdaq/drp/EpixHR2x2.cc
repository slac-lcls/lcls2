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

void EpixHR2x2::_connect(PyObject* mbytes)
{
    m_para->serNo = _string_from_PyDict(mbytes,"serno");
}

unsigned EpixHR2x2::enable(XtcData::Xtc& xtc, const nlohmann::json& info)
{
    monStreamDisable();
    return 0;
}

unsigned EpixHR2x2::disable(XtcData::Xtc& xtc, const nlohmann::json& info)
{
    monStreamEnable();
    return 0;
}

json EpixHR2x2::connectionInfo()
{
    // Exclude connection info until lcls2-epix-hr-pcie timingTxLink is fixed
    logging::error("Returning NO XPM link; implementation incomplete");
    return json({});

    return BEBDetector::connectionInfo();
}

unsigned EpixHR2x2::_configure(XtcData::Xtc& xtc,XtcData::ConfigIter& configo)
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
        Names& eventNames = *new(xtc) Names(configNames.detName(), alg, 
                                            configNames.detType(),
                                            configNames.detId(), 
                                            nid,
                                            m_para->detSegment);
            
        eventNames.add(xtc, epixHRPanelDef);
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

//
//  Subframes:  0:   Event header
//              2:   Timing frame detailed
//              3-6: ASIC[0:3]
//
void EpixHR2x2::_event(XtcData::Xtc& xtc, std::vector< XtcData::Array<uint8_t> >& subframes)
{
    unsigned shape[MaxRank] = {0,0,0,0,0};
  
    //  A super row crosses 2 elements; each element contains 2x2 ASICs
    const unsigned elemRows     = 144;
    const unsigned elemRowSize  = 192;

    //  The epix10kT unit cell is 2x2 ASICs
    CreateData cd(xtc, m_namesLookup, m_evtNamesId[0]);
    logging::debug("Writing panel event src 0x%x",unsigned(m_evtNamesId[0]));
    shape[0] = elemRows*2; shape[1] = elemRowSize*2;
    Array<uint16_t> aframe = cd.allocate<uint16_t>(EpixHRPanelDef::raw, shape);
    memset(aframe.data(),0,4*elemRows*elemRowSize*2);
    
    //
    //    A1   |   A3       (A1,A3) rotated 180deg
    // --------+--------
    //    A0   |   A2
    //
    char dline[280];
    for(unsigned q=0; (q+3)<subframes.size(); q+=2) {
        logging::debug("asic[%d] nelem[%d]",q,subframes[q+3].num_elem());
        if (!subframes[q+3].num_elem()) {
            if (m_asics & q)
                xtc.damage.increase(XtcData::Damage::MissingData);
            continue;
        }
        const uint16_t* u = reinterpret_cast<const uint16_t*>(subframes[q+3].data());
        for(unsigned i=0; i<32; i++)
            sprintf(&dline[i*5]," %04x", u[i+19200]);
        logging::debug("asic[%d] sz[%d] %s",q,subframes[q+3].num_elem(),dline);
        u += 6;
        for(unsigned row=0, e=0; row<elemRows; row++, e+=elemRowSize) {
            uint16_t* dst = &aframe(row+elemRows,elemRowSize*(q>>1));  
            for(unsigned m=0; m<elemRowSize; m++) {
                //  special fixup for the last two columns
                if (row > 1 && (m&0x1f) > 0x1d)
                    dst[m] = u[e+6*(m&0x1f)+(m>>5)-elemRowSize];
                else
                    dst[m] = u[e+6*(m&0x1f)+(m>>5)];
            }
        }
    }
    for(unsigned q=1; (q+3)<subframes.size(); q+=2) {
        logging::debug("asic[%d] nelem[%d]",q,subframes[q+3].num_elem());
        if (!subframes[q+3].num_elem()) {
            if (m_asics & q)
                xtc.damage.increase(XtcData::Damage::MissingData);
            continue;
        }
        const uint16_t* u = reinterpret_cast<const uint16_t*>(subframes[q+3].data());
        u += 6;
        for(unsigned row=0, e=0; row<elemRows; row++, e+=elemRowSize) {
            uint16_t* dst = &aframe(elemRows-1-row,elemRowSize*(1-(q>>1)));
            for(unsigned m=0; m<elemRowSize; m++) {
                //  special fixup for the last two columns
                if (row > 1 && (m&0x1f) > 0x1d)
                    dst[elemRowSize-1-m] = u[e+6*(m&0x1f)+(m>>5)-elemRowSize];
                else
                    dst[elemRowSize-1-m] = u[e+6*(m&0x1f)+(m>>5)];
            }
        }
    }
}

void     EpixHR2x2::slowupdate(XtcData::Xtc& xtc)
{
    this->Detector::slowupdate(xtc);
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
