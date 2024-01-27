#include "Epix100.hh"
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

static unsigned cpocount=0;

namespace Drp {

#define ADD_FIELD(name,ntype,ndim)  NameVec.push_back({#name, Name::ntype, ndim})

    class Epix100PanelDef : public VarDef
    {
    public:
        //        enum index { raw, aux, numfields };
        enum index { raw, numfields };

        Epix100PanelDef() {
            ADD_FIELD(raw              ,UINT16,2);
            //            ADD_FIELD(aux              ,UINT16,2);
        }
    } epix100PanelDef;

    class Epix100HwDef : public VarDef
    {
    public:
        Epix100HwDef() {
            ADD_FIELD(var, UINT32, 0);
        }
    } epix100HwDef;

    class Epix100Def : public VarDef
    {
    public:
        enum index { sht31Hum, sht31TempC,
                     num_fields };

        Epix100Def() {
            ADD_FIELD(sht31Hum         ,FLOAT,0);
            ADD_FIELD(sht31TempC       ,FLOAT,0);
        }
    } epix100Def;
};

#undef ADD_FIELD

using Drp::Epix100;

static Epix100* epix = 0;

static void sigHandler(int signal)
{
  psignal(signal, "epix100 received signal");
  epix->monStreamEnable();
  ::exit(signal);
}

Epix100::Epix100(Parameters* para, MemPool* pool) :
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

Epix100::~Epix100()
{
}

void Epix100::_connectionInfo(PyObject* mbytes)
{
    m_para->serNo = _string_from_PyDict(mbytes,"serno");
}

unsigned Epix100::enable(XtcData::Xtc& xtc, const void* bufEnd, const nlohmann::json& info)
{
    // monStreamDisable();
    return 0;
}

unsigned Epix100::disable(XtcData::Xtc& xtc, const void* bufEnd, const nlohmann::json& info)
{
    // monStreamEnable();
    return 0;
}

unsigned Epix100::_configure(XtcData::Xtc& xtc, const void* bufEnd, XtcData::ConfigIter& configo)
{
    // set up the names for L1Accept data
    // Generic panel data
    {
        Alg alg("raw", 2, 0, 1);
        // copy the detName, detType, detId from the Config Names
        Names& configNames = configo.namesLookup()[NamesId(nodeId, ConfigNamesIndex)].names();
        NamesId nid = m_evtNamesId[0] = NamesId(nodeId, EventNamesIndex);
        logging::debug("Constructing panel eventNames src 0x%x ids %s",
                       unsigned(nid),configNames.detId());
        Names& eventNames = *new(xtc, bufEnd) Names(bufEnd,
                                                    configNames.detName(), alg,
                                                    configNames.detType(),
                                                    configNames.detId(),
                                                    nid,
                                                    m_para->detSegment);

        eventNames.add(xtc, bufEnd, epix100PanelDef);
        m_namesLookup[nid] = NameIndex(eventNames);
    }

    // cpo: in case we need to iterate over the config object
    // {
    //     XtcData::Names&    names    = detector::configNames(configo);
    //     XtcData::DescData& descdata = configo.desc_shape();
    //     for(unsigned i=0; i< names.num(); i++) {
    //         XtcData::Name& name = names.get(i);
    //         //if (strcmp(name.name(),"user.asic_enable")==0)
    //         //    m_asics = descdata.get_value<uint32_t>(name.name());
    //     }
    // }

    return 0;
}

//
//  Subframes:  0:   Event header
//              2:   Timing frame detailed
//              3:   epix100
//
void Epix100::_event(XtcData::Xtc& xtc, const void* bufEnd, std::vector< XtcData::Array<uint8_t> >& subframes)
{
    unsigned shape[MaxRank] = {0,0,0,0,0};

    // data structure in lcls1's psddldata/data/epix.ddl::ElementV3
    // epix100 has 32 byte header, frame data, 2 environmental rows,
    // 2 calibration rows, and 12 bytes on the end (4 2-byte
    // temperatures, and a 4 byte trailer)
    const unsigned headersize  = 32;
    const unsigned nrows = 704;
    const unsigned ncols = 768;
    const unsigned nenvironmentalrows = 2;
    const unsigned ncalibrationrows   = 2;
    const unsigned trailersize = 12;
    const unsigned nasicrows = nrows/2;
    const unsigned expectedsize = headersize+(nrows+nenvironmentalrows+ncalibrationrows)*ncols*sizeof(uint16_t)+trailersize;

    CreateData cd(xtc, bufEnd, m_namesLookup, m_evtNamesId[0]);
    logging::debug("Writing panel event src 0x%x",unsigned(m_evtNamesId[0]));
    shape[0] = nrows; shape[1] = ncols;
    Array<uint16_t> aframe = cd.allocate<uint16_t>(Epix100PanelDef::raw, shape);

    cpocount++;
    if (cpocount%100==0) {
        printf("*** event %d subframes.size: %zd\n",cpocount,subframes.size());
        for (unsigned i=0; i<subframes.size(); i++) {
            printf("*** subframes[%d].num_elem(): %zd\n",i,subframes[i].num_elem());
        }
    }

    if (subframes[3].num_elem() != expectedsize) {
        printf("*** incorrect size %zd %d\n",subframes[3].num_elem(),expectedsize);
        // raise damage here
    }

    uint16_t* indata = (uint16_t*)(subframes[3].data()+headersize);
    // unscramble asics like lcls1's pds/epix100a/Epix100aServer.cc
    for(unsigned i=0; i<nasicrows; i++) {
        memcpy(aframe.data()+(nasicrows+i+0)*ncols, indata+(2*i+0)*ncols, ncols*sizeof(uint16_t));
        memcpy(aframe.data()+(nasicrows-i-1)*ncols, indata+(2*i+1)*ncols, ncols*sizeof(uint16_t));
    }

    // feels like we need to unscramble environmental/calibration rows too?
}

void     Epix100::slowupdate(XtcData::Xtc& xtc, const void* bufEnd)
{
  this->Detector::slowupdate(xtc, bufEnd);
}

bool     Epix100::scanEnabled()
{
    //  Only needed this when the TimingSystem could not be used
    //    return true;
    return false;
}

void     Epix100::shutdown()
{
}

void     Epix100::monStreamEnable()
{
    PyObject* pDict = _check(PyModule_GetDict(m_module));
    char func_name[64];
    sprintf(func_name,"%s_disable",m_para->detType.c_str());
    PyObject* pFunc = _check(PyDict_GetItemString(pDict, (char*)func_name));

    // returns new reference
    PyObject* mybytes = _check(PyObject_CallFunction(pFunc, "O", m_root));
    Py_DECREF(mybytes);
}

void     Epix100::monStreamDisable()
{
    PyObject* pDict = _check(PyModule_GetDict(m_module));
    char func_name[64];
    sprintf(func_name,"%s_enable",m_para->detType.c_str());
    PyObject* pFunc = _check(PyDict_GetItemString(pDict, (char*)func_name));

    // returns new reference
    PyObject* mybytes = _check(PyObject_CallFunction(pFunc, "O", m_root));
    Py_DECREF(mybytes);
}
