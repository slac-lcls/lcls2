#include "EpixM320.hh"
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

    class EpixMPanelDef : public VarDef
    {
    public:
        enum index { raw, numfields };

        EpixMPanelDef() {
            ADD_FIELD(raw, UINT16, 3);
        }
    } epixMPanelDef;

    class EpixMHwDef : public VarDef
    {
    public:
        EpixMHwDef() {
            ADD_FIELD(var, UINT32, 0);
        }
    } epixMHwDef;

    class EpixMDef : public VarDef
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

        EpixMDef() {
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
    } epixMDef;
};

#undef ADD_FIELD

using Drp::EpixM320;

static EpixM320* epix = 0;

static void sigHandler(int signal)
{
  psignal(signal, "epixm320 received signal");
  epix->monStreamEnable();
  ::exit(signal);
}

EpixM320::EpixM320(Parameters* para, MemPool* pool) :
    BEBDetector   (para, pool),
    m_env_sem     (Pds::Semaphore::FULL),
    m_env_empty   (true)
{
    // VC 0 is used with XilinxKcu1500Pgp4_10Gbps
    // VC 1 is used with Lcls2EpixHrXilinxKcu1500Pgp4_10Gbps, which is the default
    virtChan = 0;

    _init(para->detName.c_str());  // an argument is required here

    m_descramble = true;

    epix = this;

    struct sigaction sa;
    sa.sa_handler = sigHandler;
    sa.sa_flags = SA_RESETHAND;

    sigaction(SIGINT ,&sa,NULL);
    sigaction(SIGABRT,&sa,NULL);
    sigaction(SIGKILL,&sa,NULL);
    sigaction(SIGSEGV,&sa,NULL);
}

EpixM320::~EpixM320()
{
}

void EpixM320::_connectionInfo(PyObject* mbytes)
{
    m_para->serNo = _string_from_PyDict(mbytes,"serno");
}

unsigned EpixM320::enable(XtcData::Xtc& xtc, const void* bufEnd, const nlohmann::json& info)
{
    logging::debug("EpixM320 enable");
    monStreamDisable();
    return 0;
}

unsigned EpixM320::disable(XtcData::Xtc& xtc, const void* bufEnd, const nlohmann::json& info)
{
    logging::debug("EpixM320 disable");
    monStreamEnable();
    return 0;
}

unsigned EpixM320::_configure(XtcData::Xtc& xtc, const void* bufEnd, XtcData::ConfigIter& configo)
{
    // set up the names for L1Accept data
    // Generic panel data
    {
        Alg alg("raw", 0, 0, 0);
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

        eventNames.add(xtc, bufEnd, epixMPanelDef);
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

Pds::TimingHeader* EpixM320::getTimingHeader(uint32_t index) const
{
    EvtBatcherHeader* ebh = static_cast<EvtBatcherHeader*>(m_pool->dmaBuffers[index]);
    ebh = reinterpret_cast<EvtBatcherHeader*>(ebh->next());
    //  This may get called multiple times, so we can't overwrite input we need
    uint32_t* p = reinterpret_cast<uint32_t*>(ebh);

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
        // Descrambling will be done in firmware
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
void EpixM320::_event(XtcData::Xtc& xtc, const void* bufEnd, std::vector< XtcData::Array<uint8_t> >& subframes)
{
    //  A super row crosses 2 elements; each element contains 2x2 ASICs
    const unsigned numAsics    = 4;
    const unsigned elemRows    = 192;
    const unsigned elemRowSize = 384;
    const size_t   headerSize  = 24;

    //  The epix10kT unit cell is 2x2 ASICs
    CreateData cd(xtc, bufEnd, m_namesLookup, m_evtNamesId[0]);
    logging::debug("Writing panel event src 0x%x",unsigned(m_evtNamesId[0]));
    unsigned shape[MaxRank] = { numAsics, elemRows, elemRowSize, 0,0 };
    Array<uint16_t> aframe = cd.allocate<uint16_t>(EpixMPanelDef::raw, shape);

    if (subframes.size() != 6) {
        logging::error("Missing data: subframe size %d vs 6 expected\n",
                       subframes.size());
        xtc.damage.increase(XtcData::Damage::MissingData);
        return;
    }

    logging::debug("m_asics[%d] subframes.num_elem[%d]",m_asics,subframes.size());

    //  Validate timing headers

    //
    //    A1   |   A3
    // --------+--------
    //    A0   |   A2
    //

    //  Missing ASICS are padded with zeroes
    const unsigned asicSize = elemRows*elemRowSize;
    memset(aframe.data(), 0, numAsics*asicSize*sizeof(uint16_t));

    //  Check which ASICs are in the streams
    unsigned q_asics = m_asics;
    for(unsigned q=0; q<numAsics; q++) {
        if (q_asics & (1<<q)) {
            if (subframes.size() < (q+2)) {
                logging::error("Missing data from asic %d\n", q);
                xtc.damage.increase(XtcData::Damage::MissingData);
                q_asics ^= (1<<q);
            }
            else if (subframes[q+2].num_elem() != 2*(asicSize+headerSize)) {
                logging::error("Wrong size frame %d [%d] from asic %d\n",
                               subframes[q+2].num_elem()/2, asicSize+headerSize, q);
                xtc.damage.increase(XtcData::Damage::MissingData);
                q_asics ^= (1<<q);
            }
        }
    }

    // Descramble the data
#if 1
    // Reorder banks from:                 to:
    // 18    19    20    21    22    23        3     7    11    15    19    23
    // 12    13    14    15    16    17        2     6    10    14    18    22
    //  6     7     8     9    10    11        1     5     9    13    17    21
    //  0     1     2     3     4     5        0     4     8    12    16    20

    //const unsigned asicOrder[] = {0, 2, 1, 3};
    const unsigned numBanks    = 24;
    const unsigned bankRows    = 4;
    const unsigned bankCols    = 6;
    const unsigned bankHeight  = elemRows / bankRows;
    const unsigned bankWidth   = elemRowSize / bankCols;
    const unsigned hw          = bankWidth / 2;
    const unsigned hb          = bankHeight * hw;

    for (unsigned q = 0; q < numAsics; ++q) {
        if ((q_asics & (1<<q))==0)
            continue;
        //auto src = reinterpret_cast<const uint16_t*>(subframes[2 + asicOrder[q]].data()) + headerSize;
        auto src = reinterpret_cast<const uint16_t*>(subframes[2 + q].data()) + headerSize;
        for (unsigned bankRow = 0; bankRow < bankRows; ++bankRow) {
            for (unsigned r = 0; r < bankHeight; ++r) {
                // ASIC firmware bug: Rows are shifted up by one in ring buffer fashion
                auto row  = r == 0 ? bankHeight - 1 : r - 1; // Compensate
                auto dst  = &aframe(q, bankRow * bankHeight + r, 0);
                for (unsigned bankCol = 0; bankCol < bankCols; ++bankCol) {
                    unsigned bank = bankWidth * bankCol + bankRow;
                    for (unsigned col = 0; col < bankWidth; ++col) {
                        //          (even cols w/ offset + row offset + inc every 2 cols) * fill one pixel / bank + bank inc
                        auto idx = (((col+1) % 2) * hb   +  hw * row  + int(col / 2))     * numBanks              + bank;
                        *dst++ = src[idx];
                    }
                }
            }
        }
    }
#else
    auto frame = aframe.data();
    for (unsigned asic = 0; asic < numAsics; ++asic) {
        auto src = reinterpret_cast<const uint16_t*>(subframes[2 + asic].data()) + headerSize;
        auto dst = &frame[asic * asicSize];
        memcpy(dst, src, elemRows*elemRowSize*sizeof(uint16_t));
    }
#endif
}

void     EpixM320::slowupdate(XtcData::Xtc& xtc, const void* bufEnd)
{
    this->Detector::slowupdate(xtc, bufEnd);
}

bool     EpixM320::scanEnabled()
{
    //  Only needed this when the TimingSystem could not be used
    //    return true;
    return false;
}

void     EpixM320::shutdown()
{
}

void     EpixM320::monStreamEnable()
{
    PyObject* pDict = _check(PyModule_GetDict(m_module));
    char func_name[64];
    sprintf(func_name,"%s_disable",m_para->detType.c_str());
    PyObject* pFunc = _check(PyDict_GetItemString(pDict, (char*)func_name));

    // returns new reference
    PyObject* mybytes = _check(PyObject_CallFunction(pFunc, "O", m_root));
    Py_DECREF(mybytes);
}

void     EpixM320::monStreamDisable()
{
    PyObject* pDict = _check(PyModule_GetDict(m_module));
    char func_name[64];
    sprintf(func_name,"%s_enable",m_para->detType.c_str());
    PyObject* pFunc = _check(PyDict_GetItemString(pDict, (char*)func_name));

    // returns new reference
    PyObject* mybytes = _check(PyObject_CallFunction(pFunc, "O", m_root));
    Py_DECREF(mybytes);
}
