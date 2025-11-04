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
        enum { HeaderSize  = 48 };
        enum { TrailerSize = 48 };
        enum index { header, raw, trailer, numfields };

        EpixMPanelDef() {
            ADD_FIELD(header,  UINT8,  2);
            ADD_FIELD(raw,     UINT16, 3);
            ADD_FIELD(trailer, UINT8,  2);
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
static struct sigaction old_actions[64];

static void sigHandler(int signal)
{
  psignal(signal, "epixm320 received signal");
  epix->monStreamEnable();

  sigaction(signal,&old_actions[signal],NULL);
  raise(signal);
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
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_RESETHAND;

#define REGISTER(t) {                               \
        if (sigaction(t, &sa, &old_actions[t]) > 0) \
            printf("Couldn't set up #t handler\n"); \
    }

    REGISTER(SIGINT);
    REGISTER(SIGABRT);
    REGISTER(SIGKILL);
    REGISTER(SIGSEGV);

#undef REGISTER
}

EpixM320::~EpixM320()
{
}

void EpixM320::_connectionInfo(PyObject* mbytes)
{
    m_para->serNo = _string_from_PyDict(mbytes,"serno");
}

unsigned EpixM320::enable(Xtc& xtc, const void* bufEnd, const nlohmann::json& info)
{
    logging::debug("EpixM320 enable");
    monStreamDisable();
    return 0;
}

unsigned EpixM320::disable(Xtc& xtc, const void* bufEnd, const nlohmann::json& info)
{
    logging::debug("EpixM320 disable");
    monStreamEnable();
    return 0;
}

unsigned EpixM320::_configure(Xtc& xtc, const void* bufEnd, ConfigIter& configo)
{
    // copy the detName, detType, detId from the Config Names
    Names& configNames = configo.namesLookup()[NamesId(nodeId, ConfigNamesIndex+1)].names();

    // set up the names for L1Accept data
    // Generic panel data
    {
        Alg alg("raw", 0, 1, 0);
        m_evtNamesId[0] = NamesId(nodeId, EventNamesIndex);
        logging::debug("Constructing panel eventNames src 0x%x",
                       unsigned(m_evtNamesId[0]));
        Names& names = *new(xtc, bufEnd) Names(bufEnd,
                                               configNames.detName(), alg,
                                               configNames.detType(), configNames.detId(), m_evtNamesId[0], m_para->detSegment);
        names.add(xtc, bufEnd, epixMPanelDef);
        m_namesLookup[m_evtNamesId[0]] = NameIndex(names);
    }

    {
        Names&    names    = detector::configNames(configo);
        DescData& descdata = configo.desc_shape();
        for(unsigned i=0; i< names.num(); i++) {
            Name& name = names.get(i);
            if (strcmp(name.name(),"user.asic_enable")==0)
                m_asics = descdata.get_value<uint32_t>(name.name());
        }
    }

    // Rearm logging of certain messages
    m_logOnce = true;

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
        //  Padding has been removed as of 3/25/24
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
//  Subframes:  2:   ASIC0
//                   0:  Timing
//                   2:  ASIC0 2B interleaved
//              3:   ASIC1
//                   0:  Timing
//                   2:  ASIC1 2B interleaved
//              4:   ASIC2
//                   0:  Timing
//                   2:  ASIC2 2B interleaved
//              5:   ASIC3
//                   0:  Timing
//                   2:  ASIC3 2B interleaved
//
void EpixM320::_event(Xtc& xtc, const void* bufEnd, uint64_t l1count, std::vector< Array<uint8_t> >& subframes)
{
    //  A super row crosses 2 elements; each element contains 2x2 ASICs
    const size_t headerSize  = EpixMPanelDef::HeaderSize;
    const auto   asicSize    = ElemRows*ElemRowSize*sizeof(uint16_t);
    const size_t trailerSize = EpixMPanelDef::TrailerSize;

    CreateData cd(xtc, bufEnd, m_namesLookup, m_evtNamesId[0]);
    logging::debug("Writing panel event src 0x%x",unsigned(m_evtNamesId[0]));
    unsigned hShape[MaxRank] = { NumAsics, 48, 0,0,0 };
    unsigned fShape[MaxRank] = { NumAsics, ElemRows, ElemRowSize, 0,0 };
    unsigned tShape[MaxRank] = { NumAsics, 48, 0,0,0 };
    // Maintain this order to keep these contiguous:
    Array<uint8_t>  aHeader  = cd.allocate<uint8_t> (EpixMPanelDef::header,  hShape);
    Array<uint16_t> aFrame   = cd.allocate<uint16_t>(EpixMPanelDef::raw,     fShape);
    Array<uint8_t>  aTrailer = cd.allocate<uint8_t> (EpixMPanelDef::trailer, tShape);

    unsigned nSubframes = __builtin_popcount(m_asics)+2;
    if (subframes.size() < nSubframes) {
        logging::error("Missing data: subframe size %d vs %d expected\n",
                        subframes.size(), nSubframes);
        xtc.damage.increase(Damage::MissingData);
        return;
    } else if (subframes.size() > nSubframes) {
        if (m_logOnce) {
            logging::warning("Ignoring extra data: subframe size %d vs %d expected\n",
                             subframes.size(), nSubframes);
            m_logOnce = false;
        }
    }

    logging::debug("m_asics[%d] subframes.num_elem[%d]",m_asics,subframes.size());

    //  Validate timing headers

    //
    //    A1   |   A3
    // --------+--------
    //    A0   |   A2
    //

    //  Check which ASICs are in the streams
    unsigned ia = 2;                    // Point to the first ASIC subframe
    unsigned q_asics = m_asics;
    for(unsigned q=0; q<NumAsics; q++) {
        if (q_asics & (1<<q)) {
            if (subframes.size() < ia) {
                logging::error("Missing data from asic %d\n", q);
                xtc.damage.increase(Damage::MissingData);
                q_asics ^= (1<<q);
            }
            else if (subframes[ia].num_elem() != headerSize+asicSize+trailerSize) {
                logging::error("Wrong size frame %d [%d] from asic %d\n",
                               subframes[ia].num_elem()/2, (headerSize+asicSize+trailerSize)/2, q);
                xtc.damage.increase(Damage::MissingData);
                q_asics ^= (1<<q);
            }
            ++ia;                       // On to the next ASIC in the subframes
        }
    }

    ia = 2;                             // Point to the first ASIC subframes
    for (unsigned q = 0; q < NumAsics; ++q) {
        if ((q_asics & (1<<q))==0) {
            //  Missing ASICS are padded with zeroes
            memset(&aHeader (q, 0),    0, headerSize);
            memset(&aFrame  (q, 0, 0), 0, asicSize);
            memset(&aTrailer(q, 0),    0, trailerSize);
            continue;
        }

        // Pack the header into the Xtc
        auto header = subframes[ia].data();
        memcpy(&aHeader(q, 0), header, headerSize);

        // Pack the image data into the Xtc
        auto src = reinterpret_cast<const uint16_t*>(subframes[ia].data() + headerSize);
        auto dst = &aFrame(q, 0, 0);
#if 1
        _descramble(dst, src);
#else
        memcpy(dst, src, asicSize);
#endif

        // Pack the trailer into the Xtc
        auto trailer = subframes[ia].data() + headerSize + asicSize;
        memcpy(&aTrailer(q, 0), trailer, trailerSize);

        ++ia;                           // On to the next ASIC in the subframes
    }
}

void     EpixM320::_descramble(uint16_t* dst, const uint16_t* src) const
{
    // Reorder banks from:                 to:
    // 18    19    20    21    22    23        3     7    11    15    19    23
    // 12    13    14    15    16    17        2     6    10    14    18    22
    //  6     7     8     9    10    11        1     5     9    13    17    21
    //  0     1     2     3     4     5        0     4     8    12    16    20

    const unsigned bankHeight = ElemRows / BankRows;
    const unsigned bankWidth  = ElemRowSize / BankCols;
    const unsigned hw         = bankWidth / 2;
    const unsigned hb         = bankHeight * hw;

    for (unsigned bankRow = 0; bankRow < BankRows; ++bankRow) {
        for (unsigned row = 0; row < bankHeight; ++row) {
            auto rowFix = row == 0 ? bankHeight - 1 : row - 1; // Shift rows up by one for ASIC f/w bug
            for (unsigned bankCol = 0; bankCol < BankCols; ++bankCol) {
                unsigned bank = BankRows * bankCol + bankRow;  // Given (column, row), reorder banks
                for (unsigned col = 0; col < bankWidth; ++col) {
                    //          (even cols w/ offset + row offset  + inc every 2 cols) * fill one pixel / bank + bank inc
                    auto idx = (((col+1) % 2) * hb   + hw * rowFix + int(col / 2))     * NumBanks              + bank;
                    *dst++ = src[idx];
                }
            }
        }
    }
}

void     EpixM320::slowupdate(Xtc& xtc, const void* bufEnd)
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
