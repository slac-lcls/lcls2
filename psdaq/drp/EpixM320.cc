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
        //        enum index { raw, aux, numfields };
        enum index { raw, numfields };

        EpixMPanelDef() {
            ADD_FIELD(raw              ,UINT16,2);
            //            ADD_FIELD(aux              ,UINT16,2);
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

#define BYTES_PER_PIXEL 2
#define ROW_PER_FRAME 192
#define COLS_PER_FRAME 384
#define ROWS_PER_BANK 48
#define COLS_PER_BANK 64
#define BANKS_PER_ROW 6
#define NUM_DATA_CYCLES 3072
#define IO_STREAM_WIDTH_BYTES NUM_BANKS * BYTES_PER_PIXEL
#define IO_STREAM_WIDTH_BITS IO_STREAM_WIDTH_BYTES * 8
#define ASIC_DATA_WIDTH_BITS BYTES_PER_PIXEL * 8

void EpixM320::_descramble(uint16_t inputImageAxiStreamBunches[3073][NUM_BANKS], /* first cycle header and rest is body */
                           uint16_t outputImageAxiStreamBunches[3072][NUM_BANKS])
{
    // Unused: uint16_t descrambledImage[NUM_BANKS][ROWS_PER_BANK][COLS_PER_BANK];
    uint16_t descrambledImageFlattened[NUM_BANKS * ROWS_PER_BANK * COLS_PER_BANK];

    /* Reorder banks from
    # 18    19    20    21    22    23
    # 12    13    14    15    16    17
    #  6     7     8     9    10    11
    #  0     1     2     3     4     5
    #
    #                To
    #  3     7    11    15    19    23
    #  2     6    10    14    18    22
    #  1     5     9    13    17    21
    #  0     4     8    12    16    20
    */
    const int bankRemapping[24] = {0, 6, 12, 18, 1, 7, 13, 19, 2, 8, 14, 20, 3, 9, 15, 21, 4, 10, 16, 22, 5, 11, 17, 23};
    const int colOffset[24] = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5 };
    const int rowOffset[24] = {0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5 };

    int rowPerBankIndex = 0;
    int colPerBankIndex = 1;
    bool even = false;

    unsigned idx = 0;
    for (unsigned imageAxiStreamBunchesIndex = 1; imageAxiStreamBunchesIndex < 3073; imageAxiStreamBunchesIndex++)
    {
        for (unsigned bankIndex = 0; bankIndex < NUM_BANKS; bankIndex++)
        {
            //descrambledImage[bankRemapping[bankIndex]][rowPerBankIndex][colPerBankIndex] = inputImageAxiStreamBunches[imageAxiStreamBunchesIndex][bankIndex];
            //printf("%u : %u.%u | ", idx++, imageAxiStreamBunchesIndex, bankIndex);  fflush(stdout);
            auto br = bankRemapping[bankIndex];
            //printf("b %u | ", br);  fflush(stdout);
            //printf("r %u : %u | ", rowOffset[br], rowOffset[br] * ROWS_PER_BANK * COLS_PER_BANK);  fflush(stdout);
            //printf("c %u : %u | ", colOffset[br], colOffset[br] * COLS_PER_BANK);  fflush(stdout);
            //printf("rpbi %u: %u | ", rowPerBankIndex, rowPerBankIndex * BANKS_PER_ROW * COLS_PER_BANK);  fflush(stdout);
            //printf("cpbi %u: %u | ", colPerBankIndex, colPerBankIndex);  fflush(stdout);
            auto sum = rowOffset[br] * ROWS_PER_BANK * COLS_PER_BANK +
                       colOffset[br] * COLS_PER_BANK +
                       rowPerBankIndex * BANKS_PER_ROW * COLS_PER_BANK +
                       colPerBankIndex;
            //printf("sum %u\n", sum);
            if (sum > NUM_BANKS * ROWS_PER_BANK * COLS_PER_BANK)
            {
                printf(">>> sum too big: %u > max b %u * r %u * c %u = %u\n", sum, NUM_BANKS, ROWS_PER_BANK, COLS_PER_BANK, NUM_BANKS * ROWS_PER_BANK * COLS_PER_BANK);
            }
            descrambledImageFlattened[sum] = inputImageAxiStreamBunches[imageAxiStreamBunchesIndex][bankIndex];
        }

        if (colPerBankIndex >= COLS_PER_BANK-2)
            colPerBankIndex = even ? 0 : 1;

        if (rowPerBankIndex >= ROWS_PER_BANK-1)
            rowPerBankIndex = 0;

        if ((colPerBankIndex == COLS_PER_BANK-1) && (rowPerBankIndex == ROWS_PER_BANK-1))
        {
            even = true;
            colPerBankIndex = 0;
        }
        if ((colPerBankIndex == COLS_PER_BANK-2) && (rowPerBankIndex == ROWS_PER_BANK-1))
        {
            printf("%u : %u.%u\n", idx, imageAxiStreamBunchesIndex, NUM_BANKS-1);
            break;
        }

        // Update counters
        colPerBankIndex += 2;
        rowPerBankIndex++;
    }

    for (unsigned imageAxiStreamBunchesIndex = 0; imageAxiStreamBunchesIndex < 3072; imageAxiStreamBunchesIndex++)
    {
        for (unsigned bankIndex = 0; bankIndex < NUM_BANKS; bankIndex++)
        {
            //printf("%u.%u\n", imageAxiStreamBunchesIndex, bankIndex);
            outputImageAxiStreamBunches[imageAxiStreamBunchesIndex][bankIndex] = descrambledImageFlattened[imageAxiStreamBunchesIndex * NUM_BANKS + bankIndex];
        }
    }
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
    unsigned shape[MaxRank] = {0,0,0,0,0};

    //  A super row crosses 2 elements; each element contains 2x2 ASICs
    const unsigned elemRows     = 384;
    const unsigned elemRowSize  = 192;
    const size_t   headerSize   = 24 * sizeof(uint16_t);

    //  The epix10kT unit cell is 2x2 ASICs
    CreateData cd(xtc, bufEnd, m_namesLookup, m_evtNamesId[0]);
    logging::debug("Writing panel event src 0x%x",unsigned(m_evtNamesId[0]));
    shape[0] = elemRows; shape[1] = elemRowSize;
    Array<uint16_t> aframe = cd.allocate<uint16_t>(EpixMPanelDef::raw, shape);

    if (subframes.size() != 3) {
        logging::error("Missing data: subframe size %d [3]\n",
                       subframes.size());
        xtc.damage.increase(XtcData::Damage::MissingData);
        return;
    }

    logging::debug("m_asics[%d] subframes.num_elem[%d]",m_asics,subframes.size());
    //for (unsigned i = 0; i < subframes.size(); ++i)
    //{
    //  logging::debug("subframes[%u].rank %u\n", i, subframes[i].rank());
    //  logging::debug("subframes[%u].num_elem %lu\n", i, subframes[i].num_elem());
    //}

    //  Validate timing headers

    // Descramble the data
    //auto p = subframes[2].data();
    //for(unsigned i=0; i<16*8; i++)
    //    printf("%08x%c",reinterpret_cast<uint32_t*>(p)[i], (i&7)==7 ? '\n':' ');

    //_descramble((uint16_t(*)[NUM_BANKS])(subframes[2].data()),
    //            (uint16_t(*)[NUM_BANKS])(aframe.data()));
    memcpy(aframe.data(), subframes[2].data() + headerSize, elemRows*elemRowSize*sizeof(uint16_t));

    ////uint16_t* f = (uint16_t*)aframe.data();
    //uint16_t* f = (uint16_t*)(subframes[2].data() + headerSize);
    //for (unsigned i = 0; i < 1; ++i)  // < 4
    //{
    //  for (unsigned j = 0; j < 3; ++j)
    //  {
    //    unsigned k = i*384+j;
    //    printf("%u %3u: %5hu %5hu %5hu ... %5hu %5hu %5hu\n", i, j, f[k*192+  0], f[k*192+  1], f[k*192+  2],
    //                                                                f[k*192+189], f[k*192+190], f[k*192+191]);
    //  }
    //  printf("       ...\n");
    //  for (unsigned j = 381; j < 384; ++j)
    //  {
    //    unsigned k = i*384+j;
    //    printf("%u %3u: %5hu %5hu %5hu ... %5hu %5hu %5hu\n", i, j, f[k*192+  0], f[k*192+  1], f[k*192+  2],
    //                                                                f[k*192+189], f[k*192+190], f[k*192+191]);
    //  }
    //  printf("\n");
    //}
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
