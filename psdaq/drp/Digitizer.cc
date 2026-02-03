#include "Digitizer.hh"
#include "PythonConfigScanner.hh"
#include "psdaq/service/EbDgram.hh"
#include "xtcdata/xtc/VarDef.hh"
#include "xtcdata/xtc/DescData.hh"
#include "xtcdata/xtc/NamesLookup.hh"
#include "psdaq/service/Json2Xtc.hh"
#include "rapidjson/document.h"
#include "xtcdata/xtc/XtcIterator.hh"
#include "psalg/digitizer/Hsd.hh"
#include "psdaq/aes-stream-drivers/DataDriver.h"
#include "Si570.hh"
#include "XpmInfo.hh"
#include "psdaq/mmhw/Pgp3Axil.hh"
#include "psdaq/mmhw/Reg.hh"
#include "psalg/utils/SysLog.hh"

#include <Python.h>
#include <stdint.h>
#include <stdio.h>
#include <assert.h>
#include <fstream>
#include <unistd.h>
#include <netdb.h>
#include <arpa/inet.h>

using namespace XtcData;
using namespace rapidjson;
using logging = psalg::SysLog;

using json = nlohmann::json;
using Pds::Mmhw::Reg;

namespace Drp {

class HsdDef : public VarDef
{
public:
    enum index {EventHeader = 0};
    HsdDef(unsigned lane_mask)
    {
        Alg alg("fpga", 1, 2, 3);
        const unsigned nameLen = 7;
        char chanName[nameLen];
        NameVec.push_back({"eventHeader", Name::UINT32, 1});
        for (unsigned i = 0; i < sizeof(lane_mask)*sizeof(uint8_t); i++){
            if (!((1<<i)&lane_mask)) continue;
            snprintf(chanName, nameLen, "chan%2.2d", i);
            NameVec.push_back({chanName, alg});
        }
    }
};

static PyObject* _check(PyObject* obj) {
    if (!obj) {
        PyErr_Print();
        logging::critical("**** python error");
        abort();
    }
    return obj;
}

Digitizer::Digitizer(Parameters* para, MemPool* pool) :
    Detector    (para, pool),
    m_epics_name(para->kwargs["hsd_epics_prefix"]),
    m_strm_limit(1000000),
    m_strm_limit_sem(Pds::Semaphore::FULL)
{
    printf("*** found epics name %s\n",m_epics_name.c_str());

    if (para->kwargs.find("strm_limit")!=para->kwargs.end()) 
        m_strm_limit = std::stoul(const_cast<Parameters&>(*para).kwargs["strm_limit"]);

    //
    //  Initialize python calls
    //
    m_module = _check(PyImport_ImportModule("psdaq.configdb.hsd_config"));

    PyObject* pDict = _check(PyModule_GetDict(m_module));
    PyObject* pFunc = _check(PyDict_GetItemString(pDict, "hsd_init"));

    // returns new reference
    PyObject* mbytes = _check(PyObject_CallFunction(pFunc,"ss",
                                                    m_epics_name.c_str(), para->device.c_str()));
    Py_DECREF(mbytes);

    m_configScanner = new PythonConfigScanner(*m_para,*m_module);
}

Digitizer::~Digitizer()
{
    delete m_configScanner;
    Py_DECREF(m_module);
}

json Digitizer::connectionInfo(const nlohmann::json& msg)
{
    PyObject* pDict = _check(PyModule_GetDict(m_module));
    PyObject* pFunc = _check(PyDict_GetItemString(pDict, "hsd_connect"));
        
    // returns new reference
    PyObject* mbytes = _check(PyObject_CallFunction(pFunc,"s", msg.dump().c_str()));
    unsigned paddr = PyLong_AsLong(PyDict_GetItemString(mbytes, "paddr"));
    Py_DECREF(mbytes);

    // there is currently a failure mode where the register reads
    // back as zero or 0xffffffff (incorrectly). This is not the best
    // longterm fix, but throw here to highlight the problem. the
    // difficulty is that Matt says this register has to work
    // so that an automated software solution would know which
    // xpm TxLink's to reset (a chicken-and-egg problem) - cpo
    // Also, register is corrupted when port number > 15 - Ric
    if (!paddr || paddr==0xffffffff || (paddr & 0xff) > 15) {
        logging::critical("XPM Remote link id register illegal value: 0x%x. Try XPM TxLink reset.",paddr);
        abort();
    }
    else
        logging::info("paddr %x",paddr);
    
    return xpmInfo(paddr);
}

unsigned Digitizer::_addJson(Xtc& xtc, const void* bufEnd, NamesId& configNamesId, const std::string& config_alias) {

  timespec tv_b; clock_gettime(CLOCK_REALTIME,&tv_b);

#define CHECK_TIME(s) {                                                 \
    timespec tv; clock_gettime(CLOCK_REALTIME,&tv);                     \
    printf("%s %f seconds\n",#s,                                        \
           double(tv.tv_sec-tv_b.tv_sec)+1.e-9*(double(tv.tv_nsec)-double(tv_b.tv_nsec))); }


  // returns borrowed reference
  PyObject* pDict = _check(PyModule_GetDict(m_module));
  // returns borrowed reference
  PyObject* pFunc = _check(PyDict_GetItemString(pDict, "hsd_config"));
  
  CHECK_TIME(PyDict_Get);

  // returns new reference
  PyObject* mybytes = _check(PyObject_CallFunction(pFunc,"ssssii",
                                                   m_connect_json.c_str(),
                                                   m_epics_name.c_str(),
                                                   config_alias.c_str(),
                                                   m_para->detName.c_str(),
                                                   m_para->detSegment,
                                                   m_readoutGroup));
  
  CHECK_TIME(PyObj_Call);

  // returns new reference
  PyObject * json_bytes = _check(PyUnicode_AsASCIIString(mybytes));
  char* json = (char*)PyBytes_AsString(json_bytes);
  printf("json: %s\n",json);

    // convert to json to xtc
  const unsigned BUFSIZE = 1024*1024;
  char buffer[BUFSIZE];
  unsigned len = Pds::translateJson2Xtc(json, buffer, &buffer[BUFSIZE], configNamesId, m_para->detName.c_str(), m_para->detSegment);
  if (len>BUFSIZE) {
      throw "**** Config json output too large for buffer";
  }
  if (len <= 0) {
      throw "**** Config json translation error";
  }
  
  CHECK_TIME(translateJson);
  
  // append the config xtc info to the dgram
  Xtc& jsonxtc = *(Xtc*)buffer;
  auto payload = xtc.alloc(jsonxtc.sizeofPayload(), bufEnd);
  memcpy(payload,(const void*)jsonxtc.payload(),jsonxtc.sizeofPayload());

  // get the lane mask from the json
  unsigned lane_mask = 1;
  printf("hsd lane_mask is 0x%x\n",lane_mask);
  
  Py_DECREF(mybytes);
  Py_DECREF(json_bytes);
  
  CHECK_TIME(Done);
  
  return lane_mask;
}

void Digitizer::connect(const json& connect_json, const std::string& collectionId)
{
  m_connect_json = connect_json.dump();
  m_readoutGroup = connect_json["body"]["drp"][collectionId]["det_info"]["readout"];
}

unsigned Digitizer::configure(const std::string& config_alias, Xtc& xtc, const void* bufEnd)
{
    unsigned lane_mask;
    m_evtNamesId = NamesId(nodeId, EventNamesIndex);
    // set up the names for the configuration data
    NamesId configNamesId(nodeId,ConfigNamesIndex);
    lane_mask = Digitizer::_addJson(xtc, bufEnd, configNamesId, config_alias);

    // set up the names for L1Accept data
    Alg alg("raw", 3, 0, 0);
    Names& eventNames = *new(xtc, bufEnd) Names(bufEnd,
                                                m_para->detName.c_str(), alg,
                                                m_para->detType.c_str(), m_para->serNo.c_str(), m_evtNamesId, m_para->detSegment);
    HsdDef myHsdDef(lane_mask);
    eventNames.add(xtc, bufEnd, myHsdDef);
    m_namesLookup[m_evtNamesId] = NameIndex(eventNames);
    return 0;
}

void Digitizer::event(XtcData::Dgram& dgram, const void* bufEnd, PGPEvent* event, uint64_t l1count)
{
    CreateData hsd(dgram.xtc, bufEnd, m_namesLookup, m_evtNamesId);

    // HSD data includes two uint32_t "event header" words
    unsigned data_size;
    unsigned shape[MaxRank];
    shape[0] = 2;
    Array<uint32_t> arrayH = hsd.allocate<uint32_t>(HsdDef::EventHeader, shape);

    int lane = __builtin_ffs(event->mask) - 1;
    uint32_t dmaIndex = event->buffers[lane].index;
    Pds::TimingHeader* timing_header = (Pds::TimingHeader*)m_pool->dmaBuffers[dmaIndex];
    arrayH(0) = timing_header->_opaque[0];
    arrayH(1) = timing_header->_opaque[1];

    if ((timing_header->_opaque[1] & (1<<31))==0)  // check JESD status bit
      dgram.xtc.damage.increase(Damage::UserDefined);

    for (int i=0; i<PGP_MAX_LANES; i++) {
        if (event->mask & (1 << i)) {
            data_size = event->buffers[i].size - sizeof(Pds::TimingHeader);
            if (data_size >= m_strm_limit) {
              logging::warning("Invalid lane %d DMA size %u (0x%x).  Skipping event.",
                            i, data_size, data_size);
              //              _dump(timing_header,data_size);
              dgram.xtc.damage.increase(Damage::UserDefined);
              continue;
            }
            shape[0] = data_size;
            Array<uint8_t> arrayT = hsd.allocate<uint8_t>(i+1, shape);
            uint32_t dmaIndex = event->buffers[i].index;
            memcpy(arrayT.data(), (uint8_t*)m_pool->dmaBuffers[dmaIndex] + sizeof(Pds::TimingHeader), data_size);

            //
            // Check the overflow bit in the stream headers
            //
            const uint8_t* p = (const uint8_t*)m_pool->dmaBuffers[dmaIndex]+sizeof(Pds::TimingHeader);
            const uint8_t* const p_end = p + data_size;
            //  Add some error checking in this iteration
            const Pds::HSD::EventHeader* _EH = reinterpret_cast<const Pds::HSD::EventHeader*>(m_pool->dmaBuffers[dmaIndex]);
            unsigned streams = _EH->streams();
            do {
                if (streams==0) {
                    logging::warning("Exiting stream header error checking at %p (end = %p)", p, p_end);
                    _EH->dump();
                    break;
                }
                const Pds::HSD::StreamHeader& stream = *reinterpret_cast<const Pds::HSD::StreamHeader*>(p);
                if (stream.overflow()) {
                    logging::critical("Header error: overflow %c  unlocked %c",
                                      stream.overflow()?'T':'F',
                                      stream.unlocked()?'T':'F');
                    abort();
                }
                else if (stream.unlocked()) {
                    logging::debug("Header error: overflow %c  unlocked %c",
                                     stream.overflow()?'T':'F',
                                     stream.unlocked()?'T':'F');
                    dgram.xtc.damage.increase(Damage::UserDefined);
                    break;
                }
                streams &= ~(1<<stream.stream_id());
                p += sizeof(stream);                        // step past the header
                p += stream.num_samples()*sizeof(uint16_t); //   and its payload
            } while( p < p_end);

            logging::debug("Exited stream header error checking at %p (end = %p), streams = %x", p, p_end, streams);

            // example showing how to use psalg Hsd code to extract data.
            // we are now not using this code since it was too complex
            // (see comment at top of Hsd.hh)
            // Pds::HSD::Channel channel(&m_allocator,
            //                           (const uint32_t*)arrayH.data(),
            //                           (const uint8_t*)arrayT.data());
            //printf("*** npeaks %d\n",channel.npeaks());
         }
    }
}

void Digitizer::shutdown()
{
    // returns borrowed reference
    PyObject* pDict = _check(PyModule_GetDict(m_module));
    // returns borrowed reference
    PyObject* pFunc = _check(PyDict_GetItemString(pDict, "hsd_unconfig"));

    // returns new reference
    PyObject_CallFunction(pFunc,"s",
                          m_epics_name.c_str());
}

unsigned Digitizer::configureScan(const json& scan_keys,
                                  Xtc&        xtc,
                                  const void* bufEnd)
{
    NamesId namesId(nodeId,UpdateNamesIndex);
    return m_configScanner->configure(scan_keys,xtc,bufEnd,namesId,m_namesLookup);
}

unsigned Digitizer::stepScan(const json& stepInfo, Xtc& xtc, const void* bufEnd)
{
    NamesId namesId(nodeId,UpdateNamesIndex);
    return m_configScanner->step(stepInfo,xtc,bufEnd,namesId,m_namesLookup);
}

void Digitizer::_dump(const void* hdr, unsigned data_size)
{
    m_strm_limit_sem.take();
    printf("== data_size = %u ==\n",data_size);
    const uint32_t* p = reinterpret_cast<const uint32_t*>(hdr);
    for(unsigned k=0; k<data_size/4; k++) {
        if ((k%8)==0)
            printf("\n[%08x]",k);
        printf(" %08x",p[k]);
    }
    printf("\n");
    m_strm_limit_sem.give();
}

}
