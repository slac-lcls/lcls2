#include "BEBDetector.hh"
#include "PythonConfigScanner.hh"
#include "XpmInfo.hh"
#include "psdaq/service/EbDgram.hh"
#include "xtcdata/xtc/VarDef.hh"
#include "xtcdata/xtc/DescData.hh"
#include "xtcdata/xtc/NamesLookup.hh"
#include "psdaq/service/Json2Xtc.hh"
#include "xtcdata/xtc/XtcIterator.hh"
#include "psalg/utils/SysLog.hh"
#include "psdaq/aes-stream-drivers/AxisDriver.h"

#include <thread>
#include <fcntl.h>
#include <Python.h>
#include <stdint.h>
#include <stdio.h>
#include <assert.h>
#include <type_traits>

using namespace XtcData;
using namespace rapidjson;
using logging = psalg::SysLog;

using json = nlohmann::json;

/**
 * \brief Split a delimited string of detector segment/serial numbers into a vector.
 * May denote DAQ detector segments or serial numbers etc.
 * \param inStr Delimited string, e.g. `0_1_2_3`
 * \param delimiter Delimiter. Currently "_".
 * \returns segs A vector of T that contains the parsed segment numbers.
 */
template <class T>
static std::vector<T> split_segs(const std::string& inStr,
                                 const std::string& delimiter=std::string("_"))
{
    std::vector<T> segs;
    size_t nextPos = 0;
    size_t lastPos = 0;
    std::string subStr;
    while ((nextPos = inStr.find(delimiter, lastPos)) != std::string::npos) {
        subStr = inStr.substr(lastPos, nextPos-lastPos);
        if constexpr(std::is_same_v<T, unsigned>) {
            segs.push_back(static_cast<T>(std::stoul(subStr)));
        } else {
            segs.push_back(subStr);
        }
        lastPos = nextPos + 1;
        std::cout << subStr << ", ";
    }
    subStr = inStr.substr(lastPos);
    if constexpr(std::is_same_v<T, unsigned>) {
        segs.push_back(static_cast<T>(std::stoul(subStr)));
    } else {
        segs.push_back(subStr);
    }
    std::cout << subStr << "." << std::endl;
    return segs;
}

namespace Drp {

PyObject* BEBDetector::_check(PyObject* obj) {
    if (!obj) {
        PyErr_Print();
        logging::critical("**** python error");
        abort();
    }
    return obj;
}

std::string BEBDetector::_string_from_PyDict(PyObject* dict, const char* key)
{
    PyObject* item      = _check(PyDict_GetItemString(dict,key));
    PyObject * item_str = _check(PyUnicode_AsASCIIString(item));
    std::string s(PyBytes_AsString(item_str));
    Py_DECREF(item_str);
    return s;
}


BEBDetector::BEBDetector(Parameters* para, MemPool* pool) :
    Detector      (para, pool),
    m_connect_json(""),
    m_module      (0),
    m_configScanner (0),
    m_debatch     (false)
{
    virtChan = 1;
}

BEBDetector::~BEBDetector()
{
    delete m_configScanner;
    if (m_module)  Py_DECREF(m_module);
}

void BEBDetector::_init(const char* arg)
{
    char module_name[64];
    sprintf(module_name,"psdaq.configdb.%s_config",m_para->detType.c_str());

    char func_name[64];

    // returns new reference
    m_module = _check(PyImport_ImportModule(module_name));

    PyObject* pDict = _check(PyModule_GetDict(m_module));
    {
      sprintf(func_name,"%s_init",m_para->detType.c_str());
      PyObject* pFunc = _check(PyDict_GetItemString(pDict, (char*)func_name));

      // returns new reference
#define MLOOKUP(m,name,dflt) (m.find(name)==m.end() ? dflt : m[name].c_str())
      const char* xpmpv = MLOOKUP(m_para->kwargs,"xpmpv",0);
      const char* timebase = MLOOKUP(m_para->kwargs,"timebase","186M");

      m_root = _check(PyObject_CallFunction(pFunc,"ssissi",
                                            arg,
                                            m_para->device.c_str(),
                                            m_para->laneMask,
                                            xpmpv,
                                            timebase,
                                            m_para->verbose));

      // check if m_root has "virtChan" member and set accordingly
      if (m_root) {
          PyObject* o_virtChan = PyDict_GetItemString(m_root,"virtChan");
          if (o_virtChan)
              virtChan = PyLong_AsLong(o_virtChan);
      }
    }

    m_configScanner = new PythonConfigScanner(*m_para,*m_module);
}

void BEBDetector::_init_feb()
{
    PyObject* pDict = _check(PyModule_GetDict(m_module));

    char func_name[64];
    sprintf(func_name,"%s_init_feb",m_para->detType.c_str());
    PyObject* pFunc = _check(PyDict_GetItemString(pDict, (char*)func_name));

#define MLOOKUP(m,name,dflt) (m.find(name)==m.end() ? dflt : m[name].c_str())
    const char* lane = MLOOKUP(m_para->kwargs,"feb_lane"   ,0);
    const char* chan = MLOOKUP(m_para->kwargs,"feb_channel",0);
    PyObject* mybytes = _check(PyObject_CallFunction(pFunc, "ss", lane, chan));
    Py_DECREF(mybytes);
}

json BEBDetector::connectionInfo(const json& msg)
{
    std::string alloc_json = msg.dump();
    char func_name[64];
    PyObject* pDict = _check(PyModule_GetDict(m_module));
    {
      sprintf(func_name,"%s_connectionInfo",m_para->detType.c_str());
      PyObject* pFunc = _check(PyDict_GetItemString(pDict, (char*)func_name));

      // returns new reference
      PyObject* mbytes = _check(PyObject_CallFunction(pFunc,"Os",m_root,alloc_json.c_str()));

      m_paddr = PyLong_AsLong(PyDict_GetItemString(mbytes, "paddr"));
      printf("*** BebDetector: paddr is %08x = %u\n", m_paddr, m_paddr);

      // there is currently a failure mode where the register reads
      // back as zero or 0xffffffff (incorrectly). This is not the best
      // longterm fix, but throw here to highlight the problem. the
      // difficulty is that Matt says this register has to work
      // so that an automated software solution would know which
      // xpm TxLink's to reset (a chicken-and-egg problem) - cpo
      // Also, register is corrupted when port number > 15 - Ric
      if (!m_paddr || m_paddr==0xffffffff || (m_paddr & 0xff) > 15) {
          logging::critical("XPM Remote link id register illegal value: 0x%x. Try XPM TxLink reset.",m_paddr);
          abort();
      }

      _connectionInfo(mbytes);

      Py_DECREF(mbytes);
    }

    return xpmInfo(m_paddr);
}

void BEBDetector::connectionShutdown()
{
    char func_name[64];
    PyObject* pDict = _check(PyModule_GetDict(m_module));
    sprintf(func_name,"%s_connectionShutdown",m_para->detType.c_str());
    PyObject* pFunc = PyDict_GetItemString(pDict, (char*)func_name);
    if (pFunc) {
        Py_DECREF(_check(PyObject_CallFunction(pFunc,"")));
    }
}

void BEBDetector::connect(const json& connect_json, const std::string& collectionId)
{
    logging::info("BEBDetector connect");
    m_connect_json = connect_json.dump();
    m_readoutGroup = connect_json["body"]["drp"][collectionId]["det_info"]["readout"];
}

unsigned BEBDetector::configure(const std::string& config_alias,
                                Xtc&               xtc,
                                const void*        bufEnd)
{
    PyObject* pDict = _check(PyModule_GetDict(m_module));

    char func_name[64];
    sprintf(func_name,"%s_config",m_para->detType.c_str());
    PyObject* pFunc = _check(PyDict_GetItemString(pDict, (char*)func_name));

    // returns new reference
    PyObject* mybytes;
    std::vector<unsigned> segNums;
    std::vector<std::string> segSerNos; // Assume multiSegment serNos are in m_para->serNo "_" delimited
    if (!m_multiSegment) {
        mybytes = _check(PyObject_CallFunction(pFunc, "Osssii",
                                               m_root, m_connect_json.c_str(), config_alias.c_str(),
                                               m_para->detName.c_str(), m_para->detSegment, m_readoutGroup));
    } else {
        // m_segNoStr in derived class, like jungfrau
        mybytes = _check(PyObject_CallFunction(pFunc, "Ossssi",
                                               m_root, m_connect_json.c_str(), config_alias.c_str(),
                                               m_para->detName.c_str(), m_segNoStr.c_str(), m_readoutGroup));
        segNums = split_segs<unsigned>(m_segNoStr); // Underscore delimited passed from cnf in sub-class constructor
        segSerNos = split_segs<std::string>(m_para->serNo); // Underscore delimited created during connect
    }

    char* buffer = new char[m_para->maxTrSize];
    const void* end = buffer + m_para->maxTrSize;
    Xtc& jsonxtc = *new (buffer, end) Xtc(TypeId(TypeId::Parent, 0));

    logging::debug("PyList_Check");
    if (PyList_Check(mybytes)) {
        logging::debug("PyList_Check true");
        for(unsigned seg=0; seg<PyList_Size(mybytes); seg++) {
            unsigned detSegment;
            std::string serNo;
            if (!m_multiSegment) {
                detSegment = m_para->detSegment;
                serNo = std::string("");
            } else {
                detSegment = segNums[seg];
                serNo = segSerNos[seg];
            }
            if (!m_multiSegment)
                logging::debug("seg %d",seg);
            else
                logging::debug("seg %d", detSegment);
            PyObject* item = PyList_GetItem(mybytes,seg);
            logging::debug("item %p",item);
            NamesId namesId(nodeId,ConfigNamesIndex+seg);
            if (Pds::translateJson2Xtc( item, jsonxtc, end, namesId, detSegment, serNo ))
                return -1;
        }
    }
    else if ( Pds::translateJson2Xtc( mybytes, jsonxtc, end, NamesId(nodeId,ConfigNamesIndex) ) )
        return -1;

    if (jsonxtc.extent>m_para->maxTrSize) {
        logging::critical("Config json output too large (%zu) for buffer (%zu)",
                          jsonxtc.extent, m_para->maxTrSize);
        throw "**** Config json output too large for buffer\n";
    }

    XtcData::ConfigIter iter(&jsonxtc, end);
    unsigned r = _configure(xtc,bufEnd,iter);

    // append the config xtc info to the dgram
    auto payload = xtc.alloc(jsonxtc.sizeofPayload(), bufEnd);
    memcpy(payload,(const void*)jsonxtc.payload(),jsonxtc.sizeofPayload());

    Py_DECREF(mybytes);
    delete[] buffer;

    return r;
}

unsigned BEBDetector::configureScan(const json& scan_keys,
                                    Xtc&        xtc,
                                    const void* bufEnd)
{
    NamesId namesId(nodeId,UpdateNamesIndex);
    return m_configScanner->configure(scan_keys,xtc,bufEnd,namesId,m_namesLookup);
}

unsigned BEBDetector::stepScan(const json& stepInfo, Xtc& xtc, const void* bufEnd)
{
    NamesId namesId(nodeId,UpdateNamesIndex);
    return m_configScanner->step(stepInfo,xtc,bufEnd,namesId,m_namesLookup);
}

Pds::TimingHeader* BEBDetector::getTimingHeader(uint32_t index) const
{
    EvtBatcherHeader* ebh = static_cast<EvtBatcherHeader*>(m_pool->dmaBuffers[index]);
    if (m_debatch) ebh = reinterpret_cast<EvtBatcherHeader*>(ebh->next());
    return static_cast<Pds::TimingHeader*>(ebh->next());
}

std::vector< XtcData::Array<uint8_t> > BEBDetector::_subframes(void* buffer, unsigned length)
{
    EvtBatcherIterator ebit = EvtBatcherIterator((EvtBatcherHeader*)buffer, length);
    EvtBatcherSubFrameTail* ebsft = ebit.next();
    unsigned nsubs = ebsft->tdest()+1;
    std::vector< XtcData::Array<uint8_t> > subframes(nsubs, XtcData::Array<uint8_t>(0, 0, 1) );
    do {
        logging::debug("Deb::event: array[%d] sz[%d]\n",ebsft->tdest(),ebsft->size());
        subframes[ebsft->tdest()] = XtcData::Array<uint8_t>(ebsft->data(), &ebsft->size(), 1);
    } while ((ebsft=ebit.next()));
    return subframes;
}

std::vector< XtcData::Array<uint8_t> > BEBDetector::_subframes(void* buffer, unsigned length, size_t nsubhint)
{
    EvtBatcherIterator ebit = EvtBatcherIterator((EvtBatcherHeader*)buffer, length);
    EvtBatcherSubFrameTail* ebsft = ebit.next();

    std::vector<XtcData::Array<uint8_t>> subframes;
    if (nsubhint > 0)
        subframes.reserve(nsubhint);

    do {
        subframes.emplace_back(ebsft->data(), &ebsft->size(), 1);
    } while((ebsft=ebit.next()));

    return subframes;
}

void BEBDetector::event(XtcData::Dgram& dgram, const void* bufEnd, PGPEvent* event, uint64_t l1count)
{
    // The subframes structure will be 
    // elem[0] = Timing Header, 
    // elem[1] = Auxilliary timing header, 
    // elem[2 and on] = frame data in lane order
    auto mask = event->mask;
    std::vector< XtcData::Array<uint8_t> > subframes(0);

    while (mask) {
        auto lane = __builtin_ffs(mask) - 1;
        auto dmaIndex = event->buffers[lane].index;
        auto data_size = event->buffers[lane].size;

        std::vector< XtcData::Array<uint8_t> > sf = _subframes(m_pool->dmaBuffers[dmaIndex], data_size);
        if (m_debatch)
            sf = _subframes(sf[2].data(), sf[2].shape()[0]);

        if (sf.size()>2) {
            if (subframes.size()==0)
                subframes = sf;
            else
                subframes.push_back(sf[2]);
        }
        mask &= mask - 1;
    }

    _event(dgram.xtc, bufEnd, l1count, subframes);
}

void BEBDetector::shutdown()
{
    // returns borrowed reference
    PyObject* pDict = _check(PyModule_GetDict(m_module));

    char func_name[64];
    sprintf(func_name,"%s_unconfig",m_para->detType.c_str());
    // returns borrowed reference
    PyObject* pFunc = _check(PyDict_GetItemString(pDict, (char*)func_name));

    // returns new reference
    Py_DECREF(_check(PyObject_CallFunction(pFunc,"O",m_root)));
}

}
