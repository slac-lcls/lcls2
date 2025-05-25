#include "PvaDetector.hh"

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <getopt.h>
#include <cassert>
#include <bitset>
#include <chrono>
#include <unistd.h>
#include <map>
#include <algorithm>
#include <limits>
#include <thread>
#include <fstream>      // std::ifstream
#include <cctype>       // std::isspace
#include <regex>
#include <Python.h>
#include "psdaq/aes-stream-drivers/DataDriver.h"
#include "RunInfoDef.hh"
#include "xtcdata/xtc/Damage.hh"
#include "xtcdata/xtc/DescData.hh"
#include "xtcdata/xtc/ShapesData.hh"
#include "xtcdata/xtc/NamesLookup.hh"
#include "psdaq/service/kwargs.hh"
#include "psdaq/service/EbDgram.hh"
#include "psdaq/service/Json2Xtc.hh"
#include "psdaq/eb/TebContributor.hh"
#include "psalg/utils/SysLog.hh"
#include "psdaq/service/fast_monotonic_clock.hh"
#include "psalg/utils/trim.hh"

#ifndef POSIX_TIME_AT_EPICS_EPOCH
#define POSIX_TIME_AT_EPICS_EPOCH 631152000u
#endif

using namespace XtcData;
using namespace Pds;
using json = nlohmann::json;
using logging = psalg::SysLog;
using ms_t = std::chrono::milliseconds;
using us_t = std::chrono::microseconds;

namespace Drp {

struct PvParameters : public Parameters
{
    std::vector<std::string> pvSpecs;
};

};

static const TimeStamp TimeMax(std::numeric_limits<unsigned>::max(),
                               std::numeric_limits<unsigned>::max());
static unsigned tsMatchDegree = 2;

//
//  Put all the ugliness of non-global timestamps here
//
static int _compare(const TimeStamp& ts1,
                    const TimeStamp& ts2) {
  int result = 0;

  if ((tsMatchDegree == 0) && !(ts2 == TimeMax))
      return result;

  if (tsMatchDegree == 1) {
    /*
    **  Mask out the fiducial
    */
    const uint64_t mask = 0xfffffffffffe0000ULL;
    uint64_t ts1m = ts1.value()&mask;
    uint64_t ts2m = ts2.value()&mask;

    const uint64_t delta = 10000000; // 10 ms!
    if      (ts1m > ts2m)  result = ts1m - ts2m > delta ?  1 : 0;
    else if (ts2m > ts1m)  result = ts2m - ts1m > delta ? -1 : 0;

    return result;
  }

  if      (ts1 > ts2) result = 1;
  else if (ts2 > ts1) result = -1;
  return result;
}


namespace Drp {

static PyObject* pyCheckErr(PyObject* obj)
{
    if (!obj) {
        PyErr_Print();
        logging::critical("*** python error");
        abort();
    }
    return obj;
}

static const Name::DataType xtype[] = {
  Name::UINT8 , // pvBoolean
  Name::INT8  , // pvByte
  Name::INT16 , // pvShort
  Name::INT32 , // pvInt
  Name::INT64 , // pvLong
  Name::UINT8 , // pvUByte
  Name::UINT16, // pvUShort
  Name::UINT32, // pvUInt
  Name::UINT64, // pvULong
  Name::FLOAT , // pvFloat
  Name::DOUBLE, // pvDouble
  Name::CHARSTR, // pvString
};

struct DataDsc {
    uint32_t shape[MaxRank];
    void*    data;
};

class RawDef : public VarDef
{
public:
    enum index { field };
    RawDef(std::string& field, Name::DataType dType, int rank)
    {
        NameVec.push_back({field.c_str(), dType, rank});
    }
};

class InfoDef : public VarDef
{
public:
    enum index { keys, detName };
    InfoDef(std::string& detName)
    {
        NameVec.push_back({"keys",          Name::CHARSTR, 1});
        NameVec.push_back({detName.c_str(), Name::CHARSTR, 1});
    }
};

// ---

PvMonitor::PvMonitor(const PvParameters&      para,
                     const std::string&       alias,
                     const std::string&       pvName,
                     const std::string&       provider,
                     const std::string&       request,
                     const std::string&       field,
                     unsigned                 id,
                     size_t                   nBuffers,
                     unsigned                 type,
                     size_t                   nelem,
                     size_t                   rank,
                     uint32_t                 firstDim,
                     const std::atomic<bool>& running) :
    Pds_Epics::PvMonitorBase(pvName, provider, request, field),
    m_para                  (para),
    m_state                 (NotReady),
    m_id                    (id),
    m_type                  (type),
    m_nelem                 (nelem),
    m_rank                  (rank),
    m_payloadSize           (0),
    m_firstDimOverride      (firstDim),
    m_alias                 (alias),
    m_running               (running),
    pvQueue                 (nBuffers),
    bufferFreelist          (pvQueue.size()),
    m_notifySocket          {&m_context, ZMQ_PUSH},
    m_nUpdates              (0),
    m_nMissed               (0),
    m_latency               (0)
{
    // ZMQ socket for reporting errors
    m_notifySocket.connect({"tcp://" + m_para.collectionHost + ":" + std::to_string(CollectionApp::zmq_base_port + m_para.partition)});
}

int PvMonitor::getParams(std::string&    fieldName,
                         Name::DataType& xtcType,
                         int&            rank,
                         size_t          bufferSize)
{
    std::unique_lock<std::mutex> lock(m_mutex);

    if (m_state == NotReady) {
        // Wait for PV to connect
        const std::chrono::seconds tmo(3);
        m_condition.wait_for(lock, tmo, [this] { return m_state == Ready; });
        if (m_state != Ready) {
            auto msg("PV "+name()+" hasn't connected");
            json jmsg = createAsyncWarnMsg(m_para.alias, msg);
            m_notifySocket.send(jmsg.dump());
            if (m_type == -1u || m_nelem == -1ul || m_rank == -1ul) {
                // Parameter(s) weren't defaulted in PV specs so can't Configure
                logging::error("Failed to get parameters for PV %s", name().c_str());
                return 1;
            }
            logging::warning("%s; using defaulted parameter values\n", msg.c_str(), name().c_str());
        }
    }

    fieldName = m_fieldName;
    xtcType   = xtype[m_type];
    rank      = m_rank;
    if (m_firstDimOverride && m_rank != 2)
    {
        rank = 2;                       // Override rank
        logging::warning("%s rank overridden from %zu to %zu\n",
                         name().c_str(), m_rank, rank);
    }

    m_payloadSize = m_nelem * Name::get_element_size(xtcType);
    if (m_payloadSize > bufferSize) {
        auto msg("PV "+name()+" size too big; see log");
        json jmsg = createAsyncErrMsg(m_para.alias, msg);
        m_notifySocket.send(jmsg.dump());
        logging::error("PV %s size (%zu) is too large to fit in buffer of size %zu; "
                       "increase pebbleBufSize", name().c_str(), m_payloadSize, bufferSize);
        return 1;
    }

    return 0;
}

void PvMonitor::startup()
{
    pvQueue.startup();
    bufferFreelist.startup();
    size_t bufSize = sizeof(EbDgram) + sizeof(DataDsc) + m_payloadSize;
    m_buffer.resize(pvQueue.size() * bufSize);
    for (unsigned i = 0; i < pvQueue.size(); ++i) {
        bufferFreelist.push(reinterpret_cast<Dgram*>(&m_buffer[i * bufSize]));
    }
}

void PvMonitor::shutdown()
{
    pvQueue.shutdown();
    bufferFreelist.shutdown();

    m_nUpdates = 0;
    m_nMissed  = 0;
}

void PvMonitor::onConnect()
{
    logging::debug("PV  %s connected", name().c_str());

    if (m_state == NotReady) {
        std::lock_guard<std::mutex> lock(m_mutex);
        pvd::ScalarType type;
        size_t nelem;
        size_t rank;
        if (!this->Pds_Epics::PvMonitorBase::getParams(type, nelem, rank))  {
            if (m_type  == -1u)   m_type  = type;
            if (m_nelem == -1ul)  m_nelem = nelem;
            if (m_rank  == -1ul)  m_rank  = rank;
            if (type == m_type && nelem == m_nelem && rank == m_rank) {
                m_state = Ready;
            }
            else {
                logging::critical("PV's defaulted and introspected parameter(s) don't match: "
                                  "type %d vs %d, nelem %zu vs %zu, rank %zu vs %zu",
                                  m_type, type, m_nelem, nelem, m_rank, rank);
                abort();                // Crash so user can fix config file
            }
        }
        m_condition.notify_one();
    }

    if (m_para.verbose > 1) {           // Use -vv to get increased detail
        if (printStructure())
            logging::error("onConnect: printStructure() failed");
    }
}

void PvMonitor::onDisconnect()
{
    // This unfortunately seems not to get called for CA PVs
    m_state = NotReady;

    auto msg("PV "+ name() + " disconnected");
    logging::warning("%s", msg.c_str());
    json jmsg = createAsyncWarnMsg(m_para.alias, msg);
    m_notifySocket.send(jmsg.dump());
}

void PvMonitor::updated()
{
    // Queue updates only when Ready and in Running
    if (m_state == Ready && m_running.load(std::memory_order_relaxed)) {
        int64_t seconds;
        int32_t nanoseconds;
        getTimestampEpics(seconds, nanoseconds);
        TimeStamp timestamp(seconds, nanoseconds);

        ++m_nUpdates;
        m_latency = Eb::latency<us_t>(timestamp); // Grafana plots latency in us
        logging::debug("%s updated @ %u.%09u, latency %ld ms", name().c_str(), timestamp.seconds(), timestamp.nanoseconds(), m_latency/1000);

        Dgram* dgram;
        if (bufferFreelist.try_pop(dgram)) { // If a buffer is available...
            //static uint64_t last_ts = 0;
            //uint64_t ts = timestamp.to_ns();
            //int64_t  dT = ts - last_ts;
            //printf("  PV:  %u.%09u, dT %9ld, ts %18lu, last %18lu\n", timestamp.seconds(), timestamp.nanoseconds(), dT, ts, last_ts);
            //if (dT > 0)  last_ts = ts;

            dgram->time = timestamp;         // Save the PV's timestamp
            dgram->xtc = {{TypeId::Parent, 0}, {m_id}};

            size_t      bufSize = sizeof(DataDsc) + m_payloadSize;
            const void* bufEnd  = dgram->xtc.payload() + bufSize;
            DataDsc*    payload = (DataDsc*)dgram->xtc.alloc(bufSize, bufEnd);
            auto        size    = getData(&payload->data, m_payloadSize, payload->shape);
            if (size > m_payloadSize) {      // Check actual size vs available size
                logging::debug("Truncated: Buffer of size %zu is too small for payload of size %zu for %s\n",
                               m_payloadSize, size, name().c_str());
                dgram->xtc.damage.increase(Damage::Truncated);
            }

            if (m_firstDimOverride != 0) {
                payload->shape[1] = payload->shape[0] / m_firstDimOverride;
                payload->shape[0] = m_firstDimOverride;
            }
            pvQueue.push(dgram);
        }
        else {
            ++m_nMissed;                     // Else count it as missed
        }
    }
}

void PvMonitor::timeout(const PgpReader& pgp, ms_t timeout)
{
    Dgram* pvDg;
    if (pvQueue.peek(pvDg)) {
        if (pgp.age(pvDg->time) > timeout) {
            logging::debug("PV timed out!! "
                           "TimeStamp:  %u.%09u [0x%08x%04x.%05x], age %ld ms",
                           pvDg->time.seconds(),  pvDg->time.nanoseconds(),
                           pvDg->time.seconds(), (pvDg->time.nanoseconds()>>16)&0xfffe, pvDg->time.nanoseconds()&0x1ffff,
                           Eb::latency<ms_t>(pvDg->time));
            pvQueue.try_pop(pvDg);      // Actually consume the element
            bufferFreelist.push(pvDg);  // Return buffer to freelist
        }
    }
}

// ---

Pgp::Pgp(const Parameters& para, MemPool& pool, Detector* det) :
    PgpReader(para, pool, MAX_RET_CNT_C, 32),
    m_det(det),
    m_available(0), m_current(0), m_nDmaRet(0)
{
    if (pool.setMaskBytes(para.laneMask, det->virtChan)) {
        logging::error("Failed to allocate lane/vc");
    }
}

EbDgram* Pgp::_handle(uint32_t& pebbleIndex)
{
    const TimingHeader* timingHeader = handle(m_det, m_current);
    if (!timingHeader)  return nullptr;

    uint32_t pgpIndex = timingHeader->evtCounter & (m_pool.nDmaBuffers() - 1);
    PGPEvent* event = &m_pool.pgpEvents[pgpIndex];
    // No need to check for a broken event since we don't get indices for those

    // make new dgram in the pebble
    // It must be an EbDgram in order to be able to send it to the MEB
    pebbleIndex = event->pebbleIndex;
    Src src = m_det->nodeId;
    EbDgram* dgram = new(m_pool.pebble[pebbleIndex]) EbDgram(*timingHeader, src, m_para.rogMask);

    // Collect indices of DMA buffers that can be recycled and reset event
    freeDma(event);

    return dgram;
}

EbDgram* Pgp::next(uint32_t& evtIndex)
{
    // get new buffers
    if (m_current == m_available) {
        m_current = 0;
        m_available = read();
        m_nDmaRet = m_available;
        if (m_available == 0) {
            return nullptr;
        }
    }

    EbDgram* dgram = _handle(evtIndex);
    m_current++;
    return dgram;
}

// ---

PvDetector::PvDetector(PvParameters& para, MemPoolCpu& pool) :
    XpmDetector(&para, &pool)
{
    virtChan = 0;

    const char* module_name = "psdaq.configdb.pvadetector_config";
    m_pyModule = PyImport_ImportModule(module_name);
    if (!m_pyModule) {
        PyErr_Print();
        abort();
    }
}

PvDetector::~PvDetector()
{
    if (m_pyModule) {
        Py_DECREF(m_pyModule);
    }
}

unsigned PvDetector::connect(const json& connectJson, const std::string& collectionId, std::string& msg)
{
    unsigned rc = 0;
    XpmDetector::connect(connectJson, collectionId);
    m_connectJson = connectJson.dump();

    // Check for a default first dimension specification
    uint32_t firstDimDef = 0;
    if (m_para->kwargs.find("firstdim") != m_para->kwargs.end()) {
        firstDimDef = std::stoul(m_para->kwargs["firstdim"]);
    }

    unsigned id = 0;
    for (const auto& pvSpec : static_cast<PvParameters*>(m_para)->pvSpecs) {
        // Parse the pvSpec string of the forms
        //   "[<alias>=][<provider>/]<PV name>[.<field>][,firstDim]"
        //   "[<alias>=][<provider>/]<PV name>[.<field>][<shape>][(<type>)]"
        try {
            std::string alias     = m_para->detName;
            std::string pvName    = pvSpec;
            std::string provider  = "pva";
            std::string field     = "value";
            uint32_t    firstDim  = firstDimDef;
            uint32_t    secondDim = 0;
            auto pos = pvName.find("=", 0);
            if (pos != std::string::npos) { // Parse alias
                alias  = pvName.substr(0, pos);
                pvName = pvName.substr(pos+1);
            }
            pos = pvName.find("/", 0);
            if (pos != std::string::npos) { // Parse provider
                provider = pvName.substr(0, pos);
                pvName   = pvName.substr(pos+1);
            }
            pos = pvName.find(",", 0);
            if (pos != std::string::npos) { // Parse firstDim value
                // Let '[dim1,dim2]' syntax take precedence
                if (pvName.find("[", 0) == std::string::npos) {
                    firstDim = std::stoul(pvName.substr(pos+1));
                    pvName   = pvName.substr(0, pos);
                }
            }
            pos = pvName.find(".", 0);
            if (pos != std::string::npos) { // Parse field name
                field  = pvName.substr(pos+1);
                pvName = pvName.substr(0, pos);
            }
            // Provided values will be clobbered when PV connects
            // I think this is fine since it means things are working
            // firstDim will still work as originally
            // PV shape (optionally) provided as [dim1,dim2]
            // Shape comes before type [shape](type)
            std::regex pattern("\\[(.*?)\\]");
            std::smatch matches;
            ssize_t rank = -1;
            size_t nelem = -1ul;
            if (std::regex_search(pvSpec, matches, pattern)) {
                // Cleanup pvName or field
                pos = pvName.find("[", 0);
                if (pos != std::string::npos)
                    pvName = pvName.substr(0, pos);
                else
                    field = field.substr(0, field.find("[",0));
                // Extract shape and rank
                auto dataShape = matches[1].str();
                pos = dataShape.find(",", 0);
                if (pos != std::string::npos) { // Rank 2
                    auto tmp = dataShape.substr(0,pos);
                    firstDim = std::stoul(tmp);
                    secondDim = std::stoul(dataShape.substr(pos+1));
                    rank = 2;
                    nelem = firstDim * secondDim;
                } else { // Rank 0/1
                    // Check if scalar or array
                    nelem = std::stoul(dataShape);
                    rank = nelem == 1 ? 0 : 1;
                    // firstDim should remain 0 or rank = 2 results
                }
            }
            // PV type (optionally) provided as (type)
            pattern = std::regex("\\((.*?)\\)");
            unsigned type = -1u;
            if (std::regex_search(pvSpec, matches, pattern)) {
                // Cleanup pvName or field
                pos = pvName.find("(", 0);
                if (pos != std::string::npos)
                    pvName = pvName.substr(0, pos);
                else
                    field = field.substr(0, field.find("(",0));
                if (matches[1].str().compare("bool") == 0)
                    type = pvd::pvBoolean;
                else if (matches[1].str().compare("byte") == 0)
                    type = pvd::pvByte;
                else if (matches[1].str().compare("short") == 0)
                    type = pvd::pvShort;
                else if (matches[1].str().compare("int") == 0)
                    type = pvd::pvInt;
                else if (matches[1].str().compare("long") == 0)
                    type = pvd::pvLong;
                else if (matches[1].str().compare("ubyte") == 0)
                    type = pvd::pvUByte;
                else if (matches[1].str().compare("ushort") == 0)
                    type = pvd::pvUShort;
                else if (matches[1].str().compare("uint") == 0)
                    type = pvd::pvUInt;
                else if (matches[1].str().compare("ulong") == 0)
                    type = pvd::pvULong;
                else if (matches[1].str().compare("float") == 0)
                    type = pvd::pvFloat;
                else if (matches[1].str().compare("double") == 0)
                    type = pvd::pvDouble;
                else if (matches[1].str().compare("string") == 0) {
                    type = pvd::pvString;
                    nelem = MAX_STRING_SIZE;
                }
                else {
                    throw std::string("Unrecognized type '" + matches[1].str() + "'");
                }
                // ...
            }
            std::string request = provider == "pva" ? "field(value,timeStamp,dimension)"
                                                    : "field(value,timeStamp)";
            if (field != "value" && field != "timeStamp" && (provider != "pva" || field != "dimension")) {
                pos = request.find(")", 0);
                if (pos  != std::string::npos) {
                    request = request.substr(0, pos) + "," + field + ")";
                }
            }

            logging::debug("For '%s', alias '%s', provider '%s', PV '%s', field '%s', "
                           "type %d, firstDim %u, nelem %zd, rank %zd, request '%s'",
                           pvSpec.c_str(), alias.c_str(), provider.c_str(), pvName.c_str(), field.c_str(),
                           type, firstDim, nelem, rank, request.c_str());
            auto pvMonitor = std::make_shared<PvMonitor>(*static_cast<PvParameters*>(m_para),
                                                         alias, pvName, provider, request, field,
                                                         id++, m_pool->nbuffers(), type, nelem, rank, firstDim,
                                                         m_running);
            m_pvMonitors.push_back(pvMonitor);
        }
        catch(std::string& error) {
            logging::error("Failed to create PvMonitor for '%s': %s",
                           pvSpec.c_str(), error.c_str());
            m_pvMonitors.clear();
            msg = error;
            rc = 1;
        }
    }

    return rc;
}

unsigned PvDetector::disconnect()
{
    XpmDetector::shutdown();

    m_pvMonitors.clear();
    return 0;
}

unsigned PvDetector::configure(const std::string& config_alias, Xtc& xtc, const void* bufEnd)
{
    logging::info("PvDetector configure");

    if (XpmDetector::configure(config_alias, xtc, bufEnd))
        return 1;

    for (auto& pvMonitor : m_pvMonitors) {
        // Set up the names for L1Accept data
        unsigned uvsn = m_para->kwargs.find("data_vsn") != m_para->kwargs.end() ? std::stoul(m_para->kwargs["data_vsn"],NULL,0) : 0x010000;
        AlgVersion& vsn = *reinterpret_cast<AlgVersion*>(&uvsn);
        logging::debug("AlgVersion %d.%d.%d",vsn.major(),vsn.minor(),vsn.micro());
        Alg     rawAlg("raw", vsn.major(), vsn.minor(), vsn.micro());
        NamesId rawNamesId(nodeId, RawNamesIndex + pvMonitor->id());
        Names&  rawNames = *new(xtc, bufEnd) Names(bufEnd,
                                                   pvMonitor->alias().c_str(), rawAlg,
                                                   m_para->detType.c_str(), m_para->serNo.c_str(), rawNamesId);
        std::string    fieldName;
        Name::DataType xtcType;
        int            rank;
        if (pvMonitor->getParams(fieldName, xtcType, rank, m_pool->bufferSize())) {
            return 1;
        }

        RawDef rawDef(fieldName, xtcType, rank);
        rawNames.add(xtc, bufEnd, rawDef);
        m_namesLookup[rawNamesId] = NameIndex(rawNames);

        // Create configuration object -> Only works for one PV per executable currently
        if (m_para->detType != "pv") {
            PyObject* funcDict = pyCheckErr(PyModule_GetDict(m_pyModule));
            const char* funcName = "pvadetector_config";
            PyObject* configFunc = pyCheckErr(PyDict_GetItemString(funcDict, funcName));

            PyObject* pyjsoncfg = pyCheckErr(PyObject_CallFunction(configFunc,
                                                                   "sssi",
                                                                   m_connectJson.c_str(),
                                                                   config_alias.c_str(),
                                                                   m_para->detName.c_str(),
                                                                   m_para->detSegment));

            // pvadetector_config returns None if no retrieval from configdb
            if (pyjsoncfg != Py_None) {
                char* buffer = new char[m_para->maxTrSize];
                const void* end = buffer + m_para->maxTrSize;

                Xtc& jsonxtc = *new (buffer, end) Xtc(TypeId(TypeId::Parent, 0));
                NamesId cfgNamesId(nodeId, ConfigNamesIndex + pvMonitor->id());
                if (translateJson2Xtc(pyjsoncfg, jsonxtc, end, cfgNamesId)) {
                    return -1;
                }

                if (jsonxtc.extent > m_para->maxTrSize) {
                    logging::critical("Config JSON (%u) too large for buffer (%u)!",
                                      jsonxtc.extent, m_para->maxTrSize);
                    abort();
                }

                logging::info("Adding config object for PV detector %s", m_para->detName.c_str());
                auto jsonXtcPayload = xtc.alloc(jsonxtc.sizeofPayload(), bufEnd);
                memcpy(jsonXtcPayload, (const void*) jsonxtc.payload(), jsonxtc.sizeofPayload());
                delete[] buffer;
            } else {
                logging::info("No config object for PV detector %s", m_para->detName.c_str());
            }
            Py_DECREF(pyjsoncfg);
        }
    }

    // Set up the names for PvDetector informational data
    Alg     infoAlg("pvdetinfo", 1, 0, 0);
    NamesId infoNamesId(nodeId, InfoNamesIndex);
    Names&  infoNames = *new(xtc, bufEnd) Names(bufEnd,
                                                ("pvdetinfo_" + m_para->detName).c_str(), infoAlg,
                                                "pvdetinfo", "detnum1234", infoNamesId);
    InfoDef infoDef(m_para->detName);
    infoNames.add(xtc, bufEnd, infoDef);
    m_namesLookup[infoNamesId] = NameIndex(infoNames);

    // add dictionary of information for each epics detname above.
    // first name is required to be "keys".  keys and values
    // are delimited by ",".
    CreateData cd(xtc, bufEnd, m_namesLookup, infoNamesId);
    std::string str = "";
    for (auto& pvMonitor : m_pvMonitors)
        str = str + pvMonitor->alias() + ",";
    cd.set_string(InfoDef::keys,    str.substr(0, str.length()-1).c_str());
    str = "";
    for (auto& pvMonitor : m_pvMonitors)
        str = str + pvMonitor->name() + "\n";
    cd.set_string(InfoDef::detName, str.substr(0, str.length()-1).c_str());

    return 0;
}

unsigned PvDetector::unconfigure()
{
    m_namesLookup.clear();   // erase all elements

    return 0;
}

void PvDetector::enable()
{
    m_running = true;
}

void PvDetector::disable()
{
    m_running = false;
}

void PvDetector::event_(Dgram& dgram, const void* bufEnd, const Xtc& pvXtc)
{
    NamesId namesId(nodeId, RawNamesIndex + pvXtc.src.value());
    CreateData cd(dgram.xtc, bufEnd, m_namesLookup, namesId);
    DataDsc* payload = reinterpret_cast<DataDsc*>(pvXtc.payload());
    void* data = &payload->data;
    Name& name = cd.nameindex().names().get(RawDef::field);
    if (name.rank()) {                  // Handle vectors and arrays, etc.
        uint32_t* shape = payload->shape;
        size_t size = pvXtc.extent - ((char*)data - (char*)payload); // Exclude shape info
        void* ptr = cd.get_ptr(); // Fetch a pointer to the next part of contiguous memory
        cd.set_array_shape(RawDef::field, shape); // Allocate the space before filling it
        memcpy(ptr, data, size);  // size is the same as the amount of space allocated
    }
    else {                              // Handle scalars
        switch (name.type()) {
            case Name::INT8:      cd.set_value(RawDef::field, *static_cast<int8_t  *>(data));  break;
            case Name::INT16:     cd.set_value(RawDef::field, *static_cast<int16_t *>(data));  break;
            case Name::INT32:
            case Name::ENUMVAL:
            case Name::ENUMDICT:  cd.set_value(RawDef::field, *static_cast<int32_t *>(data));  break;
            case Name::INT64:     cd.set_value(RawDef::field, *static_cast<int64_t *>(data));  break;
            case Name::UINT8:     cd.set_value(RawDef::field, *static_cast<uint8_t *>(data));  break;
            case Name::UINT16:    cd.set_value(RawDef::field, *static_cast<uint16_t*>(data));  break;
            case Name::UINT32:    cd.set_value(RawDef::field, *static_cast<uint32_t*>(data));  break;
            case Name::UINT64:    cd.set_value(RawDef::field, *static_cast<uint64_t*>(data));  break;
            case Name::FLOAT:     cd.set_value(RawDef::field, *static_cast<float   *>(data));  break;
            case Name::DOUBLE:    cd.set_value(RawDef::field, *static_cast<double  *>(data));  break;
            default: {
                logging::critical("Unsupported scalar type %d for %s\n", name.type(), name.name());
                abort();
                break;
            }
        }
    }
    dgram.xtc.damage.increase(pvXtc.damage.value());
}

// ---

PvDrp::PvDrp(PvParameters& para, MemPoolCpu& pool, PvDetector& det, ZmqContext& context) :
    DrpBase    (para, pool, det, context),
    m_para     (para),
    m_det      (det),
    m_pgp      (para, pool, &det),
    m_evtQueue (pool.nbuffers()),
    m_terminate(false)
{
}

std::string PvDrp::configure(const json& msg)
{
    std::string errorMsg = DrpBase::configure(msg);
    if (!errorMsg.empty()) {
        return errorMsg;
    }

    // Start the worker thread
    m_workerThread = std::thread{&PvDrp::_worker, this};

    return std::string();
}

unsigned PvDrp::unconfigure()
{
    DrpBase::unconfigure(); // TebContributor must be shut down before the worker

    // Stop the worker thread
    m_terminate.store(true, std::memory_order_release);
    if (m_workerThread.joinable()) {
        m_workerThread.join();
    }

    return 0;
}

int PvDrp::_setupMetrics(const std::shared_ptr<MetricExporter> exporter)
{
    std::map<std::string, std::string> labels{{"instrument", m_para.instrument},
                                              {"partition", std::to_string(m_para.partition)},
                                              {"detname", m_para.detName},
                                              {"detseg", std::to_string(m_para.detSegment)},
                                              {"alias", m_para.alias}};
    m_nEvents = 0;
    exporter->add("drp_event_rate", labels, MetricType::Rate,
                  [&](){return m_nEvents;});
    m_nUpdates = 0;
    exporter->add("drp_update_rate", labels, MetricType::Rate,
                  [&](){ uint64_t nUpdates = 0;
                         for (const auto& pvMonitor : m_det.pvMonitors())
                             nUpdates += pvMonitor->nUpdates();
                         m_nUpdates = nUpdates;
                         return m_nUpdates; });
    m_nMatch = 0;
    exporter->add("drp_match_count", labels, MetricType::Counter,
                  [&](){return m_nMatch;});
    m_nEmpty = 0;
    exporter->add("drp_empty_count", labels, MetricType::Counter,
                  [&](){return m_nEmpty;});
    m_nMissed = 0;
    exporter->add("drp_miss_count", labels, MetricType::Counter,
                  [&](){ uint64_t nMissed = 0;
                         for (const auto& pvMonitor : m_det.pvMonitors())
                             nMissed += pvMonitor->nMissed();
                         m_nMissed = nMissed;
                         return m_nMissed; });
    m_nTooOld = 0;
    exporter->add("drp_tooOld_count", labels, MetricType::Counter,
                  [&](){return m_nTooOld;});
    m_nTimedOut = 0;
    exporter->add("drp_timeout_count", labels, MetricType::Counter,
                  [&](){return m_nTimedOut;});
    m_timeDiff = 0;
    exporter->add("drp_time_diff", labels, MetricType::Gauge,
                  [&](){return m_timeDiff;});

    exporter->add("drp_worker_input_queue", labels, MetricType::Gauge,
                  [&](){return m_evtQueue.guess_size();});
    exporter->constant("drp_worker_queue_depth", labels, m_evtQueue.size());

    // Borrow this for awhile
    exporter->add("drp_worker_output_queue", labels, MetricType::Gauge,
                  [&](){return m_det.pvMonitors()[0]->pvQueue.guess_size();});

    // @todo: Support multiple PVs
    exporter->add("drp_pv_latency", labels, MetricType::Gauge,
                  [&](){return m_det.pvMonitors()[0]->latency();});

    exporter->add("drp_num_dma_ret", labels, MetricType::Gauge,
                  [&](){return m_pgp.nDmaRet();});
    exporter->add("drp_pgp_byte_rate", labels, MetricType::Rate,
                  [&](){return m_pgp.dmaBytes();});
    exporter->add("drp_dma_size", labels, MetricType::Gauge,
                  [&](){return m_pgp.dmaSize();});
    exporter->add("drp_th_latency", labels, MetricType::Gauge,
                  [&](){return m_pgp.latency();});
    exporter->add("drp_num_dma_errors", labels, MetricType::Gauge,
                  [&](){return m_pgp.nDmaErrors();});
    exporter->add("drp_num_no_common_rog", labels, MetricType::Gauge,
                  [&](){return m_pgp.nNoComRoG();});
    exporter->add("drp_num_missing_rogs", labels, MetricType::Gauge,
                  [&](){return m_pgp.nMissingRoGs();});
    exporter->add("drp_num_th_error", labels, MetricType::Gauge,
                  [&](){return m_pgp.nTmgHdrError();});
    exporter->add("drp_num_pgp_jump", labels, MetricType::Gauge,
                  [&](){return m_pgp.nPgpJumps();});
    exporter->add("drp_num_no_tr_dgram", labels, MetricType::Gauge,
                  [&](){return m_pgp.nNoTrDgrams();});

    return 0;
}

void PvDrp::_worker()
{
    m_terminate.store(false, std::memory_order_release);

    // Avoid running off the end of the word
    uint64_t mask = 1ul << (m_det.pvMonitors().size() - 1);
    uint64_t contract = mask | (mask - 1ul);

    const ms_t tmo{ m_para.kwargs.find("match_tmo_ms") != m_para.kwargs.end()            ?
                    std::stoul(const_cast<PvParameters&>(m_para).kwargs["match_tmo_ms"]) :
                    1500 };

    logging::info("Worker thread is starting with process ID %lu", syscall(SYS_gettid));

    // Reset counters to avoid 'jumping' errors on reconfigures
    pool.resetCounters();
    m_pgp.resetEventCounter();

    // (Re)initialize the queues
    m_evtQueue.startup();
    for (auto& pvMonitor : m_det.pvMonitors()) {
      pvMonitor->startup();
    }

    // Set up monitoring
    auto exporter = std::make_shared<MetricExporter>();
    if (exposer()) {
        exposer()->RegisterCollectable(exporter);

        if (_setupMetrics(exporter))  return;
    }

    while (true) {
        if (m_terminate.load(std::memory_order_relaxed)) {
            break;
        }

        uint32_t index;
        if (m_pgp.next(index)) {
            m_nEvents++;

            m_evtQueue.push({contract, index});

            _matchUp();
        }
        else {
            // If there are any PGP datagrams stacked up, try to match them
            // up with any PV updates that may have arrived
            _matchUp();

            // Time out older PVs and datagrams
            _timeout(tmo);

            // Time out batches for the TEB
            tebContributor().timeout();
        }
    }

    // Flush the DMA buffers
    m_pgp.flush();

    for (auto& pvMonitor : m_det.pvMonitors()) {
      pvMonitor->shutdown();
    }
    m_evtQueue.shutdown();

    if (exposer())  exporter.reset();

    logging::info("Worker thread finished");
}

void PvDrp::_matchUp()
{
    while (true) {
        if (m_evtQueue.is_empty())  break;
        Event& evt = m_evtQueue.front();

        EbDgram* evtDg = reinterpret_cast<EbDgram*>(pool.pebble[evt.index]);
        TransitionId::Value service = evtDg->service();
        if (service == TransitionId::L1Accept) {
            uint64_t remaining = evt.remaining;
            while (remaining) {
                unsigned id = __builtin_ffsl(remaining) - 1;
                remaining &= ~(1ull << id);

                auto& pvMonitor = m_det.pvMonitors()[id];

                Dgram* pvDg;
                if (!pvMonitor->pvQueue.peek(pvDg))  continue;

                m_timeDiff = evtDg->time.to_ns() - pvDg->time.to_ns();

                int result = _compare(evtDg->time, pvDg->time);

                logging::debug("PGP: %u.%09d %c PV: %u.%09d, PGP - PV: %12ld ns, "
                               "pid %014lx, svc %2d, PGP age %ld ms",
                               evtDg->time.seconds(), evtDg->time.nanoseconds(),
                               result == 0 ? '=' : (result < 0 ? '<' : '>'),
                               pvDg->time.seconds(), pvDg->time.nanoseconds(),
                               m_timeDiff, evtDg->pulseId(), evtDg->service(),
                               Eb::latency<ms_t>(evtDg->time));

                if      (result == 0) { _tEvtEqPv(pvMonitor, *evtDg, *pvDg);  evt.remaining &= ~(1ull << id); }
                else if (result  < 0) { _tEvtLtPv(pvMonitor, *evtDg, *pvDg);  evt.remaining &= ~(1ull << id); }
                else                  { _tEvtGtPv(pvMonitor, *evtDg, *pvDg); }
            }
            if (evt.remaining)  break;  // Break so the timeout routine can run
        }
        else {
            // Find the transition dgram in the pool
            EbDgram* trDg = pool.transitionDgrams[evt.index];
            if (trDg)                   // nullptr can happen during shutdown
                _handleTransition(*evtDg, *trDg);
        }

        Event event;
        m_evtQueue.try_pop(event);      // Actually consume the element
        assert(event.index == evt.index);

        _sendToTeb(*evtDg, evt.index);
    }
}

void PvDrp::_handleTransition(EbDgram& evtDg, EbDgram& trDg)
{
    // Initialize the transition dgram's header
    trDg = evtDg;

    TransitionId::Value service = trDg.service();
    if (service != TransitionId::SlowUpdate) {
        // copy the temporary xtc created on phase 1 of the transition
        // into the real location
        Xtc& trXtc = m_det.transitionXtc();
        trDg.xtc = trXtc; // Preserve header info, but allocate to check fit
        const void* bufEnd = (char*)&trDg + m_para.maxTrSize;
        auto payload = trDg.xtc.alloc(trXtc.sizeofPayload(), bufEnd);
        memcpy(payload, (const void*)trXtc.payload(), trXtc.sizeofPayload());

        // Enable/disable PV updates
        if (service == TransitionId::Enable) {
            m_det.enable();
        }
        else if (service == TransitionId::Disable) {
            m_det.disable();
        }
    }
}

void PvDrp::_tEvtEqPv(std::shared_ptr<PvMonitor> pvMonitor, EbDgram& evtDg, const Dgram& pvDg)
{
    auto bufEnd = (char*)&evtDg + pool.pebble.bufferSize();
    m_det.event_(evtDg, bufEnd, pvDg.xtc);

    ++m_nMatch;
    logging::debug("PV matches PGP!!  "
                   "TimeStamps: PV %u.%09u == PGP %u.%09u",
                   pvDg.time.seconds(), pvDg.time.nanoseconds(),
                   evtDg.time.seconds(), evtDg.time.nanoseconds());

    Dgram* dgram;
    pvMonitor->pvQueue.try_pop(dgram);     // Actually consume the element
    pvMonitor->bufferFreelist.push(dgram); // Return buffer to freelist
}

void PvDrp::_tEvtLtPv(std::shared_ptr<PvMonitor> pvMonitor, EbDgram& evtDg, const Dgram& pvDg)
{
    // Because PVs show up in time order, when the most recent PV is younger
    // than the PGP event (t(PV) > t(PGP)), we know that no older PV will show
    // up to possibly match the PGP timestamp.  Thus, mark the event damaged and
    // leave the PV on the queue to perhaps be matched with a newer PGP event.
    evtDg.xtc.damage.increase(Damage::MissingData);

    ++m_nEmpty;
    logging::debug("PV too young!!    "
                   "TimeStamps: PGP %u.%09u < PV %u.%09u",
                   evtDg.time.seconds(), evtDg.time.nanoseconds(),
                   pvDg.time.seconds(), pvDg.time.nanoseconds());
}

void PvDrp::_tEvtGtPv(std::shared_ptr<PvMonitor> pvMonitor, EbDgram& evtDg, const Dgram& pvDg)
{
    // Because PGP events show up in time order, when the most recent PV is older
    // than the PGP event (t(PV) < t(PGP)), we know that no older PGP event will
    // show up to match the PV's timestamp.  Thus, the PV is discarded and
    // the PGP event is left on the queue to perhaps be matched with a newer PV.
    ++m_nTooOld;
    logging::debug("PV too old!!      "
                   "TimeStamps: PGP %u.%09u > PV %u.%09u [0x%08x%04x.%05x > 0x%08x%04x.%05x]",
                   evtDg.time.seconds(), evtDg.time.nanoseconds(),
                   pvDg.time.seconds(), pvDg.time.nanoseconds(),
                   evtDg.time.seconds(), (evtDg.time.nanoseconds()>>16)&0xfffe, evtDg.time.nanoseconds()&0x1ffff,
                   pvDg.time.seconds(), (pvDg.time.nanoseconds()>>16)&0xfffe, pvDg.time.nanoseconds()&0x1ffff);

    Dgram* dgram;
    pvMonitor->pvQueue.try_pop(dgram);     // Actually consume the element
    pvMonitor->bufferFreelist.push(dgram); // Return buffer to freelist
}

void PvDrp::_timeout(ms_t timeout)
{
    // Try to clear out as many of the older queue entries as we can in one go
    while (true) {
        // Time out older PV updates
        for (auto& pvMonitor : m_det.pvMonitors()) {
            pvMonitor->timeout(m_pgp, timeout);
        }

        // Time out older pending PGP datagrams
        Event event;
        if (!m_evtQueue.peek(event))  break;

        EbDgram& dgram = *reinterpret_cast<EbDgram*>(pool.pebble[event.index]);
        if (m_pgp.age(dgram.time) < timeout)  break;

        logging::debug("Event timed out!! "
                       "TimeStamp:  %u.%09u [0x%08x%04x.%05x], age %ld ms, svc %u",
                       dgram.time.seconds(), dgram.time.nanoseconds(),
                       dgram.time.seconds(), (dgram.time.nanoseconds()>>16)&0xfffe, dgram.time.nanoseconds()&0x1ffff,
                       Eb::latency<ms_t>(dgram.time), dgram.service());

        if (dgram.service() == TransitionId::L1Accept) {
            // No PV data so mark event as damaged
            dgram.xtc.damage.increase(Damage::TimedOut);
            ++m_nTimedOut;
        }
        else {
            // Find the transition dgram in the pool
            EbDgram* trDg = pool.transitionDgrams[event.index];
            if (trDg)                   // nullptr can happen during shutdown
                _handleTransition(dgram, *trDg);
        }

        Event evt;
        m_evtQueue.try_pop(evt);        // Actually consume the element
        assert(evt.index == event.index);

        _sendToTeb(dgram, event.index);
    }
}

void PvDrp::_sendToTeb(const EbDgram& dgram, uint32_t index)
{
    // Make sure the datagram didn't get too big
    const size_t size = sizeof(dgram) + dgram.xtc.sizeofPayload();
    const size_t maxSize = (dgram.service() == TransitionId::L1Accept)
                         ? pool.pebble.bufferSize()
                         : m_para.maxTrSize;
    if (size > maxSize) {
        logging::critical("%s Dgram of size %zd overflowed buffer of size %zd", TransitionId::name(dgram.service()), size, maxSize);
        throw "Dgram overflowed buffer";
    }

    auto l3InpBuf = tebContributor().fetch(index);
    EbDgram* l3InpDg = new(l3InpBuf) EbDgram(dgram);
    if (l3InpDg->isEvent()) {
        auto trgPrimitive = triggerPrimitive();
        if (trgPrimitive) { // else this DRP doesn't provide input
            const void* bufEnd = (char*)l3InpDg + sizeof(*l3InpDg) + trgPrimitive->size();
            trgPrimitive->event(pool, index, dgram.xtc, l3InpDg->xtc, bufEnd); // Produce
        }
    }
    tebContributor().process(l3InpDg);
}

// ---

PvApp::PvApp(PvParameters& para) :
    CollectionApp(para.collectionHost, para.partition, "drp", para.alias),
    m_para(para),
    m_pool(para),
    m_unconfigure(false)
{
    Py_Initialize();                    // for use by configuration

    m_det = std::make_unique<PvDetector>(m_para, m_pool);
    m_drp = std::make_unique<PvDrp>(para, m_pool, *m_det, context());

    logging::info("Ready for transitions");
}

PvApp::~PvApp()
{
    // Try to take things down gracefully when an exception takes us off the
    // normal path so that the most chance is given for prints to show up
    handleReset(json({}));

    Py_Finalize();                      // for use by configuration
}

void PvApp::_disconnect()
{
    m_drp->disconnect();
    m_det->disconnect();
}

void PvApp::_unconfigure()
{
    m_drp->pool.shutdown();        // Release Tr buffer pool
    m_drp->unconfigure();
    m_det->unconfigure();
    m_unconfigure = false;
}

json PvApp::connectionInfo(const json& msg)
{
    std::string ip = m_para.kwargs.find("ep_domain") != m_para.kwargs.end()
                   ? getNicIp(m_para.kwargs["ep_domain"])
                   : getNicIp(m_para.kwargs["forceEnet"] == "yes");
    logging::debug("nic ip  %s", ip.c_str());
    json body = {{"connect_info", {{"nic_ip", ip}}}};
    json info = static_cast<Detector&>(*m_det).connectionInfo(msg);
    body["connect_info"].update(info);
    json bufInfo = m_drp->connectionInfo(ip);
    body["connect_info"].update(bufInfo);
    return body;
}

void PvApp::connectionShutdown()
{
    static_cast<Detector&>(*m_det).connectionShutdown();
    m_drp->shutdown();
}

void PvApp::_error(const std::string& which, const json& msg, const std::string& errorMsg)
{
    json body = json({});
    body["err_info"] = errorMsg;
    json answer = createMsg(which, msg["header"]["msg_id"], getId(), body);
    reply(answer);
}

void PvApp::handleConnect(const json& msg)
{
    std::string errorMsg = m_drp->connect(msg, getId());
    if (!errorMsg.empty()) {
        logging::error(("DrpBase::connect: " + errorMsg).c_str());
        _error("connect", msg, errorMsg);
        return;
    }

    m_det->nodeId = m_drp->nodeId();
    unsigned rc = m_det->connect(msg, std::to_string(getId()), errorMsg);
    if (!errorMsg.empty()) {
        if (!rc) {
            logging::warning(("PvDetector::connect: " + errorMsg).c_str());
            json warning = createAsyncWarnMsg(m_para.alias, errorMsg);
            reply(warning);
        }
        else {
            logging::error(("PvDetector::connect: " + errorMsg).c_str());
            _error("connect", msg, errorMsg);
            return;
        }
    }

    json body = json({});
    json answer = createMsg("connect", msg["header"]["msg_id"], getId(), body);
    reply(answer);
}

void PvApp::handleDisconnect(const json& msg)
{
    // Carry out the queued Unconfigure, if there was one
    if (m_unconfigure) {
        _unconfigure();
    }

    _disconnect();

    json body = json({});
    reply(createMsg("disconnect", msg["header"]["msg_id"], getId(), body));
}

void PvApp::handlePhase1(const json& msg)
{
    std::string key = msg["header"]["key"];
    logging::debug("handlePhase1 for %s in PvDetectorApp", key.c_str());

    Xtc& xtc = m_det->transitionXtc();
    xtc = {{TypeId::Parent, 0}, {m_det->nodeId}};
    auto bufEnd = m_det->trXtcBufEnd();

    json phase1Info{ "" };
    if (msg.find("body") != msg.end()) {
        if (msg["body"].find("phase1Info") != msg["body"].end()) {
            phase1Info = msg["body"]["phase1Info"];
        }
    }

    json body = json({});

    if (key == "configure") {
        if (m_unconfigure) {
            _unconfigure();
        }

        // Configure the detector first
        std::string config_alias = msg["body"]["config_alias"];
        unsigned error = m_det->configure(config_alias, xtc, bufEnd);
        if (error) {
            std::string errorMsg = "Failed transition phase 1";
            logging::error("%s", errorMsg.c_str());
            _error(key, msg, errorMsg);
            return;
        }

        // Next, configure the DRP
        std::string errorMsg = m_drp->configure(msg);
        if (!errorMsg.empty()) {
            errorMsg = "Phase 1 error: " + errorMsg;
            logging::error("%s", errorMsg.c_str());
            _error(key, msg, errorMsg);
            return;
        }

        m_drp->runInfoSupport(xtc, bufEnd, m_det->namesLookup());
        m_drp->chunkInfoSupport(xtc, bufEnd, m_det->namesLookup());
    }
    else if (key == "unconfigure") {
        // "Queue" unconfiguration until after phase 2 has completed
        m_unconfigure = true;
    }
    else if (key == "beginrun") {
        RunInfo runInfo;
        std::string errorMsg = m_drp->beginrun(phase1Info, runInfo);
        if (!errorMsg.empty()) {
            logging::error("%s", errorMsg.c_str());
            _error(key, msg, errorMsg);
            return;
        }

        m_drp->runInfoData(xtc, bufEnd, m_det->namesLookup(), runInfo);
    }
    else if (key == "endrun") {
        std::string errorMsg = m_drp->endrun(phase1Info);
        if (!errorMsg.empty()) {
            logging::error("%s", errorMsg.c_str());
            _error(key, msg, errorMsg);
            return;
        }
    }
    else if (key == "enable") {
        bool chunkRequest;
        ChunkInfo chunkInfo;
        std::string errorMsg = m_drp->enable(phase1Info, chunkRequest, chunkInfo);
        if (!errorMsg.empty()) {
            body["err_info"] = errorMsg;
            logging::error("%s", errorMsg.c_str());
        } else if (chunkRequest) {
            logging::debug("handlePhase1 enable found chunkRequest");
            m_drp->chunkInfoData(xtc, bufEnd, m_det->namesLookup(), chunkInfo);
        }
        unsigned error = static_cast<Detector&>(*m_det).enable(xtc, bufEnd, phase1Info);
        if (error) {
            std::string errorMsg = "Phase 1 error in Detector::enable()";
            body["err_info"] = errorMsg;
            logging::error("%s", errorMsg.c_str());
        }
        logging::debug("handlePhase1 enable complete");
    }

    json answer = createMsg(key, msg["header"]["msg_id"], getId(), body);
    reply(answer);
}

void PvApp::handleReset(const json& msg)
{
    unsubscribePartition();    // ZMQ_UNSUBSCRIBE
    _unconfigure();
    _disconnect();
    connectionShutdown();
}

} // namespace Drp


int main(int argc, char* argv[])
{
    Drp::PvParameters para;
    std::string kwargs_str;
    std::string filename;
    int c;
    while((c = getopt(argc, argv, "p:o:l:D:S:C:d:u:k:P:M:01f:v")) != EOF) {
        switch(c) {
            case 'p':
                para.partition = std::stoi(optarg);
                break;
            case 'o':
                para.outputDir = optarg;
                break;
            case 'l':
                para.laneMask = std::stoul(optarg, nullptr, 16);
                break;
            case 'D':
                para.detType = optarg;  // Defaults to 'pv'
                break;
            case 'S':
                para.serNo = optarg;
                break;
            case 'u':
                para.alias = optarg;
                break;
            case 'C':
                para.collectionHost = optarg;
                break;
            case 'd':
                para.device = optarg;
                break;
            case 'k':
                kwargs_str = kwargs_str.empty()
                           ? optarg
                           : kwargs_str + "," + optarg;
                break;
            case 'P':
                para.instrument = optarg;
                break;
            case 'M':
                para.prometheusDir = optarg;
                break;
            //  Indicate level of timestamp matching (ugh)
            case '0':
                tsMatchDegree = 0;
                break;
            case '1':
                fprintf(stderr, "Option -1 is disabled\n");  exit(EXIT_FAILURE);
                tsMatchDegree = 1;
                break;
            case 'f':
                filename = optarg;
                break;
            case 'v':
                ++para.verbose;
                break;
            default:
                return 1;
        }
    }

    switch (para.verbose) {
        case 0:  logging::init(para.instrument.c_str(), LOG_INFO);   break;
        default: logging::init(para.instrument.c_str(), LOG_DEBUG);  break;
    }
    logging::info("logging configured");
    if (para.instrument.empty()) {
        logging::warning("-P: instrument name is missing");
    }
    // Check required parameters
    if (para.partition == unsigned(-1)) {
        logging::critical("-p: partition is mandatory");
        return 1;
    }
    if (para.device.empty()) {
        logging::critical("-d: device is mandatory");
        return 1;
    }
    if (para.alias.empty()) {
        logging::critical("-u: alias is mandatory");
        return 1;
    }

    // Only one lane is supported by this DRP
    if (std::bitset<PGP_MAX_LANES>(para.laneMask).count() != 1) {
        logging::critical("-l: lane mask must have only 1 bit set");
        return 1;
    }

    // Allow detType to be overridden, but generally, psana will expect 'pv'
    if (para.detType.empty()) {
      para.detType = "pv";
    }

    // Alias must be of form <detName>_<detSegment>
    size_t found = para.alias.rfind('_');
    if ((found == std::string::npos) || !isdigit(para.alias.back())) {
        logging::critical("-u: alias must have _N suffix");
        return 1;
    }
    para.detName = para.alias.substr(0, found);
    para.detSegment = std::stoi(para.alias.substr(found+1, para.alias.size()));

    // Read PVs from a file, if specified
    unsigned nPVs = 0;
    if (!filename.empty()) {
        unsigned lineNo = 0;
        std::ifstream pvFile(filename.c_str());
        if (!pvFile) {
            logging::critical("Failed to open file %s", filename.c_str());
            return 1;
        }
        logging::debug("Processing file %s", filename.c_str());
        while (!pvFile.eof()) {
            std::string line;
            std::getline(pvFile, line);
            ++lineNo;

            // Remove whitespaces
            std::string pvSpec = psalg::strip(line);

            // Skip blank and commented out lines
            if (pvSpec.size() == 0 || pvSpec[0] == '#')  continue;

            // Remove trailing comments
            pvSpec = pvSpec.substr(0, pvSpec.find("#", 0));

            logging::debug("line %u, PV spec: '%s'\n", lineNo, pvSpec.c_str());

            para.pvSpecs.push_back({pvSpec});
            ++nPVs;
        }
    }
    // Also read PVs from the command line
    while (optind < argc) {
        para.pvSpecs.push_back({argv[optind++]});
        ++nPVs;
    }
    if (para.pvSpecs.empty()) {
        // Provider is "pva" (default) or "ca"
        logging::critical("At least one PV is required");
        return 1;
    }
    if (nPVs > 64) { // Limit is set by the number of bits in contract/remaining variables
        logging::critical("Found %u PVs when max supported is %u", nPVs, 64);
        return 1;
    }

    para.maxTrSize = 256 * 1024;
    try {
        get_kwargs(kwargs_str, para.kwargs);
        for (const auto& kwargs : para.kwargs) {
            if (kwargs.first == "forceEnet")      continue;
            if (kwargs.first == "ep_fabric")      continue;
            if (kwargs.first == "ep_domain")      continue;
            if (kwargs.first == "ep_provider")    continue;
            if (kwargs.first == "sim_length")     continue;  // XpmDetector
            if (kwargs.first == "timebase")       continue;  // XpmDetector
            if (kwargs.first == "pebbleBufSize")  continue;  // DrpBase
            if (kwargs.first == "pebbleBufCount") continue;  // DrpBase
            if (kwargs.first == "batching")       continue;  // DrpBase
            if (kwargs.first == "directIO")       continue;  // DrpBase
            if (kwargs.first == "pva_addr")       continue;  // DrpBase
            if (kwargs.first == "firstdim")       continue;
            if (kwargs.first == "match_tmo_ms")   continue;
            if (kwargs.first == "data_vsn")       continue;
            logging::critical("Unrecognized kwarg '%s=%s'\n",
                              kwargs.first.c_str(), kwargs.second.c_str());
            return 1;
        }

        Drp::PvApp(para).run();
        return 0;
    }
    catch (std::exception& e)  { logging::critical("%s", e.what()); }
    catch (std::string& e)     { logging::critical("%s", e.c_str()); }
    catch (char const* e)      { logging::critical("%s", e); }
    catch (...)                { logging::critical("Default exception"); }
    return EXIT_FAILURE;
}
